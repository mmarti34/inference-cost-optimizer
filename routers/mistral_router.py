import json
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mistralai.client import MistralClient
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key
from utils.openai_compatible import call_openai_compatible_embeddings

router = APIRouter()

class PromptPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    prompt_id: str

class VisionPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    image_url: str
    prompt_id: str

def handle_vision(payload: VisionPayload) -> dict:
    """Vision: image + prompt -> text. Mistral uses OpenAI-style content array."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "mistral")
    client = MistralClient(api_key=api_key)
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    content = [
        {"type": "text", "text": (payload.prompt or "Describe this image.").strip() or "Describe this image."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    try:
        response = client.chat(
            model=(payload.model or "mistral-small-3.2").strip() or "mistral-small-3.2",
            messages=[{"role": "user", "content": content}],
        )
        reply = response.choices[0].message.content
        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)
        total_tokens = input_tokens + output_tokens
        pricing = get_pricing("mistral", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
        log_usage(payload.org_id, "Mistral", payload.model, payload.prompt[:200], reply[:200], payload.prompt_id, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, cost_usd=cost_usd)
        return {"response": reply, "output": reply, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "cost_usd": cost_usd, "latency_ms": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral vision failed: {e}")


class EmbeddingPayload(BaseModel):
    org_id: str
    text: str
    model: str
    prompt_id: str


def handle_embedding(payload: EmbeddingPayload) -> dict:
    """Embedding via Mistral mistral-embed. Returns output (JSON list), cost_usd, latency_ms."""
    import json
    import time
    import httpx
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "mistral")
    model = (payload.model or "mistral-embed").strip() or "mistral-embed"
    text = (payload.text or "").strip() or " "
    start = time.perf_counter()
    try:
        r = httpx.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": [text]},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral embedding failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    emb = data.get("data", [{}])[0].get("embedding", []) if data.get("data") else []
    usage = data.get("usage", {})
    num_tokens = usage.get("total_tokens", len(text) // 4)
    pricing = get_pricing("mistral", model)
    cost_usd = (num_tokens / 1000.0) * (pricing.get("input") or 0.0001)
    log_usage(payload.org_id, "Mistral", model, payload.text[:200], f"[{len(emb)}d]", payload.prompt_id, cost_usd=cost_usd)
    return {"output": json.dumps(emb), "embedding": emb, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    client = MistralClient(api_key=api_key)

    try:
        _t0_provider = time.perf_counter()
        response = client.chat(
            model=payload.model,
            messages=[{"role": "user", "content": payload.prompt}]
        )
        _provider_latency_ms = int((time.perf_counter() - _t0_provider) * 1000)

        reply = response.choices[0].message.content

        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        pricing = get_pricing("mistral", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        log_usage(
            payload.org_id,
            "Mistral",
            payload.model,
            payload.prompt,
            reply,
            payload.prompt_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd
        )

        return {
            "status": "success",
            "response": reply,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "provider_latency_ms": _provider_latency_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral call failed: {str(e)}")
