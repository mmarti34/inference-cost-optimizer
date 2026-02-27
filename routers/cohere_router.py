from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cohere
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key

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
    """Vision: image + prompt -> text. Cohere Command A Vision uses messages with content array."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "cohere")
    client = cohere.Client(api_key=api_key)
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    prompt = (payload.prompt or "Describe this image.").strip() or "Describe this image."
    model = (payload.model or "command-a-vision-07-2025").strip() or "command-a-vision-07-2025"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}]
    try:
        response = client.chat(model=model, messages=messages)
        reply = response.text
        input_tokens = getattr(response, "token_count", 0)
        output_tokens = getattr(response, "generation_token_count", 0)
        total_tokens = input_tokens + output_tokens
        pricing = get_pricing("cohere", model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
        log_usage(payload.org_id, "Cohere", model, prompt[:200], reply[:200], payload.prompt_id, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, cost_usd=cost_usd)
        return {"response": reply, "output": reply, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "cost_usd": cost_usd, "latency_ms": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere vision failed: {e}")


class EmbeddingPayload(BaseModel):
    org_id: str
    text: str
    model: str
    prompt_id: str


def handle_embedding(payload: EmbeddingPayload) -> dict:
    """Embedding via Cohere embed-english-v3.0 or similar. Returns output (JSON list), cost_usd, latency_ms."""
    import json
    import time
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "cohere")
    client = cohere.Client(api_key=api_key)
    model = (payload.model or "embed-english-v3.0").strip() or "embed-english-v3.0"
    text = (payload.text or "").strip() or " "
    start = time.perf_counter()
    try:
        response = client.embed(texts=[text], model=model, input_type="search_document")
        emb = response.embeddings[0] if response.embeddings else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere embedding failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    tokens = getattr(response, "meta", None) and getattr(response.meta, "billed_units", None)
    num_tokens = tokens if isinstance(tokens, (int, float)) else len(text) // 4
    pricing = get_pricing("cohere", model)
    cost_usd = (num_tokens / 1000.0) * (pricing.get("input") or 0.0001)
    log_usage(payload.org_id, "Cohere", model, payload.text[:200], f"[{len(emb)}d]", payload.prompt_id, cost_usd=cost_usd)
    return {"output": json.dumps(emb), "embedding": list(emb) if hasattr(emb, "__iter__") and not isinstance(emb, str) else emb, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    client = cohere.Client(api_key)

    try:
        # 2. Send prompt
        response = client.chat(
            model=payload.model,
            message=payload.prompt
        )
        reply = response.text

        # 3. Extract token usage (Cohere returns token_count and generation_token_count)
        input_tokens = getattr(response, "token_count", 0)
        output_tokens = getattr(response, "generation_token_count", 0)
        total_tokens = input_tokens + output_tokens

        # 4. Pricing
        pricing = get_pricing("cohere", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        # 5. Log usage
        log_usage(
            payload.org_id,
            "Cohere",
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
            "cost_usd": cost_usd
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere call failed: {str(e)}")
