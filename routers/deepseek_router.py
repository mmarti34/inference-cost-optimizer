from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import supabase
from utils.encryption import decrypt_api_key
from utils.usage_logger import log_usage
import json
from utils.openai_compatible import call_openai_compatible, call_openai_compatible_embeddings

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


class EmbeddingPayload(BaseModel):
    org_id: str
    text: str
    model: str
    prompt_id: str


def handle_embedding(payload: EmbeddingPayload) -> dict:
    """Embedding via DeepSeek OpenAI-compatible /embeddings."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "deepseek")
    model = (payload.model or "deepseek-embedding-v2").strip() or "deepseek-embedding-v2"
    text = (payload.text or "").strip() or " "
    out = call_openai_compatible_embeddings("https://api.deepseek.com", api_key, "deepseek", model, text)
    log_usage("", "DeepSeek", model, payload.text[:200], f"[{len(out.get('embedding', []))}d]", payload.prompt_id, cost_usd=out["cost_usd"], org_id=payload.org_id)
    return {"output": json.dumps(out.get("embedding", [])), "embedding": out.get("embedding", []), "cost_usd": out["cost_usd"], "latency_ms": out["latency_ms"]}


def handle_vision(payload: VisionPayload) -> dict:
    """Vision: image + prompt -> text. OpenAI-compatible content array (DeepSeek VL)."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "deepseek")
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    prompt = (payload.prompt or "Describe this image.").strip() or "Describe this image."
    model = (payload.model or "deepseek-chat").strip() or "deepseek-chat"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]},
    ]
    out = call_openai_compatible("https://api.deepseek.com", api_key, "deepseek", model, messages)
    log_usage("", "DeepSeek", model, prompt[:200], out["response"], payload.prompt_id, input_tokens=out["input_tokens"], output_tokens=out["output_tokens"], total_tokens=out["total_tokens"], cost_usd=out["cost_usd"], org_id=payload.org_id)
    return {"response": out["response"], "output": out["response"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"], "cost_usd": out["cost_usd"], "latency_ms": out.get("latency_ms", 0)}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    out = call_openai_compatible(
        "https://api.deepseek.com",
        api_key,
        "deepseek",
        payload.model,
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": payload.prompt}],
    )
    log_usage(
        "", "DeepSeek", payload.model, payload.prompt, out["response"], payload.prompt_id,
        input_tokens=out["input_tokens"], output_tokens=out["output_tokens"], total_tokens=out["total_tokens"], cost_usd=out["cost_usd"], org_id=payload.org_id
    )
    return {"status": "success", "response": out["response"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"], "cost_usd": out["cost_usd"], "provider_latency_ms": out.get("provider_latency_ms")}
