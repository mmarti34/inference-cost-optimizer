import base64
import re
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key

router = APIRouter()

class PromptPayload(BaseModel):
    user_id: str
    org_id: str
    provider: str
    model: str
    prompt: str

class VisionPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    image_url: str  # data:image/...;base64,... or https URL (we use base64 for Anthropic)
    prompt_id: str

def _parse_data_url(image_url: str) -> tuple[str, str]:
    """Return (media_type, base64_data). Supports data:image/jpeg;base64,xxx."""
    s = (image_url or "").strip()
    if s.startswith("data:"):
        m = re.match(r"data:([^;]+);base64,(.+)", s, re.DOTALL)
        if m:
            return (m.group(1).strip() or "image/jpeg", m.group(2))
    return ("image/jpeg", "")

def handle_vision(payload: VisionPayload) -> dict:
    """Vision: image + prompt -> text. Image as data URL or we pass URL (Anthropic needs base64)."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "anthropic")
    client = Anthropic(api_key=api_key)
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    media_type, b64 = _parse_data_url(image_url)
    if not b64 and image_url.startswith("http"):
        import requests
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        b64 = base64.b64encode(r.content).decode("ascii")
        media_type = "image/jpeg"
    elif not b64:
        raise HTTPException(status_code=400, detail="Vision image_url must be data:...;base64,... or https URL")
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        {"type": "text", "text": (payload.prompt or "Describe this image.").strip() or "Describe this image."},
    ]
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=(payload.model or "claude-sonnet-4-5-20250929").strip() or "claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        reply = response.content[0].text
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        pricing = get_pricing("anthropic", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log_usage(getattr(payload, "user_id", None), "Anthropic", payload.model, payload.prompt[:200], reply[:200], getattr(payload, "prompt_id", ""), input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, cost_usd=cost_usd)
        return {"response": reply, "output": reply, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "cost_usd": cost_usd, "latency_ms": elapsed_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic vision failed: {e}")

def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    client = Anthropic(api_key=api_key)

    try:
        _t0_provider = time.perf_counter()
        response = client.messages.create(
            model=payload.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": payload.prompt}]
        )
        _provider_latency_ms = int((time.perf_counter() - _t0_provider) * 1000)

        reply = response.content[0].text

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        pricing = get_pricing("anthropic", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        log_usage(
            user_id=payload.user_id,
            provider="Anthropic",
            model=payload.model,
            prompt=payload.prompt,
            response=reply,
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
        raise HTTPException(status_code=500, detail=f"Anthropic call failed: {str(e)}")
