import base64
import time
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import supabase
from utils.encryption import decrypt_api_key
from utils.usage_logger import log_usage
from utils.pricing import get_non_token_rate
from utils.openai_compatible import call_openai_compatible, call_openai_compatible_with_tools

router = APIRouter()

# Non-token prices (images, audio, embeddings) live in shared/providers.json
# under non_token_rates.groq, read via get_non_token_rate(). The second
# argument is the historical constant, kept as a fallback so behaviour is
# unchanged if the config section is missing.

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
    """Vision: image + prompt -> text. OpenAI-compatible content array."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "groq")
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    prompt = (payload.prompt or "Describe this image.").strip() or "Describe this image."
    model = (payload.model or "llama-3.2-90b-vision-preview").strip() or "llama-3.2-90b-vision-preview"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]},
    ]
    out = call_openai_compatible("https://api.groq.com/openai/v1", api_key, "groq", model, messages)
    log_usage("", "Groq", model, prompt[:200], out["response"], payload.prompt_id, input_tokens=out["input_tokens"], output_tokens=out["output_tokens"], total_tokens=out["total_tokens"], cost_usd=out["cost_usd"], org_id=payload.org_id)
    return {"response": out["response"], "output": out["response"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"], "cost_usd": out["cost_usd"], "latency_ms": out.get("latency_ms", 0)}


class TTSPayload(BaseModel):
    org_id: str
    text: str
    model: str
    voice: str
    prompt_id: str


def handle_tts(payload: TTSPayload) -> dict:
    """TTS via Groq Orpheus/PlayAI. Returns base64 data URL (audio/wav), cost_usd, latency_ms."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "groq")
    model = (payload.model or "canopylabs/orpheus-v1-english").strip() or "canopylabs/orpheus-v1-english"
    voice = (payload.voice or "austin").strip() or "austin"
    text = (payload.text or "").strip() or " "
    start = time.perf_counter()
    try:
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": text, "voice": voice, "response_format": "wav"},
            timeout=60,
        )
        r.raise_for_status()
        audio_bytes = r.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq TTS failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    data_url = f"data:audio/wav;base64,{b64}"
    cost_usd = (len(text) / 1000.0) * get_non_token_rate("groq", "audio.tts.per_1k_chars", 0.022)
    log_usage(None, "Groq", model, text[:200], "[audio]", payload.prompt_id, cost_usd=cost_usd)
    return {"output": data_url, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    out = call_openai_compatible(
        "https://api.groq.com/openai/v1",
        api_key,
        "groq",
        payload.model,
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": payload.prompt}],
    )
    log_usage(
        "", "Groq", payload.model, payload.prompt, out["response"], payload.prompt_id,
        input_tokens=out["input_tokens"], output_tokens=out["output_tokens"], total_tokens=out["total_tokens"], cost_usd=out["cost_usd"], org_id=payload.org_id
    )
    return {"status": "success", "response": out["response"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"], "cost_usd": out["cost_usd"], "provider_latency_ms": out.get("provider_latency_ms")}


def handle_prompt_with_tools(payload, tools: list[dict], *, system_message: str = "", max_iterations: int = 5, tool_executor=None, can_parallelize_tool=None):
    """Tool calling via Groq (OpenAI-compatible)."""
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, "groq")
    return call_openai_compatible_with_tools(
        "https://api.groq.com/openai/v1", api_key, "groq", payload.model,
        payload.prompt, tools, tool_executor=tool_executor,
        system_message=system_message, max_iterations=max_iterations,
        prompt_id=payload.prompt_id, org_id=payload.org_id,
        can_parallelize_tool=can_parallelize_tool,
    )
