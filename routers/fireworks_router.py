import base64
import io
import time
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import supabase
from utils.encryption import decrypt_api_key
from utils.usage_logger import log_usage
import json
from utils.openai_compatible import call_openai_compatible, call_openai_compatible_embeddings

router = APIRouter()

FLUX_COST_PER_IMAGE = 0.0014
FIREWORKS_STT_COST_PER_MIN = 0.002
# Whisper-style limit; reject oversized audio before calling API.
WHISPER_MAX_FILE_BYTES = 24 * 1024 * 1024


class PromptPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    prompt_id: str


class ImageGenerationPayload(BaseModel):
    org_id: str
    prompt: str
    model: str
    prompt_id: str
    negative_prompt: str | None = None


class STTPayload(BaseModel):
    org_id: str
    audio_base64: str
    model: str
    prompt_id: str
    prompt: str | None = None


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
    """Embedding via Fireworks OpenAI-compatible /embeddings."""
    api_key = _get_fireworks_api_key(payload.org_id)
    model = (payload.model or "nomic-ai/nomic-embed-text-v1.5").strip() or "nomic-ai/nomic-embed-text-v1.5"
    text = (payload.text or "").strip() or " "
    out = call_openai_compatible_embeddings("https://api.fireworks.ai/inference/v1", api_key, "fireworks", model, text)
    log_usage(None, "Fireworks", model, payload.text[:200], f"[{len(out.get('embedding', []))}d]", payload.prompt_id, cost_usd=out["cost_usd"], org_id=payload.org_id)
    return {"output": json.dumps(out.get("embedding", [])), "embedding": out.get("embedding", []), "cost_usd": out["cost_usd"], "latency_ms": out["latency_ms"]}


def handle_vision(payload: VisionPayload) -> dict:
    """Vision: image + prompt -> text. OpenAI-compatible content array."""
    api_key = _get_fireworks_api_key(payload.org_id)
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    prompt = (payload.prompt or "Describe this image.").strip() or "Describe this image."
    model = (payload.model or "accounts/fireworks/models/llama-v3p2-11b-vision-instruct").strip() or "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]},
    ]
    out = call_openai_compatible("https://api.fireworks.ai/inference/v1", api_key, "fireworks", model, messages)
    log_usage(None, "Fireworks", model, prompt[:200], out["response"][:200], payload.prompt_id, input_tokens=out["input_tokens"], output_tokens=out["output_tokens"], total_tokens=out["total_tokens"], cost_usd=out["cost_usd"], org_id=payload.org_id)
    return {"response": out["response"], "output": out["response"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"], "cost_usd": out["cost_usd"], "latency_ms": out.get("latency_ms", 0)}


def _get_fireworks_api_key(org_id: str) -> str:
    from api_key_cache import get_provider_api_key
    return get_provider_api_key(org_id, "fireworks")


def handle_image_generation(payload: ImageGenerationPayload) -> dict:
    """Generate image via Fireworks FLUX. Returns data URL (base64), cost_usd, latency_ms."""
    api_key = _get_fireworks_api_key(payload.org_id)
    model = (payload.model or "accounts/fireworks/models/flux-1-schnell").strip()
    # Map model id to workflow path (schnell FP8 returns image in body)
    if "flux" in model.lower() or "schnell" in model.lower():
        workflow_path = "accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    else:
        workflow_path = model.replace("accounts/fireworks/models/", "") + "/text_to_image" if "/" in model else "accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    url = f"https://api.fireworks.ai/inference/v1/workflows/{workflow_path}"
    start = time.perf_counter()
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "image/png"},
            json={"prompt": payload.prompt},
            timeout=60,
        )
        r.raise_for_status()
        image_bytes = r.content
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fireworks image generation failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    log_usage(None, "Fireworks", model, payload.prompt[:200], "[image]", payload.prompt_id, cost_usd=FLUX_COST_PER_IMAGE)
    return {"url": data_url, "cost_usd": FLUX_COST_PER_IMAGE, "latency_ms": elapsed_ms}


def handle_stt(payload: STTPayload) -> dict:
    """Speech-to-text via Fireworks Whisper. audio_base64: raw audio bytes. Returns text, cost_usd, latency_ms."""
    api_key = _get_fireworks_api_key(payload.org_id)
    model = (payload.model or "whisper-v3-turbo").strip().lower()
    try:
        raw = base64.b64decode(payload.audio_base64 or "", validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio_base64 for STT")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio for STT")
    if len(raw) > WHISPER_MAX_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file is too large ({len(raw) / (1024*1024):.1f} MB). Maximum 25 MB. Use a shorter clip or extract audio from video.",
        )
    # Fireworks: whisper-v3 -> audio-prod, whisper-v3-turbo -> audio-turbo
    base_url = "https://audio-turbo.api.fireworks.ai" if "turbo" in model else "https://audio-prod.api.fireworks.ai"
    url = f"{base_url}/v1/audio/transcriptions"
    start = time.perf_counter()
    try:
        files = {"file": ("audio.mp3", io.BytesIO(raw), "audio/mpeg")}
        data: dict = {"model": model}
        if payload.prompt:
            data["prompt"] = payload.prompt
        r = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, files=files, data=data, timeout=120)
        r.raise_for_status()
        out = r.json()
        text = out.get("text", "") if isinstance(out, dict) else str(out)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fireworks transcription failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    cost_usd = max(0.0001, (len(raw) / (16000 * 2 * 60)) * FIREWORKS_STT_COST_PER_MIN)
    log_usage(None, "Fireworks", model, "[audio]", text[:200], payload.prompt_id, cost_usd=cost_usd)
    return {"output": text, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    out = call_openai_compatible(
        "https://api.fireworks.ai/inference/v1",
        api_key,
        "fireworks",
        payload.model,
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": payload.prompt}],
    )
    log_usage(
        "", "Fireworks", payload.model, payload.prompt, out["response"], payload.prompt_id,
        input_tokens=out["input_tokens"], output_tokens=out["output_tokens"], total_tokens=out["total_tokens"], cost_usd=out["cost_usd"], org_id=payload.org_id
    )
    return {"status": "success", "response": out["response"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"], "cost_usd": out["cost_usd"]}
