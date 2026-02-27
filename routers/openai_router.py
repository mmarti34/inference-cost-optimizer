import base64
import io
import json
import time
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from supabase_client import supabase
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key

router = APIRouter()

# DALL·E 3 approximate cost per image (USD) for usage logging; actual pricing may vary.
DALLE3_STANDARD_1024_COST = 0.040
DALLE3_HD_1024_COST = 0.080
DALLE3_STANDARD_OTHER_COST = 0.080
DALLE3_HD_OTHER_COST = 0.120

# TTS: ~$0.015/1K chars (tts-1), ~$0.030/1K (tts-1-hd). Approximate.
TTS_COST_PER_1K_CHARS = 0.015
# Whisper: ~$0.006/min. Approximate per second.
WHISPER_COST_PER_SEC = 0.0001
# Whisper API max file size 25 MB; use 24 MB to stay safe.
WHISPER_MAX_FILE_BYTES = 24 * 1024 * 1024
STT_REQUEST_TIMEOUT = 120.0
# Embedding: text-embedding-3-small ~$0.00002/1K tokens.
EMBEDDING_COST_PER_1K = 0.00002

class PromptPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    prompt_id: str


class ImageGenerationPayload(BaseModel):
    org_id: str
    prompt: str
    size: str
    model: str
    quality: str
    prompt_id: str
    negative_prompt: str | None = None


class TTSPayload(BaseModel):
    org_id: str
    text: str
    model: str
    voice: str
    prompt_id: str


class STTPayload(BaseModel):
    org_id: str
    audio_base64: str
    model: str
    prompt_id: str
    prompt: str | None = None


class EmbeddingPayload(BaseModel):
    org_id: str
    text: str
    model: str
    prompt_id: str


class VisionPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    image_url: str  # URL or data:image/...;base64,...
    prompt_id: str


def _get_openai_client(org_id: str) -> OpenAI:
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(org_id, "openai")
    return OpenAI(api_key=api_key)


def handle_image_generation(payload: ImageGenerationPayload) -> dict:
    """Generate image via OpenAI Images API (DALL·E 2/3). Returns url, cost_usd, latency_ms."""
    client = _get_openai_client(payload.org_id)
    size = (payload.size or "1024x1024").strip() or "1024x1024"
    model = (payload.model or "dall-e-3").strip().lower() or "dall-e-3"
    quality = (payload.quality or "standard").strip().lower() or "standard"
    if model not in ("dall-e-3", "dall-e-2"):
        model = "dall-e-3"
    if model == "dall-e-2":
        size = size if size in ("256x256", "512x512", "1024x1024") else "1024x1024"
    else:
        size = size if size in ("1024x1024", "1792x1024", "1024x1792") else "1024x1024"
    kwargs = {"prompt": payload.prompt, "n": 1, "size": size, "response_format": "url", "model": model}
    if model == "dall-e-3":
        kwargs["quality"] = "hd" if quality == "hd" else "standard"
    start = time.perf_counter()
    try:
        response = client.images.generate(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI image generation failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    url = response.data[0].url if response.data else ""
    if model == "dall-e-2":
        cost_usd = 0.020
    else:
        if quality == "hd":
            cost_usd = DALLE3_HD_1024_COST if size == "1024x1024" else DALLE3_HD_OTHER_COST
        else:
            cost_usd = DALLE3_STANDARD_1024_COST if size == "1024x1024" else DALLE3_STANDARD_OTHER_COST
    log_usage(
        None,
        "OpenAI",
        model,
        payload.prompt[:200],
        url[:200] if url else "[image]",
        payload.prompt_id,
        cost_usd=cost_usd,
    )
    return {"url": url, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_tts(payload: TTSPayload) -> dict:
    """Text-to-speech via OpenAI Audio API. Returns base64 data URL (audio/mp3), cost_usd, latency_ms."""
    client = _get_openai_client(payload.org_id)
    model = (payload.model or "tts-1").strip() or "tts-1"
    voice = (payload.voice or "alloy").strip().lower() or "alloy"
    text = (payload.text or "").strip() or " "
    start = time.perf_counter()
    try:
        response = client.audio.speech.create(model=model, voice=voice, input=text)
        b = response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI TTS failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    b64 = base64.b64encode(b).decode("ascii")
    data_url = f"data:audio/mp3;base64,{b64}"
    cost_usd = (len(text) / 1000.0) * TTS_COST_PER_1K_CHARS
    log_usage(None, "OpenAI", model, text[:200], "[audio]", payload.prompt_id, cost_usd=cost_usd)
    return {"output": data_url, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_stt(payload: STTPayload) -> dict:
    """Speech-to-text via OpenAI Whisper. audio_base64: raw audio bytes (any format Whisper supports). Returns text, cost_usd, latency_ms."""
    client = _get_openai_client(payload.org_id)
    model = (payload.model or "whisper-1").strip() or "whisper-1"
    try:
        raw = base64.b64decode(payload.audio_base64 or "", validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio_base64 for STT")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio for STT")
    if len(raw) > WHISPER_MAX_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file is too large ({len(raw) / (1024*1024):.1f} MB). Whisper supports up to 25 MB. Use a shorter clip or convert to audio-only (e.g. extract audio from video).",
        )
    start = time.perf_counter()
    try:
        # OpenAI client expects a file-like object with name for format detection
        f = io.BytesIO(raw)
        f.name = "audio.mp3"
        kwargs = {"model": model, "file": f, "timeout": STT_REQUEST_TIMEOUT}
        if payload.prompt:
            kwargs["prompt"] = payload.prompt
        transcription = client.audio.transcriptions.create(**kwargs)
        text = transcription.text if hasattr(transcription, "text") else str(transcription)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Whisper failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    # Approximate cost by length (Whisper ~$0.006/min)
    cost_usd = max(0.0001, (len(raw) / (16000 * 2 * 60)) * 0.006)
    log_usage(None, "OpenAI", model, "[audio]", text[:200], payload.prompt_id, cost_usd=cost_usd)
    return {"output": text, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_embedding(payload: EmbeddingPayload) -> dict:
    """Embedding via OpenAI. Returns embedding (list of floats as JSON string for workflow), cost_usd, latency_ms."""
    client = _get_openai_client(payload.org_id)
    model = (payload.model or "text-embedding-3-small").strip() or "text-embedding-3-small"
    text = (payload.text or "").strip() or " "
    start = time.perf_counter()
    try:
        response = client.embeddings.create(input=text, model=model)
        emb = response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    tokens = getattr(response, "usage", None)
    num_tokens = (getattr(tokens, "total_tokens", 0) or 0) if tokens else len(text) // 4
    cost_usd = (num_tokens / 1000.0) * EMBEDDING_COST_PER_1K
    log_usage(None, "OpenAI", model, text[:200], f"[{len(emb)}d]", payload.prompt_id, cost_usd=cost_usd)
    return {"output": json.dumps(emb), "embedding": emb, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_vision(payload: VisionPayload) -> dict:
    """Vision: image + prompt -> text. image_url can be https URL or data:image/...;base64,..."""
    client = _get_openai_client(payload.org_id)
    model = (payload.model or "gpt-4o-mini").strip() or "gpt-4o-mini"
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    content: list = [
        {"type": "text", "text": (payload.prompt or "Describe this image.").strip() or "Describe this image."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI vision failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    reply = (completion.choices[0].message.content or "").strip()
    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    pricing = get_pricing("openai", model)
    cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
    log_usage(None, "OpenAI", model, payload.prompt[:200], reply[:200], payload.prompt_id, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)
    return {"response": reply, "output": reply, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI client: {e}")

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": payload.prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model=payload.model,
            messages=messages
        )
        reply = completion.choices[0].message.content
        input_tokens = getattr(completion.usage, "prompt_tokens", 0)
        output_tokens = getattr(completion.usage, "completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        pricing = get_pricing("openai", payload.model)
        cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

        log_usage(
            payload.org_id,
            "OpenAI",
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
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")
