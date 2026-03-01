import base64
import json
import re
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key

router = APIRouter()

# New Gemini SDK (Imagen, embeddings). Optional to avoid breaking if not installed.
try:
    from google import genai as genai_new
    _HAS_GENAI_NEW = True
except ImportError:
    genai_new = None
    _HAS_GENAI_NEW = False

IMAGEN_COST_PER_IMAGE = 0.03
EMBEDDING_COST_PER_1K = 0.000025

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
    image_url: str
    prompt_id: str


def _parse_image_data_url(image_url: str) -> tuple[str, str]:
    """Return (mime_type, base64_data)."""
    s = (image_url or "").strip()
    if s.startswith("data:"):
        m = re.match(r"data:([^;]+);base64,(.+)", s, re.DOTALL)
        if m:
            mt = m.group(1).strip() or "image/jpeg"
            return (mt, m.group(2))
    return ("image/jpeg", "")


def handle_vision(payload: VisionPayload) -> dict:
    """Vision: image + prompt -> text. Uses existing genai (google.generativeai)."""
    api_key = _get_gemini_api_key(payload.org_id)
    genai.configure(api_key=api_key)
    model_name = (payload.model or "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    image_url = (payload.image_url or "").strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="Vision requires image_url")
    mime_type, b64 = _parse_image_data_url(image_url)
    if not b64 and image_url.startswith("http"):
        import requests
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        b64 = base64.b64encode(r.content).decode("ascii")
        mime_type = "image/jpeg"
    elif not b64:
        raise HTTPException(status_code=400, detail="Vision image_url must be data:...;base64,... or https URL")
    prompt = (payload.prompt or "Describe this image.").strip() or "Describe this image."
    contents = [{"inline_data": {"mime_type": mime_type, "data": b64}}, prompt]
    model = genai.GenerativeModel(model_name)
    start = time.perf_counter()
    try:
        response = model.generate_content(contents)
        reply = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini vision failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
    total_tokens = input_tokens + output_tokens
    pricing = get_pricing("gemini", model_name)
    cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
    log_usage(None, "Gemini", model_name, prompt[:200], reply[:200], payload.prompt_id, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, cost_usd=cost_usd)
    return {"response": reply, "output": reply, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def _get_gemini_api_key(org_id: str) -> str:
    from api_key_cache import get_provider_api_key
    return get_provider_api_key(org_id, "gemini")


def handle_image_generation(payload: ImageGenerationPayload) -> dict:
    """Generate image via Gemini/Imagen API. Returns data URL (base64), cost_usd, latency_ms."""
    if not _HAS_GENAI_NEW:
        raise HTTPException(status_code=501, detail="google-genai package required for Gemini image generation. pip install google-genai")
    api_key = _get_gemini_api_key(payload.org_id)
    model = (payload.model or "imagen-3.0-generate-002").strip() or "imagen-3.0-generate-002"
    start = time.perf_counter()
    try:
        client = genai_new.Client(api_key=api_key)
        response = client.models.generate_images(
            model=model,
            prompt=payload.prompt,
            config=genai_new.types.GenerateImagesConfig(number_of_images=1),
        )
        if not response.generated_images:
            raise HTTPException(status_code=500, detail="Imagen returned no image")
        img = response.generated_images[0]
        image_bytes = img.image.image_bytes if hasattr(img.image, "image_bytes") else getattr(img, "image_bytes", None)
        if not image_bytes:
            raise HTTPException(status_code=500, detail="Imagen image bytes missing")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Imagen failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    log_usage(None, "Gemini", model, payload.prompt[:200], "[image]", payload.prompt_id, cost_usd=IMAGEN_COST_PER_IMAGE)
    return {"url": data_url, "cost_usd": IMAGEN_COST_PER_IMAGE, "latency_ms": elapsed_ms}


def handle_embedding(payload: EmbeddingPayload) -> dict:
    """Embedding via Gemini. Returns output (JSON list), cost_usd, latency_ms."""
    if not _HAS_GENAI_NEW:
        raise HTTPException(status_code=501, detail="google-genai package required for Gemini embeddings. pip install google-genai")
    api_key = _get_gemini_api_key(payload.org_id)
    model = (payload.model or "text-embedding-005").strip() or "text-embedding-005"
    text = (payload.text or "").strip() or " "
    start = time.perf_counter()
    try:
        client = genai_new.Client(api_key=api_key)
        result = client.models.embed_content(model=model, contents=text)
        emb = result.embeddings[0].values if result.embeddings else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini embedding failed: {e}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if isinstance(emb, list):
        pass
    else:
        emb = list(emb) if hasattr(emb, "__iter__") else []
    cost_usd = (len(text) / 1000.0) * EMBEDDING_COST_PER_1K
    log_usage(None, "Gemini", model, text[:200], f"[{len(emb)}d]", payload.prompt_id, cost_usd=cost_usd)
    return {"output": json.dumps(emb), "embedding": emb, "cost_usd": cost_usd, "latency_ms": elapsed_ms}


def handle_prompt(payload: PromptPayload):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(payload.org_id, payload.provider)

    # 2. Configure Gemini client
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(payload.model)

    try:
        # 3. Generate content
        _t0_provider = time.perf_counter()
        response = model.generate_content(payload.prompt)
        _provider_latency_ms = int((time.perf_counter() - _t0_provider) * 1000)

        reply = response.text

        # 4. Get token usage (may be None depending on model/version)
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
        total_tokens = input_tokens + output_tokens

        # 5. Pricing
        pricing = get_pricing("gemini", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        # 6. Log usage
        log_usage(
            payload.org_id,
            "Gemini",
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
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {str(e)}")
