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
from utils.pricing import get_pricing, get_non_token_rate
from utils.encryption import decrypt_api_key

router = APIRouter()

# Non-token prices (images, audio, embeddings) live in shared/providers.json
# under non_token_rates.openai and are read via get_non_token_rate(). The
# second argument to each call is the historical constant, kept as a fallback
# so behaviour is unchanged if the config section is missing.

# Whisper API max file size 25 MB; use 24 MB to stay safe.
WHISPER_MAX_FILE_BYTES = 24 * 1024 * 1024
STT_REQUEST_TIMEOUT = 120.0

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


def _get_openai_client(org_id: str, *, base_url: str | None = None, api_key_override: str | None = None, default_headers: dict[str, str] | None = None) -> OpenAI:
    """
    Create an OpenAI client.

    For standard provider usage, only org_id is needed (key fetched from cache).
    For custom endpoints (openai_compatible_endpoint), pass base_url +
    api_key_override + optional default_headers to point at a different server.
    """
    from api_key_cache import get_provider_api_key

    api_key = api_key_override or get_provider_api_key(org_id, "openai")
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if default_headers:
        kwargs["default_headers"] = default_headers
    return OpenAI(**kwargs)


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
        cost_usd = get_non_token_rate("openai", "image.dall-e-2.per_image", 0.020)
    else:
        tier = "hd" if quality == "hd" else "standard"
        dim = "1024" if size == "1024x1024" else "other"
        _fallbacks = {("hd", "1024"): 0.080, ("hd", "other"): 0.120,
                      ("standard", "1024"): 0.040, ("standard", "other"): 0.080}
        cost_usd = get_non_token_rate(
            "openai", f"image.dall-e-3.{tier}.{dim}", _fallbacks[(tier, dim)]
        )
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
    cost_usd = (len(text) / 1000.0) * get_non_token_rate("openai", "audio.tts.per_1k_chars", 0.015)
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
    _per_min = get_non_token_rate("openai", "audio.stt.per_min", 0.006)
    _min_cost = get_non_token_rate("openai", "audio.stt.min_cost", 0.0001)
    cost_usd = max(_min_cost, (len(raw) / (16000 * 2 * 60)) * _per_min)
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
    cost_usd = (num_tokens / 1000.0) * get_non_token_rate("openai", "embedding.per_1k_tokens", 0.00002)
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


def handle_prompt(payload: PromptPayload, *, target=None):
    """
    Unified prompt handler.  Works for both:
      1. Standard provider calls  (target is None) — uses api_key_cache as before.
      2. Custom endpoint calls    (target is a ModelTarget with base_url/auth) — uses
         the OpenAI SDK pointed at the custom server.

    The ``target`` parameter is optional so every existing call site that does
    ``openai_router.handle_prompt(payload)`` keeps working unchanged.

    Header merge precedence (highest wins):
      1. SDK-required headers (Content-Type, etc.)
      2. Endpoint default_headers from custom_endpoints table
      3. Auth header (either SDK api_key or custom header)
    We never send duplicate Authorization headers.
    """
    from api_key_cache import get_provider_api_key

    # ── Build client kwargs ──────────────────────────────────────────────
    is_custom = target is not None and target.base_url is not None
    client_kwargs: dict = {}

    if is_custom:
        client_kwargs["base_url"] = target.base_url

        # Merge endpoint default_headers first (lower precedence)
        extra_headers: dict[str, str] = dict(target.default_headers or {})

        auth_name = target.auth_header_name or "Authorization"
        auth_value = target.auth_header_value or ""

        if auth_name == "Authorization":
            # Auth goes through SDK's api_key parameter.
            # "Bearer " prefix is stripped because the SDK adds its own.
            raw_key = auth_value
            if raw_key.lower().startswith("bearer "):
                raw_key = raw_key[7:]
            client_kwargs["api_key"] = raw_key
            # Ensure we don't ALSO send Authorization via default_headers
            extra_headers.pop("Authorization", None)
            extra_headers.pop("authorization", None)
        else:
            # Custom header (e.g. X-Api-Key).
            # SDK still requires a non-empty api_key; use a placeholder.
            client_kwargs["api_key"] = "custom-endpoint-placeholder"
            extra_headers[auth_name] = auth_value
            # Remove any accidental Authorization in default_headers
            extra_headers.pop("Authorization", None)
            extra_headers.pop("authorization", None)

        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
    else:
        client_kwargs["api_key"] = get_provider_api_key(payload.org_id, payload.provider)

    # ── Create client & call ─────────────────────────────────────────────
    try:
        client = OpenAI(**client_kwargs)
    except Exception as e:
        label = f"custom endpoint ({target.base_url})" if is_custom else "OpenAI"
        raise HTTPException(status_code=500, detail=f"Failed to initialize {label} client: {e}")

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": payload.prompt}
    ]

    try:
        # ── Provider latency: measure only the outbound SDK request ───────
        _t0_provider = time.perf_counter()
        completion = client.chat.completions.create(
            model=payload.model,
            messages=messages
        )
        _provider_latency_ms = int((time.perf_counter() - _t0_provider) * 1000)

        reply = completion.choices[0].message.content
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        # Pricing: best-effort fallback for custom endpoints
        try:
            pricing = get_pricing("openai", payload.model)
        except Exception:
            pricing = {"input": 0, "output": 0}
        cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

        # Log provider label — "custom:<host>" for custom endpoints so dashboards distinguish
        if is_custom:
            try:
                from urllib.parse import urlparse
                host = urlparse(target.base_url).hostname or target.base_url
            except Exception:
                host = target.base_url
            provider_label = f"custom:{host}"
        else:
            provider_label = "OpenAI"

        log_usage(
            None,  # user_id — this is an org-scoped call; org_id goes in org_id=
            provider_label,
            payload.model,
            payload.prompt[:200],
            (reply or "")[:200],
            payload.prompt_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            org_id=payload.org_id,
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
        label = f"custom endpoint ({target.base_url})" if is_custom else "OpenAI"
        raise HTTPException(status_code=500, detail=f"{label} call failed: {e}")


def handle_prompt_with_tools(payload: PromptPayload, tools: list[dict], *, target=None, system_message: str = "", max_iterations: int = 5, tool_executor=None, can_parallelize_tool=None):
    """
    LLM call with tool/function calling support (OpenAI format).

    Loops up to ``max_iterations`` times: if the model returns tool_calls,
    we execute each tool via ``tool_executor(name, arguments_dict)`` and feed
    the results back.  Once the model returns a text response (or the loop
    exhausts), we return the final answer plus all tool-call records.

    ``tool_executor`` signature:  (name: str, arguments: dict) -> (result_str, latency_ms)
    """
    from api_key_cache import get_provider_api_key

    # ── Build client kwargs (same as handle_prompt) ───────────────────────
    is_custom = target is not None and getattr(target, "base_url", None) is not None
    client_kwargs: dict = {}

    if is_custom:
        client_kwargs["base_url"] = target.base_url
        extra_headers: dict[str, str] = dict(target.default_headers or {})
        auth_name = target.auth_header_name or "Authorization"
        auth_value = target.auth_header_value or ""
        if auth_name == "Authorization":
            raw_key = auth_value
            if raw_key.lower().startswith("bearer "):
                raw_key = raw_key[7:]
            client_kwargs["api_key"] = raw_key
            extra_headers.pop("Authorization", None)
            extra_headers.pop("authorization", None)
        else:
            client_kwargs["api_key"] = "custom-endpoint-placeholder"
            extra_headers[auth_name] = auth_value
            extra_headers.pop("Authorization", None)
            extra_headers.pop("authorization", None)
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
    else:
        client_kwargs["api_key"] = get_provider_api_key(payload.org_id, payload.provider)

    try:
        client = OpenAI(**client_kwargs)
    except Exception as e:
        label = f"custom endpoint ({target.base_url})" if is_custom else "OpenAI"
        raise HTTPException(status_code=500, detail=f"Failed to initialize {label} client: {e}")

    # ── Convert tools to OpenAI format ────────────────────────────────────
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
        if t.get("name")
    ]

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message or "You are a helpful assistant."},
        {"role": "user", "content": payload.prompt},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_provider_latency_ms = 0
    all_tool_calls: list[dict] = []
    iteration_count = 0
    reply = ""

    try:
        for iteration_count in range(1, max_iterations + 1):
            _t0 = time.perf_counter()
            completion = client.chat.completions.create(
                model=payload.model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
            )
            _provider_ms = int((time.perf_counter() - _t0) * 1000)
            total_provider_latency_ms += _provider_ms

            usage = getattr(completion, "usage", None)
            total_input_tokens += getattr(usage, "prompt_tokens", 0) if usage else 0
            total_output_tokens += getattr(usage, "completion_tokens", 0) if usage else 0

            choice = completion.choices[0]
            finish = getattr(choice, "finish_reason", None)

            if finish == "tool_calls" and choice.message.tool_calls:
                # Append the assistant message with tool_calls to the conversation
                messages.append(choice.message)  # type: ignore[arg-type]

                # Parse all tool calls first
                parsed_tcs = []
                for tc in choice.message.tool_calls:
                    tc_name = tc.function.name
                    try:
                        tc_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        tc_args = {"raw": tc.function.arguments}
                    parsed_tcs.append((tc.id, tc_name, tc_args))

                # Classify into parallelizable vs sequential
                if can_parallelize_tool and tool_executor and len(parsed_tcs) > 1:
                    parallel_tcs = [(tid, n, a) for tid, n, a in parsed_tcs if can_parallelize_tool(n)]
                    sequential_tcs = [(tid, n, a) for tid, n, a in parsed_tcs if not can_parallelize_tool(n)]
                else:
                    parallel_tcs = []
                    sequential_tcs = parsed_tcs

                # Execute parallelizable tools concurrently
                results_map: dict[str, tuple[str, int]] = {}
                if parallel_tcs:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=min(len(parallel_tcs), 8)) as pool:
                        future_to_tc = {
                            pool.submit(tool_executor, n, a): (tid, n, a)
                            for tid, n, a in parallel_tcs
                        }
                        for future in as_completed(future_to_tc):
                            tid, _n, _a = future_to_tc[future]
                            try:
                                results_map[tid] = future.result()
                            except Exception as exc:
                                results_map[tid] = (f"Tool execution error: {exc}", 0)

                # Execute sequential tools one at a time
                for tid, n, a in sequential_tcs:
                    if tool_executor:
                        results_map[tid] = tool_executor(n, a)
                    else:
                        results_map[tid] = ("No tool executor configured", 0)

                # Append results in original order (matching tool_call_ids)
                for tc in choice.message.tool_calls:
                    tc_name = tc.function.name
                    try:
                        tc_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        tc_args = {"raw": tc.function.arguments}
                    result_str, tool_latency_ms = results_map.get(tc.id, ("No result", 0))

                    all_tool_calls.append({
                        "name": tc_name,
                        "arguments": tc_args,
                        "result": result_str[:2000],
                        "latency_ms": tool_latency_ms,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str[:4000],
                    })
            else:
                # Final text response
                reply = choice.message.content or ""
                break
        else:
            # Loop exhausted — use last response
            reply = reply or (completion.choices[0].message.content or "") if completion else ""

        # ── Pricing ──────────────────────────────────────────────────────
        try:
            pricing = get_pricing("openai", payload.model)
        except Exception:
            pricing = {"input": 0, "output": 0}
        total_cost = (total_input_tokens * pricing["input"] + total_output_tokens * pricing["output"]) / 1000

        if is_custom:
            try:
                from urllib.parse import urlparse
                host = urlparse(target.base_url).hostname or target.base_url
            except Exception:
                host = target.base_url
            provider_label = f"custom:{host}"
        else:
            provider_label = "OpenAI"

        log_usage(
            payload.org_id, provider_label, payload.model,
            payload.prompt[:200], (reply or "")[:200], payload.prompt_id,
            input_tokens=total_input_tokens, output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens, cost_usd=total_cost,
        )

        return {
            "status": "success",
            "response": reply,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "cost_usd": total_cost,
            "provider_latency_ms": total_provider_latency_ms,
            "tool_calls": all_tool_calls,
            "tool_calls_count": len(all_tool_calls),
            "iterations": iteration_count,
        }

    except Exception as e:
        label = f"custom endpoint ({target.base_url})" if is_custom else "OpenAI"
        raise HTTPException(status_code=500, detail=f"{label} tool-call failed: {e}")
