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


def handle_prompt_with_tools(payload, tools: list[dict], *, system_message: str = "", max_iterations: int = 5, tool_executor=None, can_parallelize_tool=None):
    """
    Tool calling via Gemini using google.generativeai SDK.
    Converts generic tool format to Gemini function declarations, runs iteration loop.
    """
    api_key = _get_gemini_api_key(payload.org_id)
    genai.configure(api_key=api_key)

    # Convert tools to Gemini function declarations
    function_declarations = []
    for t in tools:
        if not t.get("name"):
            continue
        params = t.get("parameters") or {"type": "object", "properties": {}}
        # Gemini uses a slightly different schema format — pass as-is, SDK handles it
        function_declarations.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": params,
        })

    gemini_tools = None
    if function_declarations:
        gemini_tools = [genai.types.Tool(function_declarations=function_declarations)]

    model_name = (payload.model or "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_message or "You are a helpful assistant.",
    )

    chat = model.start_chat()
    total_input_tokens = 0
    total_output_tokens = 0
    total_provider_latency_ms = 0
    all_tool_calls = []
    iteration_count = 0
    reply = ""

    try:
        # First message
        _t0 = time.perf_counter()
        response = chat.send_message(payload.prompt, tools=gemini_tools)
        _provider_ms = int((time.perf_counter() - _t0) * 1000)
        total_provider_latency_ms += _provider_ms

        usage = getattr(response, "usage_metadata", None)
        total_input_tokens += getattr(usage, "prompt_token_count", 0) if usage else 0
        total_output_tokens += getattr(usage, "candidates_token_count", 0) if usage else 0

        for iteration_count in range(1, max_iterations + 1):
            # Check for function calls in response
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                break

            parts = candidate.content.parts if hasattr(candidate.content, "parts") else []
            function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call.name]

            if not function_calls:
                # Text response — done
                reply = response.text if hasattr(response, "text") else ""
                break

            # Parse all function calls
            parsed_fcs = [(fc_part, fc_part.function_call.name, dict(fc_part.function_call.args) if fc_part.function_call.args else {}) for fc_part in function_calls]

            # Classify into parallelizable vs sequential
            if can_parallelize_tool and tool_executor and len(parsed_fcs) > 1:
                parallel_fcs = [(fp, n, a) for fp, n, a in parsed_fcs if can_parallelize_tool(n)]
                sequential_fcs = [(fp, n, a) for fp, n, a in parsed_fcs if not can_parallelize_tool(n)]
            else:
                parallel_fcs = []
                sequential_fcs = parsed_fcs

            # Execute parallelizable tools concurrently
            results_map: dict[str, tuple[str, int]] = {}
            if parallel_fcs:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=min(len(parallel_fcs), 8)) as pool:
                    future_to_fc = {pool.submit(tool_executor, n, a): (n, a) for _, n, a in parallel_fcs}
                    for future in as_completed(future_to_fc):
                        _n, _a = future_to_fc[future]
                        try:
                            results_map[_n] = future.result()
                        except Exception as exc:
                            results_map[_n] = (f"Tool execution error: {exc}", 0)

            # Execute sequential tools
            for _, n, a in sequential_fcs:
                if tool_executor:
                    results_map[n] = tool_executor(n, a)
                else:
                    results_map[n] = ("No tool executor configured", 0)

            # Build results in original order
            function_responses = []
            for fc_part, tc_name, tc_args in parsed_fcs:
                result_str, tool_latency_ms = results_map.get(tc_name, ("No result", 0))

                all_tool_calls.append({
                    "name": tc_name,
                    "arguments": tc_args,
                    "result": result_str[:2000],
                    "latency_ms": tool_latency_ms,
                })

                function_responses.append(
                    genai.types.Part.from_function_response(
                        name=tc_name,
                        response={"result": result_str[:4000]},
                    )
                )

            # Send function results back
            _t0 = time.perf_counter()
            response = chat.send_message(function_responses, tools=gemini_tools)
            _provider_ms = int((time.perf_counter() - _t0) * 1000)
            total_provider_latency_ms += _provider_ms

            usage = getattr(response, "usage_metadata", None)
            total_input_tokens += getattr(usage, "prompt_token_count", 0) if usage else 0
            total_output_tokens += getattr(usage, "candidates_token_count", 0) if usage else 0
        else:
            # Loop exhausted
            reply = response.text if hasattr(response, "text") else ""

        # If we broke out with text
        if not reply and hasattr(response, "text"):
            reply = response.text

        # Pricing
        try:
            pricing = get_pricing("gemini", model_name)
        except Exception:
            pricing = {"input": 0, "output": 0}
        total_cost = (total_input_tokens * pricing["input"] + total_output_tokens * pricing["output"]) / 1000

        log_usage(
            payload.org_id, "Gemini", model_name,
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
        raise HTTPException(status_code=500, detail=f"Gemini tool-call failed: {e}")
