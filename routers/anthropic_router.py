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


def handle_prompt_with_tools(payload, tools: list[dict], *, system_message: str = "", max_iterations: int = 5, tool_executor=None, can_parallelize_tool=None):
    """
    LLM call with tool use support (Anthropic format).

    Loops up to ``max_iterations`` times: if the model returns tool_use
    content blocks, we execute each tool via ``tool_executor(name, arguments_dict)``
    and feed the results back.

    ``tool_executor`` signature:  (name: str, arguments: dict) -> (result_str, latency_ms)
    """
    from api_key_cache import get_provider_api_key
    import json as _json

    api_key = get_provider_api_key(payload.org_id, getattr(payload, "provider", "anthropic"))
    client = Anthropic(api_key=api_key)

    # ── Convert tools to Anthropic format ─────────────────────────────────
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
        }
        for t in tools
        if t.get("name")
    ]

    messages: list[dict] = [{"role": "user", "content": payload.prompt}]

    total_input_tokens = 0
    total_output_tokens = 0
    total_provider_latency_ms = 0
    all_tool_calls: list[dict] = []
    iteration_count = 0
    reply = ""

    try:
        for iteration_count in range(1, max_iterations + 1):
            _t0 = time.perf_counter()
            response = client.messages.create(
                model=payload.model,
                max_tokens=1024,
                system=system_message or "You are a helpful assistant.",
                messages=messages,
                tools=anthropic_tools if anthropic_tools else [],
            )
            _provider_ms = int((time.perf_counter() - _t0) * 1000)
            total_provider_latency_ms += _provider_ms

            total_input_tokens += getattr(response.usage, "input_tokens", 0)
            total_output_tokens += getattr(response.usage, "output_tokens", 0)

            if response.stop_reason == "tool_use":
                # Append the full assistant response (text + tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                # Extract tool_use blocks
                tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

                # Classify into parallelizable vs sequential
                if can_parallelize_tool and tool_executor and len(tool_use_blocks) > 1:
                    parallel_blocks = [b for b in tool_use_blocks if can_parallelize_tool(b.name)]
                    sequential_blocks = [b for b in tool_use_blocks if not can_parallelize_tool(b.name)]
                else:
                    parallel_blocks = []
                    sequential_blocks = tool_use_blocks

                # Execute parallelizable tools concurrently
                results_map: dict[str, tuple[str, int]] = {}
                if parallel_blocks:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=min(len(parallel_blocks), 8)) as pool:
                        future_to_block = {
                            pool.submit(tool_executor, b.name, b.input if isinstance(b.input, dict) else {}): b
                            for b in parallel_blocks
                        }
                        for future in as_completed(future_to_block):
                            block = future_to_block[future]
                            try:
                                results_map[block.id] = future.result()
                            except Exception as exc:
                                results_map[block.id] = (f"Tool execution error: {exc}", 0)

                # Execute sequential tools one at a time
                for block in sequential_blocks:
                    tc_args = block.input if isinstance(block.input, dict) else {}
                    if tool_executor:
                        results_map[block.id] = tool_executor(block.name, tc_args)
                    else:
                        results_map[block.id] = ("No tool executor configured", 0)

                # Build results in original order
                tool_result_blocks: list[dict] = []
                for block in tool_use_blocks:
                    tc_name = block.name
                    tc_args = block.input if isinstance(block.input, dict) else {}
                    result_str, tool_latency_ms = results_map.get(block.id, ("No result", 0))

                    all_tool_calls.append({
                        "name": tc_name,
                        "arguments": tc_args,
                        "result": result_str[:2000],
                        "latency_ms": tool_latency_ms,
                    })
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str[:4000],
                    })

                # Send all tool results back
                messages.append({"role": "user", "content": tool_result_blocks})
            else:
                # Final text response — extract text blocks
                text_parts = []
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(block.text)
                reply = "\n".join(text_parts)
                break
        else:
            # Loop exhausted
            if not reply:
                text_parts = []
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(block.text)
                reply = "\n".join(text_parts)

        # ── Pricing ──────────────────────────────────────────────────────
        pricing = get_pricing("anthropic", payload.model)
        total_cost = (total_input_tokens / 1000 * pricing["input"]) + (total_output_tokens / 1000 * pricing["output"])

        log_usage(
            user_id=getattr(payload, "user_id", None),
            provider="Anthropic",
            model=payload.model,
            prompt=payload.prompt[:200],
            response=(reply or "")[:200],
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            cost_usd=total_cost,
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
        raise HTTPException(status_code=500, detail=f"Anthropic tool-call failed: {str(e)}")
