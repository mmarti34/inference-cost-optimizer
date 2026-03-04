"""
Generic caller for any OpenAI-compatible API (Groq, Together, DeepSeek, Fireworks).
Cost formula: (prompt_tokens * input_per_1k + completion_tokens * output_per_1k) / 1000.
"""
import json
import time
import httpx
from fastapi import HTTPException
from utils.pricing import get_pricing
from utils.usage_logger import log_usage


def call_openai_compatible(
    api_base: str,
    api_key: str,
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> dict:
    """Sync call. Returns dict with response, input_tokens, output_tokens, cost_usd, etc."""
    start = time.perf_counter()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    _t0_provider = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    _provider_latency_ms = int((time.perf_counter() - _t0_provider) * 1000)

    latency_ms = int((time.perf_counter() - start) * 1000)
    data = response.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    pricing = get_pricing(provider, model)
    cost_usd = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000

    return {
        "response": content,
        "output": content,
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "input_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "provider_latency_ms": _provider_latency_ms,
        "status": "success",
    }


def call_openai_compatible_with_tools(
    api_base: str,
    api_key: str,
    provider: str,
    model: str,
    prompt: str,
    tools: list[dict],
    *,
    tool_executor=None,
    system_message: str = "",
    max_iterations: int = 5,
    prompt_id: str = "",
    org_id: str = "",
    timeout: float = 120.0,
) -> dict:
    """
    OpenAI-compatible tool calling loop for Groq, Together, DeepSeek, Fireworks, Mistral.

    Sends messages + tools → if model returns tool_calls, execute them and loop →
    until text response or max_iterations exhausted.

    Returns dict matching openai_router.handle_prompt_with_tools output format.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{api_base.rstrip('/')}/chat/completions"

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

    messages: list[dict] = [
        {"role": "system", "content": system_message or "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_provider_latency_ms = 0
    all_tool_calls: list[dict] = []
    iteration_count = 0
    reply = ""

    try:
        with httpx.Client(timeout=timeout) as client:
            for iteration_count in range(1, max_iterations + 1):
                payload: dict = {"model": model, "messages": messages}
                if openai_tools:
                    payload["tools"] = openai_tools

                _t0 = time.perf_counter()
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                _provider_ms = int((time.perf_counter() - _t0) * 1000)
                total_provider_latency_ms += _provider_ms

                data = response.json()
                usage = data.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)

                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason", "")

                if finish_reason == "tool_calls" and message.get("tool_calls"):
                    # Append assistant message with tool_calls to conversation
                    messages.append(message)

                    for tc in message["tool_calls"]:
                        tc_func = tc.get("function", {})
                        tc_name = tc_func.get("name", "")
                        try:
                            tc_args = json.loads(tc_func.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            tc_args = {"raw": tc_func.get("arguments", "")}

                        if tool_executor:
                            result_str, tool_latency_ms = tool_executor(tc_name, tc_args)
                        else:
                            result_str, tool_latency_ms = "No tool executor configured", 0

                        all_tool_calls.append({
                            "name": tc_name,
                            "arguments": tc_args,
                            "result": result_str[:2000],
                            "latency_ms": tool_latency_ms,
                        })

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": result_str[:4000],
                        })
                else:
                    reply = message.get("content", "") or ""
                    break
            else:
                # Loop exhausted
                reply = reply or (data.get("choices", [{}])[0].get("message", {}).get("content", "") if data else "")

        # Pricing
        try:
            pricing = get_pricing(provider, model)
        except Exception:
            pricing = {"input": 0, "output": 0}
        total_cost = (total_input_tokens * pricing["input"] + total_output_tokens * pricing["output"]) / 1000

        log_usage(
            org_id, provider.capitalize(), model,
            prompt[:200], (reply or "")[:200], prompt_id,
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

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"{provider} tool-call failed: {e.response.text[:500]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{provider} tool-call failed: {e}")


def call_openai_compatible_embeddings(
    api_base: str,
    api_key: str,
    provider: str,
    model: str,
    input_text: str,
    timeout: float = 30.0,
) -> dict:
    """POST to OpenAI-compatible /embeddings. Returns embedding (list), input_tokens, cost_usd, latency_ms."""
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": input_text}
    url = f"{api_base.rstrip('/')}/embeddings"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    data = response.json()
    emb_data = data.get("data", [{}])
    embedding = emb_data[0].get("embedding", []) if emb_data else []
    usage = data.get("usage", {})
    prompt_tokens = usage.get("total_tokens", usage.get("prompt_tokens", len(input_text) // 4))
    try:
        pricing = get_pricing(provider, model)
        cost_usd = (prompt_tokens / 1000.0) * (pricing.get("input") or 0.0001)
    except Exception:
        cost_usd = (prompt_tokens / 1000.0) * 0.0001
    return {"embedding": embedding, "input_tokens": prompt_tokens, "cost_usd": cost_usd, "latency_ms": elapsed_ms}
