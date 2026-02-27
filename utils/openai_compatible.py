"""
Generic caller for any OpenAI-compatible API (Groq, Together, DeepSeek, Fireworks).
Cost formula: (prompt_tokens * input_per_1k + completion_tokens * output_per_1k) / 1000.
"""
import time
import httpx
from utils.pricing import get_pricing


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
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()

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
        "status": "success",
    }


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
