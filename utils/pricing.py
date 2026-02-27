# utils/pricing.py — single source of truth: shared/providers.json (USD per 1K tokens)
# Tries package-local (inference-cost-optimizer/shared/) first, then monorepo root shared/.
import json
import os

_utils_dir = os.path.dirname(os.path.abspath(__file__))
# 1) Package-local: inference-cost-optimizer/shared/providers.json (used when backend is deployed alone)
# 2) Monorepo: optiml/shared/providers.json
_CANDIDATE_PATHS = [
    os.path.join(_utils_dir, "..", "shared", "providers.json"),
    os.path.join(_utils_dir, "..", "..", "shared", "providers.json"),
]
_config = None


def _load():
    global _config
    if _config is None:
        path = None
        for p in _CANDIDATE_PATHS:
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                "providers.json not found. Tried: "
                + ", ".join(_CANDIDATE_PATHS)
                + ". Add inference-cost-optimizer/shared/providers.json or run from monorepo with shared/providers.json."
            )
        with open(path) as f:
            _config = json.load(f)
    return _config


def get_pricing(provider: str, model: str) -> dict:
    cfg = _load()
    p = cfg.get("providers", {}).get(provider.lower(), {})
    models = p.get("models", {})

    # Exact match
    if model in models:
        m = models[model]
        return {"input": m["input_per_1k"], "output": m["output_per_1k"]}

    # Fuzzy: prefix match, case-insensitive
    ml = model.lower()
    for mid, m in models.items():
        if mid.lower() == ml or mid.lower().startswith(ml) or ml.startswith(mid.lower()):
            return {"input": m["input_per_1k"], "output": m["output_per_1k"]}

    return {"input": 0.001, "output": 0.002}  # safe default


def get_provider_for_model(model: str) -> str | None:
    cfg = _load()
    for pid, p in cfg.get("providers", {}).items():
        if model in p.get("models", {}):
            return pid
    # Heuristic fallbacks
    prefixes = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
        "claude": "anthropic",
        "gemini": "gemini",
        "gemma": "gemini",
        "mistral": "mistral",
        "codestral": "mistral",
        "open-mistral": "mistral",
        "command": "cohere",
        "llama": "groq",
        "mixtral-8x7b-32768": "groq",
        "deepseek": "deepseek",
    }
    for prefix, prov in prefixes.items():
        if model.lower().startswith(prefix):
            return prov
    if "/" in model:
        return "together"  # Together uses org/model format
    if model.startswith("accounts/fireworks"):
        return "fireworks"
    return None


def get_all_providers() -> dict:
    return _load().get("providers", {})


def suggest_model(prompt: str) -> dict:
    length = len(prompt.split())
    if length <= 50:
        model = "gpt-3.5-turbo"
        tier = "cheapest (suitable for short/simple prompts)"
    elif length <= 200:
        model = "gpt-4-turbo"
        tier = "mid-tier (for moderate prompts)"
    else:
        model = "gpt-4o"
        tier = "high-tier (for long/complex prompts)"
    pricing = get_pricing("openai", model)
    return {
        "provider": "openai",
        "model": model,
        "reason": tier,
        "pricing": pricing,
    }
