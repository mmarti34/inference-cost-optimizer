# utils/pricing.py — single source of truth: shared/providers.json (USD per 1K tokens)
# Tries package-local (inference-cost-optimizer/shared/) first, then monorepo root shared/.
#
# Model-id resolution is DETERMINISTIC and ordered (see get_pricing):
#   1. exact id match
#   2. explicit alias (short name -> canonical registry id)
#   3. longest registry id that is a prefix of the requested id (dated variants)
#   4. loud fallback default, flagged as estimated
#
# It is NEVER acceptable to resolve a request for "gpt-4" onto "gpt-4o-mini":
# that mis-prices the call by ~200x and corrupts every savings number downstream.
# Rule 3 only matches in the direction registry_id -> requested_id, i.e. the
# requested id must be a *more specific* form of a registry id.
from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

_utils_dir = os.path.dirname(os.path.abspath(__file__))
# 1) Package-local: inference-cost-optimizer/shared/providers.json (used when backend is deployed alone)
# 2) Monorepo: optiml/shared/providers.json
_CANDIDATE_PATHS = [
    os.path.join(_utils_dir, "..", "shared", "providers.json"),
    os.path.join(_utils_dir, "..", "..", "shared", "providers.json"),
]
_config = None

# Fallback used when a model id cannot be resolved to a real price. Any cost
# computed from this is a GUESS — callers must treat pricing["estimated"] as
# "exclude or flag this row in savings math".
DEFAULT_PRICING = {"input": 0.001, "output": 0.002}

# Short/legacy names that are not prefixes of their canonical registry id.
# Keyed by provider, then by lowercased requested model id.
_MODEL_ALIASES: dict[str, dict[str, str]] = {
    "openai": {
        "chatgpt-4o-latest": "gpt-4o",
        "gpt-4-turbo-preview": "gpt-4-turbo",
        "gpt-3.5": "gpt-3.5-turbo",
    },
    "anthropic": {
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    },
    "mistral": {
        "mistral-small": "mistral-small-latest",
        "mistral-large": "mistral-large-latest",
        "codestral": "codestral-latest",
    },
    "gemini": {
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
    },
}

# Countable signal for unknown models. Keyed by (provider, model) -> hit count.
# Read with get_pricing_miss_stats(); surfaced by GET /observability/pricing-misses.
_pricing_misses: dict[tuple[str, str], int] = {}
_pricing_miss_lock = threading.Lock()


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


def _record_pricing_miss(provider: str, model: str) -> int:
    """Count an unresolved (provider, model). Warn loudly the first time, then keep counting."""
    key = (provider or "", model or "")
    with _pricing_miss_lock:
        count = _pricing_misses.get(key, 0) + 1
        _pricing_misses[key] = count
    if count == 1:
        logger.warning(
            "PRICING MISS: no price for provider=%r model=%r. Falling back to "
            "estimated default $%s/$%s per 1K tokens. Costs logged for this model "
            "are GUESSES — add it to shared/providers.json.",
            provider, model, DEFAULT_PRICING["input"], DEFAULT_PRICING["output"],
        )
    else:
        logger.debug("PRICING MISS (x%d): provider=%r model=%r", count, provider, model)
    return count


def get_pricing_miss_stats() -> dict:
    """Countable signal for unknown models: total misses + per-model counts."""
    with _pricing_miss_lock:
        misses = dict(_pricing_misses)
    return {
        "distinct_models": len(misses),
        "total_misses": sum(misses.values()),
        "misses": [
            {"provider": p, "model": m, "count": c}
            for (p, m), c in sorted(misses.items(), key=lambda kv: -kv[1])
        ],
    }


def reset_pricing_miss_stats() -> None:
    """Test helper: clear the miss counters."""
    with _pricing_miss_lock:
        _pricing_misses.clear()


def get_pricing(provider: str, model: str) -> dict:
    """
    Resolve token pricing (USD per 1K tokens).

    Returns {"input", "output", "source", "estimated", "resolved_model"} where
    source is one of "exact" | "alias" | "prefix" | "default" and estimated is
    True only for "default" — meaning the number is a guess and must not be
    presented as a known price.
    """
    cfg = _load()
    prov = (provider or "").strip().lower()
    p = cfg.get("providers", {}).get(prov, {})
    models = p.get("models", {})
    requested = (model or "").strip()
    ml = requested.lower()

    def _out(mid: str, source: str) -> dict:
        m = models[mid]
        return {
            "input": m["input_per_1k"],
            "output": m["output_per_1k"],
            "source": source,
            "estimated": False,
            "resolved_model": mid,
        }

    # 1) Exact id
    if requested in models:
        return _out(requested, "exact")

    # 1b) Exact id, case-insensitive (registry ids like "Qwen/Qwen2.5-72B-Instruct-Turbo")
    for mid in models:
        if mid.lower() == ml:
            return _out(mid, "exact")

    # 2) Explicit alias
    alias = _MODEL_ALIASES.get(prov, {}).get(ml)
    if alias and alias in models:
        return _out(alias, "alias")

    # 3) Longest registry id that is a prefix of the requested id.
    #    "gpt-4o-2024-08-06" -> "gpt-4o" (not "gpt-4"), deterministic by length.
    best = None
    for mid in models:
        mid_l = mid.lower()
        if ml.startswith(mid_l) and (best is None or len(mid_l) > len(best.lower())):
            best = mid
    if best is not None:
        return _out(best, "prefix")

    # 4) Unknown — loud, counted, and flagged as estimated.
    _record_pricing_miss(prov or "(unknown)", requested or "(empty)")
    return {
        "input": DEFAULT_PRICING["input"],
        "output": DEFAULT_PRICING["output"],
        "source": "default",
        "estimated": True,
        "resolved_model": None,
    }


def is_estimated_pricing(provider: str, model: str) -> bool:
    """True when we have no real price for this model and would fall back to a guess."""
    try:
        return bool(get_pricing(provider, model).get("estimated"))
    except Exception:
        return True


def get_non_token_rate(provider: str, key: str, default: float | None = None) -> float:
    """
    Non-token unit price (images, audio seconds/characters, embeddings) from
    shared/providers.json -> non_token_rates. Keys are flat and provider-scoped,
    e.g. "image.dall-e-3.hd.1024", "audio.tts.per_1k_chars".

    Falls back to non_token_rates._defaults, then to `default`. This is a
    lookup table move, not a re-pricing: values match the constants they replaced.
    """
    try:
        cfg = _load()
    except Exception:
        return 0.0 if default is None else default
    rates = cfg.get("non_token_rates", {}) or {}
    prov = (provider or "").strip().lower()
    for table in (rates.get(prov) or {}, rates.get("_defaults") or {}):
        if key in table:
            try:
                return float(table[key])
            except (TypeError, ValueError):
                pass
    if default is None:
        _record_pricing_miss(prov or "(unknown)", f"non_token:{key}")
        return 0.0
    return default


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
