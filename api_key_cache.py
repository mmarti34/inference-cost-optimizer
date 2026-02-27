"""
In-memory TTL cache for decrypted provider API keys.

Each router currently hits the DB + runs AES decryption on every single request.
Since provider keys rarely change, caching them with a short TTL eliminates
~50-200ms of overhead per AI step.

Usage (drop-in replacement for the duplicated pattern in each router):
    from api_key_cache import get_provider_api_key
    api_key = get_provider_api_key(org_id, "openai")
"""

import logging
import threading
import time
from typing import Optional

from fastapi import HTTPException
from supabase_client import supabase
from utils.encryption import decrypt_api_key

logger = logging.getLogger(__name__)

# --- Configuration ---
_CACHE_TTL_SECONDS = 60          # keys are cached for 60 seconds
_CACHE_MAX_ENTRIES = 500         # safety cap (shouldn't hit this in normal use)

# --- Cache internals ---
# Key: (org_id, provider) -> Value: (decrypted_key, expiry_timestamp)
_cache: dict[tuple[str, str], tuple[str, float]] = {}
_lock = threading.Lock()


def _evict_expired() -> None:
    """Remove entries whose TTL has passed. Called under _lock."""
    now = time.monotonic()
    expired = [k for k, (_, exp) in _cache.items() if now >= exp]
    for k in expired:
        del _cache[k]


def get_provider_api_key(org_id: str, provider: str) -> str:
    """
    Return the decrypted provider API key for (org_id, provider).

    Checks the in-memory cache first; on miss, fetches from Supabase,
    decrypts, caches with TTL, and returns.

    Raises HTTPException(404) if no key is configured.
    Raises HTTPException(500) if decryption fails.
    """
    cache_key = (org_id, provider.lower())
    now = time.monotonic()

    # Fast path: cache hit (no lock needed for read in CPython due to GIL,
    # but we use a lock for correctness across implementations)
    with _lock:
        entry = _cache.get(cache_key)
        if entry is not None:
            decrypted, expiry = entry
            if now < expiry:
                return decrypted
            # Expired — remove it
            del _cache[cache_key]

    # Cache miss — fetch from DB
    try:
        result = (
            supabase.table("api_keys")
            .select("api_key")
            .eq("org_id", org_id)
            .eq("provider", provider)
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.warning("API key DB lookup failed for %s/%s: %s", org_id, provider, type(e).__name__)
        raise HTTPException(status_code=500, detail="Failed to look up API key.") from e

    keys = result.data or []
    if not keys:
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

    try:
        decrypted = decrypt_api_key(keys[0]["api_key"])
    except Exception as e:
        logger.warning("API key decryption failed for %s/%s: %s", org_id, provider, type(e).__name__)
        raise HTTPException(status_code=500, detail="Failed to decrypt API key.") from e

    # Store in cache
    with _lock:
        # Evict expired entries periodically to prevent unbounded growth
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            _evict_expired()
        _cache[cache_key] = (decrypted, now + _CACHE_TTL_SECONDS)

    return decrypted


def invalidate_provider_key(org_id: str, provider: str) -> None:
    """
    Explicitly invalidate a cached key (e.g., after key rotation).
    Call this from the /store-key and /delete-key endpoints.
    """
    cache_key = (org_id, provider.lower())
    with _lock:
        _cache.pop(cache_key, None)


def invalidate_all_keys_for_org(org_id: str) -> None:
    """Invalidate all cached keys for an org (e.g., after org deletion)."""
    with _lock:
        to_remove = [k for k in _cache if k[0] == org_id]
        for k in to_remove:
            del _cache[k]
