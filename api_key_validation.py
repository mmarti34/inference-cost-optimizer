"""
Validate Universal API key (Bearer token) for public endpoints.
Returns OrgContext; never exposes hashed_key.
Only active keys are accepted. On success, last_used_at is updated.
Do NOT use select("*"). Use explicit columns only.
"""
import hmac
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from supabase_client import supabase
from utils.encryption import hash_service_api_key, verify_service_api_key, decrypt_api_key

logger = logging.getLogger(__name__)

_COLS = "id, org_id, hashed_key, api_key, key_type, rate_limit_per_minute, status"


@dataclass
class OrgContext:
    org_id: str
    key_type: str  # "live" | "test"
    rate_limit_per_minute: int


def _update_last_used(key_id: str) -> None:
    """Update last_used_at for the key. No-op on failure."""
    try:
        supabase.table("service_api_keys").update({
            "last_used_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", key_id).execute()
    except Exception:
        pass


def _build_org_context(row: dict) -> Optional[OrgContext]:
    """Extract OrgContext from a matched row. Returns None if org_id missing."""
    org_id = row.get("org_id")
    if not org_id:
        return None
    key_type = (row.get("key_type") or "live").strip() or "live"
    if key_type not in ("live", "test"):
        key_type = "live"
    rate = row.get("rate_limit_per_minute", 60)
    try:
        rate = int(rate)
    except (TypeError, ValueError):
        rate = 60
    rate = max(1, min(rate, 10000))
    return OrgContext(org_id=str(org_id), key_type=key_type, rate_limit_per_minute=rate)


def validate_api_key(token: str) -> OrgContext:
    """
    Validate Bearer token against service_api_keys.
    Only keys with status='active' are considered.

    Strategy (fast path first):
      1. Hash the incoming token, query for exact hashed_key match (single row lookup).
      2. If no match, fall back to scanning legacy rows that have encrypted api_key
         but no hashed_key (pre-migration keys). This path is O(N_legacy) and will
         shrink to zero as keys are rotated.
      3. For legacy matches, backfill the hashed_key so future lookups use the fast path.

    Raises 401 if invalid. Never exposes hashed_key.
    """
    if not token or not token.strip():
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    clean_token = token.strip()
    token_hash = hash_service_api_key(clean_token)

    # ── Fast path: direct lookup by hash ─────────────────────────────────
    try:
        result = (
            supabase.table("service_api_keys")
            .select(_COLS)
            .eq("hashed_key", token_hash)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
    except Exception:
        # Fallback: status column may not exist pre-migration
        try:
            result = (
                supabase.table("service_api_keys")
                .select(_COLS)
                .eq("hashed_key", token_hash)
                .limit(1)
                .execute()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error validating API key.") from e

    rows = result.data or []
    for row in rows:
        if row.get("status") and row.get("status") != "active":
            continue
        # Constant-time verification even though we already matched by hash,
        # to guard against hash collisions and ensure timing safety.
        stored_hash = row.get("hashed_key", "")
        if hmac.compare_digest(stored_hash, token_hash):
            ctx = _build_org_context(row)
            if ctx:
                _update_last_used(row["id"])
                return ctx

    # ── Legacy path: scan rows with encrypted api_key but no hashed_key ──
    try:
        legacy_result = (
            supabase.table("service_api_keys")
            .select(_COLS)
            .is_("hashed_key", "null")
            .eq("status", "active")
            .execute()
        )
    except Exception:
        try:
            legacy_result = (
                supabase.table("service_api_keys")
                .select(_COLS)
                .is_("hashed_key", "null")
                .execute()
            )
        except Exception:
            legacy_result = None

    if legacy_result and legacy_result.data:
        for row in legacy_result.data:
            if row.get("status") and row.get("status") != "active":
                continue
            enc = row.get("api_key")
            if not enc:
                continue
            try:
                decrypted = decrypt_api_key(enc)
                # Use constant-time comparison for the legacy path too
                if hmac.compare_digest(decrypted.encode(), clean_token.encode()):
                    ctx = _build_org_context(row)
                    if ctx:
                        _update_last_used(row["id"])
                        # Backfill: write hashed_key so future lookups use the fast path
                        try:
                            supabase.table("service_api_keys").update({
                                "hashed_key": token_hash,
                            }).eq("id", row["id"]).execute()
                        except Exception:
                            logger.warning("Failed to backfill hashed_key for key %s", row["id"])
                        return ctx
            except Exception:
                continue

    raise HTTPException(status_code=401, detail="Invalid API key.")
