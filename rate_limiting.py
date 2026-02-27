"""
DB-backed rate limiting for public workflow endpoint (per org, per minute).
Uses a Postgres RPC function for atomic increment (no TOCTOU race).
Requires the `increment_rate_limit` function from migration_p2_fixes.sql.
Falls back to application-level check-then-increment if the RPC is not yet deployed.
"""
import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from supabase_client import supabase

logger = logging.getLogger(__name__)


def _minute_bucket_utc() -> str:
    """Current time truncated to minute in UTC, ISO format for DB."""
    n = datetime.now(timezone.utc)
    return n.replace(second=0, microsecond=0).isoformat()


def check_and_increment_usage(
    org_id: str,
    endpoint_slug: str,
    rate_limit_per_minute: int,
) -> None:
    """
    Atomically increment the request count for (org_id, endpoint_slug, this minute).
    If the new count exceeds rate_limit_per_minute, raise 429.

    Uses a Postgres function `increment_rate_limit` that does:
      INSERT ... ON CONFLICT DO UPDATE SET request_count = request_count + 1
      RETURNING request_count
    This is a single atomic statement — no read-then-write race.
    """
    bucket = _minute_bucket_utc()

    try:
        # Call the atomic RPC function
        result = supabase.rpc("increment_rate_limit", {
            "p_org_id": org_id,
            "p_endpoint_slug": endpoint_slug,
            "p_minute_bucket": bucket,
        }).execute()

        new_count = result.data
        # rpc returns the scalar integer directly
        if isinstance(new_count, list) and len(new_count) > 0:
            new_count = new_count[0]
        if isinstance(new_count, dict):
            new_count = new_count.get("increment_rate_limit", new_count.get("request_count", 0))

        new_count = int(new_count) if new_count is not None else 1

        if new_count > rate_limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": "60"},
            )

    except HTTPException:
        raise
    except Exception as e:
        # RPC may not exist yet (pre-migration). Fall back to application-level logic.
        logger.warning("Rate limit RPC failed (%s), falling back to app-level check", type(e).__name__)
        _check_and_increment_fallback(org_id, endpoint_slug, bucket, rate_limit_per_minute)


def _check_and_increment_fallback(
    org_id: str,
    endpoint_slug: str,
    bucket: str,
    rate_limit_per_minute: int,
) -> None:
    """
    Fallback: application-level check-then-increment.
    Has a minor TOCTOU race under high concurrency but works without the RPC function.
    This path is only used before the migration is applied.
    """
    try:
        r = (
            supabase.table("api_key_usage")
            .select("id, request_count")
            .eq("org_id", org_id)
            .eq("endpoint_slug", endpoint_slug)
            .eq("minute_bucket", bucket)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Rate limit check failed.") from e

    row = (r.data or [None])[0] if r.data else None
    current = int(row["request_count"]) if row else 0

    if current >= rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": "60"},
        )

    if row:
        try:
            supabase.table("api_key_usage").update({
                "request_count": current + 1,
            }).eq("id", row["id"]).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Rate limit update failed.") from e
    else:
        try:
            supabase.table("api_key_usage").insert({
                "org_id": org_id,
                "endpoint_slug": endpoint_slug,
                "minute_bucket": bucket,
                "request_count": 1,
            }).execute()
        except Exception:
            # Concurrent insert race — row now exists, just update it
            try:
                r2 = (
                    supabase.table("api_key_usage")
                    .select("id, request_count")
                    .eq("org_id", org_id)
                    .eq("endpoint_slug", endpoint_slug)
                    .eq("minute_bucket", bucket)
                    .limit(1)
                    .execute()
                )
                row2 = (r2.data or [None])[0] if r2.data else None
                if row2:
                    current2 = int(row2["request_count"])
                    if current2 >= rate_limit_per_minute:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
                    supabase.table("api_key_usage").update({
                        "request_count": current2 + 1,
                    }).eq("id", row2["id"]).execute()
            except HTTPException:
                raise
            except Exception as e2:
                raise HTTPException(status_code=500, detail="Rate limit update failed.") from e2
