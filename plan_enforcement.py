"""
Plan enforcement — shared helpers for checking resource limits and monthly API usage.

Used by project_management, workflow_management, and public_execution.
Reads plan tier from the organizations.plan column; falls back to 'free'.
Monthly usage is tracked in the monthly_usage table (atomic upsert via RPC).
"""
import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from supabase_client import supabase
from org_access_control import PLAN_LIMITS, get_upgrade_suggestion

logger = logging.getLogger(__name__)


def _month_key() -> str:
    """Current month as 'YYYY-MM' in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def get_org_plan_tier(org_id: str) -> str:
    """Look up the organization's plan tier. Returns 'free' if not found."""
    try:
        result = (
            supabase.table("organizations")
            .select("plan")
            .eq("id", org_id)
            .single()
            .execute()
        )
        if result.data:
            return result.data.get("plan") or "free"
    except Exception as e:
        logger.warning("Failed to get org plan for %s: %s", org_id, type(e).__name__)
    return "free"


def _get_plan_limit(plan: str, resource: str) -> int:
    """Get the limit for a resource on a given plan. Returns -1 for unlimited."""
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    return limits.get(resource, 0)


# ── Resource limit checks ──────────────────────────────────────────


def check_project_limit(org_id: str) -> None:
    """Raise 403 if the org has reached its project limit."""
    plan = get_org_plan_tier(org_id)
    limit = _get_plan_limit(plan, "projects")
    if limit == -1:
        return  # unlimited

    try:
        result = (
            supabase.table("projects")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .execute()
        )
        current = result.count if result.count is not None else len(result.data or [])
    except Exception as e:
        logger.warning("Failed to count projects for org %s: %s", org_id, e)
        return  # fail open — don't block on count errors

    if current >= limit:
        upgrade = get_upgrade_suggestion(plan)
        raise HTTPException(
            status_code=403,
            detail=f"Project limit reached. Your plan ({plan}) allows {limit} project(s).{upgrade}",
        )


def check_workflow_limit(org_id: str) -> None:
    """Raise 403 if the org has reached its workflow limit."""
    plan = get_org_plan_tier(org_id)
    limit = _get_plan_limit(plan, "workflows")
    if limit == -1:
        return  # unlimited

    try:
        result = (
            supabase.table("workflows")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .execute()
        )
        current = result.count if result.count is not None else len(result.data or [])
    except Exception as e:
        logger.warning("Failed to count workflows for org %s: %s", org_id, e)
        return  # fail open

    if current >= limit:
        upgrade = get_upgrade_suggestion(plan)
        raise HTTPException(
            status_code=403,
            detail=f"Workflow limit reached. Your plan ({plan}) allows {limit} workflow(s).{upgrade}",
        )


def check_server_key_limit(org_id: str) -> None:
    """Raise 403 if the org has reached its server API key limit."""
    plan = get_org_plan_tier(org_id)
    limit = _get_plan_limit(plan, "server_keys")
    if limit == -1:
        return  # unlimited

    try:
        result = (
            supabase.table("service_api_keys")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("status", "active")
            .execute()
        )
        current = result.count if result.count is not None else len(result.data or [])
    except Exception as e:
        logger.warning("Failed to count server keys for org %s: %s", org_id, e)
        return

    if current >= limit:
        upgrade = get_upgrade_suggestion(plan)
        raise HTTPException(
            status_code=403,
            detail=f"Server key limit reached. Your plan ({plan}) allows {limit} key(s).{upgrade}",
        )


# ── Monthly request tracking ───────────────────────────────────────


def check_monthly_request_limit(org_id: str) -> None:
    """Raise 429 if the org has exceeded its monthly API request limit."""
    plan = get_org_plan_tier(org_id)
    limit = _get_plan_limit(plan, "requests_per_month")
    if limit == -1:
        return  # unlimited

    month = _month_key()
    try:
        result = (
            supabase.table("monthly_usage")
            .select("request_count")
            .eq("org_id", org_id)
            .eq("month", month)
            .single()
            .execute()
        )
        current = int(result.data.get("request_count", 0)) if result.data else 0
    except Exception:
        current = 0  # no row yet or query error — treat as 0

    if current >= limit:
        upgrade = get_upgrade_suggestion(plan)
        raise HTTPException(
            status_code=429,
            detail=f"Monthly API request limit reached. Your plan ({plan}) allows {limit:,} requests/month.{upgrade}",
            headers={"Retry-After": "86400"},
        )


def increment_monthly_usage(org_id: str) -> None:
    """Atomically increment the monthly request count for an org. Fire-and-forget."""
    month = _month_key()
    try:
        supabase.rpc("increment_monthly_usage", {
            "p_org_id": org_id,
            "p_month": month,
        }).execute()
    except Exception as e:
        # RPC may not exist yet — fall back to manual upsert
        logger.warning("increment_monthly_usage RPC failed (%s), falling back", type(e).__name__)
        try:
            result = (
                supabase.table("monthly_usage")
                .select("id, request_count")
                .eq("org_id", org_id)
                .eq("month", month)
                .limit(1)
                .execute()
            )
            if result.data:
                row = result.data[0]
                supabase.table("monthly_usage").update({
                    "request_count": int(row["request_count"]) + 1,
                }).eq("id", row["id"]).execute()
            else:
                supabase.table("monthly_usage").insert({
                    "org_id": org_id,
                    "month": month,
                    "request_count": 1,
                }).execute()
        except Exception as e2:
            logger.warning("monthly_usage fallback failed: %s", e2)


def get_monthly_usage(org_id: str) -> int:
    """Return the current month's request count for an org."""
    month = _month_key()
    try:
        result = (
            supabase.table("monthly_usage")
            .select("request_count")
            .eq("org_id", org_id)
            .eq("month", month)
            .single()
            .execute()
        )
        return int(result.data.get("request_count", 0)) if result.data else 0
    except Exception:
        return 0
