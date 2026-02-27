"""
Fire-and-forget logging of every public API request to api_request_log.
Never blocks or raises; used for error rate, experiment metrics, rollback checks.
"""
import logging
from typing import Any, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)


def get_api_request_log_sync(request_id: str) -> Optional[dict]:
    """Fetch a single api_request_log row by id. Returns None if not found."""
    try:
        r = (
            supabase.table("api_request_log")
            .select("id, org_id, custom_metrics")
            .eq("id", request_id)
            .limit(1)
            .execute()
        )
        if r.data and len(r.data) > 0:
            return r.data[0]
    except Exception:
        pass
    return None


def log_api_request_sync(entry: dict[str, Any]) -> None:
    """Insert one row into api_request_log (sync). Catch and log any failure; never raise."""
    try:
        if entry.get("org_id") is None:
            return
        row = {"org_id": entry["org_id"]}
        if entry.get("id"):
            row["id"] = entry["id"]
        row.update({
            "endpoint_slug": entry.get("endpoint_slug") or "",
            "served_version": entry.get("served_version"),
            "experiment_id": entry.get("experiment_id"),
            "variant_name": entry.get("variant_name"),
            "http_status": int(entry.get("http_status", 500)),
            "success": bool(entry.get("success", False)),
            "error_type": entry.get("error_type"),
            "error_message": (entry.get("error_message") or "")[:500] if entry.get("error_message") else None,
            "total_latency_ms": entry.get("total_latency_ms"),
            "total_cost": entry.get("total_cost"),
            "workflow_run_id": entry.get("workflow_run_id"),
            "custom_metrics": entry.get("custom_metrics") if isinstance(entry.get("custom_metrics"), dict) else {},
        })
        supabase.table("api_request_log").insert(row).execute()
    except Exception as e:
        logger.warning("api_request_log insert failed: %s", e, exc_info=False)


async def log_api_request(entry: dict[str, Any]) -> None:
    """Fire-and-forget: run sync insert in thread so response is not blocked."""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, log_api_request_sync, entry)


def update_api_request_log_metrics_sync(request_id: str, custom_metrics: dict[str, Any]) -> None:
    """Merge custom_metrics into api_request_log.custom_metrics for the given request_id. Never raise."""
    try:
        r = supabase.table("api_request_log").select("custom_metrics").eq("id", request_id).limit(1).execute()
        if not r.data or len(r.data) == 0:
            return
        existing = r.data[0].get("custom_metrics") or {}
        if not isinstance(existing, dict):
            existing = {}
        merged = {**existing, **custom_metrics}
        supabase.table("api_request_log").update({"custom_metrics": merged}).eq("id", request_id).execute()
    except Exception as e:
        logger.warning("api_request_log custom_metrics update failed: %s", e, exc_info=False)


async def update_api_request_log_metrics(request_id: str, custom_metrics: dict[str, Any]) -> None:
    """Async wrapper for update_api_request_log_metrics_sync."""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, update_api_request_log_metrics_sync, request_id, custom_metrics)
