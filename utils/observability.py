"""
Observability utilities for request logging, tracing, and metrics.
"""
import os
import uuid
import hashlib
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from supabase_client import supabase
from routers.base import Outcome, FailureReason


# Environment flags
STORE_PROMPTS = os.getenv("OPTIML_STORE_PROMPTS", "false").lower() == "true"
ENABLE_REPLAY = os.getenv("OPTIML_ENABLE_REPLAY", "false").lower() == "true"


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID"""
    return str(uuid.uuid4())


def hash_user_identifier(user_id: Optional[str]) -> Optional[str]:
    """Hash user identifier for privacy"""
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def log_request(
    request_id: str,
    trace_id: str,
    org_id: str,
    project_id: Optional[str],
    user_id: Optional[str],
    prompt_id: Optional[str],
    route_policy_name: str,
    route_policy_version: str,
    chosen_provider: str,
    chosen_model: str,
    reason_code: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    cost_usd: Optional[float],
    latency_ms: int,
    outcome: Outcome,
    failure_reason: Optional[FailureReason] = None,
    json_valid: Optional[bool] = None,
    schema_valid: Optional[bool] = None,
    prompt_text: Optional[str] = None,
    response_text: Optional[str] = None,
) -> None:
    """
    Log a request to the request_logs table.
    """
    try:
        data = {
            "request_id": request_id,
            "trace_id": trace_id,
            "org_id": org_id,
            "project_id": project_id,
            "user_id": user_id,
            "user_identifier_hash": hash_user_identifier(user_id),
            "prompt_id": prompt_id,
            "route_policy_name": route_policy_name,
            "route_policy_version": route_policy_version,
            "chosen_provider": chosen_provider,
            "chosen_model": chosen_model,
            "reason_code": reason_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "outcome": outcome.value,
            "failure_reason": failure_reason.value if failure_reason else None,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
        }
        
        # Only store prompts if flag is enabled
        if STORE_PROMPTS:
            if prompt_text:
                # Optionally truncate or redact sensitive data
                data["prompt_text"] = prompt_text[:10000]  # Limit length
            if response_text:
                data["response_text"] = response_text[:10000]  # Limit length
        
        supabase.table("request_logs").insert(data).execute()
    except Exception as e:
        # Don't fail the request if logging fails
        print(f"[Observability] Failed to log request: {e}")


def log_span(
    request_id: str,
    trace_id: str,
    span_type: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    attempt_number: int = 1,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    duration_ms: Optional[int] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> None:
    """
    Log a trace span to the request_spans table.
    """
    try:
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        if end_time is None and duration_ms is not None:
            end_time = start_time.replace(microsecond=start_time.microsecond + duration_ms * 1000)
        if duration_ms is None and end_time is not None:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        data = {
            "request_id": request_id,
            "trace_id": trace_id,
            "span_type": span_type,
            "provider": provider,
            "model": model,
            "attempt_number": attempt_number,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "duration_ms": duration_ms,
            "status": status,
            "error_message": error_message,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }
        
        supabase.table("request_spans").insert(data).execute()
    except Exception as e:
        # Don't fail the request if logging fails
        print(f"[Observability] Failed to log span: {e}")


def update_daily_stats(
    date: datetime,
    org_id: str,
    project_id: Optional[str],
    provider: str,
    model: str,
    request_count: int = 1,
    success_count: int = 0,
    failure_count: int = 0,
    fallback_count: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: Optional[int] = None,
    timeout_count: int = 0,
    rate_limit_count: int = 0,
    provider_5xx_count: int = 0,
    invalid_json_count: int = 0,
    tool_error_count: int = 0,
    safety_count: int = 0,
    json_valid_count: int = 0,
    schema_valid_count: int = 0,
    refusal_count: int = 0,
) -> None:
    """
    Update or insert daily statistics for a model/provider.
    This is typically called asynchronously or in a background job.
    """
    try:
        date_str = date.date().isoformat()
        
        # Try to get existing record
        existing = supabase.table("model_stats_daily") \
            .select("id, date, org_id, project_id, provider, model, request_count, success_count, failure_count, fallback_count, total_prompt_tokens, total_completion_tokens, total_tokens, total_cost_usd, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms, timeout_count, rate_limit_count, provider_5xx_count, invalid_json_count, tool_error_count, safety_count, json_valid_count, schema_valid_count, refusal_count, created_at") \
            .eq("date", date_str) \
            .eq("org_id", org_id) \
            .eq("provider", provider) \
            .eq("model", model) \
            .execute()
        
        if existing.data:
            # Update existing record
            record = existing.data[0]
            new_data = {
                "request_count": (record.get("request_count", 0) or 0) + request_count,
                "success_count": (record.get("success_count", 0) or 0) + success_count,
                "failure_count": (record.get("failure_count", 0) or 0) + failure_count,
                "fallback_count": (record.get("fallback_count", 0) or 0) + fallback_count,
                "total_prompt_tokens": (record.get("total_prompt_tokens", 0) or 0) + prompt_tokens,
                "total_completion_tokens": (record.get("total_completion_tokens", 0) or 0) + completion_tokens,
                "total_tokens": (record.get("total_tokens", 0) or 0) + total_tokens,
                "total_cost_usd": (record.get("total_cost_usd", 0) or 0) + cost_usd,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            # Update latency percentiles (simplified - in production, use proper percentile calculation)
            if latency_ms is not None:
                current_avg = record.get("avg_latency_ms")
                if current_avg:
                    new_data["avg_latency_ms"] = int((current_avg + latency_ms) / 2)
                else:
                    new_data["avg_latency_ms"] = latency_ms
            
            # Update error counts
            new_data["timeout_count"] = (record.get("timeout_count", 0) or 0) + timeout_count
            new_data["rate_limit_count"] = (record.get("rate_limit_count", 0) or 0) + rate_limit_count
            new_data["provider_5xx_count"] = (record.get("provider_5xx_count", 0) or 0) + provider_5xx_count
            new_data["invalid_json_count"] = (record.get("invalid_json_count", 0) or 0) + invalid_json_count
            new_data["tool_error_count"] = (record.get("tool_error_count", 0) or 0) + tool_error_count
            new_data["safety_count"] = (record.get("safety_count", 0) or 0) + safety_count
            new_data["json_valid_count"] = (record.get("json_valid_count", 0) or 0) + json_valid_count
            new_data["schema_valid_count"] = (record.get("schema_valid_count", 0) or 0) + schema_valid_count
            new_data["refusal_count"] = (record.get("refusal_count", 0) or 0) + refusal_count
            
            supabase.table("model_stats_daily") \
                .update(new_data) \
                .eq("id", record["id"]) \
                .execute()
        else:
            # Insert new record
            data = {
                "date": date_str,
                "org_id": org_id,
                "project_id": project_id,
                "provider": provider,
                "model": model,
                "request_count": request_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "fallback_count": fallback_count,
                "total_prompt_tokens": prompt_tokens,
                "total_completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": cost_usd,
                "avg_latency_ms": latency_ms,
                "timeout_count": timeout_count,
                "rate_limit_count": rate_limit_count,
                "provider_5xx_count": provider_5xx_count,
                "invalid_json_count": invalid_json_count,
                "tool_error_count": tool_error_count,
                "safety_count": safety_count,
                "json_valid_count": json_valid_count,
                "schema_valid_count": schema_valid_count,
                "refusal_count": refusal_count,
            }
            
            supabase.table("model_stats_daily").insert(data).execute()
    except Exception as e:
        # Don't fail if stats update fails
        print(f"[Observability] Failed to update daily stats: {e}")

