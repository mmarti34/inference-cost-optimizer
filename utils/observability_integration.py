"""
Helper functions to integrate observability logging into existing endpoints.
"""
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import Request
from routers.base import RouteInput, Outcome, FailureReason
from routers.router_manager import get_router
from utils.observability import log_request, log_span, update_daily_stats


def log_request_with_observability(
    request: Request,
    org_id: str,
    project_id: Optional[str],
    user_id: Optional[str],
    prompt_id: Optional[str],
    provider: str,
    model: str,
    prompt_text: str,
    result: Dict[str, Any],
    start_time: float,
    outcome: Outcome,
    failure_reason: Optional[FailureReason] = None,
) -> None:
    """
    Log a complete request with observability.
    This should be called after the LLM call completes.
    """
    request_id = getattr(request.state, 'request_id', None)
    trace_id = getattr(request.state, 'trace_id', None)
    
    if not request_id or not trace_id:
        # Generate IDs if middleware didn't add them
        from utils.observability import generate_request_id, generate_trace_id
        request_id = generate_request_id()
        trace_id = generate_trace_id()
        request.state.request_id = request_id
        request.state.trace_id = trace_id
    
    # Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Get router decision (use baseline for now)
    router = get_router()
    route_input = RouteInput(
        prompt=prompt_text,
        provider=provider,
        model=model,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        prompt_id=prompt_id,
    )
    route_decision = router.select_model(route_input)
    
    # Extract metrics from result
    prompt_tokens = result.get("input_tokens") or result.get("prompt_tokens")
    completion_tokens = result.get("output_tokens") or result.get("completion_tokens")
    total_tokens = result.get("total_tokens") or (prompt_tokens + completion_tokens if prompt_tokens and completion_tokens else None)
    cost_usd = result.get("cost_usd")
    response_text = result.get("response")
    
    # Log the request
    log_request(
        request_id=request_id,
        trace_id=trace_id,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        prompt_id=prompt_id,
        route_policy_name=route_decision.route_policy_name,
        route_policy_version=route_decision.route_policy_version,
        chosen_provider=provider,
        chosen_model=model,
        reason_code=route_decision.reason_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        outcome=outcome,
        failure_reason=failure_reason,
        prompt_text=prompt_text,
        response_text=response_text,
    )
    
    # Update daily stats (async would be better, but this works for now)
    if cost_usd and total_tokens:
        try:
            update_daily_stats(
                date=datetime.now(timezone.utc),
                org_id=org_id,
                project_id=project_id,
                provider=provider,
                model=model,
                request_count=1,
                success_count=1 if outcome == Outcome.SUCCESS else 0,
                failure_count=1 if outcome == Outcome.FAILURE else 0,
                fallback_count=1 if outcome == Outcome.FALLBACK_SUCCESS else 0,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                total_tokens=total_tokens or 0,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )
        except Exception as e:
            # Don't fail the request if stats update fails
            print(f"[Observability] Failed to update daily stats: {e}")


def log_gateway_span(request: Request, start_time: datetime) -> None:
    """Log gateway receive span."""
    request_id = getattr(request.state, 'request_id', None)
    trace_id = getattr(request.state, 'trace_id', None)
    
    if request_id and trace_id:
        log_span(
            request_id=request_id,
            trace_id=trace_id,
            span_type="gateway_receive",
            start_time=start_time,
            duration_ms=1,  # Gateway receive is instant
            status="success",
        )


def log_route_decision_span(request: Request, route_decision, start_time: datetime, duration_ms: int) -> None:
    """Log route decision span."""
    request_id = getattr(request.state, 'request_id', None)
    trace_id = getattr(request.state, 'trace_id', None)
    
    if request_id and trace_id:
        log_span(
            request_id=request_id,
            trace_id=trace_id,
            span_type="route_decision",
            start_time=start_time,
            duration_ms=duration_ms,
            status="success",
        )


def log_provider_call_span(
    request: Request,
    provider: str,
    model: str,
    start_time: datetime,
    duration_ms: int,
    status: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Log provider call span."""
    request_id = getattr(request.state, 'request_id', None)
    trace_id = getattr(request.state, 'trace_id', None)
    
    if request_id and trace_id:
        log_span(
            request_id=request_id,
            trace_id=trace_id,
            span_type="provider_call",
            provider=provider,
            model=model,
            attempt_number=1,
            start_time=start_time,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

