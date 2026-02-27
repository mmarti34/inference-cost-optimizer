"""
Provider call wrapper with span tracking, fallback support, and observability.
"""
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
from fastapi import HTTPException
from utils.observability import log_span
from utils.pricing import get_pricing
from routers.base import Outcome, FailureReason


class ProviderCallResult:
    """Result of a provider call"""
    def __init__(
        self,
        success: bool,
        response: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        error: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
    ):
        self.success = success
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms
        self.error = error
        self.failure_reason = failure_reason


def classify_failure_reason(error: Exception) -> FailureReason:
    """Classify error type into failure reason"""
    error_str = str(error).lower()
    
    if "timeout" in error_str or "timed out" in error_str:
        return FailureReason.TIMEOUT
    elif "rate limit" in error_str or "429" in error_str:
        return FailureReason.RATE_LIMIT
    elif "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return FailureReason.PROVIDER_5XX
    elif "json" in error_str or "parse" in error_str:
        return FailureReason.INVALID_JSON
    elif "tool" in error_str or "function" in error_str:
        return FailureReason.TOOL_ERROR
    elif "safety" in error_str or "content policy" in error_str:
        return FailureReason.SAFETY
    else:
        return FailureReason.UNKNOWN


def is_retriable_error(failure_reason: FailureReason) -> bool:
    """Determine if an error is retriable (should trigger fallback)"""
    retriable = [
        FailureReason.TIMEOUT,
        FailureReason.RATE_LIMIT,
        FailureReason.PROVIDER_5XX,
    ]
    return failure_reason in retriable


def call_provider_with_span(
    request_id: str,
    trace_id: str,
    provider: str,
    model: str,
    call_func: Callable,
    attempt_number: int = 1,
    span_type: str = "provider_call",
) -> ProviderCallResult:
    """
    Call a provider function with span tracking and error handling.
    
    Args:
        request_id: Request ID for logging
        trace_id: Trace ID for grouping spans
        provider: Provider name
        model: Model name
        call_func: Function to call that returns (response_text, usage_dict)
        attempt_number: Attempt number (1 for primary, 2+ for fallbacks)
        span_type: Type of span (provider_call or fallback_call)
    
    Returns:
        ProviderCallResult with success status and metrics
    """
    start_time = datetime.now(timezone.utc)
    start_ts = time.time()
    
    try:
        # Call the provider function
        response_text, usage_dict = call_func()
        
        # Calculate latency
        end_ts = time.time()
        latency_ms = int((end_ts - start_ts) * 1000)
        end_time = datetime.now(timezone.utc)
        
        # Extract token usage
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)
        total_tokens = usage_dict.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Calculate cost
        pricing = get_pricing(provider, model)
        cost_usd = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000
        
        # Log successful span
        log_span(
            request_id=request_id,
            trace_id=trace_id,
            span_type=span_type,
            provider=provider,
            model=model,
            attempt_number=attempt_number,
            start_time=start_time,
            end_time=end_time,
            duration_ms=latency_ms,
            status="success",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )
        
        return ProviderCallResult(
            success=True,
            response=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        # Calculate latency even on failure
        end_ts = time.time()
        latency_ms = int((end_ts - start_ts) * 1000)
        end_time = datetime.now(timezone.utc)
        
        # Classify failure
        failure_reason = classify_failure_reason(e)
        error_message = str(e)
        
        # Determine status
        if failure_reason == FailureReason.TIMEOUT:
            status = "timeout"
        elif failure_reason == FailureReason.RATE_LIMIT:
            status = "rate_limited"
        else:
            status = "failure"
        
        # Log failed span
        log_span(
            request_id=request_id,
            trace_id=trace_id,
            span_type=span_type,
            provider=provider,
            model=model,
            attempt_number=attempt_number,
            start_time=start_time,
            end_time=end_time,
            duration_ms=latency_ms,
            status=status,
            error_message=error_message,
        )
        
        return ProviderCallResult(
            success=False,
            latency_ms=latency_ms,
            error=error_message,
            failure_reason=failure_reason,
        )


def execute_with_fallbacks(
    request_id: str,
    trace_id: str,
    primary_provider: str,
    primary_model: str,
    primary_call_func: Callable,
    fallback_chain: List[Dict[str, str]],
    max_retries: int = 3,
) -> ProviderCallResult:
    """
    Execute provider call with fallback chain support.
    
    Args:
        request_id: Request ID
        trace_id: Trace ID
        primary_provider: Primary provider name
        primary_model: Primary model name
        primary_call_func: Function to call primary provider
        fallback_chain: List of {provider, model} dicts for fallbacks
        max_retries: Maximum number of retries (including primary)
    
    Returns:
        ProviderCallResult from first successful call, or last failure
    """
    # Try primary provider
    result = call_provider_with_span(
        request_id=request_id,
        trace_id=trace_id,
        provider=primary_provider,
        model=primary_model,
        call_func=primary_call_func,
        attempt_number=1,
        span_type="provider_call",
    )
    
    if result.success:
        return result
    
    # If primary failed and error is retriable, try fallbacks
    if is_retriable_error(result.failure_reason) and len(fallback_chain) > 0:
        for idx, fallback in enumerate(fallback_chain[:max_retries - 1], start=2):
            fallback_provider = fallback.get("provider")
            fallback_model = fallback.get("model")
            
            if not fallback_provider or not fallback_model:
                continue
            
            # Create fallback call function (this would need to be passed or constructed)
            # For now, we'll need to handle this at the router level
            # This is a placeholder - actual implementation depends on how routers are structured
            pass
    
    # Return the last failure if all attempts failed
    return result

