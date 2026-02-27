"""
Retry + timeout wrapper for LLM provider calls.

Usage in workflow_runtime.py:
    from provider_resilience import call_with_resilience
    result = call_with_resilience(lambda: _execute_model_node(node_id, node, prompt, org_id))

Retry policy:
  - Retries on transient errors (429, 502, 503, 504, timeout, connection errors).
  - Does NOT retry on 400/401/403/404 (client errors that won't self-heal).
  - Exponential backoff: 1s, 2s, 4s (3 attempts total by default).

Timeout:
  - Each individual attempt is capped at ATTEMPT_TIMEOUT_SECONDS.
  - If the provider SDK doesn't respect the timeout internally, we enforce it
    by running the call in a thread and joining with a deadline.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, TypeVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")

# -- Configuration --
MAX_RETRIES = 3              # total attempts (1 initial + 2 retries)
INITIAL_BACKOFF_S = 1.0      # first retry waits 1s
BACKOFF_MULTIPLIER = 2.0     # each subsequent wait doubles
ATTEMPT_TIMEOUT_S = 120.0    # per-attempt timeout (2 minutes)

# HTTP status codes that are considered transient (worth retrying)
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

# Exception class names that indicate transient/connection failures
_RETRYABLE_EXCEPTION_NAMES = {
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "TimeoutError",
    "APIConnectionError",      # OpenAI SDK
    "APITimeoutError",         # OpenAI SDK
    "RateLimitError",          # OpenAI SDK
    "InternalServerError",     # OpenAI SDK
    "OverloadedError",         # Anthropic SDK
    "APIStatusError",          # shared
}

# Shared thread pool for timeout enforcement.
# Using a small pool because each provider call can be long-running.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="provider-call")


def _is_retryable(exc: Exception) -> bool:
    """Decide whether an exception is transient and worth retrying."""
    # FastAPI HTTPException with retryable status
    if isinstance(exc, HTTPException) and exc.status_code in _RETRYABLE_STATUS_CODES:
        return True

    # Provider SDK exceptions (match by class name to avoid import dependency)
    exc_name = type(exc).__name__
    if exc_name in _RETRYABLE_EXCEPTION_NAMES:
        return True

    # Some SDKs wrap the status code in a `status_code` attribute
    status = getattr(exc, "status_code", None)
    if status and int(status) in _RETRYABLE_STATUS_CODES:
        return True

    # Timeout from concurrent.futures
    if isinstance(exc, (FuturesTimeoutError, TimeoutError)):
        return True

    return False


def call_with_resilience(
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_S,
    attempt_timeout: float = ATTEMPT_TIMEOUT_S,
    context_label: str = "provider call",
) -> T:
    """
    Execute `fn()` with retry + per-attempt timeout.

    Returns the result of fn() on success.
    Raises the last exception if all attempts fail.
    Non-retryable exceptions are raised immediately without retry.
    """
    last_exc: Exception | None = None
    backoff = initial_backoff

    for attempt in range(1, max_retries + 1):
        try:
            # Run fn in a thread with a timeout to enforce the deadline even if
            # the SDK call hangs.  If fn is already running in the main thread
            # (workflow_runtime executes synchronously), this is safe.
            future = _executor.submit(fn)
            return future.result(timeout=attempt_timeout)

        except FuturesTimeoutError:
            last_exc = HTTPException(
                status_code=504,
                detail=f"Provider call timed out after {attempt_timeout}s",
            )
            logger.warning(
                "%s attempt %d/%d timed out after %.0fs",
                context_label, attempt, max_retries, attempt_timeout,
            )

        except Exception as exc:
            last_exc = exc

            if not _is_retryable(exc):
                # Non-retryable (400, 401, 403, 404, etc.) — fail immediately
                raise

            logger.warning(
                "%s attempt %d/%d failed with %s: %s — retrying in %.1fs",
                context_label, attempt, max_retries,
                type(exc).__name__, str(exc)[:200], backoff,
            )

        # Don't sleep after the last attempt
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER

    # All attempts exhausted
    if last_exc is not None:
        raise last_exc
    # Should never reach here, but just in case
    raise HTTPException(status_code=500, detail="All retry attempts exhausted")
