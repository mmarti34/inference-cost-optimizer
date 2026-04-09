"""
Synthetic Mind — Observer subsystem.

Watches every request/response flowing through OptiML and extracts structured
observations asynchronously.  NEVER adds latency to the hot path.

Usage (from public_execution.py):
    from synthetic_mind.observer import observe_request
    asyncio.create_task(observe_request(observation_payload))
"""
import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity extraction (lightweight, regex-based for Phase 1)
# ---------------------------------------------------------------------------

# Common model name patterns
_MODEL_RE = re.compile(
    r"\b(gpt-4[o0]?(?:-mini|-turbo)?|gpt-3\.5-turbo|claude-(?:opus|sonnet|haiku)"
    r"(?:-[0-9.]+)?|gemini-(?:pro|flash|ultra)(?:-[0-9.]+)?|llama-?[0-9]+"
    r"(?:\.[0-9]+)?(?:b)?|mistral-(?:large|medium|small|tiny)(?:-[0-9.]+)?)\b",
    re.IGNORECASE,
)

# Provider names
_PROVIDER_RE = re.compile(
    r"\b(openai|anthropic|google|gemini|groq|together|mistral|cohere|deepseek|fireworks)\b",
    re.IGNORECASE,
)


def _extract_entities(text: str) -> list[str]:
    """Extract notable entities from text (models, providers, etc.)."""
    if not text:
        return []
    entities: set[str] = set()
    for m in _MODEL_RE.finditer(text):
        entities.add(m.group(1).lower())
    for m in _PROVIDER_RE.finditer(text):
        entities.add(m.group(1).lower())
    return sorted(entities)


def _summarize(text: str, max_len: int = 500) -> str:
    """Truncate to max_len while keeping whole words."""
    if not text or len(text) <= max_len:
        return text or ""
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------------
# Core observation builder
# ---------------------------------------------------------------------------

def build_observation(
    *,
    request_id: str,
    org_id: str,
    workflow_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    node_results: Optional[list[dict]] = None,
    total_cost: Optional[float] = None,
    total_latency_ms: Optional[int] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> dict[str, Any]:
    """Build an observation record ready for DB insertion."""

    input_summary = _summarize(input_text)
    output_summary = _summarize(output_text)

    # Merge entities from input and output
    entities = _extract_entities((input_text or "") + " " + (output_text or ""))

    return {
        "id": str(uuid.uuid4()),
        "request_id": request_id,
        "org_id": org_id,
        "workflow_id": workflow_id,
        "endpoint_slug": endpoint_slug,
        "user_id": user_id,
        "session_id": session_id,
        "observation_type": "response" if success else "error",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "entities_mentioned": entities,
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost,
        "total_latency_ms": total_latency_ms,
        "error_message": (error_message or "")[:500] if error_message else None,
        "consolidated": False,
    }


# ---------------------------------------------------------------------------
# Persistence (fire-and-forget)
# ---------------------------------------------------------------------------

# Auto-consolidation threshold: consolidate after this many unconsolidated
# observations accumulate for an org. Keeps the mind fresh without manual triggers.
AUTO_CONSOLIDATE_THRESHOLD = 50


def _save_observation_sync(obs: dict[str, Any]) -> None:
    """Insert observation into DB, then auto-consolidate if threshold reached. Never raises."""
    try:
        if not obs.get("org_id"):
            return
        supabase.table("sm_observations").insert(obs).execute()

        # Auto-consolidation: check if we've hit the threshold
        _maybe_auto_consolidate(obs["org_id"])
    except Exception as e:
        logger.warning("sm_observations insert failed: %s", e, exc_info=False)


def _maybe_auto_consolidate(org_id: str) -> None:
    """Run consolidation if unconsolidated observations exceed threshold."""
    try:
        r = (
            supabase.table("sm_observations")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("consolidated", False)
            .execute()
        )
        count = r.count or 0
        if count >= AUTO_CONSOLIDATE_THRESHOLD:
            from synthetic_mind.consolidation import consolidate_org
            from synthetic_mind.forgetting import run_forgetting_cycle
            logger.info("Auto-consolidating org %s (%d observations)", org_id, count)
            consolidate_org(org_id)
            run_forgetting_cycle(org_id)
    except Exception as e:
        logger.warning("Auto-consolidation check failed: %s", e, exc_info=False)


async def observe_request(
    *,
    request_id: str,
    org_id: str,
    workflow_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    node_results: Optional[list[dict]] = None,
    total_cost: Optional[float] = None,
    total_latency_ms: Optional[int] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    """
    Async entry point — build observation and persist.
    Called via asyncio.create_task() so it never blocks the response.
    """
    obs = build_observation(
        request_id=request_id,
        org_id=org_id,
        workflow_id=workflow_id,
        endpoint_slug=endpoint_slug,
        user_id=user_id,
        session_id=session_id,
        input_text=input_text,
        output_text=output_text,
        node_results=node_results,
        total_cost=total_cost,
        total_latency_ms=total_latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        provider=provider,
        success=success,
        error_message=error_message,
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _save_observation_sync, obs)
