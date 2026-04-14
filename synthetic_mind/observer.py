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
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity & topic extraction
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

# Stopwords to skip when extracting content keywords
_STOPWORDS = frozenset(
    "i me my we our you your he she it they them their this that these those "
    "a an the is am are was were be been being have has had do does did will "
    "would shall should can could may might must need dare to of in for on "
    "with at by from as into through during before after above below between "
    "out off over under again further then once here there when where why how "
    "all each every both few more most other some such no nor not only own same "
    "so than too very just don doesn didn won t s d ll ve re m about also but "
    "and or if what which who whom what hi hello hey thanks thank please help "
    "question quick know get set up want like".split()
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


def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract meaningful content keywords from text.
    Uses simple word-frequency after filtering stopwords.
    """
    if not text:
        return []
    # Tokenize: lowercase, alpha-only words 3+ chars
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    filtered = [w for w in words if w not in _STOPWORDS and len(w) <= 30]
    counts = Counter(filtered)
    # Return top keywords by frequency
    return [word for word, _ in counts.most_common(max_keywords)]


def _classify_intent(text: str) -> str:
    """Classify the user's intent from their input text."""
    if not text:
        return "unknown"
    t = text.lower().strip()
    if any(w in t for w in ["how do i", "how to", "walk me through", "can you show", "steps to", "how does"]):
        return "how_to"
    if any(w in t for w in ["price", "pricing", "cost", "plan", "tier", "upgrade", "billing", "subscribe", "pay", "refund"]):
        return "pricing"
    if any(w in t for w in ["error", "bug", "broken", "not working", "can't", "won't", "issue",
                            "problem", "help me", "fix", "stuck", "failing", "disappeared", "stopped",
                            "not syncing", "slow", "loading", "crash", "wrong"]):
        return "troubleshooting"
    if any(w in t for w in ["what is", "what's", "what are", "does it", "do you", "does ",
                            "is there", "can i", "can you", "support", "have a", "have an"]):
        return "feature_question"
    if any(w in t for w in ["hello", "hi ", "hi!", "hey", "new user", "just signed", "getting started", "where do i start"]):
        return "onboarding"
    return "general"


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
    scope_key: Optional[str] = None,
    scope_value: Optional[str] = None,
) -> dict[str, Any]:
    """Build an observation record ready for DB insertion."""

    input_summary = _summarize(input_text)
    output_summary = _summarize(output_text)

    # Extract entities from both input and output (model names, providers)
    combined_text = (input_text or "") + " " + (output_text or "")
    entities = _extract_entities(combined_text)

    # Extract content keywords primarily from INPUT (user's actual question)
    # Output keywords are de-prioritized because outputs are often repetitive
    # (e.g., a chatbot greeting is the same every time)
    input_keywords = _extract_keywords(input_text, max_keywords=8)
    output_keywords = _extract_keywords(output_text, max_keywords=4)
    # Input keywords first, then unique output keywords
    seen = set(input_keywords)
    keywords = input_keywords + [k for k in output_keywords if k not in seen]
    intent = _classify_intent(input_text)

    # Extract KB asset IDs and agent tool calls from node_results for Phase 2/3
    kb_asset_ids = _extract_kb_asset_ids(node_results)
    agent_tool_calls = _extract_agent_tool_calls(node_results)

    # SM v2 Phase 4: capture variable content hash + char count for variable consolidation
    # When scope_key is set, we track the variable data so the consolidation engine
    # can detect repeated large variable payloads and consolidate them.
    variable_chars = 0
    variable_hash = None
    if scope_value and input_text:
        variable_chars = len(input_text)
        import hashlib as _hl
        variable_hash = _hl.sha256(input_text.encode()).hexdigest()[:16]

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
        "entities_mentioned": entities + keywords,  # combine model entities + content keywords
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost,
        "total_latency_ms": total_latency_ms,
        "error_message": (error_message or "")[:500] if error_message else None,
        "consolidated": False,
        # Store structured metadata for richer consolidation
        "metadata": {
            "intent": intent,
            "keywords": keywords[:10],
            "kb_asset_ids": kb_asset_ids,
            "agent_tool_calls": agent_tool_calls,
            "scope_key": scope_key,
            "scope_value": scope_value,
            "variable_hash": variable_hash,
            "variable_chars": variable_chars,
        },
    }


# ---------------------------------------------------------------------------
# SM v2: Extract KB and agent tool data from node_results
# ---------------------------------------------------------------------------

def _extract_kb_asset_ids(node_results: list[dict] | None) -> list[str]:
    """Extract KB asset IDs from context_trace in node_results."""
    if not node_results:
        return []
    asset_ids: list[str] = []
    for nr in node_results:
        trace = nr.get("context_trace") or {}
        if not trace.get("enabled"):
            continue
        for item in (trace.get("items_used") or []):
            ref = item.get("source_ref") or ""
            # Skip consolidation refs and non-asset refs
            if ref and not ref.startswith("sm_consolidation:") and item.get("source_type") == "knowledge_asset":
                asset_ids.append(ref)
    return sorted(set(asset_ids))


def _extract_agent_tool_calls(node_results: list[dict] | None) -> list[dict]:
    """Extract agent tool call summaries from reasoning_steps in node_results."""
    if not node_results:
        return []
    tool_calls: list[dict] = []
    for nr in node_results:
        if nr.get("type") != "agent":
            continue
        for step in (nr.get("reasoning_steps") or []):
            if step.get("type") != "act":
                continue
            tool_name = step.get("tool_name", "")
            tool_input = step.get("tool_input") or {}
            # Build a compact record for consolidation
            input_key = ""
            if tool_name == "get_knowledge_asset":
                input_key = tool_input.get("asset_id", "")
            elif tool_name == "search_knowledge_base":
                input_key = tool_input.get("query", "")[:100]

            # Find the corresponding observe step for output summary
            output_summary = ""
            step_num = step.get("step")
            if step_num is not None:
                for obs_step in (nr.get("reasoning_steps") or []):
                    if obs_step.get("type") == "observe" and obs_step.get("step") == step_num:
                        output_summary = _summarize(obs_step.get("content", ""), 200)
                        break

            tool_calls.append({
                "tool_name": tool_name,
                "input_key": input_key,
                "output_summary": output_summary,
                "observation_id": nr.get("node_id", ""),
            })
    return tool_calls[:20]  # Cap to prevent bloat


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
        # Extract metadata before insert (not a DB column yet, stored in entities)
        obs_copy = {k: v for k, v in obs.items() if k != "metadata"}
        supabase.table("sm_observations").insert(obs_copy).execute()

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
    scope_key: Optional[str] = None,
    scope_value: Optional[str] = None,
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
        scope_key=scope_key,
        scope_value=scope_value,
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _save_observation_sync, obs)
