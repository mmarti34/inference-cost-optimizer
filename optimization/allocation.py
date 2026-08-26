"""
Allocation decisions: "OptiML chose strategy X for workload Y because of
objective Z under policy P."

Deliberately small. This records decisions the product already makes today
(which candidate a benchmark selected, which recommendation was acted on) so
that the record exists and is auditable. It is NOT an allocation engine, and
nothing here selects a strategy at request time.

Its value is that a future allocation engine — one that routes a unit of work
to a model, an agent, deterministic software or a person — reads and writes
exactly this table. Recording the REJECTED options alongside the selected one is
what makes such a decision reviewable rather than a black box.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import domain

logger = logging.getLogger(__name__)

ALLOCATION_COLS = (
    "id, org_id, workload_id, recommendation_id, policy_id, decision_kind, objective, "
    "objective_config, considered_strategies, selected_strategy_id, expected_cost_usd, "
    "expected_quality, expected_latency_p95_ms, confidence, reason, decided_at, "
    "actual_result, resolved_at, created_at"
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_decision(
    org_id: str,
    *,
    workload_id: Optional[str],
    decision_kind: str = "recommendation",
    objective: str = "cost",
    considered: Optional[list[dict]] = None,
    selected_strategy_id: Optional[str] = None,
    policy_id: Optional[str] = None,
    recommendation_id: Optional[str] = None,
    expected_cost_usd: Optional[float] = None,
    expected_quality: Optional[float] = None,
    expected_latency_p95_ms: Optional[int] = None,
    confidence: Optional[float] = None,
    reason: Optional[str] = None,
    objective_config: Optional[dict] = None,
) -> Optional[dict]:
    """
    Record one decision. Never raises: an audit-trail failure must not break the
    decision it was recording.

    `considered` should include the options that were REJECTED and why. A
    decision that only records its winner cannot be reviewed.
    """
    row: dict[str, Any] = {
        "org_id": org_id,
        "workload_id": workload_id,
        "recommendation_id": recommendation_id,
        "policy_id": policy_id,
        "decision_kind": decision_kind,
        "objective": objective if domain.is_valid_objective(objective) else "cost",
        "objective_config": objective_config or {},
        "considered_strategies": considered or [],
        "selected_strategy_id": selected_strategy_id,
        "expected_cost_usd": expected_cost_usd,
        "expected_quality": expected_quality,
        "expected_latency_p95_ms": expected_latency_p95_ms,
        "confidence": confidence,
        "reason": reason,
        "decided_at": _iso_now(),
    }
    try:
        result = supabase.table("allocation_decisions").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("record_decision failed: %s", type(exc).__name__)
        return None


def resolve_decision(org_id: str, decision_id: str, actual_result: dict) -> Optional[dict]:
    """
    Backfill what actually happened, so expected-vs-actual calibration becomes
    measurable. Never pre-filled with the expectation.
    """
    try:
        result = (
            supabase.table("allocation_decisions")
            .update({"actual_result": actual_result, "resolved_at": _iso_now()})
            .eq("id", decision_id)
            .eq("org_id", org_id)
            .execute()
        )
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("resolve_decision failed: %s", type(exc).__name__)
        return None


def list_decisions(
    org_id: str, *, workload_id: Optional[str] = None, limit: int = 100
) -> list[dict]:
    try:
        q = supabase.table("allocation_decisions").select(ALLOCATION_COLS).eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        resp = q.order("decided_at", desc=True).limit(max(1, min(limit, 200))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_decisions failed: %s", type(exc).__name__)
        return []
