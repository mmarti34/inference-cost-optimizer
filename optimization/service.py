"""
Recommendation CRUD, lifecycle transitions and the audit trail.

Two invariants this module enforces:

  EVERY STATE CHANGE IS AUDITED. `transition` appends to the `audit` JSONB array
  (who, what, when, from -> to, why). The array is append-only; nothing here
  rewrites history.

  ONLY LEGAL TRANSITIONS HAPPEN. domain.LEGAL_TRANSITIONS is the state machine
  and there is no edge from 'verified' straight to 'canary': human approval is
  the default path, and production is never changed autonomously unless an
  optimization_policies.automation flag says so.

And one relationship it respects:

  EVIDENCE PRECEDES RECOMMENDATION. A recommendation CITES benchmarks through
  `recommendation_evidence`; it does not own them. Creating a recommendation
  requires pointing at the evidence that justified it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import domain, policies

logger = logging.getLogger(__name__)

RECOMMENDATION_COLS = (
    "id, org_id, project_id, workload_id, status, title, dimensions, "
    "baseline_strategy, candidate_strategy, baseline_strategy_id, candidate_strategy_id, "
    "baseline_version, candidate_version, generator, rationale, evidence_source, "
    "evidence_strength, sample_size, baseline_cost, candidate_cost, "
    "projected_savings_usd, verified_savings_usd, realized_savings_usd, "
    "baseline_quality, candidate_quality, quality_provenance, success_signal, "
    "baseline_latency_p95_ms, candidate_latency_p95_ms, baseline_error_rate, "
    "candidate_error_rate, confidence, constraints, objective, objective_config, "
    "policy_id, parent_recommendation_id, supersedes_id, bundle_id, baseline_reference, "
    "approval_required, approved_by, approved_at, promoted_at, decided_by, decided_at, "
    "deployment_id, experiment_id, rolled_back_at, monitoring_status, "
    "realized_window_start, realized_window_end, realized_metrics, audit, "
    "quality_safety, created_at, updated_at"
)

EVIDENCE_COLS = (
    "id, org_id, recommendation_id, benchmark_id, evidence_role, created_at"
)

STRATEGY_COLS = (
    "id, org_id, workload_id, name, description, kind, steps, surface_binding, "
    "dimensions, fingerprint, created_at, updated_at"
)


class RecommendationError(ValueError):
    """Invalid recommendation operation."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def upsert_strategy(
    org_id: str,
    strategy,
    *,
    workload_id: Optional[str] = None,
    kind: str = "candidate",
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[dict]:
    """
    Persist an execution strategy, deduped on (org_id, fingerprint).

    Identical candidates produced by different generators collapse to one row,
    so a customer is not shown the same change twice under two names.
    """
    fingerprint = strategy.fingerprint()
    try:
        existing = (
            supabase.table("execution_strategies")
            .select(STRATEGY_COLS)
            .eq("org_id", org_id)
            .eq("fingerprint", fingerprint)
            .limit(1)
            .execute()
        )
        rows = getattr(existing, "data", None) or []
        if rows:
            return rows[0]

        payload = strategy.to_dict()
        result = supabase.table("execution_strategies").insert({
            "org_id": org_id,
            "workload_id": workload_id,
            "name": name or f"strategy:{fingerprint[:12]}",
            "description": description,
            "kind": kind,
            "steps": payload["steps"],
            "surface_binding": payload["surface_binding"],
            "dimensions": payload["dimensions"],
            "fingerprint": fingerprint,
            "updated_at": _iso_now(),
        }).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("upsert_strategy failed: %s", type(exc).__name__)
        return None


def get_strategy(org_id: str, strategy_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("execution_strategies")
            .select(STRATEGY_COLS)
            .eq("id", strategy_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("get_strategy failed: %s", type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def list_recommendations(
    org_id: str,
    *,
    status: Optional[str] = None,
    workload_id: Optional[str] = None,
    bundle_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    try:
        q = (
            supabase.table("optimization_recommendations")
            .select(RECOMMENDATION_COLS)
            .eq("org_id", org_id)
        )
        if status:
            q = q.eq("status", status)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        if bundle_id:
            q = q.eq("bundle_id", bundle_id)
        resp = q.order("created_at", desc=True).limit(max(1, min(limit, 200))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_recommendations failed: %s", type(exc).__name__)
        return []


def get_recommendation(org_id: str, rec_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("optimization_recommendations")
            .select(RECOMMENDATION_COLS)
            .eq("id", rec_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("get_recommendation failed: %s", type(exc).__name__)
        return None


def find_ancestor_by_strategy_fingerprint(
    org_id: str, workload_id: str, fingerprint: Optional[str]
) -> Optional[dict]:
    """
    The recommendation, if any, whose CANDIDATE configuration is the one now
    serving as this workload's baseline.

    A chained optimization is measured against its parent's result, so the
    parent's savings are already embedded in the baseline the child improved on.
    Summing both claims the same dollars twice. Recording the ancestor here lets
    `domain.attributable_savings` drop it from the total — which is why the
    identification has to happen at creation time, while the baseline strategy
    that was actually measured is still in hand.

    Returns None when no live recommendation produced this baseline, which is
    the ordinary case for a first optimization.
    """
    if not fingerprint:
        return None
    for status in domain.LIVE_STATUSES:
        for row in list_recommendations(
            org_id, status=status, workload_id=workload_id, limit=200
        ):
            cand = row.get("candidate_strategy")
            if isinstance(cand, dict) and cand.get("fingerprint") == fingerprint:
                return row
    return None


def create_recommendation(
    org_id: str,
    *,
    workload_id: str,
    title: str,
    candidate_strategy,
    baseline_strategy,
    dimensions: list[str],
    generator: Optional[str] = None,
    rationale: Optional[str] = None,
    objective: str = "cost",
    project_id: Optional[str] = None,
    evidence_benchmark_ids: Optional[list[str]] = None,
    projected_savings_usd: Optional[float] = None,
    baseline_reference: Optional[dict] = None,
    parent_recommendation_id: Optional[str] = None,
    bundle_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> Optional[dict]:
    """
    Create a recommendation from candidate evidence.

    `projected_savings_usd` is an EXTRAPOLATION and is written only to the
    projected column. Verified and realized savings are written elsewhere, by
    the benchmark and by post-promotion monitoring respectively — never here.

    `approval_required` follows the workload's policy and defaults to TRUE.
    """
    policy = policies.get_effective_policy(org_id, workload_id)

    baseline_row = upsert_strategy(
        org_id, baseline_strategy, workload_id=workload_id, kind="baseline",
        name="Current configuration",
    )
    candidate_row = upsert_strategy(
        org_id, candidate_strategy, workload_id=workload_id, kind="candidate", name=title,
    )

    row: dict[str, Any] = {
        "org_id": org_id,
        "project_id": project_id,
        "workload_id": workload_id,
        "status": domain.STATUS_DISCOVERED,
        "title": title,
        "dimensions": dimensions,
        "baseline_strategy": baseline_strategy.to_dict(),
        "candidate_strategy": candidate_strategy.to_dict(),
        "baseline_strategy_id": (str(baseline_row["id"]) if baseline_row else None),
        "candidate_strategy_id": (str(candidate_row["id"]) if candidate_row else None),
        "generator": generator,
        # Prose. Explicitly not evidence, and never sufficient for 'verified'.
        "rationale": rationale,
        "evidence_source": "none",
        "evidence_strength": 0,
        "objective": objective if domain.is_valid_objective(objective) else "cost",
        "policy_id": (str(policy["id"]) if policy and policy.get("id") else None),
        "constraints": policies.constraints_of(policy),
        "approval_required": policies.approval_required(policy),
        "projected_savings_usd": projected_savings_usd,
        "baseline_reference": baseline_reference or {},
        "parent_recommendation_id": parent_recommendation_id,
        "bundle_id": bundle_id,
        "audit": [{
            "at": _iso_now(),
            "actor": actor,
            "action": "created",
            "from_status": None,
            "to_status": domain.STATUS_DISCOVERED,
            "reason": f"generator:{generator}" if generator else None,
        }],
        "updated_at": _iso_now(),
    }

    try:
        result = supabase.table("optimization_recommendations").insert(row).execute()
        created = (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("create_recommendation failed: %s", type(exc).__name__)
        return None

    if created and evidence_benchmark_ids:
        for bid in evidence_benchmark_ids:
            cite_evidence(org_id, str(created["id"]), bid)

    return created


def transition(
    org_id: str,
    rec_id: str,
    to_status: str,
    *,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
    extra_fields: Optional[dict] = None,
) -> dict:
    """
    Move a recommendation to a new state, validating the transition and
    appending to the audit trail.

    Raises RecommendationError on an unknown recommendation and
    domain.IllegalTransition on an illegal move. Both are deliberate: silently
    ignoring an illegal transition would let a candidate reach production
    without passing through approval.
    """
    current = get_recommendation(org_id, rec_id)
    if current is None:
        raise RecommendationError("Recommendation not found for this organization.")

    from_status = current.get("status")
    domain.assert_transition(from_status, to_status)

    audit = current.get("audit")
    if not isinstance(audit, list):
        audit = []
    audit.append({
        "at": _iso_now(),
        "actor": actor,
        "action": "transition",
        "from_status": from_status,
        "to_status": to_status,
        "reason": reason,
    })

    patch: dict[str, Any] = {
        "status": to_status,
        "audit": audit,
        "updated_at": _iso_now(),
    }
    patch.update(extra_fields or {})

    # Timestamps that record a human decision, set exactly once at the moment
    # the decision was taken.
    if to_status in (domain.STATUS_REJECTED, domain.STATUS_CANARY, domain.STATUS_SHADOWING):
        patch.setdefault("decided_by", actor)
        patch.setdefault("decided_at", _iso_now())
    if to_status in (domain.STATUS_CANARY, domain.STATUS_SHADOWING):
        patch.setdefault("approved_by", actor)
        patch.setdefault("approved_at", _iso_now())
    if to_status == domain.STATUS_PROMOTED:
        patch.setdefault("promoted_at", _iso_now())
        # Realized measurement starts at promotion; it is NOT the benchmark
        # result carried forward.
        patch.setdefault("monitoring_status", "monitoring")
        patch.setdefault("realized_window_start", _iso_now())
    if to_status == domain.STATUS_ROLLED_BACK:
        patch.setdefault("rolled_back_at", _iso_now())
        patch.setdefault("monitoring_status", "stopped")

    try:
        result = (
            supabase.table("optimization_recommendations")
            .update(patch)
            .eq("id", rec_id)
            .eq("org_id", org_id)
            .execute()
        )
        updated = (result.data or [None])[0]
        if updated is None:
            raise RecommendationError("Failed to update the recommendation.")
        return updated
    except RecommendationError:
        raise
    except Exception as exc:  # pragma: no cover
        logger.warning("transition failed: %s", type(exc).__name__)
        raise RecommendationError("Failed to update the recommendation.") from exc


def require_evidence(org_id: str, rec_id: str) -> list[dict]:
    """
    The benchmarks this recommendation CITES.

    A recommendation may only be approved on the back of cited evidence — the
    whole product thesis in one function.
    """
    try:
        resp = (
            supabase.table("recommendation_evidence")
            .select(EVIDENCE_COLS)
            .eq("org_id", org_id)
            .eq("recommendation_id", rec_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("require_evidence failed: %s", type(exc).__name__)
        return []


def cite_evidence(
    org_id: str, rec_id: str, benchmark_id: str, *, evidence_role: str = "primary"
) -> Optional[dict]:
    """Link a benchmark as evidence for a recommendation. Idempotent."""
    try:
        result = supabase.table("recommendation_evidence").insert({
            "org_id": org_id,
            "recommendation_id": rec_id,
            "benchmark_id": benchmark_id,
            "evidence_role": evidence_role,
        }).execute()
        return (result.data or [None])[0]
    except Exception:
        # Unique index: already cited. Citing twice is a no-op.
        return None


def supersede(org_id: str, old_rec_id: str, new_rec_id: str, *, actor: Optional[str] = None) -> None:
    """
    Mark an older recommendation as superseded by a newer one.

    Keeps savings attribution non-overlapping: a superseded recommendation's
    savings are already embedded in the baseline the newer one measured against.
    """
    try:
        supabase.table("optimization_recommendations").update({
            "supersedes_id": old_rec_id, "updated_at": _iso_now(),
        }).eq("id", new_rec_id).eq("org_id", org_id).execute()
        transition(
            org_id, old_rec_id, domain.STATUS_SUPERSEDED,
            actor=actor, reason=f"superseded_by:{new_rec_id}",
        )
    except Exception as exc:
        logger.warning("supersede failed: %s", exc)


def recommendation_row_to_response(row: dict, *, evidence: Optional[list[dict]] = None) -> dict:
    """
    API shape.

    Every measured field may be null, and null always means NOT MEASURED. The
    three savings figures are returned under separate keys with their meanings
    attached, so a client cannot read a projection as a realized result.
    """
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "project_id": (str(row["project_id"]) if row.get("project_id") else None),
        "workload_id": (str(row["workload_id"]) if row.get("workload_id") else None),
        "status": row.get("status"),
        "title": row.get("title"),
        "dimensions": row.get("dimensions") or [],
        "objective": row.get("objective"),
        "objective_config": row.get("objective_config") or {},
        "generator": row.get("generator"),
        # Prose written to explain the proposal. NOT evidence.
        "rationale": row.get("rationale"),
        "evidence": {
            "source": row.get("evidence_source"),
            "strength": row.get("evidence_strength"),
            "sample_size": row.get("sample_size"),
            "confidence": row.get("confidence"),
            "confidence_band": domain.confidence_band(row.get("confidence")),
            "benchmarks": [
                {
                    "benchmark_id": str(e["benchmark_id"]),
                    "role": e.get("evidence_role"),
                    "cited_at": e.get("created_at"),
                }
                for e in (evidence or [])
            ],
        },
        "cost": {
            "baseline": row.get("baseline_cost"),
            "candidate": row.get("candidate_cost"),
        },
        "savings": {
            "projected_usd": row.get("projected_savings_usd"),
            "verified_usd": row.get("verified_savings_usd"),
            "realized_usd": row.get("realized_savings_usd"),
            "meanings": {
                "projected_usd": "Extrapolated from measured per-call delta x observed traffic volume.",
                "verified_usd": "Measured inside a benchmark or canary, over the sample only.",
                "realized_usd": "Observed in production after promotion.",
            },
            "baseline_reference": row.get("baseline_reference") or {},
        },
        "quality": {
            "baseline": row.get("baseline_quality"),
            "candidate": row.get("candidate_quality"),
            "provenance": row.get("quality_provenance"),
            "success_signal": row.get("success_signal") or {},
            # Whether a material regression against the baseline was RULED OUT,
            # with the paired evidence behind it. NULL means not established —
            # never "fine". A client reading candidate=0.90 next to
            # baseline=1.00 must be able to see that this was never checked, or
            # that it was checked and failed.
            "safety": row.get("quality_safety"),
            "regression_ruled_out": (
                bool((row.get("quality_safety") or {}).get("established"))
                if row.get("quality_safety") else None
            ),
        },
        "latency": {
            "baseline_p95_ms": row.get("baseline_latency_p95_ms"),
            "candidate_p95_ms": row.get("candidate_latency_p95_ms"),
        },
        "reliability": {
            "baseline_error_rate": row.get("baseline_error_rate"),
            "candidate_error_rate": row.get("candidate_error_rate"),
        },
        "strategies": {
            "baseline_id": (str(row["baseline_strategy_id"]) if row.get("baseline_strategy_id") else None),
            "candidate_id": (str(row["candidate_strategy_id"]) if row.get("candidate_strategy_id") else None),
            "baseline": row.get("baseline_strategy"),
            "candidate": row.get("candidate_strategy"),
        },
        "governance": {
            "policy_id": (str(row["policy_id"]) if row.get("policy_id") else None),
            "constraints": row.get("constraints") or {},
            "approval_required": bool(row.get("approval_required", True)),
            "approved_by": (str(row["approved_by"]) if row.get("approved_by") else None),
            "approved_at": row.get("approved_at"),
            "decided_by": (str(row["decided_by"]) if row.get("decided_by") else None),
            "decided_at": row.get("decided_at"),
        },
        "rollout": {
            "deployment_id": (str(row["deployment_id"]) if row.get("deployment_id") else None),
            "experiment_id": (str(row["experiment_id"]) if row.get("experiment_id") else None),
            "promoted_at": row.get("promoted_at"),
            "rolled_back_at": row.get("rolled_back_at"),
            "monitoring_status": row.get("monitoring_status"),
            "realized_window_start": row.get("realized_window_start"),
            "realized_window_end": row.get("realized_window_end"),
            "realized_metrics": row.get("realized_metrics"),
        },
        "lineage": {
            "parent_recommendation_id": (
                str(row["parent_recommendation_id"]) if row.get("parent_recommendation_id") else None
            ),
            "supersedes_id": (str(row["supersedes_id"]) if row.get("supersedes_id") else None),
            "bundle_id": (str(row["bundle_id"]) if row.get("bundle_id") else None),
        },
        "audit": row.get("audit") or [],
        "allowed_transitions": list(domain.LEGAL_TRANSITIONS.get(row.get("status") or "", ())),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
