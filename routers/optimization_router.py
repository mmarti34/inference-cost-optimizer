"""
Optimization layer API.

Everything is guarded with `Depends(require_org_member)`, takes `org_id` in the
PATH, and re-filters every query by that org_id. There is no `select("*")`
anywhere in this package.

CONTRACT NOTE: this API returns CODES AND FACTS, never customer-facing prose.
A conclusion is a stable `conclusion` code plus a `reasons` array whose entries
carry the underlying facts (observed, required, constraint, unit). All wording
is derived by the frontend, so rephrasing a sentence is never an API change.

Routes:
  GET  /api/optimization/{org_id}/summary
  GET  /api/optimization/{org_id}/workloads
  POST /api/optimization/{org_id}/workloads/discover
  GET  /api/optimization/{org_id}/recommendations
  GET  /api/optimization/{org_id}/recommendations/{rec_id}
  POST /api/optimization/{org_id}/recommendations/{rec_id}/benchmark
  POST /api/optimization/{org_id}/recommendations/{rec_id}/reject
  POST /api/optimization/{org_id}/recommendations/{rec_id}/accept
  GET  /api/optimization/{org_id}/benchmarks
  GET  /api/optimization/{org_id}/benchmarks/{benchmark_id}
  POST /api/optimization/{org_id}/benchmarks/{benchmark_id}/reevaluate
  GET  /api/optimization/{org_id}/candidate-results
  POST /api/optimization/{org_id}/workloads/{workload_id}/benchmark
  GET  /api/optimization/{org_id}/policies
  POST /api/optimization/{org_id}/policies
  PUT  /api/optimization/{org_id}/policies/{policy_id}
  GET  /api/optimization/{org_id}/executors
  POST /api/optimization/{org_id}/executors/sync
  GET  /api/optimization/{org_id}/outcomes
  POST /api/optimization/{org_id}/outcomes
  POST /api/optimization/{org_id}/outcomes/{outcome_id}/correct
  POST /v1/outcomes                        (customer-facing alias)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

from auth_dependency import AuthenticatedUser, require_org_member
from supabase_client import supabase

from optimization import (
    allocation,
    benchmark as benchmark_mod,
    domain,
    evidence as evidence_mod,
    executors as executors_mod,
    outcomes as outcomes_mod,
    policies as policies_mod,
    service,
    strategy as strategy_mod,
    workloads as workloads_mod,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["optimization"])

# The customer-facing alias lives outside /api and carries no org in the path;
# org identity comes from the authenticated principal via require_org_member
# (X-Org-Id header or body), and every query is re-filtered by the org that
# dependency actually VERIFIED.
public_router = APIRouter(prefix="/v1", tags=["optimization-public"])


def _verified_org(user: AuthenticatedUser, org_id: str) -> str:
    """
    The org the auth dependency actually verified.

    require_org_member already checks membership against the path org_id; this
    re-asserts the match so a handler can never widen scope by trusting a value
    from the request body.
    """
    verified = getattr(user, "_verified_org_id", None)
    if verified and str(verified) != str(org_id):
        raise HTTPException(status_code=403, detail="Organization mismatch.")
    return org_id


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

class RejectPayload(BaseModel):
    reason: str

    class Config:
        extra = "ignore"


class BenchmarkPayload(BaseModel):
    objective: Optional[str] = None
    min_sample_size: Optional[int] = None

    class Config:
        extra = "ignore"


class PolicyPayload(BaseModel):
    workload_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    constraints: Optional[dict] = None
    automation: Optional[dict] = None
    success_signal: Optional[dict] = None
    materiality: Optional[dict] = None

    class Config:
        extra = "ignore"


class OutcomePayload(BaseModel):
    """
    Attach an outcome to an EARLIER attempt.

    `occurred_at` is when it happened in the world; the server records
    `recorded_at` itself. The gap between them is expected and may be days.
    `idempotency_key` is REQUIRED: outcome feeds are webhooks and webhooks retry.
    """

    outcome_type: str
    idempotency_key: str
    org_id: Optional[str] = None
    attempt_id: Optional[str] = None
    request_id: Optional[str] = None
    attempt_source: Optional[str] = None
    workload_id: Optional[str] = None
    workload_key: Optional[str] = None
    value: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    success: Optional[bool] = None
    outcome_category: Optional[str] = None
    outcome_key: Optional[str] = None
    source: Optional[str] = None
    provenance: Optional[str] = None
    signal_strength: Optional[float] = None
    confidence: Optional[float] = None
    occurred_at: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        extra = "ignore"


class OutcomeCorrectionPayload(BaseModel):
    idempotency_key: str
    correction_reason: str
    org_id: Optional[str] = None
    value: Optional[float] = None
    value_text: Optional[str] = None
    success: Optional[bool] = None
    provenance: Optional[str] = None
    signal_strength: Optional[float] = None
    confidence: Optional[float] = None
    occurred_at: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        extra = "ignore"


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

@router.get("/{org_id}/workloads")
async def list_workloads(
    org_id: str,
    surface: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 200,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    rows = workloads_mod.list_workloads(
        org_id, surface=surface, project_id=project_id, limit=limit
    )
    return [workloads_mod.workload_row_to_response(r) for r in rows]


@router.post("/{org_id}/workloads/discover")
async def discover_workloads(
    org_id: str,
    lookback_days: int = 30,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Structural discovery from observed production traffic. Returns measured
    counts plus a `coverage` block stating the window and whether the scan was
    truncated.
    """
    org_id = _verified_org(user, org_id)
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: workloads_mod.discover_workloads(org_id, lookback_days=lookback_days)
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@router.get("/{org_id}/recommendations")
async def list_recommendations(
    org_id: str,
    status: Optional[str] = None,
    workload_id: Optional[str] = None,
    bundle_id: Optional[str] = None,
    limit: int = 100,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    if status and status not in domain.STATUSES:
        raise HTTPException(status_code=400, detail=f"Unknown status '{status}'.")
    rows = service.list_recommendations(
        org_id, status=status, workload_id=workload_id, bundle_id=bundle_id, limit=limit
    )
    return [service.recommendation_row_to_response(r) for r in rows]


@router.get("/{org_id}/recommendations/{rec_id}")
async def get_recommendation(
    org_id: str, rec_id: str, user: AuthenticatedUser = Depends(require_org_member)
):
    org_id = _verified_org(user, org_id)
    row = service.get_recommendation(org_id, rec_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Recommendation not found.")
    evidence = service.require_evidence(org_id, rec_id)
    return service.recommendation_row_to_response(row, evidence=evidence)


@router.post("/{org_id}/recommendations/{rec_id}/benchmark")
async def benchmark_recommendation(
    org_id: str,
    rec_id: str,
    payload: Optional[BenchmarkPayload] = None,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Run a replay benchmark and cite it from this recommendation.

    Moves the recommendation to 'benchmarking' immediately and runs the replay
    in a worker thread — the same kickoff pattern the existing eval flow uses.
    The conclusion, when it lands, drives the next transition.
    """
    org_id = _verified_org(user, org_id)
    rec = service.get_recommendation(org_id, rec_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recommendation not found.")
    if not rec.get("workload_id"):
        raise HTTPException(
            status_code=400, detail="Recommendation is not attached to a workload."
        )

    try:
        service.transition(
            org_id, rec_id, domain.STATUS_BENCHMARKING,
            actor=user.user_id, reason="benchmark_requested",
        )
    except domain.IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except service.RecommendationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    objective = (payload.objective if payload else None) or rec.get("objective")
    min_sample = payload.min_sample_size if payload else None

    asyncio.get_event_loop().run_in_executor(
        None,
        lambda: benchmark_mod.run_benchmark(
            org_id,
            workload_id=str(rec["workload_id"]),
            recommendation_id=rec_id,
            objective=objective,
            min_sample_size=min_sample,
            actor=user.user_id,
        ),
    )

    return {
        "recommendation_id": rec_id,
        "status": domain.STATUS_BENCHMARKING,
        "workload_id": str(rec["workload_id"]),
        "objective": objective,
        "note": "Benchmark started. Poll the recommendation or /benchmarks for the conclusion.",
    }


@router.post("/{org_id}/recommendations/{rec_id}/reject")
async def reject_recommendation(
    org_id: str,
    rec_id: str,
    payload: RejectPayload,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    if not (payload.reason or "").strip():
        raise HTTPException(status_code=400, detail="reason is required.")
    try:
        row = service.transition(
            org_id, rec_id, domain.STATUS_REJECTED,
            actor=user.user_id, reason=payload.reason.strip(),
        )
    except domain.IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except service.RecommendationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return service.recommendation_row_to_response(row)


@router.post("/{org_id}/recommendations/{rec_id}/accept")
async def accept_recommendation(
    org_id: str, rec_id: str, user: AuthenticatedUser = Depends(require_org_member)
):
    """
    HUMAN APPROVAL. Creates a candidate deployment from the candidate strategy
    and moves the recommendation to 'canary'.

    Legal only from 'verified' or 'awaiting_approval', and only when cited
    evidence exists. There is deliberately no path from a recommendation that
    was never benchmarked to production.
    """
    org_id = _verified_org(user, org_id)
    rec = service.get_recommendation(org_id, rec_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recommendation not found.")

    status = rec.get("status")
    if status not in (domain.STATUS_VERIFIED, domain.STATUS_AWAITING_APPROVAL):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot accept a recommendation in status '{status}'. It must be "
                f"'{domain.STATUS_VERIFIED}' or '{domain.STATUS_AWAITING_APPROVAL}' — "
                "a candidate may only reach production on the back of benchmark evidence."
            ),
        )

    cited = service.require_evidence(org_id, rec_id)
    if not cited:
        raise HTTPException(
            status_code=409,
            detail=(
                "This recommendation cites no benchmark evidence. Run a benchmark "
                "before accepting it."
            ),
        )

    deployment = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _create_candidate_deployment(org_id, rec)
    )
    if deployment is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not build a candidate deployment for this recommendation. "
                "The workload may have no workflow behind it (direct inference), or "
                "the strategy no longer matches the current graph."
            ),
        )

    if status == domain.STATUS_VERIFIED:
        service.transition(
            org_id, rec_id, domain.STATUS_AWAITING_APPROVAL,
            actor=user.user_id, reason="submitted_for_approval",
        )

    try:
        row = service.transition(
            org_id, rec_id, domain.STATUS_CANARY,
            actor=user.user_id, reason="human_approved",
            extra_fields={"deployment_id": str(deployment["id"])},
        )
    except domain.IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    allocation.record_decision(
        org_id,
        workload_id=(str(rec["workload_id"]) if rec.get("workload_id") else None),
        decision_kind="recommendation",
        objective=rec.get("objective") or "cost",
        recommendation_id=rec_id,
        policy_id=(str(rec["policy_id"]) if rec.get("policy_id") else None),
        selected_strategy_id=(
            str(rec["candidate_strategy_id"]) if rec.get("candidate_strategy_id") else None
        ),
        expected_cost_usd=rec.get("candidate_cost"),
        expected_quality=rec.get("candidate_quality"),
        expected_latency_p95_ms=rec.get("candidate_latency_p95_ms"),
        confidence=rec.get("confidence"),
        reason="human approved; candidate deployment created",
    )

    return {
        "recommendation": service.recommendation_row_to_response(row, evidence=cited),
        "deployment_id": str(deployment["id"]),
        "deployment_version": deployment.get("version"),
        "endpoint_slug": deployment.get("endpoint_slug"),
    }


def _create_candidate_deployment(org_id: str, rec: dict) -> Optional[dict]:
    """
    Build a candidate deployment from the recommendation's candidate strategy,
    reusing the existing `workflow_deployments` mechanism.

    The candidate GRAPH is regenerated from the current promoted graph plus the
    strategy, rather than stored: a graph snapshotted at recommendation time
    could be stale by the time a human approves it, and deploying a stale graph
    would silently revert unrelated changes.
    """
    workload = workloads_mod.get_workload(org_id, str(rec["workload_id"]))
    if workload is None:
        return None
    workflow_id = workloads_mod.resolve_workflow_id(org_id, workload)
    if not workflow_id:
        return None

    baseline_graph, endpoint_slug = benchmark_mod._load_baseline_graph(
        org_id, workload, workflow_id
    )
    if baseline_graph is None:
        return None

    try:
        cand_strategy = strategy_mod.Strategy.from_dict(rec.get("candidate_strategy") or {})
        candidate_graph = strategy_mod.apply_to_graph(baseline_graph, cand_strategy)
    except (strategy_mod.UnsupportedDimension, strategy_mod.StrategyApplyError) as exc:
        logger.warning("Candidate graph could not be built: %s", exc)
        return None

    try:
        existing = (
            supabase.table("workflow_deployments")
            .select("version, endpoint_slug")
            .eq("workflow_id", workflow_id)
            .eq("org_id", org_id)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        rows = getattr(existing, "data", None) or []
        if not rows:
            return None
        next_version = int(rows[0].get("version") or 0) + 1
        slug = (rows[0].get("endpoint_slug") or endpoint_slug or "").strip()

        wf = (
            supabase.table("workflows")
            .select("project_id, org_id")
            .eq("id", workflow_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        wf_rows = getattr(wf, "data", None) or []
        project_id = wf_rows[0].get("project_id") if wf_rows else None

        insert: dict[str, Any] = {
            "workflow_id": workflow_id,
            "org_id": org_id,
            "version": next_version,
            "endpoint_slug": slug,
            "graph_json": candidate_graph,
            "status": "candidate",
        }
        if project_id is not None:
            insert["project_id"] = project_id

        result = supabase.table("workflow_deployments").insert(insert).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("_create_candidate_deployment failed: %s", type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# Benchmarks — evidence, addressable on its own
# ---------------------------------------------------------------------------

@router.post("/{org_id}/workloads/{workload_id}/benchmark")
async def benchmark_workload(
    org_id: str,
    workload_id: str,
    payload: Optional[BenchmarkPayload] = None,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Run an EXPLORATORY benchmark against a workload, with no recommendation.

    Benchmarks discover facts; recommendations propose actions. Completing this
    never creates a recommendation — only a 'safe_improvement_found' conclusion
    justifies one, and creating it stays an explicit act.
    """
    org_id = _verified_org(user, org_id)
    if workloads_mod.get_workload(org_id, workload_id) is None:
        raise HTTPException(status_code=404, detail="Workload not found.")

    asyncio.get_event_loop().run_in_executor(
        None,
        lambda: benchmark_mod.run_benchmark(
            org_id,
            workload_id=workload_id,
            objective=(payload.objective if payload else None),
            min_sample_size=(payload.min_sample_size if payload else None),
            actor=user.user_id,
        ),
    )
    return {
        "workload_id": workload_id,
        "status": "pending",
        "note": "Exploratory benchmark started. No recommendation will be created automatically.",
    }


@router.get("/{org_id}/benchmarks")
async def list_benchmarks(
    org_id: str,
    workload_id: Optional[str] = None,
    conclusion: Optional[str] = None,
    limit: int = 100,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    if conclusion and conclusion not in domain.CONCLUSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown conclusion '{conclusion}'.")
    rows = benchmark_mod.list_benchmarks(
        org_id, workload_id=workload_id, conclusion=conclusion, limit=limit
    )
    return [benchmark_mod.benchmark_row_to_response(r) for r in rows]


@router.get("/{org_id}/benchmarks/{benchmark_id}")
async def get_benchmark(
    org_id: str, benchmark_id: str, user: AuthenticatedUser = Depends(require_org_member)
):
    org_id = _verified_org(user, org_id)
    row = benchmark_mod.get_benchmark(org_id, benchmark_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Benchmark not found.")
    current = benchmark_mod.current_conclusion(org_id, benchmark_id)
    return {
        **benchmark_mod.benchmark_row_to_response(row, conclusion_row=current),
        "candidate_results": benchmark_mod.list_candidate_results(
            org_id, benchmark_id=benchmark_id
        ),
        # Every evaluation of this evidence, each bound to its policy version.
        # An older verdict is not wrong; it was correct under the policy then.
        "conclusion_history": benchmark_mod.conclusion_history(org_id, benchmark_id),
    }


@router.post("/{org_id}/benchmarks/{benchmark_id}/reevaluate")
async def reevaluate_benchmark(
    org_id: str,
    benchmark_id: str,
    objective: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Re-derive a conclusion from RETAINED candidate results under the current
    policy version. Runs no model calls and costs nothing.
    """
    org_id = _verified_org(user, org_id)
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: benchmark_mod.reevaluate(org_id, benchmark_id, objective=objective)
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark not found, or it has no retained baseline result to re-evaluate.",
        )
    return result


@router.get("/{org_id}/candidate-results")
async def list_candidate_results(
    org_id: str,
    workload_id: Optional[str] = None,
    benchmark_id: Optional[str] = None,
    limit: int = 200,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Measured candidate arms, queryable independently of any conclusion.

    A near-miss (saved 51%, missed the quality floor by 0.7pp) is retained here
    even though its benchmark concluded 'candidates_failed_policy'. Relaxing the
    threshold later is a re-read, not a re-measurement.
    """
    org_id = _verified_org(user, org_id)
    return benchmark_mod.list_candidate_results(
        org_id, workload_id=workload_id, benchmark_id=benchmark_id, limit=limit
    )


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@router.get("/{org_id}/policies")
async def list_policies(
    org_id: str,
    workload_id: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    rows = policies_mod.list_policies(org_id, workload_id=workload_id)
    return [policies_mod.policy_row_to_response(r) for r in rows]


@router.post("/{org_id}/policies")
async def create_policy(
    org_id: str, payload: PolicyPayload, user: AuthenticatedUser = Depends(require_org_member)
):
    org_id = _verified_org(user, org_id)
    row = policies_mod.create_policy(org_id, payload.dict(exclude_none=True))
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create policy.")
    return policies_mod.policy_row_to_response(row)


@router.put("/{org_id}/policies/{policy_id}")
async def update_policy(
    org_id: str,
    policy_id: str,
    payload: PolicyPayload,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Editing a policy creates a NEW VERSION. The previous version is retained
    unchanged so that historical benchmark conclusions stay reproducible as
    (evidence + policy version + objective).
    """
    org_id = _verified_org(user, org_id)
    row = policies_mod.update_policy(org_id, policy_id, payload.dict(exclude_none=True))
    if row is None:
        raise HTTPException(status_code=404, detail="Policy not found.")
    return policies_mod.policy_row_to_response(row)


# ---------------------------------------------------------------------------
# Executors — VENDOR metadata, labelled as such
# ---------------------------------------------------------------------------

@router.get("/{org_id}/executors")
async def list_executors(
    org_id: str,
    executor_type: Optional[str] = None,
    vendor: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    rows = executors_mod.list_executors(org_id, executor_type=executor_type, vendor=vendor)
    return [executors_mod.executor_row_to_response(r) for r in rows]


@router.post("/{org_id}/executors/sync")
async def sync_executors(org_id: str, user: AuthenticatedUser = Depends(require_org_member)):
    org_id = _verified_org(user, org_id)
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: executors_mod.sync_model_executors(org_id)
    )


# ---------------------------------------------------------------------------
# Outcomes — delayed arrival, plural, correctable, idempotent
# ---------------------------------------------------------------------------

def _record_outcome(org_id: str, payload: OutcomePayload) -> tuple[dict, bool]:
    attempt_ref = payload.attempt_id or payload.request_id
    attempt_source = payload.attempt_source or (
        "api_request" if (payload.request_id and not payload.attempt_id) else "workflow_run"
    )

    workload_id = payload.workload_id
    if workload_id is None and payload.workload_key:
        wl = workloads_mod.resolve_workload(
            org_id, external_key=payload.workload_key, create=True
        )
        workload_id = str(wl["id"]) if wl else None

    return outcomes_mod.record_outcome(
        org_id,
        outcome_type=payload.outcome_type,
        idempotency_key=payload.idempotency_key,
        attempt_ref=attempt_ref,
        attempt_source=attempt_source,
        workload_id=workload_id,
        value=payload.value,
        value_text=payload.value_text,
        unit=payload.unit,
        success=payload.success,
        outcome_category=payload.outcome_category,
        outcome_key=payload.outcome_key,
        source=payload.source or "api",
        provenance=payload.provenance or "unknown",
        signal_strength=payload.signal_strength,
        confidence=payload.confidence,
        occurred_at=payload.occurred_at,
        metadata=payload.metadata,
    )


@router.post("/{org_id}/outcomes")
async def create_outcome(
    org_id: str, payload: OutcomePayload, user: AuthenticatedUser = Depends(require_org_member)
):
    """
    Attach an outcome to an earlier attempt. Idempotent on
    (org_id, idempotency_key): re-POSTing the same key returns the existing row
    unchanged rather than creating a duplicate or mutating the original.
    """
    org_id = _verified_org(user, org_id)
    try:
        row, created = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _record_outcome(org_id, payload)
        )
    except outcomes_mod.OutcomeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "outcome": outcomes_mod.outcome_row_to_response(row),
        "created": created,
        "idempotent_replay": not created,
    }


@public_router.post("/outcomes")
async def create_outcome_public(
    payload: OutcomePayload,
    user: AuthenticatedUser = Depends(require_org_member),
    x_org_id: Optional[str] = Header(None),
):
    """
    Customer-facing alias of POST /api/optimization/{org_id}/outcomes.

    The org is taken from the principal that `require_org_member` VERIFIED
    (X-Org-Id header or an org_id in the body), never from an unverified field.
    """
    org_id = getattr(user, "_verified_org_id", None) or payload.org_id or x_org_id
    if not org_id:
        raise HTTPException(
            status_code=400,
            detail="org_id is required (X-Org-Id header or org_id in the body).",
        )
    org_id = _verified_org(user, str(org_id))
    try:
        row, created = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _record_outcome(str(org_id), payload)
        )
    except outcomes_mod.OutcomeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "outcome": outcomes_mod.outcome_row_to_response(row),
        "created": created,
        "idempotent_replay": not created,
    }


@router.post("/{org_id}/outcomes/{outcome_id}/correct")
async def correct_outcome(
    org_id: str,
    outcome_id: str,
    payload: OutcomeCorrectionPayload,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Revise a recorded outcome. Inserts a new revision and marks the original
    superseded — never overwrites, because savings math may already have
    consumed the old value.
    """
    org_id = _verified_org(user, org_id)
    try:
        row, created = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: outcomes_mod.correct_outcome(
                org_id, outcome_id,
                idempotency_key=payload.idempotency_key,
                correction_reason=payload.correction_reason,
                value=payload.value,
                value_text=payload.value_text,
                success=payload.success,
                provenance=payload.provenance,
                signal_strength=payload.signal_strength,
                confidence=payload.confidence,
                occurred_at=payload.occurred_at,
                metadata=payload.metadata,
            ),
        )
    except outcomes_mod.OutcomeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "outcome": outcomes_mod.outcome_row_to_response(row),
        "created": created,
        "idempotent_replay": not created,
        "revision_chain": [
            outcomes_mod.outcome_row_to_response(r)
            for r in outcomes_mod.revision_chain(org_id, str(row["id"]))
        ],
    }


@router.get("/{org_id}/outcomes")
async def list_outcomes(
    org_id: str,
    workload_id: Optional[str] = None,
    attempt_ref: Optional[str] = None,
    outcome_type: Optional[str] = None,
    include_superseded: bool = False,
    limit: int = 200,
    user: AuthenticatedUser = Depends(require_org_member),
):
    org_id = _verified_org(user, org_id)
    rows = outcomes_mod.list_outcomes(
        org_id, workload_id=workload_id, attempt_ref=attempt_ref,
        outcome_type=outcome_type, current_only=not include_superseded, limit=limit,
    )
    return [outcomes_mod.outcome_row_to_response(r) for r in rows]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@router.get("/{org_id}/summary")
async def summary(
    org_id: str,
    lookback_days: int = 30,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Executive summary. Every number is measured or explicitly null.

    Two things this endpoint will not do:

      * Report "no opportunity" for a workload we could not assess.
        `insufficient_evidence` and `benchmark_failed` land in
        `not_yet_assessable`, never in `no_opportunity`.
      * Present workload-count coverage as the headline when the objective is
        cost. SPEND coverage leads, because "8 of 10 workloads assessed" reads
        as excellent right up until the two unassessed ones are 78% of spend.
    """
    org_id = _verified_org(user, org_id)
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _build_summary(org_id, lookback_days)
    )


def _build_summary(org_id: str, lookback_days: int) -> dict:
    traffic = evidence_mod.observed_production_traffic(org_id, lookback_days=lookback_days)

    workloads = workloads_mod.list_workloads(org_id, limit=500)
    recommendations = service.list_recommendations(org_id, limit=200)

    # Latest current conclusion per workload.
    verdict_by_workload: dict[str, dict] = {}
    try:
        resp = (
            supabase.table("benchmark_conclusions")
            .select("workload_id, conclusion, confidence, created_at, is_current")
            .eq("org_id", org_id)
            .eq("is_current", True)
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        for r in (getattr(resp, "data", None) or []):
            wid = str(r.get("workload_id"))
            if wid not in verdict_by_workload:
                verdict_by_workload[wid] = r
    except Exception:  # pragma: no cover
        pass

    coverage_entries: list[dict] = []
    for wl in workloads:
        wid = str(wl["id"])
        slug = wl.get("identity_ref") if wl.get("identity_kind") == "endpoint" else None
        wl_traffic = evidence_mod.observed_production_traffic(
            org_id, endpoint_slug=slug, lookback_days=lookback_days
        ) if slug else {"total_cost_usd": None, "run_count": None}
        v = verdict_by_workload.get(wid) or {}
        coverage_entries.append({
            "workload_id": wid,
            "conclusion": v.get("conclusion"),
            "benchmark_status": None,
            "spend_usd": wl_traffic.get("total_cost_usd"),
            "volume": wl_traffic.get("run_count"),
        })

    coverage = domain.compute_coverage(coverage_entries, objective="cost")

    conclusions = [e.get("conclusion") for e in coverage_entries]
    assessed = {
        "no_opportunity": sum(1 for c in conclusions if domain.is_efficiency_finding(c)),
        "opportunity_found": sum(
            1 for c in conclusions if c == domain.CONCLUSION_SAFE_IMPROVEMENT
        ),
        "candidates_failed_policy": sum(
            1 for c in conclusions if c == domain.CONCLUSION_CANDIDATES_FAILED_POLICY
        ),
    }
    not_yet_assessable = {
        "insufficient_evidence": sum(
            1 for c in conclusions if c == domain.CONCLUSION_INSUFFICIENT_EVIDENCE
        ),
        "benchmark_failed": sum(
            1 for c in conclusions if c == domain.CONCLUSION_BENCHMARK_FAILED
        ),
        "never_benchmarked": sum(1 for c in conclusions if c is None),
    }

    projected, projected_cov = domain.attributable_savings(recommendations, "projected")
    verified, verified_cov = domain.attributable_savings(recommendations, "verified")
    realized, realized_cov = domain.attributable_savings(recommendations, "realized")

    quality_counts: dict[str, int] = {}
    for r in recommendations:
        p = r.get("quality_provenance") or "unknown"
        quality_counts[p] = quality_counts.get(p, 0) + 1

    return {
        "org_id": org_id,
        "window_days": lookback_days,
        "spend": {
            "total_usd": traffic.get("total_cost_usd"),
            "run_count": traffic.get("run_count"),
            "mean_cost_usd": traffic.get("mean_cost_usd"),
            "error_rate": traffic.get("error_rate"),
            "coverage": traffic.get("coverage"),
        },
        "opportunities": {
            "open": sum(1 for r in recommendations if r.get("status") in domain.OPEN_STATUSES),
            "awaiting_approval": sum(
                1 for r in recommendations if r.get("status") == domain.STATUS_AWAITING_APPROVAL
            ),
            "live": sum(1 for r in recommendations if r.get("status") in domain.LIVE_STATUSES),
            "by_status": _count_by(recommendations, "status"),
        },
        "savings": {
            "projected_usd": projected,
            "verified_usd": verified,
            "realized_usd": realized,
            "attribution": {
                "projected": projected_cov,
                "verified": verified_cov,
                "realized": realized_cov,
            },
            "meanings": {
                "projected_usd": "Extrapolated from a measured per-call delta x observed volume.",
                "verified_usd": "Measured inside a benchmark or canary, over the sample only.",
                "realized_usd": (
                    "Observed in production after promotion. Post-promotion monitoring "
                    "is not yet instrumented, so this is null rather than zero."
                ),
            },
        },
        "assessed": assessed,
        "not_yet_assessable": not_yet_assessable,
        "optimization_coverage": coverage,
        "quality_health": {
            "recommendations_by_quality_provenance": quality_counts,
            "measured": sum(
                n for p, n in quality_counts.items()
                if domain.provenance_rank(p) >= domain.MIN_QUALITY_PROVENANCE_RANK_FOR_CONSTRAINT
            ),
            "unmeasured": quality_counts.get("unknown", 0),
        },
        "coverage_notes": [
            {
                "code": "realized_savings_uninstrumented",
                "detail": (
                    "Post-promotion production monitoring is a documented extension "
                    "point. realized_usd is null, never zero."
                ),
            },
            {
                "code": "ignorance_excluded_from_findings",
                "detail": (
                    "insufficient_evidence and benchmark_failed are reported under "
                    "not_yet_assessable and are never counted as no_opportunity."
                ),
            },
            {
                "code": "spend_coverage_is_primary",
                "detail": (
                    "optimization_coverage.primary names the figure to lead with. "
                    "Workload-count coverage can look healthy while most spend is "
                    "unassessed."
                ),
            },
        ],
    }


def _count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = r.get(key) or "unknown"
        out[v] = out.get(v, 0) + 1
    return out
