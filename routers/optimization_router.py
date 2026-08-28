"""
Optimization layer API.

Every /api route is guarded with `Depends(require_org_member)`, takes `org_id`
in the PATH, and re-filters every query by that org_id. There is no
`select("*")` anywhere in this package.

TWO AUDIENCES, TWO CREDENTIALS. The dashboard is a signed-in human; a customer
integration is a backend holding a server API key. They do not share an auth
mechanism and are not collapsed into one endpoint:

    dashboard / internal   POST /api/optimization/{org_id}/outcomes
                           user auth (require_org_member), org in the path

    customer / server      POST /v1/outcomes
                           server API key (validate_api_key), org from the key

CONTRACT NOTE: this API returns CODES AND FACTS, never customer-facing prose.
A conclusion is a stable `conclusion` code plus a `reasons` array whose entries
carry the underlying facts (observed, required, constraint, unit). All wording
is derived by the frontend, so rephrasing a sentence is never an API change.

BENCHMARK EXECUTION IS ASYNCHRONOUS. /optimize and /benchmark return 202 with a
benchmark id and a phase, and the run continues in a worker. See
optimization/jobs.py for the job lifecycle, the idempotency key and how an
orphaned run is detected. Poll
GET /api/optimization/{org_id}/benchmarks/{benchmark_id}/status.

Routes:
  GET  /api/optimization/{org_id}/summary
  GET  /api/optimization/{org_id}/workloads
  POST /api/optimization/{org_id}/workloads/discover
  GET  /api/optimization/{org_id}/optimization-targets
  POST /api/optimization/{org_id}/workloads/{workload_id}/optimize      -> 202
  GET  /api/optimization/{org_id}/jobs
  GET  /api/optimization/{org_id}/recommendations
  GET  /api/optimization/{org_id}/recommendations/{rec_id}
  POST /api/optimization/{org_id}/recommendations/{rec_id}/benchmark
  POST /api/optimization/{org_id}/recommendations/{rec_id}/reject
  POST /api/optimization/{org_id}/recommendations/{rec_id}/accept
  GET  /api/optimization/{org_id}/benchmarks
  GET  /api/optimization/{org_id}/benchmarks/{benchmark_id}
  GET  /api/optimization/{org_id}/benchmarks/{benchmark_id}/status
  POST /api/optimization/{org_id}/benchmarks/{benchmark_id}/reevaluate
  GET  /api/optimization/{org_id}/candidate-results
  POST /api/optimization/{org_id}/workloads/{workload_id}/benchmark      -> 202
  GET  /api/optimization/{org_id}/policies
  POST /api/optimization/{org_id}/policies
  PUT  /api/optimization/{org_id}/policies/{policy_id}
  GET  /api/optimization/{org_id}/executors
  POST /api/optimization/{org_id}/executors/sync
  GET  /api/optimization/{org_id}/outcomes
  POST /api/optimization/{org_id}/outcomes
  POST /api/optimization/{org_id}/outcomes/{outcome_id}/correct
  POST /v1/outcomes                        (customer-facing, server API key)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel

import audit
from api_key_validation import validate_api_key
from auth_dependency import AuthenticatedUser, require_org_member
from supabase_client import supabase

from optimization import (
    allocation,
    benchmark as benchmark_mod,
    domain,
    evidence as evidence_mod,
    executors as executors_mod,
    jobs as jobs_mod,
    outcomes as outcomes_mod,
    policies as policies_mod,
    service,
    strategy as strategy_mod,
    workloads as workloads_mod,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["optimization"])

# The customer-facing surface lives outside /api and carries no org in the
# path. It is authenticated with a SERVER API KEY — the same credential the
# customer already uses for POST /v1/chat/completions — and the org is taken
# from that validated key alone. It does NOT use require_org_member: a
# customer's backend has no Supabase session, which is exactly why outcome
# ingestion was previously uncallable by the customer it was built for.
public_router = APIRouter(prefix="/v1", tags=["optimization-public"])

#: One message for "that attempt is not yours" AND "that attempt does not
#: exist". Any difference between the two turns this endpoint into an oracle
#: for another tenant's request ids. Matches main.py's trace endpoint.
_NOT_FOUND_DETAIL = "Request not found."


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


class OptimizePayload(BaseModel):
    """
    Input to the full optimization loop.

    `create_recommendation` defaults TRUE here and is absent from
    BenchmarkPayload on purpose: /benchmark is exploratory and must never
    produce a proposal, while /optimize exists precisely to close the loop.
    Either way the creation is gated on the EVIDENCE concluding
    safe_improvement_found — this flag can only decline it, never force it.
    """

    objective: Optional[str] = None
    min_sample_size: Optional[int] = None
    create_recommendation: bool = True

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
    #: Never an input to authorization. On /v1/outcomes the org comes from the
    #: server API key; this field is accepted only so that a MISMATCH can be
    #: reported back to the caller instead of being silently discarded.
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


@router.get("/{org_id}/optimization-targets")
async def optimization_targets(
    org_id: str,
    lookback_days: int = 30,
    min_runs: Optional[int] = None,
    min_spend_usd: Optional[float] = None,
    limit: int = 10,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Which observed workloads are worth spending a benchmark on, ranked on
    MEASURED spend then MEASURED volume.

    Read-only. Every workload passed over is returned under `skipped` with
    structured reason codes, because a workload missing from the list must not
    read as a workload that was assessed and found optimal.
    """
    org_id = _verified_org(user, org_id)
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: workloads_mod.select_optimization_targets(
            org_id,
            lookback_days=lookback_days,
            min_runs=min_runs,
            min_spend_usd=min_spend_usd,
            limit=limit,
        ),
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


@router.post("/{org_id}/recommendations/{rec_id}/benchmark", status_code=202)
async def benchmark_recommendation(
    org_id: str,
    rec_id: str,
    payload: Optional[BenchmarkPayload] = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Start a replay benchmark JOB whose evidence this recommendation will cite.

    Moves the recommendation to 'benchmarking' and returns 202 with the
    benchmark id. The conclusion, when it lands, drives the next transition.

    Unlike the two workload routes this one has a precondition with a side
    effect: the lifecycle transition. It is taken FIRST, so an illegal
    transition is a 409 and no job is created; if the job then turns out to be a
    duplicate the recommendation is already 'benchmarking', which is what the
    in-flight job it is being handed will make true anyway.
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

    row, created = await _start_benchmark_job(
        org_id,
        workload_id=str(rec["workload_id"]),
        job_kind=jobs_mod.JOB_KIND_RECOMMENDATION_BENCHMARK,
        actor=user.user_id,
        objective=(payload.objective if payload else None) or rec.get("objective"),
        min_sample_size=(payload.min_sample_size if payload else None),
        create_recommendation=False,
        recommendation_id=rec_id,
        client_key=idempotency_key,
    )
    return _job_envelope(
        row,
        created,
        extra={
            "recommendation_id": rec_id,
            "recommendation_status": domain.STATUS_BENCHMARKING,
            "creates_recommendation": False,
        },
    )


@router.post("/{org_id}/recommendations/{rec_id}/reject")
async def reject_recommendation(
    request: Request,
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
    # `payload.reason` is operator free text and stays in the recommendation's
    # own audit column; it is not copied here.
    audit.record(
        audit.RECOMMENDATION_REJECTED,
        principal=user,
        resource_type=audit.RESOURCE_RECOMMENDATION,
        resource_id=rec_id,
        metadata={"new_status": domain.STATUS_REJECTED},
        request=request,
    )
    return service.recommendation_row_to_response(row)


@router.post("/{org_id}/recommendations/{rec_id}/accept")
async def accept_recommendation(
    request: Request,
    org_id: str,
    rec_id: str,
    user: AuthenticatedUser = Depends(require_org_member),
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
        evidence_maturity=rec.get("confidence"),
        reason="human approved; candidate deployment created",
    )

    # HUMAN APPROVAL of a change that now has a candidate deployment behind it.
    # `prior_status` is the state the human actually approved from, which is the
    # fact that matters if the decision is ever questioned.
    audit.record(
        audit.RECOMMENDATION_ACCEPTED,
        principal=user,
        resource_type=audit.RESOURCE_RECOMMENDATION,
        resource_id=rec_id,
        metadata={
            "prior_status": status,
            "new_status": domain.STATUS_CANARY,
            "deployment_id": str(deployment["id"]),
        },
        request=request,
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

# ---------------------------------------------------------------------------
# Starting a benchmark job
# ---------------------------------------------------------------------------
#
# THREE ENTRY POINTS, ONE MECHANISM.
#
# Before this change there were three ways to run the same function and three
# different contracts:
#
#   POST /workloads/{id}/optimize   ran it synchronously and awaited the verdict
#   POST /workloads/{id}/benchmark  fired and forgot, returned {"status":"pending"}
#   POST /recommendations/{id}/benchmark  fired and forgot, moved the rec
#
# The first could not survive an edge timeout. The other two returned no id, so
# a caller had no way to ask what happened; a benchmark that died with its
# worker left a row saying `running` that nothing would ever correct.
#
# All three now create a JOB and return the SAME 202 envelope with the benchmark
# id. They are not merged into one route, because they express three genuinely
# different intents — and that intent is recorded on the row as `job_kind`, not
# inferred later from which fields happen to be populated:
#
#   optimize                  may create a recommendation on safe_improvement_found
#   benchmark (workload)      may NEVER create one, whatever it concludes
#   benchmark (recommendation) gathers evidence FOR an existing recommendation
#                             and drives its lifecycle transition
#
# The first two differ in exactly one boolean, which /optimize already exposes
# as `create_recommendation`. /benchmark is therefore kept as the NAMED,
# non-negotiable form of that boolean rather than deleted: a route that cannot
# produce a proposal is a stronger guarantee than a flag a caller must remember
# to send, the existing frontend already calls it, and the difference is
# recorded in the idempotency key so an exploratory run can never be handed back
# to a caller that asked for the full loop.


def _job_envelope(row: dict, created: bool, *, extra: Optional[dict] = None) -> dict:
    out = jobs_mod.job_response(row, created=created)
    if extra:
        out.update(extra)
    return out


async def _start_benchmark_job(
    org_id: str,
    *,
    workload_id: str,
    job_kind: str,
    actor: Optional[str],
    objective: Optional[str] = None,
    min_sample_size: Optional[int] = None,
    create_recommendation: bool = False,
    recommendation_id: Optional[str] = None,
    client_key: Optional[str] = None,
) -> tuple[dict, bool]:
    """
    Persist the job, then hand the run to a worker. Returns (row, created).

    `created=False` means an equivalent job was already in flight and THIS
    REQUEST STARTED NOTHING. That is the double-spend guarantee: the decision
    not to execute is taken here, before a single provider call, on the basis of
    a row the database already holds.
    """
    try:
        prepared = benchmark_mod.prepare_benchmark(
            org_id,
            workload_id=workload_id,
            objective=objective,
            min_sample_size=min_sample_size,
        )
    except benchmark_mod.BenchmarkError as exc:
        # Resolvable failures are answered NOW, while the caller is still on the
        # line. Accepting a request with 202 and failing it a second later would
        # be a worse contract than the synchronous one it replaces.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row, created, _key = jobs_mod.create_job(
        org_id,
        workload_id=workload_id,
        job_kind=job_kind,
        insert_row=lambda cols: benchmark_mod.create_benchmark_row(
            org_id, prepared, extra_columns=cols
        ),
        objective=prepared["objective"],
        min_sample_size=min_sample_size,
        create_recommendation=create_recommendation,
        recommendation_id=recommendation_id,
        client_key=client_key,
        requested_by=actor,
    )
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create the benchmark job.")
    if not created:
        return row, False

    benchmark_id = str(row["id"])
    try:
        jobs_mod.start_job(
            org_id,
            benchmark_id,
            lambda reporter: benchmark_mod.run_benchmark(
                org_id,
                workload_id=workload_id,
                objective=prepared["objective"],
                min_sample_size=min_sample_size,
                create_recommendation=create_recommendation,
                recommendation_id=recommendation_id,
                actor=actor,
                benchmark_id=benchmark_id,
                prepared=prepared,
                progress=reporter,
            ),
        )
    except Exception as exc:
        # The row exists and the worker never got it. Say so on the row rather
        # than leaving a job that is queued forever.
        logger.exception("Could not start benchmark job %s", benchmark_id)
        jobs_mod.fail_job(
            org_id, benchmark_id, code="start_failed", detail=str(exc)[:300]
        )
        raise HTTPException(
            status_code=503, detail="Benchmark job could not be started."
        ) from exc

    refreshed = jobs_mod.get_job(org_id, benchmark_id) or row
    return refreshed, True


@router.post("/{org_id}/workloads/{workload_id}/benchmark", status_code=202)
async def benchmark_workload(
    org_id: str,
    workload_id: str,
    payload: Optional[BenchmarkPayload] = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Start an EXPLORATORY benchmark job against a workload. 202 + benchmark id.

    Benchmarks discover facts; recommendations propose actions. Completing this
    never creates a recommendation — only a 'safe_improvement_found' conclusion
    justifies one, and creating it stays an explicit act. This is /optimize with
    create_recommendation permanently false; the job is identical in every other
    respect, including progress reporting and orphan recovery.
    """
    org_id = _verified_org(user, org_id)
    if workloads_mod.get_workload(org_id, workload_id) is None:
        raise HTTPException(status_code=404, detail="Workload not found.")

    row, created = await _start_benchmark_job(
        org_id,
        workload_id=workload_id,
        job_kind=jobs_mod.JOB_KIND_EXPLORE,
        actor=user.user_id,
        objective=(payload.objective if payload else None),
        min_sample_size=(payload.min_sample_size if payload else None),
        create_recommendation=False,
        client_key=idempotency_key,
    )
    return _job_envelope(row, created, extra={"creates_recommendation": False})


@router.post("/{org_id}/workloads/{workload_id}/optimize", status_code=202)
async def optimize_workload(
    org_id: str,
    workload_id: str,
    payload: Optional[OptimizePayload] = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Start the full loop as a JOB, and return 202 with the benchmark id.

    The loop itself is unchanged: generate model-substitution candidates, replay
    them and the baseline over the SAME golden inputs, judge the result against
    the workload's policy, persist every measured arm and one immutable
    conclusion, and — only on `safe_improvement_found` — create a recommendation
    that cites the benchmark. A conclusion of `insufficient_evidence` remains a
    successful outcome, not an error, and is never rendered as "your current
    configuration is optimal".

    WHAT CHANGED, AND WHY. This used to await the verdict inside the request.
    The first real production run took ~28 minutes against a 300-second edge
    timeout: the caller got a connection error, the benchmark finished, and the
    API told nobody. The verdict is still the point — it is now fetched by
    polling .../benchmarks/{benchmark_id}/status, which works whether the run
    takes 40 seconds or 40 minutes and whether or not the caller was still
    connected when it landed.

    A duplicate request while an equivalent job is in flight returns THAT job
    with `created: false` and starts nothing.

    The recommendation, if one is created, is `approval_required` per policy and
    defaults to requiring a human. Nothing here changes production.
    """
    org_id = _verified_org(user, org_id)
    if workloads_mod.get_workload(org_id, workload_id) is None:
        raise HTTPException(status_code=404, detail="Workload not found.")

    wants_recommendation = payload.create_recommendation if payload else True
    row, created = await _start_benchmark_job(
        org_id,
        workload_id=workload_id,
        job_kind=jobs_mod.JOB_KIND_OPTIMIZE,
        actor=user.user_id,
        objective=(payload.objective if payload else None),
        min_sample_size=(payload.min_sample_size if payload else None),
        create_recommendation=wants_recommendation,
        client_key=idempotency_key,
    )
    return _job_envelope(
        row, created, extra={"creates_recommendation": bool(wants_recommendation)}
    )


@router.get("/{org_id}/jobs")
async def list_jobs(
    org_id: str,
    workload_id: Optional[str] = None,
    limit: int = 100,
    user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Every benchmark job for THIS org that has not reached a verdict.

    Exists so a frontend that lost its benchmark id — a reload, a new tab, a
    different device — can find the run that is still going instead of starting
    a second one.
    """
    org_id = _verified_org(user, org_id)
    rows = jobs_mod.list_active_jobs(org_id, workload_id=workload_id, limit=limit)
    return {
        "jobs": [jobs_mod.job_response(r) for r in rows],
        "progress_states": {
            **{k: v for k, v in jobs_mod.PROGRESS_STATES.items()},
        },
        "job_kinds": jobs_mod.JOB_KINDS,
        "lease_seconds": jobs_mod.lease_seconds(),
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


@router.get("/{org_id}/benchmarks/{benchmark_id}/status")
async def benchmark_status(
    org_id: str, benchmark_id: str, user: AuthenticatedUser = Depends(require_org_member)
):
    """
    The poll target. Cheap, org-scoped, safe to call every few seconds.

    Returns the job envelope — status, progress_state, the resolved phase plan
    and the counts behind it — plus, once the job is terminal, the conclusion
    and the recommendation it created. Before that the conclusion fields are
    absent rather than provisional: a run that has measured three of seven arms
    has no verdict, and reporting a partial one would be the thing this codebase
    calls fabrication.

    404 is returned both for a benchmark that does not exist and for one
    belonging to another organization. Distinguishing them would make this
    endpoint an oracle for another tenant's benchmark ids.
    """
    org_id = _verified_org(user, org_id)
    row = jobs_mod.get_job(org_id, benchmark_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Benchmark not found.")

    envelope = jobs_mod.job_response(row)
    if row.get("status") not in jobs_mod.TERMINAL_STATUSES:
        return envelope

    full = benchmark_mod.get_benchmark(org_id, benchmark_id)
    if full is None:
        return envelope
    conclusion_row = benchmark_mod.current_conclusion(org_id, benchmark_id)
    cited_by = service.recommendations_citing(org_id, benchmark_id)
    rec_id = cited_by[0] if cited_by else None
    rec_row = service.get_recommendation(org_id, rec_id) if rec_id else None
    return {
        **envelope,
        "result": benchmark_mod.benchmark_row_to_response(
            full, conclusion_row=conclusion_row
        ),
        "recommendation_id": rec_id,
        "recommendation": (
            service.recommendation_row_to_response(
                rec_row, evidence=service.require_evidence(org_id, rec_id)
            )
            if rec_row
            else None
        ),
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
    request: Request,
    authorization: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None, alias="X-Org-Id"),
):
    """
    Customer-facing outcome ingestion. Server-to-server.

    AUTH CONTRACT — this endpoint is called by a customer's BACKEND, holding
    the same OptiML **server API key** it uses for POST /v1/chat/completions.
    It is NOT a dashboard endpoint and takes no user session:

      * The Bearer token is validated by `validate_api_key`, the single
        production key-validation primitive (see routers/openai_compat.py and
        routers/public_execution.py). No key logic is duplicated here, so
        revocation, status checks and the legacy-key path all apply unchanged
        and a disabled key fails closed with 401.
      * `org_id` comes EXCLUSIVELY from the validated key's OrgContext.
      * An `org_id` asserted anywhere in the request — body, `X-Org-Id`, or
        query string — is never trusted. If it conflicts with the key's org the
        request is REJECTED (403), because silently overriding it would let a
        customer believe they had written to an org they had not, and silently
        ignoring it would hide the same mistake.
      * The referenced attempt is verified to belong to the key's org before
        anything is attached, and a reference to another org's attempt returns
        the SAME 404 as one that does not exist anywhere. This mirrors the
        observability trace endpoint in main.py: a request id is obscurity, not
        an access control, so the endpoint must not double as an oracle for
        which ids exist in another tenant.

    The dashboard equivalent, POST /api/optimization/{org_id}/outcomes, keeps
    user/org-member auth and is unaffected.
    """
    # ── Auth: server API key → org. Never relaxed. ───────────────────────
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header."
        )
    # validate_api_key raises 401 for malformed, unknown, revoked or disabled
    # keys. Failing closed is its behaviour, not ours to re-implement.
    ctx = validate_api_key(authorization.split(" ", 1)[1].strip())
    org_id = str(ctx.org_id)

    # ── Any asserted org_id must AGREE with the key, or the call is refused ──
    for source, claimed in (
        ("body", payload.org_id),
        ("X-Org-Id header", x_org_id),
        ("org_id query parameter", request.query_params.get("org_id")),
    ):
        if claimed and str(claimed).strip() and str(claimed).strip() != org_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"org_id in the {source} does not match the organization this "
                    "API key belongs to. Omit org_id: it is derived from the key."
                ),
            )

    # ── A caller-supplied workload_id is verified, not trusted ──────────────
    if payload.workload_id:
        owned = await asyncio.get_event_loop().run_in_executor(
            None, lambda: workloads_mod.get_workload(org_id, str(payload.workload_id))
        )
        if owned is None:
            raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)

    try:
        row, created = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _record_outcome(org_id, payload)
        )
    except outcomes_mod.AttemptNotFoundError as exc:
        # Indistinguishable from "no such id anywhere". Do not echo the ref.
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL) from exc
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
            # `confidence` is NOT selected: coverage is a count of conclusions
            # by class, and the evidence-maturity index it stores is internal
            # and has no place in a customer-facing summary.
            .select("workload_id, conclusion, created_at, is_current")
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
        # A promising candidate is a real finding but NOT a determination: we
        # found something worth pursuing and cannot yet vouch for it. It belongs
        # here rather than under `opportunity_found`, or the summary would count
        # unverified candidates as verified opportunities — the same rounding-up
        # that produced the incident this bucket exists to prevent.
        "promising_candidate_unverified": sum(
            1 for c in conclusions if c == domain.CONCLUSION_PROMISING_UNVERIFIED
        ),
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
