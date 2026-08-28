"""
The evidence engine.

A benchmark executes a BASELINE arm and one or more CANDIDATE arms over the
SAME inputs, with execution_mode='eval', and measures what happened. That
sameness is what makes the result a counterfactual rather than an observation,
and a counterfactual is the only thing that may move a recommendation to
'verified'.

Three rules this module exists to enforce
-----------------------------------------

1. BENCHMARKS DISCOVER FACTS; RECOMMENDATIONS PROPOSE ACTIONS.
   A benchmark belongs to a WORKLOAD and may exist forever with no
   recommendation. Completing a benchmark never auto-creates one — only
   conclusion='safe_improvement_found' may create or advance a recommendation.

2. EVIDENCE OUTLIVES ITS INTERPRETATION.
   Every measured arm is written to `benchmark_candidate_results` regardless of
   the verdict. A candidate that saved 51% but missed the quality floor by
   0.7pp is retained as a first-class result, so relaxing the threshold later is
   a re-read (`reevaluate`) rather than a re-measurement.

3. IGNORANCE IS NEVER RENDERED AS A FINDING.
   `insufficient_evidence` is not evidence that the current configuration is
   optimal. It is structurally separated in optimization.domain and can never be
   counted as coverage or as an efficiency finding.

Reuse
-----
Execution goes through `workflow_runtime.execute_workflow` — the same primitive
the existing eval machinery uses. Quality checks are read from the same
`eval_suites.checks` rows that `workflow_management._run_eval_sync` reads, and
comparisons use its `_normalize_output`, so a quality number produced here means
the same thing it means in the eval UI. `model_graded` checks are deliberately
NOT run: that path is unimplemented in this codebase
("AI-graded check not yet implemented"), and an LLM judge could not satisfy a
hard quality floor anyway.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import (
    allocation,
    domain,
    eligibility as eligibility_mod,
    executors as executors_mod,
    # The phase vocabulary an async job reports. Imported for the CONSTANTS, so
    # a phase name written from here is the same datum the database CHECK
    # constrains and the API documents — never a string literal typed twice.
    # jobs.py depends on nothing in this package, so there is no cycle.
    jobs as jobs_mod,
    noninferiority,
    outcomes as outcomes_mod,
    policies,
    staging as staging_mod,
    strategy as strategy_mod,
    workloads as workloads_mod,
)

logger = logging.getLogger(__name__)

#: Minimum replay cases before a conclusion may be drawn. Below this the run
#: does NOT error and does NOT go quiet — it concludes 'insufficient_evidence'
#: with a `sample_size_below_threshold` reason carrying observed and required.
#:
#: 20 is a floor, not a target: it is roughly where a single anomalous case
#: stops being able to swing a mean cost delta past a 5% materiality threshold.
#: It does not make a result statistically strong; evidence maturity reports that.
DEFAULT_MIN_SAMPLE_SIZE = 20

BENCHMARK_COLS = (
    "id, org_id, workload_id, method, status, sample_size, dataset_ref, "
    "baseline_metrics, candidate_metrics, per_case_results, quality_provenance, "
    "error, started_at, completed_at, created_at, baseline_strategy_id, "
    "candidate_strategy_id, objective, policy_evaluation, success_signal, "
    "conclusion, conclusion_detail, more_data_changes_conclusion, more_data_reason, "
    "materiality_threshold, policy_id, confidence"
)

CANDIDATE_RESULT_COLS = (
    "id, org_id, benchmark_id, workload_id, strategy_id, strategy_fingerprint, arm, "
    "label, generator, dimensions, executor_refs, sample_size, mean_cost_usd, "
    "total_cost_usd, latency_p50_ms, latency_p95_ms, error_rate, quality, "
    "quality_provenance, outcome_metrics, cost_delta_pct, latency_delta_pct, "
    "quality_delta, evidence_source, per_case_results, error, created_at"
)

CONCLUSION_COLS = (
    "id, org_id, benchmark_id, workload_id, policy_id, policy_key, policy_version, "
    "objective, conclusion, reasons, confidence, confidence_band, materiality_applied, "
    "success_signal, more_data_changes_conclusion, more_data_reasons, "
    "selected_candidate_result_id, quality_safety, quality_safety_policy, "
    "frontier, consideration, is_current, created_at"
)

_GOLDEN_INPUT_COLS = (
    "id, org_id, workflow_id, name, input_text, variables, expected_output, "
    "source, source_run_id, created_at, updated_at"
)

_EVAL_SUITE_COLS = "id, org_id, workflow_id, name, checks, enabled, created_at, updated_at"


class BenchmarkError(RuntimeError):
    """A benchmark could not be started at all."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Quality checks — same definitions as the existing eval machinery
# ---------------------------------------------------------------------------

def _normalize_output(s: Optional[str]) -> str:
    """
    Mirrors workflow_management._normalize_output.

    Duplicated as three lines rather than imported, because importing
    workflow_management here would pull an entire FastAPI router module into
    every benchmark. The contract is whitespace collapsing, and it must stay
    identical: a quality number produced here has to mean the same thing it
    means in the eval UI.
    """
    if s is None:
        return ""
    return " ".join(str(s).split())


#: Check types that produce a MEASURABLE quality verdict, mapped to the
#: provenance they justify.
_QUALITY_CHECK_PROVENANCE = {
    "deterministic": "deterministic",
    "structural": "schema",
    "format": "schema",
}

#: Not a quality signal. 'regression' is a cost/latency guard, and
#: 'model_graded' is unimplemented in this codebase — and an LLM judge sits
#: below the rank required to satisfy a hard quality floor regardless.
_NON_QUALITY_CHECKS = ("regression", "model_graded")


def _run_quality_checks(checks: list[dict], output: Optional[str], expected: Optional[str]) -> dict:
    """
    Apply the workflow's own eval-suite checks to one output.

    Returns {"ran": int, "passed": int, "provenance": str|None, "details": [...]}.
    `ran == 0` means nothing measurable applied, and the caller MUST leave
    quality NULL rather than defaulting it to 1.0 for "no failures".
    """
    ran = passed = 0
    provenance: Optional[str] = None
    details: list[dict] = []

    for ch in checks:
        ctype = (ch.get("type") or "deterministic").lower()
        if ctype in _NON_QUALITY_CHECKS:
            continue
        cfg = ch.get("config") or {}

        if ctype == "deterministic":
            # An empty expected_output means there is nothing to check against.
            # The existing eval machinery passes it; here it must NOT count as a
            # measured quality point, or a suite with no expectations would
            # yield quality=1.0 out of thin air.
            if expected is None or expected == "":
                details.append({"type": ctype, "skipped": "no_expected_output"})
                continue
            ok = _normalize_output(output) == _normalize_output(expected)

        elif ctype == "structural":
            if not cfg.get("expect_json"):
                details.append({"type": ctype, "skipped": "no_structural_expectation"})
                continue
            try:
                if output and output.strip():
                    json.loads(output)
                    ok = True
                else:
                    ok = False
            except (ValueError, TypeError):
                ok = False

        elif ctype == "format":
            pattern = (cfg.get("pattern") or "").strip()
            if not pattern:
                details.append({"type": ctype, "skipped": "no_pattern"})
                continue
            try:
                ok = bool(re.compile(pattern).search(output or ""))
            except re.error:
                details.append({"type": ctype, "skipped": "invalid_pattern"})
                continue
        else:
            continue

        ran += 1
        if ok:
            passed += 1
        details.append({"type": ctype, "name": ch.get("name"), "passed": ok})

        rank_new = domain.provenance_rank(_QUALITY_CHECK_PROVENANCE[ctype])
        if provenance is None or rank_new > domain.provenance_rank(provenance):
            provenance = _QUALITY_CHECK_PROVENANCE[ctype]

    return {"ran": ran, "passed": passed, "provenance": provenance, "details": details}


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------

def _execute_arm(
    graph: dict,
    cases: list[dict],
    *,
    org_id: str,
    workflow_id: Optional[str],
    endpoint_slug: Optional[str],
    checks: list[dict],
    label: str,
    strategy: Optional[strategy_mod.Strategy] = None,
) -> dict:
    """
    Execute one arm over the shared case set and MEASURE it.

    Every metric returned is measured or None. A case that raises is recorded as
    an error case and counted in the error rate — never dropped, because
    dropping failures would make an unreliable candidate look clean.

    COST PROVENANCE. `execute_workflow` prices every call through
    `utils.pricing.get_pricing`, which returns a loud fallback guess for a model
    id it cannot resolve. A dollar figure derived from that guess is NOT a
    measurement, and comparing it against a real price manufactures a saving
    that was never observed. So the arm's pricing provenance is resolved from
    its strategy BEFORE execution, and when any model in it is unpriced the
    figures land in `mean_cost_estimated_usd` / `total_cost_estimated_usd` and
    the measured fields stay NULL. Downstream this makes the arm unrankable on
    a cost objective, which is the honest outcome: we cannot say it is cheaper.
    """
    from workflow_runtime import execute_workflow  # imported lazily: see module docstring

    pricing = executors_mod.pricing_provenance(
        [s.executor_ref for s in strategy.steps] if strategy else []
    )
    cost_is_measured = pricing["basis"] == executors_mod.COST_BASIS_MEASURED

    costs: list[float] = []
    latencies: list[float] = []
    errors = 0
    quality_ran = quality_passed = 0
    quality_provenance: Optional[str] = None
    per_case: list[dict] = []

    for case in cases:
        case_id = str(case.get("id"))
        input_text = (case.get("input_text") or "")[:5000]
        variables = case.get("variables") if isinstance(case.get("variables"), dict) else None
        expected = case.get("expected_output")

        try:
            result = execute_workflow(
                graph,
                input_text,
                org_id,
                "",
                workflow_id=workflow_id,
                endpoint_slug=endpoint_slug or None,
                execution_mode="eval",
                variables=variables,
            )
            output = (result.get("final_output") or "")[:10000]
            cost = result.get("total_cost")
            latency = result.get("total_latency_ms")

            if cost is not None:
                costs.append(float(cost))
            if latency is not None:
                latencies.append(float(latency))

            q = _run_quality_checks(checks, output, expected)
            quality_ran += q["ran"]
            quality_passed += q["passed"]
            if q["provenance"] and (
                quality_provenance is None
                or domain.provenance_rank(q["provenance"])
                > domain.provenance_rank(quality_provenance)
            ):
                quality_provenance = q["provenance"]

            per_case.append({
                "case_id": case_id,
                "arm": label,
                # An estimated price never lands in a measured field, on any row.
                "cost_usd": (cost if cost_is_measured else None),
                "cost_estimated_usd": (None if cost_is_measured else cost),
                "cost_basis": pricing["basis"],
                "latency_ms": latency,
                "error": False,
                "quality_checks_ran": q["ran"],
                "quality_checks_passed": q["passed"],
                # The PAIRED datum. Baseline and candidates replay the same
                # cases, so retaining a per-case verdict is what makes a paired
                # non-inferiority test possible instead of pretending the arm
                # means are independent samples. None when nothing measurable
                # ran on the case — "no check ran" is not "the check failed".
                "case_passed": (
                    (q["passed"] >= q["ran"]) if q["ran"] > 0 else None
                ),
                # Carried per case so an arm restricted to a PREFIX of the case
                # set can be re-summarised exactly, provenance included, rather
                # than inheriting the full run's provenance by assumption.
                "quality_provenance": q["provenance"],
            })

        except Exception as exc:
            errors += 1
            per_case.append({
                "case_id": case_id,
                "arm": label,
                "cost_usd": None,
                "cost_estimated_usd": None,
                "cost_basis": pricing["basis"],
                "latency_ms": None,
                "error": True,
                "error_detail": str(exc)[:300],
                "case_passed": None,
                "quality_provenance": None,
            })

    return _summarize_arm(label, per_case, pricing)


def _summarize_arm(label: str, per_case: list[dict], pricing: dict) -> dict:
    """
    Roll per-case rows up into the arm metrics the rest of the loop ranks on.

    Split out of `_execute_arm` because staged evaluation needs to summarise the
    BASELINE over exactly the prefix of cases a stopped candidate actually ran.
    Comparing a candidate that ran 30 cases against a baseline summarised over
    133 would be a comparison across two different case sets, which is precisely
    what the paired design exists to prevent. This function is the only place an
    arm figure is derived, so a prefix summary and a full summary are the same
    computation over different rows rather than two code paths that could drift.
    """
    cost_is_measured = pricing["basis"] == executors_mod.COST_BASIS_MEASURED

    costs: list[float] = []
    latencies: list[float] = []
    errors = quality_ran = quality_passed = 0
    quality_provenance: Optional[str] = None

    for row in per_case:
        if row.get("error"):
            errors += 1
        cost = row.get("cost_usd") if cost_is_measured else row.get("cost_estimated_usd")
        if cost is not None:
            costs.append(float(cost))
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        ran = int(row.get("quality_checks_ran") or 0)
        if ran > 0:
            quality_ran += ran
            quality_passed += int(row.get("quality_checks_passed") or 0)
        prov = row.get("quality_provenance")
        if prov and (
            quality_provenance is None
            or domain.provenance_rank(prov) > domain.provenance_rank(quality_provenance)
        ):
            quality_provenance = prov

    n = len(per_case)
    quality = (quality_passed / quality_ran) if quality_ran > 0 else None

    mean_cost = domain.mean(costs)
    total_cost = (round(sum(costs), 8) if costs else None)

    return {
        "label": label,
        "n": n,
        # Measured columns stay NULL when the price sheet had to guess.
        "mean_cost_usd": (mean_cost if cost_is_measured else None),
        "total_cost_usd": (total_cost if cost_is_measured else None),
        "mean_cost_estimated_usd": (None if cost_is_measured else mean_cost),
        "total_cost_estimated_usd": (None if cost_is_measured else total_cost),
        "cost_basis": pricing["basis"],
        "pricing_provenance": pricing,
        "latency_p50_ms": (int(domain.percentile(latencies, 50)) if latencies else None),
        "latency_p95_ms": (int(domain.percentile(latencies, 95)) if latencies else None),
        "error_rate": (round(errors / n, 4) if n else None),
        # NULL, not 1.0, when nothing measurable ran. "No check failed" and
        # "no check ran" are different facts.
        "quality": (round(quality, 4) if quality is not None else None),
        "quality_provenance": quality_provenance or "unknown",
        "quality_checks_run": quality_ran,
        "cost_variation": domain.coefficient_of_variation(costs),
        "per_case": per_case,
        "cases_measured": len(costs),
    }


def _arm_over_prefix(arm: dict, cases_run: int) -> dict:
    """
    The same arm, restricted to the first `cases_run` cases.

    Used to compare a candidate that was stopped early against the baseline over
    EXACTLY the cases both arms ran. Never a re-execution: it is a re-summary of
    rows already measured.
    """
    rows = arm.get("per_case") or []
    # 0 means "this arm stored no per-case rows", not "it ran no cases" — a
    # historical row predating per-case retention. Restricting to nothing would
    # turn a measured arm into an unmeasured one, so leave it alone.
    if cases_run <= 0 or cases_run >= len(rows):
        return arm
    return _summarize_arm(
        arm.get("label") or "baseline",
        list(rows)[:cases_run],
        arm.get("pricing_provenance") or {"basis": arm.get("cost_basis")},
    )


# ---------------------------------------------------------------------------
# Preparation — everything resolvable BEFORE a row is written
# ---------------------------------------------------------------------------
#
# Split out of run_benchmark so an ASYNCHRONOUS caller can validate the request
# and persist the job row while it still holds the HTTP connection, then hand
# the run itself to a worker. An unknown objective must be a 400 on the POST,
# not a job that is accepted with 202 and fails a second later — the caller is
# still there to be told, and telling it is free.
#
# It resolves only: the workload, the objective, the effective policy, its
# materiality and the sample-size floor. All are cheap reads. Nothing here
# executes an arm, prices a call or reaches a provider.

def prepare_benchmark(
    org_id: str,
    *,
    workload_id: str,
    objective: Optional[str] = None,
    min_sample_size: Optional[int] = None,
) -> dict:
    """Resolve a benchmark's inputs, or raise BenchmarkError. Writes nothing."""
    workload = workloads_mod.get_workload(org_id, workload_id)
    if workload is None:
        raise BenchmarkError("Workload not found for this organization.")

    objective = objective or workload.get("default_objective") or domain.DEFAULT_OBJECTIVE
    if not domain.is_valid_objective(objective):
        raise BenchmarkError(f"Unknown objective '{objective}'.")

    policy = policies.get_effective_policy(org_id, workload_id)
    materiality = policies.materiality_of(policy, objective)
    floor = int(
        min_sample_size
        if min_sample_size is not None
        else (policies.constraints_of(policy).get("min_sample_size") or DEFAULT_MIN_SAMPLE_SIZE)
    )
    return {
        "workload": workload,
        "objective": objective,
        "policy": policy,
        "materiality": materiality,
        "floor": floor,
    }


def create_benchmark_row(
    org_id: str,
    prepared: dict,
    *,
    method: str = "golden_replay",
    extra_columns: Optional[dict] = None,
) -> Optional[dict]:
    """
    Persist the benchmark row that IS the job. `extra_columns` carries the
    async-job fields (optimization/jobs.py owns their meaning; this module owns
    the row).
    """
    return _insert_benchmark(
        org_id,
        workload_id=str(prepared["workload"]["id"]),
        method=method,
        objective=prepared["objective"],
        policy=prepared["policy"],
        materiality=prepared["materiality"],
        extra=extra_columns,
    )


def _progress_emitter(progress):
    """
    Wrap a progress callback so it can never fail the run.

    The measurement is the product; the progress bar is not. A benchmark that
    has spent twenty minutes of real provider budget must not be lost because a
    JSONB status write timed out.
    """
    if progress is None:
        return lambda *_a, **_k: None

    def _emit(state: str, **facts) -> None:
        try:
            progress(state, **facts)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Progress callback raised; the benchmark continues.")

    return _emit


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    org_id: str,
    *,
    workload_id: str,
    candidates: Optional[list] = None,
    recommendation_id: Optional[str] = None,
    objective: Optional[str] = None,
    method: str = "golden_replay",
    min_sample_size: Optional[int] = None,
    actor: Optional[str] = None,
    create_recommendation: bool = False,
    benchmark_id: Optional[str] = None,
    prepared: Optional[dict] = None,
    progress: Optional[Any] = None,
) -> dict:
    """
    Run a replay benchmark for a workload and record explicit evidence.

    `recommendation_id` is OPTIONAL. A benchmark may be run exploratorily
    against a workload with no recommendation in existence; when one IS given,
    the benchmark is CITED by it via `recommendation_evidence` and the
    conclusion maps to a lifecycle transition via domain.CONCLUSION_TO_STATUS.

    `create_recommendation` closes the loop: when the evidence — and ONLY when
    the evidence — concludes `safe_improvement_found`, a recommendation is
    created that CITES this benchmark. It is opt-in rather than automatic
    because a benchmark discovering a fact and a product proposing an action are
    two different events, and the caller decides whether it wants the second.
    Every other conclusion creates nothing: there is no path by which
    `insufficient_evidence` or a policy failure produces a proposal.

    `benchmark_id`/`prepared` let an ASYNCHRONOUS caller persist the row first
    (so it can hand the id back on a 202 before the run starts) and then execute
    against it. Omitted, the behaviour is exactly what it was: resolve, insert,
    run, synchronously.

    `progress` is an optional callback invoked at real phase boundaries. It is
    reporting only — it observes work already done and can neither change what
    is measured nor fail the run.

    Returns the benchmark row plus its conclusion payload. Never raises for an
    evidence shortfall — that is a conclusion, not an error.
    """
    from optimization import candidates as candidates_mod
    from optimization import service as service_mod

    prepared = prepared or prepare_benchmark(
        org_id,
        workload_id=workload_id,
        objective=objective,
        min_sample_size=min_sample_size,
    )
    workload = prepared["workload"]
    objective = prepared["objective"]
    policy = prepared["policy"]
    materiality = prepared["materiality"]
    floor = prepared["floor"]

    if benchmark_id is None:
        benchmark = create_benchmark_row(org_id, prepared, method=method)
        if benchmark is None:
            raise BenchmarkError("Failed to create the benchmark record.")
        benchmark_id = str(benchmark["id"])
    else:
        benchmark_id = str(benchmark_id)

    try:
        return _run(
            org_id,
            benchmark_id=benchmark_id,
            workload=workload,
            objective=objective,
            policy=policy,
            materiality=materiality,
            floor=floor,
            method=method,
            explicit_candidates=candidates,
            recommendation_id=recommendation_id,
            actor=actor,
            create_recommendation=create_recommendation,
            candidates_mod=candidates_mod,
            service_mod=service_mod,
            progress=progress,
        )
    except Exception as exc:
        logger.exception("Benchmark %s failed", benchmark_id)
        return _conclude(
            org_id,
            benchmark_id=benchmark_id,
            workload=workload,
            objective=objective,
            policy=policy,
            materiality=materiality,
            conclusion=domain.CONCLUSION_BENCHMARK_FAILED,
            reasons=[domain.reason("execution_error", detail=str(exc)[:300])],
            evidence_maturity=None,
            sample_size=None,
            success_signal=domain.SuccessSignal(),
            more_data=domain.MORE_DATA_UNKNOWN,
            more_data_reasons=[],
            status="failed",
            error=str(exc)[:500],
            recommendation_id=recommendation_id,
            actor=actor,
            service_mod=service_mod,
        )


def _run(
    org_id: str,
    *,
    benchmark_id: str,
    workload: dict,
    objective: str,
    policy: Optional[dict],
    materiality: dict,
    floor: int,
    method: str,
    explicit_candidates,
    recommendation_id: Optional[str],
    actor: Optional[str],
    candidates_mod,
    service_mod,
    create_recommendation: bool = False,
    progress: Optional[Any] = None,
) -> dict:
    # Phase reporting only. Every emit() call below observes work that has
    # ALREADY happened — cases loaded, arms finished, stages run. None of them
    # estimates remaining time or predicts a verdict, and none can alter what is
    # measured. `emit` is a no-op when no callback was supplied, which is the
    # case for every synchronous caller and every existing test.
    emit = _progress_emitter(progress)

    workload_id = str(workload["id"])
    emit(jobs_mod.PROGRESS_PREPARING)
    workflow_id = workloads_mod.resolve_workflow_id(org_id, workload)

    # A runtime replay needs a workflow graph. Direct-inference workloads have
    # none: say so explicitly rather than failing obscurely.
    if not workflow_id:
        emit(jobs_mod.PROGRESS_CONCLUDING)
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "baseline_unavailable",
                detail=(
                    "No workflow is associated with this workload, so there is no "
                    "graph to replay. Golden replay currently supports the runtime "
                    "surface only."
                ),
                surface=workload.get("surface"),
            )],
            evidence_maturity=None, sample_size=0, success_signal=domain.SuccessSignal(),
            more_data=domain.MORE_DATA_UNKNOWN,
            more_data_reasons=[domain.reason(
                "baseline_unavailable",
                detail="A replay surface for this workload does not exist yet.",
            )],
            status="completed", error=None, recommendation_id=recommendation_id,
            actor=actor, service_mod=service_mod,
        )

    baseline_graph, endpoint_slug = _load_baseline_graph(org_id, workload, workflow_id)
    if baseline_graph is None:
        emit(jobs_mod.PROGRESS_CONCLUDING)
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "baseline_unavailable",
                detail="No promoted deployment or workflow graph could be resolved.",
                workflow_id=workflow_id,
            )],
            evidence_maturity=None, sample_size=0, success_signal=domain.SuccessSignal(),
            more_data=domain.MORE_DATA_UNKNOWN, more_data_reasons=[],
            status="completed", error=None, recommendation_id=recommendation_id,
            actor=actor, service_mod=service_mod,
        )

    baseline_strategy = strategy_mod.from_graph_json(baseline_graph, workflow_id=workflow_id)

    cases = _load_golden_inputs(org_id, workflow_id)
    n = len(cases)

    # ── Sample-size floor. A refusal, recorded as a CONCLUSION, not an error
    # and not silence.
    if n < floor:
        emit(jobs_mod.PROGRESS_CONCLUDING, cases_planned=n)
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "sample_size_below_threshold", observed=n, required=floor,
                unit="cases", dataset="golden_inputs",
            )],
            evidence_maturity=None, sample_size=n, success_signal=domain.SuccessSignal(),
            more_data=domain.MORE_DATA_YES,
            more_data_reasons=[domain.reason(
                "sample_size_below_threshold", observed=n, required=floor,
                detail=(
                    "Adding golden inputs (including promoting real production "
                    "runs) would let this workload be assessed."
                ),
            )],
            status="completed", error=None, recommendation_id=recommendation_id,
            actor=actor, service_mod=service_mod,
        )

    # ── Candidates
    emit(jobs_mod.PROGRESS_CANDIDATE_SCREENING, cases_planned=n)
    if explicit_candidates:
        cand_list = list(explicit_candidates)
        gen_meta = {"source": "caller_supplied", "dropped": []}
    else:
        cand_list, gen_meta = candidates_mod.generate_candidates(
            org_id, workload, baseline_strategy, workflow_id=workflow_id
        )

    # ── ELIGIBILITY PREFLIGHT — the gate between hypothesis and spend.
    #
    # Generation proposes; it does not authorise. Every candidate — generated or
    # caller-supplied — is checked against the org's credentials, the executor
    # catalog, the policy in force, the request shape it would actually send and
    # the objective, BEFORE any provider request exists for it.
    #
    # This is the o1-mini fix. That arm ran 140 cases at a 100% error rate
    # because the incompatibility was only discoverable from provider errors; it
    # is now discoverable from a declaration, and an ineligible candidate simply
    # never enters the replay loop below.
    #
    # An exclusion is NOT a benchmark failure. Each one keeps its structured
    # evidence and appears in the consideration funnel with a reason code.
    preflight = eligibility_mod.preflight(
        cand_list,
        baseline=baseline_strategy,
        objective=objective,
        policy=policy,
        materiality=materiality,
        history=gen_meta.get("history"),
        configured_providers=(
            set(gen_meta.get("configured_providers") or [])
            or candidates_mod._configured_providers(org_id)
        ),
    )
    cand_list = preflight.eligible
    gen_meta["dropped"] = (gen_meta.get("dropped") or []) + preflight.excluded
    gen_meta["opportunities"] = (
        (gen_meta.get("opportunities") or []) + preflight.opportunities
    )
    gen_meta["eligibility"] = preflight.to_dict()

    if not cand_list:
        # Nothing survived to dispatch. The funnel is still emitted in full:
        # "we considered eight and none were eligible, here is the code for
        # each" is a genuine finding, and it is the difference between a
        # customer being told nothing was looked at and being told what was.
        # arms_total=0 is the honest count: preflight ran no arm, not even the
        # baseline, so no provider request was made for this benchmark.
        emit(jobs_mod.PROGRESS_CONCLUDING, cases_planned=n, arms_total=0)
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "no_candidates_generated",
                detail="No eligible candidate strategy reached the replay stage.",
                dropped=gen_meta.get("dropped") or [],
            )],
            evidence_maturity=None, sample_size=n, success_signal=domain.SuccessSignal(),
            more_data=domain.MORE_DATA_UNKNOWN,
            more_data_reasons=[domain.reason(
                "no_candidates_generated",
                detail=(
                    "Nothing was measured, so this says nothing about whether the "
                    "current configuration is optimal."
                ),
            )],
            status="completed", error=None, recommendation_id=recommendation_id,
            actor=actor, service_mod=service_mod,
            consideration=domain.build_funnel(_dispositions(
                measured=[], safe=[], promising=[],
                opportunities=gen_meta.get("opportunities") or [],
                generation=gen_meta,
                baseline={}, objective=objective,
            )),
        )

    checks = _load_eval_checks(org_id, workflow_id)

    _update_benchmark(org_id, benchmark_id, {
        "status": "running",
        "started_at": _iso(_utc_now()),
        "sample_size": n,
        "dataset_ref": {
            "kind": "golden_inputs",
            "workflow_id": workflow_id,
            "golden_input_ids": [str(c["id"]) for c in cases],
            "requested": n,
            "used": n,
            "endpoint_slug": endpoint_slug,
        },
    })

    # ── Measure the baseline arm, over the FULL case set, first.
    #
    # The baseline is the reference and is never a candidate for elimination, so
    # it costs the same whether it is staged or not. Running it to completion up
    # front is what makes staged candidate evaluation possible at all: it means
    # the baseline's verdicts on the cases a candidate has NOT yet reached are
    # already MEASURED, so the early-stop bound can use real counts instead of
    # assuming the worst about every unrun case. See optimization/staging.py.
    emit(
        jobs_mod.PROGRESS_BASELINE_MEASUREMENT,
        cases_planned=n, arms_total=len(cand_list) + 1, arms_completed=0,
    )
    baseline_metrics = _execute_arm(
        baseline_graph, cases, org_id=org_id, workflow_id=workflow_id,
        endpoint_slug=endpoint_slug, checks=checks, label="baseline",
        strategy=baseline_strategy,
    )
    _write_candidate_result(
        org_id, benchmark_id, workload_id, arm="baseline", label="Current configuration",
        strategy=baseline_strategy, metrics=baseline_metrics, baseline=None,
        generator=None, dimensions=[],
    )

    # ── Measure each candidate arm over the SAME cases, IN THE SAME ORDER,
    # in stages. `cases` is materialised once above and every arm walks the same
    # list from index 0, so a candidate stopped at stage k has been compared
    # against the baseline on exactly the prefix it ran, and never against a
    # baseline summarised over cases it never saw.
    staging_cfg = policies.staged_evaluation_of(policy)
    stage_plan = (
        staging_mod.resolve_stages(n, staging_cfg["evaluation_stage_sizes"])
        if staging_cfg["staged_evaluation_enabled"]
        else staging_mod.resolve_stages(n, [])
    )
    if not stage_plan:
        # No cases to slice. Keep the single unstaged pass so the arm is still
        # executed and its pricing provenance resolved, exactly as before.
        stage_plan = [
            {"stage_index": 1, "start": 0, "end": n, "size": n, "cases_cumulative": n}
        ]
    quality_safety_cfg = policies.quality_safety_of(policy)
    stage_margin = quality_safety_cfg["max_quality_regression"]

    measured: list[dict] = []
    staging_records: list[dict] = []
    # The stage count is now KNOWN, so the phase plan the caller polls stops
    # being open-ended: `stages_planned` fixes how many stage_k phases exist.
    emit(
        jobs_mod.PROGRESS_BASELINE_MEASUREMENT,
        stages_planned=len(stage_plan), cases_planned=n,
        arms_total=len(cand_list) + 1, arms_completed=1,
    )
    for cand in cand_list:
        try:
            cand_graph = strategy_mod.apply_to_graph(baseline_graph, cand.strategy)
        except (strategy_mod.UnsupportedDimension, strategy_mod.StrategyApplyError) as exc:
            _write_candidate_result(
                org_id, benchmark_id, workload_id, arm="candidate", label=cand.title,
                strategy=cand.strategy, metrics=None, baseline=baseline_metrics,
                generator=cand.generator, dimensions=cand.dimensions, error=str(exc)[:300],
            )
            continue

        per_case: list[dict] = []
        pricing: Optional[dict] = None
        stage_log: list[dict] = []
        stop_decision: Optional[dict] = None

        for stage in stage_plan:
            emit(
                jobs_mod.stage_state(stage["stage_index"]),
                stage_index=stage["stage_index"],
                arms_completed=len(measured) + 1,
                cases_completed=stage["start"],
            )
            part = _execute_arm(
                cand_graph, cases[stage["start"]:stage["end"]],
                org_id=org_id, workflow_id=workflow_id, endpoint_slug=endpoint_slug,
                checks=checks, label=cand.title, strategy=cand.strategy,
            )
            per_case.extend(part["per_case"])
            pricing = part["pricing_provenance"]

            decision = staging_mod.early_stop_assessment(
                margin=stage_margin,
                baseline_per_case=baseline_metrics["per_case"],
                candidate_per_case=per_case,
            )
            so_far = _summarize_arm(cand.title, per_case, pricing)
            base_so_far = _arm_over_prefix(baseline_metrics, len(per_case))
            # WHAT WAS KNOWN WHEN. Persisted per stage so the decision to stop
            # (or not to stop) is re-checkable against the evidence that existed
            # at the moment it was taken, rather than only against the totals.
            stage_log.append({
                "stage_index": stage["stage_index"],
                "cases_this_stage": stage["size"],
                "cases_cumulative": len(per_case),
                "quality": so_far.get("quality"),
                "baseline_quality_same_cases": base_so_far.get("quality"),
                "observed_regression": decision.get("observed_regression_prefix"),
                "paired": decision.get("paired"),
                "best_case_final_paired_delta": decision.get("best_case_final_paired_delta"),
                "best_case_final_regression": decision.get("best_case_final_regression"),
                "decision": ("stop" if decision["stop"] else "continue"),
                "decision_reason_code": decision["reason_code"],
            })
            if decision["stop"]:
                stop_decision = decision
                break

        metrics = _summarize_arm(cand.title, per_case, pricing or {"basis": None})
        # The baseline restricted to the cases THIS candidate ran. For a
        # candidate that ran everything this is the baseline itself.
        paired_baseline = _arm_over_prefix(baseline_metrics, len(per_case))

        avoided = staging_mod.spend_avoided(
            cases_not_run=(n - len(per_case)),
            mean_cost_usd=metrics.get("mean_cost_usd"),
            cases_priced=metrics.get("cases_measured"),
        )
        staged_record = {
            "enabled": staging_cfg["staged_evaluation_enabled"],
            "stage_sizes": staging_cfg["evaluation_stage_sizes"],
            "stage_sizes_source": staging_cfg["evaluation_stage_sizes_source"],
            "stages_planned": len(stage_plan),
            "stages_run": len(stage_log),
            "cases_planned": n,
            "cases_run": len(per_case),
            "stopped_early": stop_decision is not None,
            "stopped_at_stage": (
                stage_log[-1]["stage_index"] if stop_decision is not None else None
            ),
            "stop_reason_code": (
                stop_decision["reason_code"] if stop_decision is not None else None
            ),
            "margin": round(float(stage_margin), 6),
            "margin_source": quality_safety_cfg["max_quality_regression_source"],
            # The bound that justified the stop, with every input it used.
            "bound": stop_decision,
            "stages": stage_log,
            **avoided,
        }
        staging_records.append({**staged_record, "candidate": cand.title})

        row = _write_candidate_result(
            org_id, benchmark_id, workload_id, arm="candidate", label=cand.title,
            strategy=cand.strategy, metrics=metrics, baseline=paired_baseline,
            generator=cand.generator, dimensions=cand.dimensions,
            staged_evaluation=staged_record,
        )
        measured.append({
            "candidate": cand,
            "metrics": metrics,
            "result_id": (str(row["id"]) if row else None),
            # Every gate that compares this candidate to the baseline uses THIS
            # baseline, not the full-run one. Pairing is the whole basis of the
            # statistics and a stopped candidate must never be scored against
            # cases it did not run.
            "paired_baseline": paired_baseline,
            "staged_evaluation": staged_record,
        })

    staging_summary = staging_mod.rollup(staging_records)

    # Every arm is measured. What follows runs no model calls: it compares what
    # was measured against the policy.
    emit(
        jobs_mod.PROGRESS_VERIFICATION,
        arms_total=len(cand_list) + 1, arms_completed=len(measured) + 1,
        cases_planned=n,
    )

    # ── Success signal: the POLICY decides, not a global constant
    observed_outcomes = outcomes_mod.list_outcomes(org_id, workload_id=workload_id, limit=500)
    signal = domain.resolve_success_signal(
        policies.success_signal_of(policy), observed_outcomes
    )

    _update_benchmark(org_id, benchmark_id, {
        "status": "completed",
        "completed_at": _iso(_utc_now()),
        "baseline_metrics": _public_metrics(baseline_metrics),
        "candidate_metrics": [_public_metrics(m["metrics"]) for m in measured],
        "per_case_results": (
            baseline_metrics["per_case"] + [c for m in measured for c in m["metrics"]["per_case"]]
        ),
        "quality_provenance": baseline_metrics["quality_provenance"],
        "success_signal": signal.to_dict(),
    })

    verdict = evaluate_conclusion(
        baseline=baseline_metrics,
        measured=measured,
        policy=policy,
        materiality=materiality,
        objective=objective,
        signal=signal,
        sample_size=n,
        traffic=_traffic_for(org_id, workload),
        opportunities=gen_meta.get("opportunities") or [],
        generation=gen_meta,
    )
    if verdict.get("winner"):
        # Carry the baseline it was measured against, so verified savings are
        # computed from the same run and never from an unrelated baseline.
        verdict["winner"]["baseline_mean_cost_usd"] = baseline_metrics.get("mean_cost_usd")
        verdict["winner"]["baseline_metrics"] = _public_metrics(baseline_metrics)

    emit(jobs_mod.PROGRESS_CONCLUDING)
    concluded = _conclude(
        org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
        policy=policy, materiality=verdict["materiality_applied"],
        conclusion=verdict["conclusion"], reasons=verdict["reasons"],
        evidence_maturity=verdict["evidence_maturity"], sample_size=n, success_signal=signal,
        more_data=verdict["more_data_changes_conclusion"],
        more_data_reasons=verdict["more_data_reasons"],
        status="completed", error=None,
        selected_result_id=(
            verdict.get("selected_result_id")
            or verdict.get("leading_candidate_result_id")
        ),
        winner=verdict.get("winner"),
        recommendation_id=recommendation_id, actor=actor, service_mod=service_mod,
        create_recommendation=create_recommendation,
        baseline_strategy=baseline_strategy,
        quality_safety=verdict.get("quality_safety"),
        quality_safety_policy=verdict.get("quality_safety_policy"),
        frontier=verdict.get("frontier"),
        consideration=verdict.get("consideration"),
    )
    # The staging rollup is DERIVED from the per-arm rows already persisted in
    # `benchmark_candidate_results.outcome_metrics`. It is surfaced on the
    # response rather than stored again, so there is exactly one written source
    # of truth for what each arm ran.
    if isinstance(concluded, dict):
        concluded["staged_evaluation"] = staging_summary
    return concluded


# ---------------------------------------------------------------------------
# The pure verdict function: evidence + policy version + objective -> conclusion
# ---------------------------------------------------------------------------

#: Which reason code explains a constraint that could not be evaluated because
#: the underlying metric was never measured. `coverage_gap` is the fallback.
_UNMEASURED_REASON_CODE = {
    "min_quality": "outcome_signal_too_weak",
    "max_quality_regression": "baseline_quality_not_measured",
}


def evaluate_conclusion(
    *,
    baseline: dict,
    measured: list[dict],
    policy: Optional[dict],
    materiality: dict,
    objective: str,
    signal: domain.SuccessSignal,
    sample_size: int,
    traffic: Optional[dict] = None,
    opportunities: Optional[list[dict]] = None,
    generation: Optional[dict] = None,
) -> dict:
    """
    PURE. Given stored evidence, a policy version and an objective, produce a
    conclusion, the quality-safety evidence behind it, the cost/quality frontier
    and the candidate consideration funnel. No I/O, no mutation.

    `opportunities` are TIER 2 items: candidates for providers this org has no
    credential for. They were never executed, so they carry no measurement, can
    never win, and can never reach `verified`. They are carried through so the
    frontier can name them with a connect-provider next action instead of
    silently dropping them.
    """
    ctx: dict = {}
    verdict = _decide(
        baseline=baseline, measured=measured, policy=policy,
        materiality=materiality, objective=objective, signal=signal,
        sample_size=sample_size, traffic=traffic, ctx=ctx,
    )
    verdict.setdefault("quality_safety", ctx.get("quality_safety"))
    verdict["quality_safety_policy"] = ctx.get("quality_safety_policy")
    verdict["frontier"] = _build_frontier(
        baseline=baseline,
        measured=measured,
        safe=ctx.get("safe") or [],
        promising=ctx.get("promising") or [],
        opportunities=opportunities or [],
        winner=verdict.get("winner"),
    )
    verdict["consideration"] = domain.build_funnel(
        _dispositions(
            measured=measured,
            safe=ctx.get("safe") or [],
            promising=ctx.get("promising") or [],
            opportunities=opportunities or [],
            generation=generation or {},
            baseline=baseline,
            objective=objective,
        )
    )
    return verdict


def _decide(
    *,
    baseline: dict,
    measured: list[dict],
    policy: Optional[dict],
    materiality: dict,
    objective: str,
    signal: domain.SuccessSignal,
    sample_size: int,
    ctx: dict,
    traffic: Optional[dict] = None,
) -> dict:
    """
    The verdict itself. `ctx` is an out-parameter carrying the intermediate
    classifications the wrapper needs to build the frontier and the funnel.

    This is what makes a verdict reproducible and re-derivable: `reevaluate`
    calls it over RETAINED candidate results under a NEW policy version, without
    re-running a single model call. A new verdict does not mean the old one was
    wrong — it was correct under the policy in force at the time, and both are
    retained.

    Returns {conclusion, reasons, evidence_maturity, more_data_changes_conclusion,
             more_data_reasons, materiality_applied, winner, selected_result_id}.
    """
    reasons: list[dict] = []
    more_data_reasons: list[dict] = []

    quality_provenance = baseline.get("quality_provenance") or "unknown"
    if not baseline.get("quality_checks_run"):
        reasons.append(domain.reason(
            "quality_not_measured",
            detail=(
                "No deterministic, structural or format check applied to this "
                "workload's replay cases, so quality was not measured."
            ),
        ))
        more_data_reasons.append(domain.reason(
            "quality_not_measured",
            detail="Adding expected outputs or an eval suite would let quality be judged.",
        ))

    if signal.outcome_type is None and policies.constraints_of(policy).get("min_quality"):
        reasons.append(domain.reason(
            "missing_primary_outcome",
            detail=(
                "The policy requires a minimum quality but no deciding outcome "
                "signal has been recorded for this workload."
            ),
        ))

    # ── Cost provenance. `utils.pricing` returns a loud fallback guess for a
    # model id it cannot resolve, and a dollar figure built on that guess is not
    # a measurement. Such an arm arrives here with mean_cost_usd already NULL,
    # so it cannot win a cost objective; this records WHY, with the models
    # responsible, and states that the fix is a price-sheet entry rather than
    # more replay cases.
    def _estimated_models(metrics: dict) -> list[str]:
        prov = metrics.get("pricing_provenance") or {}
        return [
            f"{e.get('vendor')}/{e.get('model')}"
            for e in (prov.get("estimated_models") or [])
        ]

    if baseline.get("cost_basis") == executors_mod.COST_BASIS_ESTIMATED:
        reasons.append(domain.reason(
            "cost_pricing_estimated",
            arm="baseline",
            models=_estimated_models(baseline) or None,
            observed_basis=executors_mod.COST_BASIS_ESTIMATED,
            required_basis=executors_mod.COST_BASIS_MEASURED,
        ))
        more_data_reasons.append(domain.reason(
            "cost_pricing_estimated",
            arm="baseline",
            detail=(
                "Adding a real vendor price for this model to the price sheet "
                "would make the cost comparison measurable. More replay cases "
                "would not."
            ),
        ))
    for _m in measured:
        if _m["metrics"].get("cost_basis") == executors_mod.COST_BASIS_ESTIMATED:
            reasons.append(domain.reason(
                "cost_pricing_estimated",
                arm="candidate",
                candidate=_m["candidate"].title,
                models=_estimated_models(_m["metrics"]) or None,
                observed_basis=executors_mod.COST_BASIS_ESTIMATED,
                required_basis=executors_mod.COST_BASIS_MEASURED,
            ))

    # ── SELECTION, IN THREE ORDERED STAGES.
    #
    #   1. Exclude candidates that VIOLATE a hard policy constraint. A violator
    #      is not a lower-ranked option; it is not an option.
    #   2. Exclude candidates whose quality safety is NOT ESTABLISHED against
    #      the measured baseline. This is an evidence gate, not a policy
    #      verdict: failing it means "we cannot yet vouch for this", which is a
    #      different fact from "this is bad" and reaches a different conclusion.
    #   3. Among what survives both, minimise the objective's own metric.
    #
    # The ordering is the fix. Ranking on cost first and checking quality after
    # is how a candidate 10 percentage points below baseline came to be the
    # recommendation: it was the cheapest thing that cleared an absolute floor.
    quality_safety_cfg = policies.quality_safety_of(policy)
    require_ni = bool(quality_safety_cfg["require_quality_non_inferiority"])
    # The REGIME is recorded on every conclusion, even ones where no candidate
    # got far enough to be assessed. It is what makes a stored verdict
    # reproducible: "0.05, from the default, at 95%" is part of the verdict.
    ctx["quality_safety_policy"] = quality_safety_cfg

    # ── Stage 1: hard policy constraints
    eligible: list[dict] = []
    for m in measured:
        metrics = m["metrics"]
        # A candidate stopped early ran a PREFIX of the case set. It is judged
        # against the baseline over that same prefix — `paired_baseline` — never
        # against the baseline's full-run figures. Comparing arms over different
        # case sets is not a weaker comparison, it is a different one.
        arm_baseline = m.get("paired_baseline") or baseline

        staged = m.get("staged_evaluation") or {}
        if staged.get("stopped_early"):
            # A settled verdict reached on fewer cases, NOT a shortage of
            # evidence. The bound that made it settled travels with the reason.
            bound = staged.get("bound") or {}
            reasons.append(domain.reason(
                "candidate_evaluation_stopped_early",
                candidate=m["candidate"].title,
                stopped_at_stage=staged.get("stopped_at_stage"),
                stages_planned=staged.get("stages_planned"),
                cases_run=staged.get("cases_run"),
                cases_planned=staged.get("cases_planned"),
                cases_not_run=staged.get("cases_not_run"),
                allowed_regression=staged.get("margin"),
                observed_regression=bound.get("observed_regression_prefix"),
                best_case_final_regression=bound.get("best_case_final_regression"),
                best_case_final_paired_delta=bound.get("best_case_final_paired_delta"),
                bound_method=bound.get("bound_method"),
                detail_code=staged.get("stop_reason_code"),
            ))

        evaluation = policies.evaluate(
            policy,
            measured={
                "quality": metrics.get("quality"),
                "error_rate": metrics.get("error_rate"),
                "latency_p95_ms": metrics.get("latency_p95_ms"),
                "cost_per_task_usd": metrics.get("mean_cost_usd"),
            },
            executor_refs=_executor_refs_of(m["candidate"]),
            quality_provenance=metrics.get("quality_provenance"),
            # Without the baseline arm, `max_quality_regression` is not a
            # weaker check — it is an uncheckable one.
            baseline={
                "quality": arm_baseline.get("quality"),
                "error_rate": arm_baseline.get("error_rate"),
                "latency_p95_ms": arm_baseline.get("latency_p95_ms"),
                "cost_per_task_usd": arm_baseline.get("mean_cost_usd"),
            },
        )
        m["policy_evaluation"] = evaluation
        if evaluation["eligible"]:
            eligible.append(m)
        else:
            for v in evaluation["violated"]:
                reasons.append(_violation_reason(v, m["candidate"].title))
            for u in evaluation["unmeasured"]:
                reasons.append(domain.reason(
                    _UNMEASURED_REASON_CODE.get(u["constraint"], "coverage_gap"),
                    constraint=u["constraint"], required=u["required"],
                    detail=u["reason"], candidate=m["candidate"].title,
                ))
            for u in evaluation["unenforced"]:
                reasons.append(domain.reason(
                    "constraint_unenforceable",
                    constraint=u["constraint"], required=u["required"], detail=u["reason"],
                ))

    if not eligible:
        # Did any candidate actually get measured? If none did, this is
        # ignorance, not a policy finding.
        if not measured:
            conclusion = domain.CONCLUSION_INSUFFICIENT_EVIDENCE
            more_data = domain.MORE_DATA_UNKNOWN
        else:
            any_unmeasured = any(
                m.get("policy_evaluation", {}).get("unmeasured") for m in measured
            )
            # A candidate blocked only because a required metric was never
            # measured is an evidence problem, not a policy failure.
            if any_unmeasured and not any(
                m.get("policy_evaluation", {}).get("violated") for m in measured
            ):
                conclusion = domain.CONCLUSION_INSUFFICIENT_EVIDENCE
                more_data = domain.MORE_DATA_YES
                more_data_reasons.append(domain.reason(
                    "coverage_gap",
                    detail=(
                        "Candidates could not be judged because a constraint's "
                        "metric was not measured in this run."
                    ),
                ))
            else:
                conclusion = domain.CONCLUSION_CANDIDATES_FAILED_POLICY
                more_data = domain.MORE_DATA_NO
                more_data_reasons.append(domain.reason(
                    "quality_below_threshold",
                    detail=(
                        "The measured shortfall is against a hard constraint; more "
                        "cases of the same kind would not change eligibility."
                    ),
                ))

        evidence_maturity = domain.compute_evidence_maturity(
            sample_size=sample_size, evidence_source="replay",
            quality_provenance=quality_provenance,
            variation=baseline.get("cost_variation"),
        )
        return {
            "conclusion": conclusion,
            "reasons": reasons,
            "evidence_maturity": evidence_maturity,
            "more_data_changes_conclusion": more_data,
            "more_data_reasons": more_data_reasons,
            "materiality_applied": materiality,
            "winner": None,
            "selected_result_id": None,
        }

    # ── Stage 2: QUALITY SAFETY, established against the measured baseline.
    #
    # `min_quality` and `max_quality_regression` (stage 1) are point-estimate
    # checks. They are necessary and not sufficient: 27/30 and 30/30 are also
    # compatible with a much larger true gap than the one observed. This stage
    # asks the harder question — can we RULE OUT, at the policy's confidence
    # level, that the candidate is worse than the baseline by more than the
    # allowed regression? See optimization/noninferiority.py for the test and
    # its assumptions.
    #
    # A candidate that fails here is NOT rejected. It is PROMISING and short of
    # evidence, which is a different product state with a different next action.
    safe: list[dict] = []
    promising: list[dict] = []
    for m in eligible:
        arm_baseline = m.get("paired_baseline") or baseline
        assessment = noninferiority.assess(
            baseline_per_case=arm_baseline.get("per_case"),
            candidate_per_case=m["metrics"].get("per_case"),
            margin=quality_safety_cfg["max_quality_regression"],
            confidence_level=quality_safety_cfg["quality_confidence_level"],
            baseline_quality=arm_baseline.get("quality"),
            candidate_quality=m["metrics"].get("quality"),
        )
        assessment["required"] = require_ni
        assessment["policy_source"] = quality_safety_cfg["max_quality_regression_source"]
        m["quality_safety"] = assessment
        if assessment["established"] or not require_ni:
            safe.append(m)
        else:
            promising.append(m)
    ctx["safe"] = safe
    ctx["promising"] = promising

    # ── Stage 3: rank on the objective's own metric.
    #
    # The pool is the SAFE candidates. Only when none are safe do we rank the
    # promising ones instead — not to promote them, but because the remaining
    # questions ("was any improvement even measurable?", "was it material?") are
    # answered identically either way, and answering them wrongly would let a
    # candidate whose cost was never measured come back as "promising" when the
    # truth is that we measured nothing. The `safe` list is what decides whether
    # a material improvement is reported as SAFE or as PROMISING, and that
    # decision is made at the single branch below.
    quality_safety_verified = bool(safe)
    winner = _best_by_objective(safe or promising, baseline, objective)
    if winner is not None:
        qs = winner.get("quality_safety") or {}
        ctx["quality_safety"] = qs
        if qs.get("established"):
            reasons.append(domain.reason(
                "quality_non_inferiority_established",
                candidate=winner["candidate"].title,
                method=qs.get("method"),
                observed_regression=qs.get("observed_regression"),
                allowed_regression=qs.get("allowed_regression"),
                sample_size=qs.get("n_pairs"),
                discordant_b=qs.get("discordant_b"),
                discordant_c=qs.get("discordant_c"),
                confidence_level=qs.get("confidence_level"),
                lower_confidence_bound=qs.get("lower_confidence_bound"),
            ))
        elif not require_ni:
            # The customer switched the evidence gate off. Record that this
            # verdict rests on a point estimate by their choice, so nothing
            # downstream can read it as an established safety claim.
            reasons.append(domain.reason(
                "non_inferiority_not_established",
                candidate=winner["candidate"].title,
                detail_code="non_inferiority_check_disabled_by_policy",
                detail=(
                    "The policy does not require non-inferiority evidence, so "
                    "this verdict rests on point estimates alone."
                ),
            ))

    if winner is None:
        return {
            "conclusion": domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            "reasons": reasons + [domain.reason(
                "coverage_gap",
                detail=f"The metric for objective '{objective}' was not measured in any arm.",
            )],
            "evidence_maturity": None,
            "more_data_changes_conclusion": domain.MORE_DATA_YES,
            "more_data_reasons": more_data_reasons,
            "materiality_applied": materiality,
            "winner": None,
            "selected_result_id": None,
        }

    improvement = _improvement(winner["metrics"], baseline, objective, traffic)
    # Carried so a recommendation built from this verdict extrapolates from the
    # MEASURED per-call delta and the MEASURED traffic volume, rather than from
    # the generator's pre-benchmark price-sheet guess.
    winner["improvement"] = improvement
    material, materiality_detail = domain.evaluate_materiality(improvement, materiality)

    evidence_maturity = domain.compute_evidence_maturity(
        sample_size=sample_size,
        evidence_source="replay",
        quality_provenance=winner["metrics"].get("quality_provenance") or quality_provenance,
        variation=winner["metrics"].get("cost_variation"),
    )

    if material and not quality_safety_verified:
        # ── PROMISING, NOT SAFE.
        #
        # A real, material, policy-clean saving whose quality safety we cannot
        # yet vouch for. This is the state the original failure had no name for,
        # so it was rounded UP to `safe_improvement_found` and shipped as a
        # verified recommendation.
        #
        # It returns NO winner. That is load-bearing rather than tidy:
        # `_conclude` creates a recommendation only for
        # `safe_improvement_found` WITH a winner, and `_recommendation_fields`
        # writes `verified_savings_usd` only under the same condition. Leaving
        # the winner off means there is no code path — present or future — by
        # which a promising candidate acquires a verified saving or a `verified`
        # status. The candidate is still fully described, by result id, in
        # `leading_candidate_result_id` and in the frontier.
        qs = winner.get("quality_safety") or {}
        reasons.append(domain.reason(
            "non_inferiority_not_established",
            candidate=winner["candidate"].title,
            method=qs.get("method"),
            observed_regression=qs.get("observed_regression"),
            allowed_regression=qs.get("allowed_regression"),
            sample_size=qs.get("n_pairs"),
            discordant_b=qs.get("discordant_b"),
            discordant_c=qs.get("discordant_c"),
            confidence_level=qs.get("confidence_level"),
            lower_confidence_bound=qs.get("lower_confidence_bound"),
            detail_code=qs.get("reason_code"),
        ))
        extra = qs.get("additional_cases_required")
        if extra is not None:
            # DERIVED: the smallest sample at which the SAME test would pass,
            # holding the observed discordance rates constant. Not a guessed
            # round number, and explicitly conditional on the new cases behaving
            # like the measured ones.
            more_data_reasons.append(domain.reason(
                "sample_size_below_threshold",
                observed=qs.get("n_pairs"),
                required=qs.get("required_total_cases"),
                additional_cases_required=extra,
                unit="cases",
                dataset="golden_inputs",
                derived_from=qs.get("method"),
                candidate=winner["candidate"].title,
            ))
        else:
            more_data_reasons.append(domain.reason(
                "non_inferiority_not_established",
                candidate=winner["candidate"].title,
                detail_code=qs.get("additional_cases_reason"),
                detail=(
                    "The number of additional cases needed could not be derived "
                    "from the observed data."
                ),
            ))
        return {
            "conclusion": domain.CONCLUSION_PROMISING_UNVERIFIED,
            "reasons": reasons,
            "evidence_maturity": evidence_maturity,
            "more_data_changes_conclusion": (
                domain.MORE_DATA_YES if extra is not None else domain.MORE_DATA_UNKNOWN
            ),
            "more_data_reasons": more_data_reasons,
            "materiality_applied": {**materiality, "evaluation": materiality_detail},
            "winner": None,
            "selected_result_id": None,
            "leading_candidate_result_id": winner.get("result_id"),
            "quality_safety": qs,
        }

    if material:
        return {
            "conclusion": domain.CONCLUSION_SAFE_IMPROVEMENT,
            "reasons": reasons,
            "evidence_maturity": evidence_maturity,
            "more_data_changes_conclusion": (
                domain.MORE_DATA_YES if (evidence_maturity or 0) < 0.34 else domain.MORE_DATA_NO
            ),
            "more_data_reasons": more_data_reasons + ([domain.reason(
                "sample_size_below_threshold", observed=sample_size, required=sample_size * 4,
                detail=(
                    "The improvement clears the materiality threshold but confidence "
                    "is low; more cases would firm up the estimate."
                ),
            )] if (evidence_maturity or 0) < 0.34 else []),
            "materiality_applied": {**materiality, "evaluation": materiality_detail},
            "winner": winner,
            "selected_result_id": winner.get("result_id"),
        }

    thresholds_evaluated = materiality_detail.get("thresholds") or []
    unmeasurable_all = bool(thresholds_evaluated) and all(
        t.get("unmeasurable") for t in thresholds_evaluated
    )
    if unmeasurable_all or materiality_detail.get("reason") == "no_thresholds_declared":
        # Not one materiality threshold could be evaluated. "Not worth changing"
        # is a KNOWLEDGE claim and this is not knowledge — it is the absence of
        # a comparison. Reporting it as no_material_improvement would render
        # ignorance as "your current setup is fine", which is the single thing
        # this engine exists to refuse.
        return {
            "conclusion": domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            "reasons": reasons + [domain.reason(
                "cost_not_measured" if objective in ("cost", "balanced") else "coverage_gap",
                objective=objective,
                detail=(
                    "No materiality threshold could be evaluated because the "
                    "underlying measurement was missing on at least one arm."
                ),
                thresholds=[t.get("metric") for t in thresholds_evaluated] or None,
            )],
            "evidence_maturity": evidence_maturity,
            "more_data_changes_conclusion": domain.MORE_DATA_YES,
            "more_data_reasons": more_data_reasons,
            "materiality_applied": {**materiality, "evaluation": materiality_detail},
            "winner": None,
            "selected_result_id": None,
        }

    # Eligible, measured, but not worth changing. A KNOWLEDGE state.
    for t in materiality_detail.get("thresholds", []):
        if not t.get("met"):
            reasons.append(domain.reason(
                "improvement_below_materiality",
                metric=t.get("metric"), comparator=t.get("comparator"),
                observed=t.get("observed"), required=t.get("value"), unit=t.get("unit"),
                candidate=winner["candidate"].title,
            ))

    unmeasurable = materiality_detail.get("unmeasurable_count") or 0
    return {
        "conclusion": domain.CONCLUSION_NO_MATERIAL_IMPROVEMENT,
        "reasons": reasons,
        "evidence_maturity": evidence_maturity,
        "more_data_changes_conclusion": (
            domain.MORE_DATA_YES if (unmeasurable or (evidence_maturity or 0) < 0.34)
            else domain.MORE_DATA_NO
        ),
        "more_data_reasons": more_data_reasons + ([domain.reason(
            "coverage_gap",
            detail=(
                f"{unmeasurable} materiality threshold(s) could not be evaluated "
                "because the underlying measurement was missing."
            ),
        )] if unmeasurable else []),
        "materiality_applied": {**materiality, "evaluation": materiality_detail},
        "winner": winner,
        "selected_result_id": winner.get("result_id"),
    }


def _violation_reason(v: dict, candidate_title: str) -> dict:
    code_by_constraint = {
        "min_quality": "quality_below_threshold",
        "max_quality_regression": "quality_regression_above_threshold",
        "max_error_rate": "error_rate_above_threshold",
        "max_latency_p95_ms": "latency_above_threshold",
        "max_cost_per_task_usd": "cost_above_threshold",
        "allowed_vendors": "provider_not_permitted",
        "blocked_vendors": "provider_not_permitted",
    }
    unit_by_constraint = {
        "min_quality": "score",
        "max_quality_regression": "score",
        "max_error_rate": "ratio",
        "max_latency_p95_ms": "ms",
        "max_cost_per_task_usd": "usd_per_task",
    }
    return domain.reason(
        code_by_constraint.get(v["constraint"], "coverage_gap"),
        constraint=v["constraint"],
        observed=v.get("measured"),
        required=v.get("required"),
        shortfall=v.get("shortfall"),
        unit=unit_by_constraint.get(v["constraint"]),
        candidate=candidate_title,
        # Present only on the relative quality constraint, and load-bearing
        # there: "0.90" means nothing without "against a baseline of 1.00".
        baseline_quality=v.get("baseline_quality"),
        candidate_quality=v.get("candidate_quality"),
        threshold_source=v.get("source"),
    )


def _frontier_entry(m: dict, baseline: dict, *, eligible_status: str) -> dict:
    """One measured arm as a frontier row: FACTS and CODES, never wording."""
    metrics = m["metrics"]
    evaluation = m.get("policy_evaluation") or {}
    qs = m.get("quality_safety") or {}
    staged = m.get("staged_evaluation") or {}
    # Deltas are against the baseline over the cases THIS arm ran, so a
    # stopped candidate's saving and quality delta are like-for-like.
    arm_baseline = m.get("paired_baseline") or baseline
    b_cost = arm_baseline.get("mean_cost_usd")
    c_cost = metrics.get("mean_cost_usd")
    b_q, c_q = arm_baseline.get("quality"), metrics.get("quality")

    codes: list[str] = []
    if staged.get("stopped_early"):
        codes.append("candidate_evaluation_stopped_early")
    for v in evaluation.get("violated") or []:
        codes.append(_violation_reason(v, m["candidate"].title)["code"])
    for u in evaluation.get("unmeasured") or []:
        codes.append(_UNMEASURED_REASON_CODE.get(u["constraint"], "coverage_gap"))
    if eligible_status == "quality_safe" and qs.get("established"):
        codes.append("quality_non_inferiority_established")
    if eligible_status == "promising":
        codes.append(qs.get("reason_code") or "non_inferiority_not_established")

    return {
        "label": m["candidate"].title,
        "result_id": m.get("result_id"),
        "tier": domain.TIER_EXECUTABLE,
        "evidence_source": "replay",
        "status": eligible_status,
        "reason_codes": codes,
        "mean_cost_usd": c_cost,
        "cost_delta_ratio": (
            round((float(b_cost) - float(c_cost)) / float(b_cost), 6)
            if b_cost and c_cost is not None else None
        ),
        "quality": c_q,
        "quality_delta": (
            round(float(c_q) - float(b_q), 6)
            if b_q is not None and c_q is not None else None
        ),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "error_rate": metrics.get("error_rate"),
        "cases_evaluated": metrics.get("n"),
        "staged_evaluation": ({
            "stopped_early": bool(staged.get("stopped_early")),
            "stopped_at_stage": staged.get("stopped_at_stage"),
            "stages_planned": staged.get("stages_planned"),
            "cases_run": staged.get("cases_run"),
            "cases_planned": staged.get("cases_planned"),
            "cases_not_run": staged.get("cases_not_run"),
            "stop_reason_code": staged.get("stop_reason_code"),
            "bound_method": (staged.get("bound") or {}).get("bound_method"),
        } if staged else None),
        "quality_safety": {
            k: qs.get(k) for k in (
                "established", "reason_code", "n_pairs", "discordant_b",
                "discordant_c", "observed_regression", "allowed_regression",
                "confidence_level", "lower_confidence_bound",
                "additional_cases_required", "required_total_cases",
            )
        } if qs else None,
    }


def _build_frontier(
    *,
    baseline: dict,
    measured: list[dict],
    safe: list[dict],
    promising: list[dict],
    opportunities: list[dict],
    winner: Optional[dict],
) -> dict:
    """
    The whole consideration set, with the reason each option was or was not
    eligible — not just the one that won.

    A customer shown only "we recommend X" cannot tell whether OptiML looked at
    anything else, and cannot make the trade-off themselves. So the frontier
    names, explicitly:

      largest_observed_savings   the cheapest arm we MEASURED, whether or not it
                                 is eligible. Often this is the rejected one,
                                 and saying so out loud is the honest move.
      quality_preserving         the cheapest measured arm that did not regress
                                 against the baseline at all.
      lowest_cost_rejected       the cheapest arm that is NOT adoptable, with
                                 the codes explaining why.
      selected                   what actually won, or None.
      unverified_opportunities   TIER 2. Never measured, never adoptable from
                                 here; their next action is connecting a
                                 provider.

    Every figure is measured or None. No prose.
    """
    safe_ids = {id(m) for m in safe}
    promising_ids = {id(m) for m in promising}

    entries = []
    for m in measured:
        if id(m) in safe_ids:
            status = domain.DISPOSITION_QUALITY_SAFE
        elif id(m) in promising_ids:
            status = domain.DISPOSITION_PROMISING
        elif (m.get("policy_evaluation") or {}).get("violated"):
            status = domain.DISPOSITION_FAILED_POLICY
        elif m.get("policy_evaluation"):
            status = domain.DISPOSITION_NOT_MEASURED
        else:
            status = domain.DISPOSITION_BENCHMARKED
        entries.append(_frontier_entry(m, baseline, eligible_status=status))

    def _cheapest(rows: list[dict]) -> Optional[dict]:
        priced = [r for r in rows if r.get("mean_cost_usd") is not None]
        return min(priced, key=lambda r: r["mean_cost_usd"]) if priced else None

    b_q = baseline.get("quality")
    quality_preserving = _cheapest([
        r for r in entries
        if r.get("quality") is not None and b_q is not None and r["quality"] >= b_q
    ]) if b_q is not None else None

    rejected = [r for r in entries if r["status"] != domain.DISPOSITION_QUALITY_SAFE]

    return {
        "baseline": {
            "mean_cost_usd": baseline.get("mean_cost_usd"),
            "quality": baseline.get("quality"),
            "latency_p95_ms": baseline.get("latency_p95_ms"),
            "error_rate": baseline.get("error_rate"),
            "quality_provenance": baseline.get("quality_provenance"),
        },
        "largest_observed_savings": _cheapest(entries),
        "quality_preserving": quality_preserving,
        "quality_preserving_absent_reason": (
            None if quality_preserving
            else ("baseline_quality_not_measured" if b_q is None
                  else "no_candidate_matched_baseline_quality")
        ),
        "lowest_cost_rejected": _cheapest(rejected),
        "selected": next(
            (r for r in entries if winner and r.get("result_id") == winner.get("result_id")),
            None,
        ),
        "entries": entries,
        "unverified_opportunities": list(opportunities or []),
    }


def _objective_improved(
    m: dict, baseline: dict, objective: str
) -> Optional[bool]:
    """
    Did this arm beat the MEASURED baseline on the objective's own metric?

    None when the metric was not measured on both sides — never False, because
    "not measured" is not "did not improve". Deltas are taken against the
    baseline over the cases THIS arm ran, matching _frontier_entry.
    """
    metrics = m.get("metrics") or {}
    arm_baseline = m.get("paired_baseline") or baseline or {}
    if objective in ("cost", "balanced"):
        key, lower_is_better = "mean_cost_usd", True
    elif objective == "latency":
        key, lower_is_better = "latency_p95_ms", True
    elif objective == "quality":
        key, lower_is_better = "quality", False
    else:
        # 'custom': no built-in ranking (see _best_by_objective). Refusing to
        # rank means refusing to claim an improvement.
        return None
    b, c = arm_baseline.get(key), metrics.get(key)
    if b is None or c is None:
        return None
    return float(c) < float(b) if lower_is_better else float(c) > float(b)


def _dispositions(
    *,
    measured: list[dict],
    safe: list[dict],
    promising: list[dict],
    opportunities: list[dict],
    generation: dict,
    baseline: Optional[dict] = None,
    objective: str = "cost",
) -> list[dict]:
    """
    One disposition per candidate that entered consideration, including the ones
    that never reached a benchmark.

    Candidate discovery is a funnel and the funnel counts ARE the product. Each
    record carries BOTH the disjoint disposition (where this candidate stopped —
    unchanged, the UI reads it) and the facts domain.stages_reached needs to
    count how far it GOT: `entered_replay`, `stopped_early`,
    `objective_improved`. Counting only the stop point is what let a run report
    `benchmarked: 0` while three arms had executed.
    """
    out: list[dict] = []
    safe_ids = {id(m) for m in safe}
    promising_ids = {id(m) for m in promising}

    # Where each exclusion code exits the funnel. The preflight codes come from
    # optimization.eligibility, which owns that mapping so the two cannot drift.
    drop_stage = {
        **eligibility_mod.CODE_TO_DISPOSITION,
        "strategy_not_applicable": domain.DISPOSITION_INCOMPATIBLE,
        "provider_not_permitted": domain.DISPOSITION_POLICY_BLOCKED,
        "provider_not_configured": domain.DISPOSITION_PROVIDER_NOT_CONFIGURED,
        "duplicate_strategy": domain.DISPOSITION_DUPLICATE,
        "generator_error": domain.DISPOSITION_GENERATOR_ERROR,
    }
    for d in (generation.get("dropped") or []):
        code = d.get("code")
        if code == "provider_not_configured":
            # Handled below as a TIER 2 opportunity, not as a drop.
            continue
        out.append({
            "label": d.get("title"),
            "generator": d.get("generator"),
            "tier": domain.TIER_EXECUTABLE,
            "disposition": drop_stage.get(code, domain.DISPOSITION_INCOMPATIBLE),
            "code": code,
            # Excluded before dispatch: no provider request was made for it.
            "entered_replay": False,
            "stopped_early": None,
            "objective_improved": None,
            "facts": {k: v for k, v in d.items() if k not in ("title", "generator", "code")},
        })

    for opp in (opportunities or []):
        out.append({
            "label": opp.get("label"),
            "generator": opp.get("generator"),
            "tier": domain.TIER_OPPORTUNITY,
            "disposition": domain.DISPOSITION_PROVIDER_NOT_CONFIGURED,
            "code": "provider_not_configured",
            "entered_replay": False,
            "stopped_early": None,
            "objective_improved": None,
            "facts": {"providers": opp.get("providers")},
        })

    for m in measured:
        if id(m) in safe_ids:
            stage, code = domain.DISPOSITION_QUALITY_SAFE, "quality_non_inferiority_established"
        elif id(m) in promising_ids:
            stage = domain.DISPOSITION_PROMISING
            code = (m.get("quality_safety") or {}).get("reason_code")
        elif (m.get("policy_evaluation") or {}).get("violated"):
            stage, code = domain.DISPOSITION_FAILED_POLICY, None
        elif (m.get("policy_evaluation") or {}).get("unmeasured"):
            stage, code = domain.DISPOSITION_NOT_MEASURED, "coverage_gap"
        else:
            stage, code = domain.DISPOSITION_BENCHMARKED, None
        staged = m.get("staged_evaluation") or {}
        out.append({
            "label": m["candidate"].title,
            "generator": getattr(m["candidate"], "generator", None),
            "tier": domain.TIER_EXECUTABLE,
            "disposition": stage,
            "code": code,
            # This arm was dispatched and produced a measurement — that is what
            # "entered replay" means, and it is true regardless of which
            # terminal bucket the arm ended in.
            "entered_replay": True,
            "stopped_early": bool(staged.get("stopped_early")),
            "objective_improved": _objective_improved(m, baseline or {}, objective),
            "result_id": m.get("result_id"),
            # The preflight record for an arm that DID run. Present so the
            # funnel answers "why was this one allowed to spend money?" with the
            # same structure it answers "why was that one refused?".
            "eligibility": getattr(m["candidate"], "eligibility", None),
        })

    return out


def _best_by_objective(eligible: list[dict], baseline: dict, objective: str) -> Optional[dict]:
    """Pick the winning eligible candidate on the objective's own metric."""
    def key_cost(m):
        return m["metrics"].get("mean_cost_usd")

    def key_latency(m):
        return m["metrics"].get("latency_p95_ms")

    def key_quality(m):
        return m["metrics"].get("quality")

    if objective in ("cost", "balanced"):
        scored = [m for m in eligible if key_cost(m) is not None]
        return min(scored, key=key_cost) if scored else None
    if objective == "latency":
        scored = [m for m in eligible if key_latency(m) is not None]
        return min(scored, key=key_latency) if scored else None
    if objective == "quality":
        scored = [m for m in eligible if key_quality(m) is not None]
        return max(scored, key=key_quality) if scored else None
    # 'custom': no built-in ranking. Refuse rather than silently minimising cost.
    return None


def _improvement(
    metrics: dict, baseline: dict, objective: str, traffic: Optional[dict]
) -> dict:
    """
    Improvement per materiality metric, as POSITIVE numbers in the improving
    direction. Missing measurements stay None.
    """
    out: dict[str, dict] = {}

    b_cost, c_cost = baseline.get("mean_cost_usd"), metrics.get("mean_cost_usd")
    if b_cost is not None and c_cost is not None and b_cost > 0:
        per_call = float(b_cost) - float(c_cost)
        monthly = None
        if traffic and traffic.get("run_count") and traffic.get("window_days"):
            per_day = traffic["run_count"] / float(max(1, traffic["window_days"]))
            monthly = round(per_call * per_day * 30.0, 6)
        out["cost"] = {
            "relative": round(per_call / float(b_cost), 6),
            "absolute": monthly,
            "unit": "usd_per_month",
            "per_task_absolute": round(per_call, 10),
        }

    b_lat, c_lat = baseline.get("latency_p95_ms"), metrics.get("latency_p95_ms")
    if b_lat is not None and c_lat is not None and b_lat > 0:
        out["latency_p95_ms"] = {
            "relative": round((b_lat - c_lat) / float(b_lat), 6),
            "absolute": float(b_lat - c_lat),
            "unit": "ms",
        }

    b_q, c_q = baseline.get("quality"), metrics.get("quality")
    if b_q is not None and c_q is not None:
        out["quality"] = {
            "relative": (round((c_q - b_q) / b_q, 6) if b_q else None),
            "absolute": round(float(c_q) - float(b_q), 6),
            "unit": "score",
        }
        out["outcome_rate"] = dict(out["quality"], unit="percentage_points")

    b_e, c_e = baseline.get("error_rate"), metrics.get("error_rate")
    if b_e is not None and c_e is not None:
        out["error_rate"] = {
            "relative": (round((b_e - c_e) / b_e, 6) if b_e else None),
            "absolute": round(float(b_e) - float(c_e), 6),
            "unit": "ratio",
        }

    return out


# ---------------------------------------------------------------------------
# Re-evaluation over RETAINED evidence
# ---------------------------------------------------------------------------

def reevaluate(org_id: str, benchmark_id: str, *, objective: Optional[str] = None) -> Optional[dict]:
    """
    Re-derive a conclusion from stored candidate results under the CURRENT
    policy version. Runs no model calls.

    Writes a NEW benchmark_conclusions row and flips is_current on the previous
    one. The previous verdict is retained unchanged: it was correct under the
    policy in force at the time. Both coexist, each bound to its policy version.
    """
    bench = get_benchmark(org_id, benchmark_id)
    if bench is None:
        return None

    workload = workloads_mod.get_workload(org_id, str(bench["workload_id"]))
    if workload is None:
        return None

    results = list_candidate_results(org_id, benchmark_id=benchmark_id)
    baseline_row = next((r for r in results if r.get("arm") == "baseline"), None)
    if baseline_row is None:
        return None

    objective = objective or bench.get("objective") or domain.DEFAULT_OBJECTIVE
    policy = policies.get_effective_policy(org_id, str(workload["id"]))
    materiality = policies.materiality_of(policy, objective)

    baseline_metrics = _metrics_from_result_row(baseline_row)
    # A re-read must reproduce the SAME pairing the run used. A candidate whose
    # evaluation was stopped early stored fewer per-case rows than the baseline,
    # so it is re-paired against the baseline over that same prefix — otherwise
    # a reevaluate would silently compare it against cases it never ran and
    # could reach a verdict the original evidence never supported.
    measured = []
    for r in results:
        if r.get("arm") != "candidate" or r.get("error"):
            continue
        metrics = _metrics_from_result_row(r)
        measured.append({
            "candidate": _StoredCandidate(r),
            "metrics": metrics,
            "result_id": str(r["id"]),
            "paired_baseline": _arm_over_prefix(
                baseline_metrics, len(metrics.get("per_case") or [])
            ),
            "staged_evaluation": (r.get("outcome_metrics") or {}).get("staged_evaluation"),
        })

    observed_outcomes = outcomes_mod.list_outcomes(
        org_id, workload_id=str(workload["id"]), limit=500
    )
    signal = domain.resolve_success_signal(policies.success_signal_of(policy), observed_outcomes)

    verdict = evaluate_conclusion(
        baseline=baseline_metrics,
        measured=measured,
        policy=policy,
        materiality=materiality,
        objective=objective,
        signal=signal,
        sample_size=int(bench.get("sample_size") or 0),
        traffic=_traffic_for(org_id, workload),
        # Tier-2 opportunities are a property of candidate GENERATION, which a
        # re-read does not repeat. Carrying them forward from the previous
        # conclusion would be re-asserting a fact this pass did not establish.
        opportunities=[],
        generation={},
    )

    if verdict.get("winner"):
        verdict["winner"]["baseline_mean_cost_usd"] = baseline_metrics.get("mean_cost_usd")

    return _write_conclusion(
        org_id,
        benchmark_id=benchmark_id,
        workload_id=str(workload["id"]),
        policy=policy,
        objective=objective,
        conclusion=verdict["conclusion"],
        reasons=verdict["reasons"],
        evidence_maturity=verdict["evidence_maturity"],
        materiality=verdict["materiality_applied"],
        signal=signal,
        more_data=verdict["more_data_changes_conclusion"],
        more_data_reasons=verdict["more_data_reasons"],
        selected_result_id=(
            verdict.get("selected_result_id")
            or verdict.get("leading_candidate_result_id")
        ),
        quality_safety=verdict.get("quality_safety"),
        quality_safety_policy=verdict.get("quality_safety_policy"),
        frontier=verdict.get("frontier"),
        consideration=verdict.get("consideration"),
    )


class _StoredCandidate:
    """Adapter so a stored result row satisfies the shape evaluate_conclusion expects."""

    def __init__(self, row: dict):
        self.title = row.get("label") or "candidate"
        self.generator = row.get("generator")
        self.dimensions = row.get("dimensions") or []
        self._executor_refs = row.get("executor_refs") or []

    @property
    def executor_refs(self) -> list[dict]:
        return self._executor_refs


def _executor_refs_of(candidate) -> list[dict]:
    refs = getattr(candidate, "executor_refs", None)
    if refs is not None:
        return list(refs)
    strat = getattr(candidate, "strategy", None)
    if strat is None:
        return []
    return [s.executor_ref for s in strat.steps if s.executor_ref]


def _metrics_from_result_row(row: dict) -> dict:
    return {
        "label": row.get("label"),
        "n": row.get("sample_size"),
        "mean_cost_usd": (float(row["mean_cost_usd"]) if row.get("mean_cost_usd") is not None else None),
        "total_cost_usd": (float(row["total_cost_usd"]) if row.get("total_cost_usd") is not None else None),
        "latency_p50_ms": row.get("latency_p50_ms"),
        "latency_p95_ms": row.get("latency_p95_ms"),
        "error_rate": (float(row["error_rate"]) if row.get("error_rate") is not None else None),
        "quality": (float(row["quality"]) if row.get("quality") is not None else None),
        "quality_provenance": row.get("quality_provenance") or "unknown",
        "quality_checks_run": (row.get("outcome_metrics") or {}).get("quality_checks_run", 0),
        "cost_variation": (row.get("outcome_metrics") or {}).get("cost_variation"),
        # Provenance is part of the evidence, not part of the verdict, so a
        # re-read reproduces exactly what the original run could conclude.
        "cost_basis": (
            (row.get("outcome_metrics") or {}).get("cost_basis")
            or executors_mod.COST_BASIS_MEASURED
        ),
        "mean_cost_estimated_usd": (row.get("outcome_metrics") or {}).get("mean_cost_estimated_usd"),
        "total_cost_estimated_usd": (row.get("outcome_metrics") or {}).get("total_cost_estimated_usd"),
        "pricing_provenance": (row.get("outcome_metrics") or {}).get("pricing_provenance"),
        "per_case": row.get("per_case_results") or [],
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _insert_benchmark(
    org_id, *, workload_id, method, objective, policy, materiality, extra=None
):
    """
    Create the benchmark row.

    `extra` carries the async-job columns (status/progress_state/heartbeat_at/
    idempotency_key/...) when this row is being created as a JOB. It overlays
    the defaults, so a job row starts life 'queued' rather than 'pending' while
    a synchronous run keeps the 'pending' it has always had. Exceptions
    propagate when `extra` is present: a job creation that hits the partial
    unique index on the idempotency key is a DUPLICATE, and swallowing it here
    would turn "someone else is already running this" into "the record could
    not be created", which is the opposite instruction to the caller.
    """
    row = {
        "org_id": org_id,
        "workload_id": workload_id,
        "method": method,
        "status": "pending",
        "objective": objective,
        "materiality_threshold": materiality,
        "policy_id": (str(policy["id"]) if policy and policy.get("id") else None),
        "policy_evaluation": {},
    }
    if extra:
        row.update(extra)
    try:
        result = supabase.table("optimization_benchmarks").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        if extra:
            raise
        logger.warning("_insert_benchmark failed: %s", type(exc).__name__)
        return None


def _update_benchmark(org_id: str, benchmark_id: str, patch: dict) -> None:
    try:
        supabase.table("optimization_benchmarks").update(patch).eq(
            "id", benchmark_id
        ).eq("org_id", org_id).execute()
    except Exception as exc:  # pragma: no cover
        logger.warning("_update_benchmark failed: %s", type(exc).__name__)


def _public_metrics(metrics: Optional[dict]) -> Optional[dict]:
    if metrics is None:
        return None
    return {k: v for k, v in metrics.items() if k != "per_case"}


def _write_candidate_result(
    org_id, benchmark_id, workload_id, *, arm, label, strategy, metrics, baseline,
    generator, dimensions, error: Optional[str] = None,
    staged_evaluation: Optional[dict] = None,
) -> Optional[dict]:
    """
    Persist one measured arm INDEPENDENTLY of any conclusion.

    This is what lets a near-miss survive: a candidate that saved 51% but landed
    0.7pp under the quality floor is a row here even though the run concluded
    'candidates_failed_policy'.

    `staged_evaluation` is the arm's per-stage evidence trail: which stages ran,
    what was known at the end of each, and — for a candidate stopped early — the
    bound that justified stopping, with every count it was derived from. It goes
    in `outcome_metrics` alongside the other measured facts rather than in a new
    table, because it IS a measurement of this arm, not a verdict about it.
    `baseline` here is the baseline restricted to the cases this arm actually
    ran, so every delta on the row is a like-for-like comparison.
    """
    row: dict[str, Any] = {
        "org_id": org_id,
        "benchmark_id": benchmark_id,
        "workload_id": workload_id,
        "arm": arm,
        "label": label,
        "generator": generator,
        "dimensions": dimensions or [],
        "strategy_fingerprint": strategy.fingerprint() if strategy else None,
        "executor_refs": [s.executor_ref for s in strategy.steps] if strategy else [],
        "evidence_source": "replay",
        "error": error,
    }

    if metrics is not None:
        row.update({
            "sample_size": metrics.get("n"),
            "mean_cost_usd": metrics.get("mean_cost_usd"),
            "total_cost_usd": metrics.get("total_cost_usd"),
            "latency_p50_ms": metrics.get("latency_p50_ms"),
            "latency_p95_ms": metrics.get("latency_p95_ms"),
            "error_rate": metrics.get("error_rate"),
            "quality": metrics.get("quality"),
            "quality_provenance": metrics.get("quality_provenance") or "unknown",
            "outcome_metrics": {
                "quality_checks_run": metrics.get("quality_checks_run"),
                "cost_variation": metrics.get("cost_variation"),
                "cases_measured": metrics.get("cases_measured"),
                # Cost provenance travels WITH the arm, so a later reevaluate
                # re-derives the same verdict from the row alone and can never
                # promote a guessed price into a measured one.
                "cost_basis": metrics.get("cost_basis"),
                "mean_cost_estimated_usd": metrics.get("mean_cost_estimated_usd"),
                "total_cost_estimated_usd": metrics.get("total_cost_estimated_usd"),
                "pricing_provenance": metrics.get("pricing_provenance"),
            },
            "per_case_results": metrics.get("per_case"),
        })
        if staged_evaluation is not None:
            row["outcome_metrics"]["staged_evaluation"] = staged_evaluation

        if baseline:
            # Paired discordant counts are a MEASUREMENT, not an interpretation:
            # they depend only on the two arms' per-case verdicts, never on a
            # policy. They live on the evidence row so a later re-read can
            # re-derive a verdict without re-walking every per-case blob. The
            # non-inferiority VERDICT does not live here — it is policy-versioned
            # and belongs on benchmark_conclusions.
            row["outcome_metrics"]["paired_vs_baseline"] = noninferiority.paired_counts(
                baseline.get("per_case"), metrics.get("per_case")
            )
            b_cost, c_cost = baseline.get("mean_cost_usd"), metrics.get("mean_cost_usd")
            if b_cost and c_cost is not None:
                row["cost_delta_pct"] = round((c_cost - b_cost) / b_cost * 100, 4)
            b_lat, c_lat = baseline.get("latency_p95_ms"), metrics.get("latency_p95_ms")
            if b_lat and c_lat is not None:
                row["latency_delta_pct"] = round((c_lat - b_lat) / b_lat * 100, 4)
            b_q, c_q = baseline.get("quality"), metrics.get("quality")
            if b_q is not None and c_q is not None:
                row["quality_delta"] = round(c_q - b_q, 6)

    try:
        result = supabase.table("benchmark_candidate_results").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("_write_candidate_result failed: %s", type(exc).__name__)
        return None


def _write_conclusion(
    org_id, *, benchmark_id, workload_id, policy, objective, conclusion, reasons,
    evidence_maturity, materiality, signal, more_data, more_data_reasons,
    selected_result_id: Optional[str] = None,
    quality_safety: Optional[dict] = None,
    quality_safety_policy: Optional[dict] = None,
    frontier: Optional[dict] = None,
    consideration: Optional[dict] = None,
) -> dict:
    """
    Insert an IMMUTABLE conclusion row and retire the previous current one.

    Conclusion rows are never updated. Re-evaluating the same evidence under a
    relaxed policy inserts a new row; the original verdict and the policy
    version that produced it stay readable forever.
    """
    row = {
        "org_id": org_id,
        "benchmark_id": benchmark_id,
        "workload_id": workload_id,
        "policy_id": (str(policy["id"]) if policy and policy.get("id") else None),
        "policy_key": (str(policy["policy_key"]) if policy and policy.get("policy_key") else None),
        "policy_version": (policy or {}).get("version"),
        "objective": objective,
        "conclusion": conclusion,
        "reasons": reasons,
        # STORAGE KEY, not a name for what this is. The physical columns stay
        # `confidence`/`confidence_band` so the preserved historical rows are
        # not orphaned and no data is rewritten; the value and the formula are
        # unchanged, so an old row still means what it meant. What it MEANS is
        # evidence maturity, and it is internal — see
        # migration_optimization_v10_evidence_maturity_semantics.sql and
        # optimization.domain.compute_evidence_maturity.
        "confidence": evidence_maturity,
        "confidence_band": domain.evidence_maturity_band(evidence_maturity),
        "materiality_applied": materiality,
        "success_signal": signal.to_dict() if signal else {},
        "more_data_changes_conclusion": more_data,
        "more_data_reasons": more_data_reasons,
        "selected_candidate_result_id": selected_result_id,
        # STRUCTURED, EXPLAINABLE quality evidence — deliberately NOT folded
        # into the generic evidence-maturity index. That index answers "how
        # mature is the evidence behind this measurement"; this answers "can we
        # rule out a material quality regression", which is a real statistical
        # statement. The live failure that motivated the split shipped a -10pp
        # regression with maturity 0.171 attached, and no field anywhere said
        # the regression had never been ruled out. The two are never merged and
        # only this one is customer-facing.
        "quality_safety": quality_safety,
        "quality_safety_policy": quality_safety_policy,
        "frontier": frontier,
        "consideration": consideration,
        "is_current": True,
    }

    try:
        supabase.table("benchmark_conclusions").update({"is_current": False}).eq(
            "benchmark_id", benchmark_id
        ).eq("org_id", org_id).eq("is_current", True).execute()
    except Exception:  # pragma: no cover
        pass

    try:
        result = supabase.table("benchmark_conclusions").insert(row).execute()
        return (result.data or [row])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("_write_conclusion failed: %s", type(exc).__name__)
        return row


def _conclude(
    org_id, *, benchmark_id, workload, objective, policy, materiality, conclusion,
    reasons, evidence_maturity, sample_size, success_signal, more_data, more_data_reasons,
    status, error, recommendation_id, actor, service_mod,
    selected_result_id: Optional[str] = None, winner: Optional[dict] = None,
    create_recommendation: bool = False,
    baseline_strategy: Optional[strategy_mod.Strategy] = None,
    quality_safety: Optional[dict] = None,
    quality_safety_policy: Optional[dict] = None,
    frontier: Optional[dict] = None,
    consideration: Optional[dict] = None,
) -> dict:
    """Write the conclusion, mirror it onto the benchmark, and — only when the
    benchmark is cited by a recommendation — apply the lifecycle transition."""
    workload_id = str(workload["id"])

    conclusion_row = _write_conclusion(
        org_id, benchmark_id=benchmark_id, workload_id=workload_id, policy=policy,
        objective=objective, conclusion=conclusion, reasons=reasons,
        evidence_maturity=evidence_maturity, materiality=materiality, signal=success_signal,
        more_data=more_data, more_data_reasons=more_data_reasons,
        selected_result_id=selected_result_id,
        quality_safety=quality_safety, quality_safety_policy=quality_safety_policy,
        frontier=frontier, consideration=consideration,
    )

    _update_benchmark(org_id, benchmark_id, {
        "status": status,
        "conclusion": conclusion,
        "conclusion_detail": {"reasons": reasons},
        "more_data_changes_conclusion": more_data,
        "more_data_reason": (more_data_reasons[0]["code"] if more_data_reasons else None),
        "materiality_threshold": materiality,
        # Storage key; the value is the internal evidence-maturity index.
        "confidence": evidence_maturity,
        "sample_size": sample_size,
        "error": error,
        "completed_at": _iso(_utc_now()),
        "success_signal": success_signal.to_dict() if success_signal else {},
    })

    allocation.record_decision(
        org_id,
        workload_id=workload_id,
        decision_kind="benchmark",
        objective=objective,
        policy_id=(str(policy["id"]) if policy and policy.get("id") else None),
        recommendation_id=recommendation_id,
        considered=[
            {
                "label": r.get("label"),
                "fingerprint": r.get("strategy_fingerprint"),
                "arm": r.get("arm"),
                "mean_cost_usd": r.get("mean_cost_usd"),
                "quality": r.get("quality"),
                "latency_p95_ms": r.get("latency_p95_ms"),
            }
            for r in list_candidate_results(org_id, benchmark_id=benchmark_id)
        ],
        evidence_maturity=evidence_maturity,
        reason=f"benchmark conclusion: {conclusion}",
    )

    # A benchmark NEVER auto-creates a recommendation. It only advances one that
    # already cites it.
    if recommendation_id:
        _link_evidence(org_id, recommendation_id, benchmark_id)
        target = domain.CONCLUSION_TO_STATUS.get(conclusion)
        if target:
            try:
                service_mod.transition(
                    org_id, recommendation_id, target,
                    actor=actor, reason=f"benchmark:{conclusion}",
                    extra_fields=_recommendation_fields(
                        conclusion=conclusion, evidence_maturity=evidence_maturity,
                        sample_size=sample_size, winner=winner,
                        success_signal=success_signal,
                        quality_safety=quality_safety,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Benchmark %s could not transition recommendation %s: %s",
                    benchmark_id, recommendation_id, exc,
                )

    # A recommendation is created from evidence, and from exactly one kind of
    # evidence. Any other conclusion — including every ignorance state — ends
    # here with the facts persisted and nothing proposed.
    # A recommendation is created ONLY from `safe_improvement_found` WITH a
    # winner attached. `promising_candidate_unverified` deliberately carries no
    # winner, so this guard cannot fire for it even by accident: a promising
    # candidate is a finding to pursue, never a proposal to act on.
    created_recommendation_id: Optional[str] = None
    if (
        create_recommendation
        and recommendation_id is None
        and conclusion == domain.CONCLUSION_SAFE_IMPROVEMENT
        and winner is not None
    ):
        created_recommendation_id = _create_recommendation_from_evidence(
            org_id,
            benchmark_id=benchmark_id,
            workload=workload,
            objective=objective,
            winner=winner,
            baseline_strategy=baseline_strategy,
            evidence_maturity=evidence_maturity,
            sample_size=sample_size,
            success_signal=success_signal,
            conclusion=conclusion,
            actor=actor,
            service_mod=service_mod,
            quality_safety=quality_safety,
        )

    return {
        "benchmark_id": benchmark_id,
        "workload_id": workload_id,
        "status": status,
        "sample_size": sample_size,
        "objective": objective,
        "materiality_threshold": materiality,
        "success_signal": success_signal.to_dict() if success_signal else {},
        "more_data_changes_conclusion": more_data,
        "more_data_reasons": more_data_reasons,
        "conclusion_id": str(conclusion_row.get("id")) if conclusion_row.get("id") else None,
        "recommendation_id": recommendation_id or created_recommendation_id,
        "recommendation_created": created_recommendation_id is not None,
        "quality_safety": quality_safety,
        "quality_safety_policy": quality_safety_policy,
        "frontier": frontier,
        "consideration": consideration,
        **domain.conclusion_payload(conclusion, reasons=reasons),
    }


def _create_recommendation_from_evidence(
    org_id: str,
    *,
    benchmark_id: str,
    workload: dict,
    objective: str,
    winner: dict,
    baseline_strategy: Optional[strategy_mod.Strategy],
    evidence_maturity: Optional[float],
    sample_size: Optional[int],
    success_signal,
    conclusion: str,
    actor: Optional[str],
    service_mod,
    quality_safety: Optional[dict] = None,
) -> Optional[str]:
    """
    Turn a `safe_improvement_found` verdict into a recommendation that CITES it.

    Four things this function is careful about:

    1. It refuses unless it holds an executable candidate strategy AND the
       baseline strategy that was actually measured. The `reevaluate` path
       carries a stored-result adapter with no strategy object; reconstructing
       one from a fingerprint would be inventing the thing being recommended.
    2. `projected_savings_usd` is an EXTRAPOLATION — measured per-call delta x
       measured traffic — and goes only to the projected column. It is None,
       never 0.0, when traffic volume was not measured.
    3. `verified_savings_usd` is the delta measured over THIS benchmark's sample
       and is written by the lifecycle transition below via
       `_recommendation_fields`, which routes it through
       `domain.savings_column('verified')`. The two never cross.
    4. `baseline_reference` records which recommendation, if any, produced the
       baseline this improvement was measured against, so a chain is not counted
       twice by `domain.attributable_savings`.

    `approval_required` comes from the workload's policy and defaults TRUE:
    nothing here changes production.
    """
    cand = winner.get("candidate")
    cand_strategy = getattr(cand, "strategy", None)
    if cand_strategy is None or baseline_strategy is None:
        logger.info(
            "Benchmark %s concluded %s but carries no executable candidate "
            "strategy; no recommendation created.",
            benchmark_id, conclusion,
        )
        return None

    workload_id = str(workload["id"])
    improvement = winner.get("improvement") or {}
    cost_improvement = improvement.get("cost") or {}

    # Extrapolated from measurements only. None when volume was not measured.
    projected = cost_improvement.get("absolute")
    projection_basis = {
        "kind": "projection",
        "formula": "measured_per_task_delta_usd * observed_calls_per_day * 30",
        "measured_per_task_delta_usd": cost_improvement.get("per_task_absolute"),
        "unit": cost_improvement.get("unit"),
        "source": f"benchmark:{benchmark_id}",
        "result": "projected" if projected is not None else "not_projectable",
    }
    if projected is None:
        projection_basis["reason"] = (
            "No measured production volume for this workload in the window, so a "
            "monthly figure would be invented."
        )

    ancestor = service_mod.find_ancestor_by_strategy_fingerprint(
        org_id, workload_id, baseline_strategy.fingerprint()
    )
    baseline_reference = {
        "kind": "benchmark_baseline",
        "benchmark_id": benchmark_id,
        "strategy_fingerprint": baseline_strategy.fingerprint(),
        "mean_cost_usd": winner.get("baseline_mean_cost_usd"),
        "measured_over_cases": sample_size,
        "derived_from_recommendation_id": (str(ancestor["id"]) if ancestor else None),
        "projection": projection_basis,
    }

    rec = service_mod.create_recommendation(
        org_id,
        workload_id=workload_id,
        title=getattr(cand, "title", None) or "Candidate configuration",
        candidate_strategy=cand_strategy,
        baseline_strategy=baseline_strategy,
        dimensions=list(getattr(cand, "dimensions", None) or []),
        generator=getattr(cand, "generator", None),
        rationale=getattr(cand, "rationale", None),
        objective=objective,
        project_id=(str(workload["project_id"]) if workload.get("project_id") else None),
        evidence_benchmark_ids=[benchmark_id],
        projected_savings_usd=projected,
        baseline_reference=baseline_reference,
        parent_recommendation_id=(str(ancestor["id"]) if ancestor else None),
        actor=actor,
    )
    if not rec:
        return None

    rec_id = str(rec["id"])
    fields = _recommendation_fields(
        conclusion=conclusion, evidence_maturity=evidence_maturity, sample_size=sample_size,
        winner=winner, success_signal=success_signal,
        quality_safety=quality_safety,
    )

    # discovered -> benchmarking -> verified. The intermediate hop is not
    # ceremony: the lifecycle has no edge from `discovered` straight to
    # `verified`, precisely so nothing can reach a verified state without a
    # measurement having been taken.
    try:
        service_mod.transition(
            org_id, rec_id, domain.STATUS_BENCHMARKING,
            actor=actor, reason=f"benchmark:{benchmark_id}",
        )
        service_mod.transition(
            org_id, rec_id, domain.CONCLUSION_TO_STATUS[conclusion],
            actor=actor, reason=f"benchmark:{conclusion}", extra_fields=fields,
        )
    except Exception as exc:
        logger.warning(
            "Created recommendation %s from benchmark %s but could not advance "
            "it: %s", rec_id, benchmark_id, exc,
        )

    return rec_id


def _recommendation_fields(
    *, conclusion, evidence_maturity, sample_size, winner, success_signal,
    quality_safety: Optional[dict] = None,
) -> dict:
    """
    Measured fields to write onto a recommendation from this evidence.

    verified_savings_usd is written ONLY on a safe improvement, and ONLY from
    the measured benchmark delta. It is never a projection and never a
    production observation — those are different columns with different
    meanings.
    """
    fields: dict[str, Any] = {
        "evidence_source": "replay",
        "evidence_strength": domain.evidence_strength("replay"),
        # Storage key; the value is the internal evidence-maturity index. It is
        # NOT returned by recommendation_row_to_response.
        "confidence": evidence_maturity,
        "sample_size": sample_size,
        "success_signal": success_signal.to_dict() if success_signal else {},
        # The non-inferiority evidence travels WITH the recommendation. Without
        # it a reader of the recommendations table can see `verified` and
        # `candidate_quality` but has no way to tell whether a regression
        # against baseline was ever ruled out — which is exactly how the
        # original failure reached a customer.
        "quality_safety": quality_safety,
    }
    if winner is None:
        return fields

    m = winner["metrics"]
    fields.update({
        "candidate_cost": m.get("mean_cost_usd"),
        "candidate_quality": m.get("quality"),
        "candidate_latency_p95_ms": m.get("latency_p95_ms"),
        "candidate_error_rate": m.get("error_rate"),
        "quality_provenance": m.get("quality_provenance") or "unknown",
    })

    # The baseline half of every comparison the API exposes. NULL means NOT
    # MEASURED — including the case where the baseline's price had to be
    # guessed, which leaves mean_cost_usd NULL upstream in _execute_arm.
    b_metrics = winner.get("baseline_metrics") or {}
    if b_metrics:
        fields.update({
            "baseline_cost": b_metrics.get("mean_cost_usd"),
            "baseline_quality": b_metrics.get("quality"),
            "baseline_latency_p95_ms": b_metrics.get("latency_p95_ms"),
            "baseline_error_rate": b_metrics.get("error_rate"),
        })
    if conclusion == domain.CONCLUSION_SAFE_IMPROVEMENT:
        b = winner.get("baseline_mean_cost_usd")
        c = m.get("mean_cost_usd")
        n = m.get("n")
        if b is not None and c is not None and n:
            fields[domain.savings_column("verified")] = round((float(b) - float(c)) * int(n), 8)
    return fields


def _link_evidence(org_id: str, recommendation_id: str, benchmark_id: str) -> None:
    """Cite this benchmark from the recommendation. Many-to-many by design."""
    try:
        supabase.table("recommendation_evidence").insert({
            "org_id": org_id,
            "recommendation_id": recommendation_id,
            "benchmark_id": benchmark_id,
            "evidence_role": "primary",
        }).execute()
    except Exception:
        # Already cited (unique index) — citing twice is a no-op, not an error.
        pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_baseline_graph(
    org_id: str, workload: dict, workflow_id: str
) -> tuple[Optional[dict], Optional[str]]:
    """
    The configuration currently in force, and its endpoint slug.

    Prefers the promoted deployment (what production actually serves) and falls
    back to the workflow's draft graph. A direct-inference workload has neither;
    the caller handles that before reaching here.
    """
    endpoint_slug = (
        workload.get("identity_ref") if workload.get("identity_kind") == "endpoint" else None
    )
    try:
        q = (
            supabase.table("workflow_deployments")
            .select("id, workflow_id, org_id, version, endpoint_slug, graph_json, status")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .eq("status", "promoted")
        )
        rows = getattr(q.order("version", desc=True).limit(1).execute(), "data", None) or []
        if rows:
            return rows[0].get("graph_json") or {"nodes": [], "edges": []}, (
                rows[0].get("endpoint_slug") or endpoint_slug
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("_load_baseline_graph deployment lookup failed: %s", type(exc).__name__)

    try:
        resp = (
            supabase.table("workflows")
            .select("id, org_id, graph_json")
            .eq("id", workflow_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if rows:
            return rows[0].get("graph_json") or {"nodes": [], "edges": []}, endpoint_slug
    except Exception as exc:  # pragma: no cover
        logger.warning("_load_baseline_graph workflow lookup failed: %s", type(exc).__name__)

    return None, endpoint_slug


def _load_golden_inputs(org_id: str, workflow_id: str) -> list[dict]:
    """
    The replay case set.

    Reuses `golden_inputs` — including rows created by the existing
    POST /golden-inputs/import-from-production endpoint, which promotes a real
    production run into a replayable case. That precedent is why replay evidence
    here is evidence about the customer's REAL data.
    """
    try:
        resp = (
            supabase.table("golden_inputs")
            .select(_GOLDEN_INPUT_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("_load_golden_inputs failed: %s", type(exc).__name__)
        return []


def _load_eval_checks(org_id: str, workflow_id: str) -> list[dict]:
    """The workflow's own eval-suite checks — the same rows the eval UI uses."""
    try:
        resp = (
            supabase.table("eval_suites")
            .select(_EVAL_SUITE_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if not rows:
            return []
        raw = rows[0].get("checks") or []
        return [c for c in raw if isinstance(c, dict) and c.get("enabled", True)]
    except Exception as exc:  # pragma: no cover
        logger.warning("_load_eval_checks failed: %s", type(exc).__name__)
        return []


def _traffic_for(org_id: str, workload: dict) -> dict:
    from optimization import evidence as evidence_mod

    endpoint_slug = (
        workload.get("identity_ref") if workload.get("identity_kind") == "endpoint" else None
    )
    t = evidence_mod.observed_production_traffic(
        org_id, endpoint_slug=endpoint_slug, lookback_days=30
    )
    t["window_days"] = 30
    return t


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_benchmark(org_id: str, benchmark_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("optimization_benchmarks")
            .select(BENCHMARK_COLS)
            .eq("id", benchmark_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("get_benchmark failed: %s", type(exc).__name__)
        return None


def list_benchmarks(
    org_id: str, *, workload_id: Optional[str] = None, conclusion: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    try:
        q = supabase.table("optimization_benchmarks").select(BENCHMARK_COLS).eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        if conclusion:
            q = q.eq("conclusion", conclusion)
        resp = q.order("created_at", desc=True).limit(max(1, min(limit, 200))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_benchmarks failed: %s", type(exc).__name__)
        return []


def list_candidate_results(
    org_id: str, *, benchmark_id: Optional[str] = None, workload_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """
    Measured candidate arms, queryable on their own.

    Independently queryable BY DESIGN: "every candidate we ever measured for
    this workload that beat baseline on cost, regardless of what we concluded at
    the time" must be answerable without re-running anything.
    """
    try:
        q = supabase.table("benchmark_candidate_results").select(CANDIDATE_RESULT_COLS).eq(
            "org_id", org_id
        )
        if benchmark_id:
            q = q.eq("benchmark_id", benchmark_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        resp = q.order("created_at", desc=True).limit(max(1, min(limit, 500))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_candidate_results failed: %s", type(exc).__name__)
        return []


def current_conclusion(org_id: str, benchmark_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("benchmark_conclusions")
            .select(CONCLUSION_COLS)
            .eq("org_id", org_id)
            .eq("benchmark_id", benchmark_id)
            .eq("is_current", True)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("current_conclusion failed: %s", type(exc).__name__)
        return None


def conclusion_history(org_id: str, benchmark_id: str) -> list[dict]:
    """Every evaluation of this evidence, newest first — the audit trail."""
    try:
        resp = (
            supabase.table("benchmark_conclusions")
            .select(CONCLUSION_COLS)
            .eq("org_id", org_id)
            .eq("benchmark_id", benchmark_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("conclusion_history failed: %s", type(exc).__name__)
        return []


def benchmark_row_to_response(row: dict, *, conclusion_row: Optional[dict] = None) -> dict:
    conclusion = (conclusion_row or row).get("conclusion")
    reasons = (
        (conclusion_row or {}).get("reasons")
        or ((row.get("conclusion_detail") or {}).get("reasons"))
        or []
    )
    # Read from the `confidence` column, which stores the evidence-maturity
    # index. Loaded for internal use only; conclusion_payload does not emit it.
    evidence_maturity = (conclusion_row or row).get("confidence")
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "workload_id": (str(row["workload_id"]) if row.get("workload_id") else None),
        "method": row.get("method"),
        "status": row.get("status"),
        "objective": row.get("objective"),
        "sample_size": row.get("sample_size"),
        "dataset_ref": row.get("dataset_ref") or {},
        "baseline_metrics": row.get("baseline_metrics"),
        "candidate_metrics": row.get("candidate_metrics"),
        "quality_provenance": row.get("quality_provenance"),
        "success_signal": row.get("success_signal") or {},
        "materiality_threshold": row.get("materiality_threshold") or {},
        "policy_id": (str(row["policy_id"]) if row.get("policy_id") else None),
        "more_data_changes_conclusion": row.get("more_data_changes_conclusion"),
        # Structured quality-safety evidence, the frontier and the consideration
        # funnel live on the CONCLUSION, which is policy-versioned and immutable.
        # They are surfaced here so a caller reading a benchmark does not have to
        # know that. None means the conclusion row was not loaded, not that the
        # evidence is absent.
        "quality_safety": (conclusion_row or {}).get("quality_safety"),
        "quality_safety_policy": (conclusion_row or {}).get("quality_safety_policy"),
        "frontier": (conclusion_row or {}).get("frontier"),
        "consideration": (conclusion_row or {}).get("consideration"),
        "error": row.get("error"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "created_at": row.get("created_at"),
        **domain.conclusion_payload(conclusion, reasons=reasons),
    }
