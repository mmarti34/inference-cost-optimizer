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
    outcomes as outcomes_mod,
    policies,
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
#: It does not make a result statistically strong; `confidence` reports that.
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
    "selected_candidate_result_id, is_current, created_at"
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
) -> dict:
    """
    Execute one arm over the shared case set and MEASURE it.

    Every metric returned is measured or None. A case that raises is recorded as
    an error case and counted in the error rate — never dropped, because
    dropping failures would make an unreliable candidate look clean.
    """
    from workflow_runtime import execute_workflow  # imported lazily: see module docstring

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
                "cost_usd": cost,
                "latency_ms": latency,
                "error": False,
                "quality_checks_ran": q["ran"],
                "quality_checks_passed": q["passed"],
            })

        except Exception as exc:
            errors += 1
            per_case.append({
                "case_id": case_id,
                "arm": label,
                "cost_usd": None,
                "latency_ms": None,
                "error": True,
                "error_detail": str(exc)[:300],
            })

    n = len(cases)
    quality = (quality_passed / quality_ran) if quality_ran > 0 else None

    return {
        "label": label,
        "n": n,
        "mean_cost_usd": domain.mean(costs),
        "total_cost_usd": (round(sum(costs), 8) if costs else None),
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
) -> dict:
    """
    Run a replay benchmark for a workload and record explicit evidence.

    `recommendation_id` is OPTIONAL. A benchmark may be run exploratorily
    against a workload with no recommendation in existence; when one IS given,
    the benchmark is CITED by it via `recommendation_evidence` and the
    conclusion maps to a lifecycle transition via domain.CONCLUSION_TO_STATUS.

    Returns the benchmark row plus its conclusion payload. Never raises for an
    evidence shortfall — that is a conclusion, not an error.
    """
    from optimization import candidates as candidates_mod
    from optimization import service as service_mod

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

    benchmark = _insert_benchmark(
        org_id,
        workload_id=workload_id,
        method=method,
        objective=objective,
        policy=policy,
        materiality=materiality,
    )
    if benchmark is None:
        raise BenchmarkError("Failed to create the benchmark record.")
    benchmark_id = str(benchmark["id"])

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
            candidates_mod=candidates_mod,
            service_mod=service_mod,
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
            confidence=None,
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
) -> dict:
    workload_id = str(workload["id"])
    workflow_id = workloads_mod.resolve_workflow_id(org_id, workload)

    # A runtime replay needs a workflow graph. Direct-inference workloads have
    # none: say so explicitly rather than failing obscurely.
    if not workflow_id:
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
            confidence=None, sample_size=0, success_signal=domain.SuccessSignal(),
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
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "baseline_unavailable",
                detail="No promoted deployment or workflow graph could be resolved.",
                workflow_id=workflow_id,
            )],
            confidence=None, sample_size=0, success_signal=domain.SuccessSignal(),
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
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "sample_size_below_threshold", observed=n, required=floor,
                unit="cases", dataset="golden_inputs",
            )],
            confidence=None, sample_size=n, success_signal=domain.SuccessSignal(),
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
    if explicit_candidates:
        cand_list = list(explicit_candidates)
        gen_meta = {"source": "caller_supplied", "dropped": []}
    else:
        cand_list, gen_meta = candidates_mod.generate_candidates(
            org_id, workload, baseline_strategy, workflow_id=workflow_id
        )

    if not cand_list:
        return _conclude(
            org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
            policy=policy, materiality=materiality,
            conclusion=domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            reasons=[domain.reason(
                "no_candidates_generated",
                detail="No applicable candidate strategy could be generated.",
                dropped=gen_meta.get("dropped") or [],
            )],
            confidence=None, sample_size=n, success_signal=domain.SuccessSignal(),
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

    # ── Measure the baseline arm
    baseline_metrics = _execute_arm(
        baseline_graph, cases, org_id=org_id, workflow_id=workflow_id,
        endpoint_slug=endpoint_slug, checks=checks, label="baseline",
    )
    _write_candidate_result(
        org_id, benchmark_id, workload_id, arm="baseline", label="Current configuration",
        strategy=baseline_strategy, metrics=baseline_metrics, baseline=None,
        generator=None, dimensions=[],
    )

    # ── Measure each candidate arm over the SAME cases
    measured: list[dict] = []
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

        metrics = _execute_arm(
            cand_graph, cases, org_id=org_id, workflow_id=workflow_id,
            endpoint_slug=endpoint_slug, checks=checks, label=cand.title,
        )
        row = _write_candidate_result(
            org_id, benchmark_id, workload_id, arm="candidate", label=cand.title,
            strategy=cand.strategy, metrics=metrics, baseline=baseline_metrics,
            generator=cand.generator, dimensions=cand.dimensions,
        )
        measured.append({
            "candidate": cand,
            "metrics": metrics,
            "result_id": (str(row["id"]) if row else None),
        })

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
    )
    if verdict.get("winner"):
        # Carry the baseline it was measured against, so verified savings are
        # computed from the same run and never from an unrelated baseline.
        verdict["winner"]["baseline_mean_cost_usd"] = baseline_metrics.get("mean_cost_usd")

    return _conclude(
        org_id, benchmark_id=benchmark_id, workload=workload, objective=objective,
        policy=policy, materiality=verdict["materiality_applied"],
        conclusion=verdict["conclusion"], reasons=verdict["reasons"],
        confidence=verdict["confidence"], sample_size=n, success_signal=signal,
        more_data=verdict["more_data_changes_conclusion"],
        more_data_reasons=verdict["more_data_reasons"],
        status="completed", error=None,
        selected_result_id=verdict.get("selected_result_id"),
        winner=verdict.get("winner"),
        recommendation_id=recommendation_id, actor=actor, service_mod=service_mod,
    )


# ---------------------------------------------------------------------------
# The pure verdict function: evidence + policy version + objective -> conclusion
# ---------------------------------------------------------------------------

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
) -> dict:
    """
    PURE. Given stored evidence, a policy version and an objective, produce a
    conclusion. No I/O, no mutation.

    This is what makes a verdict reproducible and re-derivable: `reevaluate`
    calls it over RETAINED candidate results under a NEW policy version, without
    re-running a single model call. A new verdict does not mean the old one was
    wrong — it was correct under the policy in force at the time, and both are
    retained.

    Returns {conclusion, reasons, confidence, more_data_changes_conclusion,
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

    # ── Which candidates are ELIGIBLE (policy) and which are BETTER (objective)
    eligible: list[dict] = []
    for m in measured:
        metrics = m["metrics"]
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
        )
        m["policy_evaluation"] = evaluation
        if evaluation["eligible"]:
            eligible.append(m)
        else:
            for v in evaluation["violated"]:
                reasons.append(_violation_reason(v, m["candidate"].title))
            for u in evaluation["unmeasured"]:
                reasons.append(domain.reason(
                    "outcome_signal_too_weak" if u["constraint"] == "min_quality"
                    else "coverage_gap",
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

        confidence = domain.compute_confidence(
            sample_size=sample_size, evidence_source="replay",
            quality_provenance=quality_provenance,
            variation=baseline.get("cost_variation"),
        )
        return {
            "conclusion": conclusion,
            "reasons": reasons,
            "confidence": confidence,
            "more_data_changes_conclusion": more_data,
            "more_data_reasons": more_data_reasons,
            "materiality_applied": materiality,
            "winner": None,
            "selected_result_id": None,
        }

    # ── Rank eligible candidates on the OBJECTIVE's own metric
    winner = _best_by_objective(eligible, baseline, objective)
    if winner is None:
        return {
            "conclusion": domain.CONCLUSION_INSUFFICIENT_EVIDENCE,
            "reasons": reasons + [domain.reason(
                "coverage_gap",
                detail=f"The metric for objective '{objective}' was not measured in any arm.",
            )],
            "confidence": None,
            "more_data_changes_conclusion": domain.MORE_DATA_YES,
            "more_data_reasons": more_data_reasons,
            "materiality_applied": materiality,
            "winner": None,
            "selected_result_id": None,
        }

    improvement = _improvement(winner["metrics"], baseline, objective, traffic)
    material, materiality_detail = domain.evaluate_materiality(improvement, materiality)

    confidence = domain.compute_confidence(
        sample_size=sample_size,
        evidence_source="replay",
        quality_provenance=winner["metrics"].get("quality_provenance") or quality_provenance,
        variation=winner["metrics"].get("cost_variation"),
    )

    if material:
        return {
            "conclusion": domain.CONCLUSION_SAFE_IMPROVEMENT,
            "reasons": reasons,
            "confidence": confidence,
            "more_data_changes_conclusion": (
                domain.MORE_DATA_YES if (confidence or 0) < 0.34 else domain.MORE_DATA_NO
            ),
            "more_data_reasons": more_data_reasons + ([domain.reason(
                "sample_size_below_threshold", observed=sample_size, required=sample_size * 4,
                detail=(
                    "The improvement clears the materiality threshold but confidence "
                    "is low; more cases would firm up the estimate."
                ),
            )] if (confidence or 0) < 0.34 else []),
            "materiality_applied": {**materiality, "evaluation": materiality_detail},
            "winner": winner,
            "selected_result_id": winner.get("result_id"),
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
        "confidence": confidence,
        "more_data_changes_conclusion": (
            domain.MORE_DATA_YES if (unmeasurable or (confidence or 0) < 0.34)
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
        "max_error_rate": "error_rate_above_threshold",
        "max_latency_p95_ms": "latency_above_threshold",
        "max_cost_per_task_usd": "cost_above_threshold",
        "allowed_vendors": "provider_not_permitted",
        "blocked_vendors": "provider_not_permitted",
    }
    unit_by_constraint = {
        "min_quality": "score",
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
    )


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
    measured = [
        {
            "candidate": _StoredCandidate(r),
            "metrics": _metrics_from_result_row(r),
            "result_id": str(r["id"]),
        }
        for r in results
        if r.get("arm") == "candidate" and not r.get("error")
    ]

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
        confidence=verdict["confidence"],
        materiality=verdict["materiality_applied"],
        signal=signal,
        more_data=verdict["more_data_changes_conclusion"],
        more_data_reasons=verdict["more_data_reasons"],
        selected_result_id=verdict.get("selected_result_id"),
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
        "per_case": row.get("per_case_results") or [],
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _insert_benchmark(org_id, *, workload_id, method, objective, policy, materiality):
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
    try:
        result = supabase.table("optimization_benchmarks").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
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
) -> Optional[dict]:
    """
    Persist one measured arm INDEPENDENTLY of any conclusion.

    This is what lets a near-miss survive: a candidate that saved 51% but landed
    0.7pp under the quality floor is a row here even though the run concluded
    'candidates_failed_policy'.
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
            },
            "per_case_results": metrics.get("per_case"),
        })

        if baseline:
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
    confidence, materiality, signal, more_data, more_data_reasons,
    selected_result_id: Optional[str] = None,
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
        "confidence": confidence,
        "confidence_band": domain.confidence_band(confidence),
        "materiality_applied": materiality,
        "success_signal": signal.to_dict() if signal else {},
        "more_data_changes_conclusion": more_data,
        "more_data_reasons": more_data_reasons,
        "selected_candidate_result_id": selected_result_id,
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
    reasons, confidence, sample_size, success_signal, more_data, more_data_reasons,
    status, error, recommendation_id, actor, service_mod,
    selected_result_id: Optional[str] = None, winner: Optional[dict] = None,
) -> dict:
    """Write the conclusion, mirror it onto the benchmark, and — only when the
    benchmark is cited by a recommendation — apply the lifecycle transition."""
    workload_id = str(workload["id"])

    conclusion_row = _write_conclusion(
        org_id, benchmark_id=benchmark_id, workload_id=workload_id, policy=policy,
        objective=objective, conclusion=conclusion, reasons=reasons,
        confidence=confidence, materiality=materiality, signal=success_signal,
        more_data=more_data, more_data_reasons=more_data_reasons,
        selected_result_id=selected_result_id,
    )

    _update_benchmark(org_id, benchmark_id, {
        "status": status,
        "conclusion": conclusion,
        "conclusion_detail": {"reasons": reasons},
        "more_data_changes_conclusion": more_data,
        "more_data_reason": (more_data_reasons[0]["code"] if more_data_reasons else None),
        "materiality_threshold": materiality,
        "confidence": confidence,
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
        confidence=confidence,
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
                        conclusion=conclusion, confidence=confidence,
                        sample_size=sample_size, winner=winner,
                        success_signal=success_signal,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Benchmark %s could not transition recommendation %s: %s",
                    benchmark_id, recommendation_id, exc,
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
        **domain.conclusion_payload(conclusion, reasons=reasons, confidence=confidence),
    }


def _recommendation_fields(*, conclusion, confidence, sample_size, winner, success_signal) -> dict:
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
        "confidence": confidence,
        "sample_size": sample_size,
        "success_signal": success_signal.to_dict() if success_signal else {},
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
    confidence = (conclusion_row or row).get("confidence")
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
        "error": row.get("error"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "created_at": row.get("created_at"),
        **domain.conclusion_payload(conclusion, reasons=reasons, confidence=confidence),
    }
