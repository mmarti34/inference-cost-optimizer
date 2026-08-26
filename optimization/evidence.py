"""
EMPIRICAL side of the layer: what was actually MEASURED on this org's own work.

Hard boundary: nothing in this module may read vendor metadata (published
prices, advertised context windows, claimed regions). Vendor claims live in
optimization/executors.py and are never evidence. A price sheet says what a
call would cost; only the tables below say what it did cost, how long it took,
and whether it worked.

Sources of measurement, in the order a query should trust them:
  outcomes            what actually happened, ranked by provenance
  workflow_runs       the existing execution record (cost, latency, errors)
  api_request_log     public-endpoint request outcomes
  optimization_benchmarks  controlled replay comparisons

This module also owns the model-performance history that used to live in
workflow_runtime.py (`_infer_provider`, `_get_model_performance_history`,
`_select_optimal_model`). It was moved here rather than copied: workflow_runtime
imports these names back from this module, so there is exactly one
implementation. This module must therefore never import workflow_runtime.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import attempts as attempts_mod
from optimization import domain

logger = logging.getLogger(__name__)

# Cap on rows pulled in a single scan. Every aggregate built from a capped scan
# reports `truncated` so a partial sample is never presented as complete.
_MAX_SCAN_ROWS = 2000

_RUN_COLS = (
    "id, workflow_id, org_id, endpoint_slug, served_version, execution_mode, "
    "experiment_id, variant_name, total_cost, total_latency_ms, node_results, created_at"
)

_RUN_COLS_LIGHT = (
    "id, workflow_id, endpoint_slug, execution_mode, total_cost, "
    "total_latency_ms, node_results, created_at"
)

_OUTCOME_COLS = (
    "id, org_id, workload_id, attempt_ref, attempt_source, outcome_type, outcome_key, "
    "outcome_value, outcome_value_text, unit, success, source, provenance, "
    "provenance_rank, confidence, occurred_at, recorded_at, metadata, "
    "idempotency_key, created_at"
)

_AI_STEP_TYPES = ("ai-step", "model", "optimizer", "tool_call")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Model performance history
#
# MOVED here from workflow_runtime.py (previously the private helpers
# `_infer_provider`, `_get_model_performance_history`, `_select_optimal_model`).
# workflow_runtime imports them back from this module — there is one copy.
# ---------------------------------------------------------------------------

def infer_provider(model: str) -> str:
    """Infer provider from model name."""
    m = (model or "").strip().lower()
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("mistral") or m.startswith("mixtral"):
        return "mistral"
    if m.startswith("command"):
        return "cohere"
    return "openai"


def get_model_performance_history(org_id: str, workflow_id: str, limit: int = 200) -> list[dict]:
    """
    Historical AI step results for this workflow, read from
    workflow_runs.node_results.

    Returns a list of { "model", "provider", "cost", "latency_ms", "error" }.
    Every field is measured. Nothing is inferred from a price sheet.
    """
    if not workflow_id or not org_id:
        return []
    history: list[dict] = []
    try:
        resp = (
            supabase.table("workflow_runs")
            .select("node_results,created_at")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception:
        return []
    data = getattr(resp, "data", None) or []
    for run in data:
        history.extend(_history_from_node_results(run.get("node_results")))
    return history


def _history_from_node_results(node_results) -> list[dict]:
    """
    Model-performance rows for one execution.

    Parsing goes through optimization.attempts.parse_step_results — the single
    node_results parser. Nothing in this module touches the JSONB directly.
    """
    out: list[dict] = []
    for step in attempts_mod.parse_step_results(node_results).steps:
        if step.step_type not in _AI_STEP_TYPES or not step.model:
            continue
        out.append({
            "model": step.model,
            "provider": step.provider or infer_provider(step.model),
            "cost": step.cost_usd,
            "latency_ms": step.latency_ms,
            "error": step.has_error,
        })
    return out


def aggregate_model_stats(history: list[dict]) -> dict[str, dict]:
    """
    Per-model measured aggregates from a history list.

    Returns {model: {model, provider, avg_cost, avg_latency, error_rate, runs,
                     cost_variation}}.

    `cost_variation` is the coefficient of variation of observed per-call cost;
    it feeds confidence so a wildly variable model is not treated as a reliable
    saving. All values are measured; a model with no observations simply does
    not appear.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"costs": [], "latencies": [], "errors": 0, "total": 0}
    )
    for h in history:
        model = h.get("model") or ""
        if not model:
            continue
        stats[model]["costs"].append(h.get("cost") or 0)
        stats[model]["latencies"].append(h.get("latency_ms") or 0)
        stats[model]["total"] += 1
        if h.get("error"):
            stats[model]["errors"] += 1

    out: dict[str, dict] = {}
    for model, s in stats.items():
        total = s["total"] or 0
        out[model] = {
            "model": model,
            "provider": infer_provider(model),
            "avg_cost": domain.mean(s["costs"]),
            "avg_latency": (
                int(sum(s["latencies"]) / len(s["latencies"])) if s["latencies"] else None
            ),
            "error_rate": (s["errors"] / total) if total > 0 else None,
            "runs": total,
            "cost_variation": domain.coefficient_of_variation(s["costs"]),
        }
    return out


def select_optimal_model(
    history: list[dict],
    priority: str,
    max_cost: Optional[float],
    max_latency: Optional[int],
    allowed_models: Optional[list[str]],
    excluded_models: Optional[list[str]],
) -> tuple[str, str, str]:
    """
    Given historical performance and constraints, pick the best model.
    Returns (model, provider, reason).

    Behaviour is preserved exactly from the original workflow_runtime helper,
    including the "gpt-4o-mini" fallback when no observed candidate meets the
    constraints — the optimizer node depends on always getting a model back.
    Callers that must not receive an unmeasured fallback should use
    `aggregate_model_stats` directly and handle the empty case themselves.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"costs": [], "latencies": [], "errors": 0, "total": 0}
    )
    for h in history:
        model = h.get("model") or ""
        if not model:
            continue
        stats[model]["costs"].append(h.get("cost") or 0)
        stats[model]["latencies"].append(h.get("latency_ms") or 0)
        stats[model]["total"] += 1
        if h.get("error"):
            stats[model]["errors"] += 1

    candidates = []
    for model, s in stats.items():
        if allowed_models and model not in allowed_models:
            continue
        if excluded_models and model in excluded_models:
            continue
        avg_cost = sum(s["costs"]) / len(s["costs"]) if s["costs"] else 999.0
        avg_latency = int(sum(s["latencies"]) / len(s["latencies"])) if s["latencies"] else 99999
        error_rate = (s["errors"] / s["total"]) if s["total"] > 0 else 1.0
        if max_cost is not None and avg_cost > max_cost:
            continue
        if max_latency is not None and avg_latency > max_latency:
            continue
        if error_rate > 0.2:
            continue
        candidates.append({
            "model": model,
            "provider": infer_provider(model),
            "avg_cost": avg_cost,
            "avg_latency": avg_latency,
            "error_rate": error_rate,
            "runs": s["total"],
        })

    if not candidates:
        return ("gpt-4o-mini", "openai", "fallback: no candidates met constraints")

    priority = (priority or "cheapest").lower()
    if priority == "cheapest":
        candidates.sort(key=lambda c: c["avg_cost"])
        winner = candidates[0]
        return (winner["model"], winner["provider"], f"cheapest at ${winner['avg_cost']:.4f}/call avg")
    if priority == "fastest":
        candidates.sort(key=lambda c: c["avg_latency"])
        winner = candidates[0]
        return (winner["model"], winner["provider"], f"fastest at {winner['avg_latency']}ms avg")
    if priority == "quality":
        candidates.sort(key=lambda c: (c["error_rate"], c["avg_latency"]))
        winner = candidates[0]
        return (
            winner["model"],
            winner["provider"],
            f"highest quality at {int((1 - winner['error_rate']) * 100)}% success rate",
        )
    winner = candidates[0]
    return (winner["model"], winner["provider"], "default selection")


# ---------------------------------------------------------------------------
# Workload-level observed traffic
# ---------------------------------------------------------------------------

def observed_production_traffic(
    org_id: str,
    *,
    endpoint_slug: Optional[str] = None,
    workflow_id: Optional[str] = None,
    lookback_days: int = 30,
) -> dict:
    """
    Measured production traffic for a workload over a window.

    Returns measured values or None — never a filled-in zero standing in for
    "we did not look". `coverage` states the window, the row cap and whether
    the scan was truncated.
    """
    since = _utc_now() - timedelta(days=max(1, lookback_days))
    coverage: dict[str, Any] = {
        "window_days": lookback_days,
        "since": _iso(since),
        "row_cap": _MAX_SCAN_ROWS,
        "truncated": False,
        "source": "workflow_runs(execution_mode=production)",
    }
    try:
        q = (
            supabase.table("workflow_runs")
            .select(_RUN_COLS_LIGHT)
            .eq("org_id", org_id)
            .eq("execution_mode", "production")
            .gte("created_at", _iso(since))
        )
        if endpoint_slug:
            q = q.eq("endpoint_slug", endpoint_slug)
        if workflow_id:
            q = q.eq("workflow_id", workflow_id)
        resp = q.order("created_at", desc=True).limit(_MAX_SCAN_ROWS).execute()
    except Exception as exc:  # pragma: no cover - network/db
        logger.warning("observed_production_traffic query failed: %s", type(exc).__name__)
        return {
            "run_count": None,
            "total_cost_usd": None,
            "mean_cost_usd": None,
            "latency_p95_ms": None,
            "error_rate": None,
            "coverage": {**coverage, "error": "query_failed"},
        }

    rows = getattr(resp, "data", None) or []
    if len(rows) >= _MAX_SCAN_ROWS:
        coverage["truncated"] = True

    if not rows:
        return {
            "run_count": 0,
            "total_cost_usd": None,
            "mean_cost_usd": None,
            "latency_p95_ms": None,
            "error_rate": None,
            "coverage": {**coverage, "note": "no production runs in window"},
        }

    costs = [float(r["total_cost"]) for r in rows if r.get("total_cost") is not None]
    latencies = [float(r["total_latency_ms"]) for r in rows if r.get("total_latency_ms")]
    errored = sum(1 for r in rows if _run_has_error(r))

    return {
        "run_count": len(rows),
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "mean_cost_usd": (round(domain.mean(costs), 8) if costs else None),
        "latency_p95_ms": (int(domain.percentile(latencies, 95)) if latencies else None),
        "error_rate": round(errored / len(rows), 4),
        "coverage": coverage,
    }


def _run_has_error(run: dict) -> bool:
    """
    Did any step of this execution error?

    Delegates to the single node_results parser, so the error rate computed here
    is literally the same number the observability, experiment and rollback
    views show.
    """
    return attempts_mod.parse_step_results(run.get("node_results")).has_error


def models_used_by_workload(
    org_id: str,
    *,
    workflow_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
    lookback_days: int = 30,
) -> dict[str, dict]:
    """
    Which models this workload has ACTUALLY been run with, and how they did.

    This is what makes "propose a model it has not tried" and "propose a model
    it has tried that measured cheaper" two genuinely different, evidence-backed
    generators. Returns {} when nothing has been observed.
    """
    since = _utc_now() - timedelta(days=max(1, lookback_days))
    try:
        q = (
            supabase.table("workflow_runs")
            .select("node_results, created_at")
            .eq("org_id", org_id)
            .gte("created_at", _iso(since))
        )
        if workflow_id:
            q = q.eq("workflow_id", workflow_id)
        if endpoint_slug:
            q = q.eq("endpoint_slug", endpoint_slug)
        resp = q.order("created_at", desc=True).limit(_MAX_SCAN_ROWS).execute()
    except Exception as exc:  # pragma: no cover
        logger.warning("models_used_by_workload query failed: %s", type(exc).__name__)
        return {}

    history: list[dict] = []
    for run in (getattr(resp, "data", None) or []):
        history.extend(_history_from_node_results(run.get("node_results")))
    return aggregate_model_stats(history)


# ---------------------------------------------------------------------------
# Outcome evidence
# ---------------------------------------------------------------------------

def outcomes_for_workload(
    org_id: str,
    workload_id: str,
    *,
    since: Optional[datetime] = None,
    outcome_type: Optional[str] = None,
    limit: int = 500,
) -> list[dict]:
    """Measured outcomes for a workload, newest occurrence first."""
    try:
        q = (
            supabase.table("outcomes")
            .select(_OUTCOME_COLS)
            .eq("org_id", org_id)
            .eq("workload_id", workload_id)
        )
        if since is not None:
            q = q.gte("occurred_at", _iso(since))
        if outcome_type:
            q = q.eq("outcome_type", outcome_type)
        resp = q.order("occurred_at", desc=True).limit(max(1, min(limit, _MAX_SCAN_ROWS))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("outcomes_for_workload query failed: %s", type(exc).__name__)
        return []


def summarize_outcomes(outcomes: list[dict]) -> dict:
    """
    Summarise outcomes WITHOUT mixing provenance tiers.

    Returns one aggregate per tier plus the strongest tier present. Callers that
    want "the" quality number should take the strongest tier and report it with
    its provenance — never average across tiers. See
    domain.group_outcomes_by_provenance for why.
    """
    if not outcomes:
        return {"tiers": {}, "strongest_provenance": None, "total": 0}

    buckets = domain.group_outcomes_by_provenance(outcomes)
    tiers: dict[str, dict] = {}
    for prov, rows in buckets.items():
        numeric = [float(r["outcome_value"]) for r in rows if r.get("outcome_value") is not None]
        successes = [r["success"] for r in rows if r.get("success") is not None]
        tiers[prov] = {
            "provenance": prov,
            "provenance_rank": domain.provenance_rank(prov),
            "n": len(rows),
            "mean_value": (round(domain.mean(numeric), 6) if numeric else None),
            "success_rate": (
                round(sum(1 for s in successes if s) / len(successes), 4) if successes else None
            ),
            "variation": domain.coefficient_of_variation(numeric),
        }
    return {
        "tiers": tiers,
        "strongest_provenance": domain.strongest_provenance(outcomes),
        "total": len(outcomes),
    }
