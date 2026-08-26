"""
Attempt: ONE execution of work.

*** THE EXISTING TRACING INFRASTRUCTURE IS THE RECORD OF TRUTH. ***
`workflow_runs` already stores runtime executions and is not duplicated,
renamed or rewritten. This module is the thin domain adapter over it (and over
the `attempts` SQL view), plus the cost/outcome edges that genuinely have no
home there.

════════════════════════════════════════════════════════════════════════════
HARD CONSTRAINT: THIS IS THE ONLY PLACE THAT PARSES `node_results`.
════════════════════════════════════════════════════════════════════════════
`workflow_runs.node_results` is JSONB with an informal, evolving per-node shape
(`cost` vs `cost_usd`, `tokens` vs `tokens_output`, `status` vs `error` vs
`output_warning`). If every caller parses it itself, that informality becomes
technical debt in a dozen files at once and the shape can never be changed.

Any new code that needs cost, tokens, latency, models, providers or error
status out of an execution MUST call a function in this module. Do not write
`run["node_results"]` anywhere else.

Deliberately NOT denormalized: no derived columns are added to workflow_runs for
hypothetical future queries. JSONB-backed aggregation is accepted for now.
Denormalize when real customer volume shows which dimensions are actually hot.

Surfaces
--------
An Attempt does NOT have to come from a workflow deployment. Direct Inference
(POST /v1/chat/completions, one changed base_url, no Studio workflow) produces
attempts with no workflow_id and no deployment version. Every function here
takes an explicit `attempt_source` and none of them assume a deployment exists.
"""
from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from supabase_client import supabase

from optimization import domain

logger = logging.getLogger(__name__)

#: Node types in node_results that represent a unit of executed work with a
#: cost. Kept in one place so the definition of "an LLM call" cannot drift.
WORK_STEP_TYPES = ("ai-step", "model", "optimizer", "tool_call")

#: Node types that call a model and therefore count toward llm_call_count.
MODEL_STEP_TYPES = ("ai-step", "model", "optimizer")

ATTEMPT_VIEW_COLS = (
    "attempt_id, external_attempt_id, attempt_source, org_id, workload_id, surface, "
    "project_id, workflow_id, endpoint_slug, served_version, execution_mode, "
    "experiment_id, variant_name, inference_cost_usd, estimated_cost_usd, "
    "cost_is_estimated, duration_ms, step_results, provider, model, prompt_tokens, "
    "completion_tokens, success, strategy_id, strategy_fingerprint, occurred_at"
)

DIRECT_LINK_COLS = (
    "id, org_id, attempt_id, workload_id, strategy_id, strategy_fingerprint, "
    "executor_id, occurred_at, created_at"
)

#: An id that is not a UUID is a customer-visible direct-inference request id.
_DIRECT_ID_PREFIX = "chatcmpl-"

_RUN_COLS = (
    "id, org_id, workflow_id, endpoint_slug, served_version, execution_mode, "
    "experiment_id, variant_name, total_cost, total_latency_ms, node_results, created_at"
)

COST_EVENT_COLS = (
    "id, org_id, workload_id, executor_id, attempt_ref, attempt_source, cost_type, "
    "amount, unit, amount_usd, quantity, quantity_unit, occurred_at, recorded_at, "
    "metadata, idempotency_key, created_at"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ═══════════════════════════════════════════════════════════════════════════
# THE node_results PARSER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepFact:
    """One executed step, normalised out of a node_results entry."""

    node_id: str
    step_type: str
    model: Optional[str] = None
    provider: Optional[str] = None
    cost_usd: float = 0.0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    has_error: bool = False
    error_kind: Optional[str] = None
    is_model_call: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttemptFacts:
    """Everything derivable from one execution, whichever surface ran it."""

    steps: list[StepFact] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    has_error: bool = False
    error_kinds: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    llm_call_count: int = 0
    #: True when node_results was missing or not a list. The caller learned
    #: nothing; it must not treat the zeros above as measurements.
    unparseable: bool = False
    #: False when cost could not be MEASURED for this execution — typically
    #: because pricing for the model was estimated rather than known. When this
    #: is False, `total_cost_usd` is NOT a measurement and must not be summed
    #: into a spend figure; use `cost_estimated_usd` and label it.
    cost_measured: bool = True
    #: The estimate, when there is one. Always paired with cost_measured=False.
    cost_estimated_usd: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def measured_cost_usd(self) -> Optional[float]:
        """Measured spend, or None. Never returns an estimate."""
        if self.unparseable or not self.cost_measured:
            return None
        return self.total_cost_usd

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        d["total_tokens"] = self.total_tokens
        d["measured_cost_usd"] = self.measured_cost_usd
        return d


def node_result_has_error(nr: Any) -> bool:
    """
    CANONICAL error check for one node_results entry.

    Covers explicit errors (status == 'error' or error truthy), output-quality
    warnings (status == 'warning'), and a non-empty output_warning. This is the
    same definition `workflow_management._nr_has_error` uses, so error rates
    computed here match the ones shown in observability, experiments and
    rollback checks. Keep exactly one definition.
    """
    if not isinstance(nr, dict):
        return False
    st = nr.get("status")
    if st == "error" or st == "warning":
        return True
    if nr.get("error"):
        return True
    if nr.get("output_warning"):
        return True
    return False


def _error_kind(nr: dict) -> Optional[str]:
    if nr.get("status") == "error" or nr.get("error"):
        return str(nr.get("error_detail") or nr.get("error_status") or "error")[:120]
    if nr.get("output_warning"):
        return str(nr.get("output_warning"))[:120]
    if nr.get("status") == "warning":
        return "warning"
    return None


def parse_step_results(node_results: Any) -> AttemptFacts:
    """
    THE parser. Normalises `workflow_runs.node_results` into AttemptFacts.

    Handles every field-name variant the runtime has emitted:
        cost         | cost_usd
        tokens_output | output_tokens | tokens   (output tokens)
        input_tokens  | tokens_input            (input tokens)
        status/error/output_warning (error signalling)

    Returns `unparseable=True` with zeroed fields when node_results is absent or
    malformed, so a caller can tell "measured zero cost" apart from "we could
    not read this". Never guess a value for a missing field.
    """
    facts = AttemptFacts()

    if not isinstance(node_results, list):
        facts.unparseable = True
        return facts

    models: list[str] = []
    providers: list[str] = []
    error_kinds: list[str] = []

    for nr in node_results:
        if not isinstance(nr, dict):
            continue
        step_type = (nr.get("type") or "").lower()

        cost = nr.get("cost_usd")
        if cost is None:
            cost = nr.get("cost")
        try:
            cost_f = float(cost or 0)
        except (TypeError, ValueError):
            cost_f = 0.0

        try:
            latency = int(nr.get("latency_ms") or 0)
        except (TypeError, ValueError):
            latency = 0

        # Field-name variants the runtime has emitted over time, plus the
        # OpenAI-style names some callers used. Order matters: the explicit
        # per-direction names win over the ambiguous bare `tokens`.
        out_tok = nr.get("tokens_output")
        if out_tok is None:
            out_tok = nr.get("output_tokens")
        if out_tok is None:
            out_tok = nr.get("tokens")
        in_tok = nr.get("input_tokens")
        if in_tok is None:
            in_tok = nr.get("tokens_input")
        try:
            out_tok_i = int(out_tok or 0)
        except (TypeError, ValueError):
            out_tok_i = 0
        try:
            in_tok_i = int(in_tok or 0)
        except (TypeError, ValueError):
            in_tok_i = 0

        model = nr.get("model")
        provider = nr.get("provider")
        errored = node_result_has_error(nr)
        kind = _error_kind(nr) if errored else None

        is_model_call = step_type in MODEL_STEP_TYPES and bool(model)

        facts.steps.append(
            StepFact(
                node_id=str(nr.get("node_id") or ""),
                step_type=step_type,
                model=(str(model).strip() if model else None),
                provider=(str(provider).strip().lower() if provider else None),
                cost_usd=cost_f,
                latency_ms=latency,
                input_tokens=in_tok_i,
                output_tokens=out_tok_i,
                has_error=errored,
                error_kind=kind,
                is_model_call=is_model_call,
            )
        )

        facts.total_cost_usd += cost_f
        facts.total_latency_ms += latency
        facts.input_tokens += in_tok_i
        facts.output_tokens += out_tok_i
        if errored:
            facts.has_error = True
            if kind:
                error_kinds.append(kind)
        if is_model_call:
            facts.llm_call_count += 1
        if model and str(model).strip() not in models:
            models.append(str(model).strip())
        if provider and str(provider).strip().lower() not in providers:
            providers.append(str(provider).strip().lower())

    facts.total_cost_usd = round(facts.total_cost_usd, 10)
    facts.models_used = models
    facts.providers_used = providers
    facts.error_kinds = error_kinds
    return facts


def sum_usage_tokens(node_results: Any) -> tuple[int, int, int]:
    """
    CANONICAL (prompt_tokens, completion_tokens, total_tokens) for an execution.

    `routers/openai_compat.py` delegates to this so the OpenAI-compatible
    `usage` block and OptiML's own accounting can never disagree.
    """
    facts = parse_step_results(node_results)
    return facts.input_tokens, facts.output_tokens, facts.total_tokens


def model_calls(node_results: Any) -> list[StepFact]:
    """Just the model-calling steps. For per-model attribution."""
    return [s for s in parse_step_results(node_results).steps if s.is_model_call]


def executors_used(node_results: Any) -> list[dict]:
    """
    Which executors performed this attempt, as executor_ref dicts.

    This is how an Attempt is linked back to `executors` without a join table:
    the execution record already names the model and provider per step.
    """
    refs: list[dict] = []
    seen: set[tuple] = set()
    for step in model_calls(node_results):
        key = ("model", step.provider, step.model)
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "executor_type": "model",
            "vendor": step.provider,
            "external_id": step.model,
        })
    return refs


# ═══════════════════════════════════════════════════════════════════════════
# Attempt domain object
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Attempt:
    """
    One execution of work, surface-agnostic.

    `workflow_id`, `endpoint_slug` and `served_version` are all Optional: a
    Direct Inference attempt has none of them. Nothing in this class assumes a
    deployment exists.
    """

    attempt_id: str
    attempt_source: str
    org_id: str
    occurred_at: Optional[str] = None
    #: The customer-visible id (``chatcmpl-...``) for a direct-inference
    #: attempt; None on the runtime surface, where attempt_id is already the
    #: durable identifier. This is what a customer quotes when attaching an
    #: outcome hours later.
    external_attempt_id: Optional[str] = None
    workload_id: Optional[str] = None
    project_id: Optional[str] = None
    surface: str = "runtime"
    workflow_id: Optional[str] = None
    endpoint_slug: Optional[str] = None
    served_version: Optional[int] = None
    execution_mode: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_name: Optional[str] = None
    duration_ms: Optional[int] = None
    strategy_id: Optional[str] = None
    strategy_fingerprint: Optional[str] = None
    facts: AttemptFacts = field(default_factory=AttemptFacts)

    def to_dict(self) -> dict:
        return {
            "attempt_id": self.attempt_id,
            "external_attempt_id": self.external_attempt_id,
            "attempt_source": self.attempt_source,
            "org_id": self.org_id,
            "occurred_at": self.occurred_at,
            "workload_id": self.workload_id,
            "strategy_id": self.strategy_id,
            "strategy_fingerprint": self.strategy_fingerprint,
            "project_id": self.project_id,
            "surface": self.surface,
            "workflow_id": self.workflow_id,
            "endpoint_slug": self.endpoint_slug,
            "served_version": self.served_version,
            "execution_mode": self.execution_mode,
            "experiment_id": self.experiment_id,
            "variant_name": self.variant_name,
            "duration_ms": self.duration_ms,
            # Measured spend only. An estimate is reported separately and
            # labelled, never returned as if it had been measured.
            "cost_usd": self.facts.measured_cost_usd,
            "cost_estimated_usd": self.facts.cost_estimated_usd,
            "cost_is_estimated": not self.facts.cost_measured,
            "input_tokens": (None if self.facts.unparseable else self.facts.input_tokens),
            "output_tokens": (None if self.facts.unparseable else self.facts.output_tokens),
            "llm_call_count": (None if self.facts.unparseable else self.facts.llm_call_count),
            "models_used": self.facts.models_used,
            "providers_used": self.facts.providers_used,
            "has_error": (None if self.facts.unparseable else self.facts.has_error),
            "executors": executors_used_from_facts(self.facts),
            "steps_unparseable": self.facts.unparseable,
        }


def executors_used_from_facts(facts: AttemptFacts) -> list[dict]:
    refs: list[dict] = []
    seen: set[tuple] = set()
    for step in facts.steps:
        if not step.is_model_call:
            continue
        key = ("model", step.provider, step.model)
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "executor_type": "model",
            "vendor": step.provider,
            "external_id": step.model,
        })
    return refs


def attempt_from_workflow_run(run: dict, *, workload_id: Optional[str] = None,
                              project_id: Optional[str] = None) -> Attempt:
    """Build an Attempt from a `workflow_runs` row. The only mapper for runtime."""
    return Attempt(
        attempt_id=str(run.get("id")),
        attempt_source="workflow_run",
        org_id=str(run.get("org_id") or ""),
        occurred_at=run.get("created_at"),
        workload_id=workload_id,
        project_id=project_id,
        surface="runtime",
        workflow_id=(str(run["workflow_id"]) if run.get("workflow_id") else None),
        endpoint_slug=run.get("endpoint_slug"),
        served_version=run.get("served_version") or run.get("version"),
        execution_mode=run.get("execution_mode"),
        experiment_id=(str(run["experiment_id"]) if run.get("experiment_id") else None),
        variant_name=run.get("variant_name"),
        duration_ms=run.get("total_latency_ms"),
        facts=parse_step_results(run.get("node_results")),
    )


def facts_from_direct_row(row: dict) -> AttemptFacts:
    """
    Build AttemptFacts for a DIRECT-INFERENCE attempt.

    The `attempts` view deliberately emits NULL for `step_results` on that
    branch rather than synthesizing a node_results array in SQL — building that
    shape in a second place would be exactly the drift this module exists to
    prevent. Instead the view exposes the raw measured columns and this function
    assembles the one step they describe. All shape knowledge stays here.

    A direct-inference request is, by construction, ONE model call.

    Cost honesty: `api_request_log.total_cost` holds MEASURED spend only and is
    NULL when pricing had to be estimated. In that case `cost_measured` is False
    and the estimate is carried separately, so a guess is never summed into a
    spend figure.
    """
    facts = AttemptFacts()

    provider = (row.get("provider") or "").strip().lower() or None
    model = (row.get("model") or "").strip() or None

    def _int(value) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    in_tok = _int(row.get("prompt_tokens"))
    out_tok = _int(row.get("completion_tokens"))
    latency = _int(row.get("duration_ms"))

    measured = row.get("inference_cost_usd")
    estimated = row.get("estimated_cost_usd")
    is_estimated = bool(row.get("cost_is_estimated"))

    if measured is not None and not is_estimated:
        cost = float(measured)
        facts.cost_measured = True
    else:
        cost = float(estimated) if estimated is not None else 0.0
        facts.cost_measured = False
        facts.cost_estimated_usd = (float(estimated) if estimated is not None else None)

    # `success` is NULL only when the log row predates this column; treat an
    # unknown outcome as "not known to have errored" rather than inventing one.
    errored = row.get("success") is False

    facts.steps.append(
        StepFact(
            node_id="direct",
            step_type="model",
            model=model,
            provider=provider,
            cost_usd=cost,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            has_error=errored,
            error_kind=("request_failed" if errored else None),
            is_model_call=bool(model),
        )
    )

    facts.total_cost_usd = cost
    facts.total_latency_ms = latency
    facts.input_tokens = in_tok
    facts.output_tokens = out_tok
    facts.has_error = errored
    facts.error_kinds = ["request_failed"] if errored else []
    facts.models_used = [model] if model else []
    facts.providers_used = [provider] if provider else []
    facts.llm_call_count = 1 if model else 0
    # We DID read this execution; there is nothing unparseable about it.
    facts.unparseable = False
    return facts


def attempt_from_view_row(row: dict) -> Attempt:
    """
    Build an Attempt from a row of the `attempts` SQL view.

    Both execution surfaces come through here, so every consumer — benchmarks,
    coverage, cost roll-ups — sees one abstraction regardless of whether the
    work ran as a Studio workflow or as a direct /v1/chat/completions call.
    """
    source = row.get("attempt_source") or "workflow_run"

    if source == "direct_inference":
        facts = facts_from_direct_row(row)
    else:
        facts = parse_step_results(row.get("step_results"))

    return Attempt(
        attempt_id=str(row.get("attempt_id")),
        external_attempt_id=row.get("external_attempt_id"),
        attempt_source=source,
        org_id=str(row.get("org_id") or ""),
        occurred_at=row.get("occurred_at"),
        workload_id=(str(row["workload_id"]) if row.get("workload_id") else None),
        project_id=(str(row["project_id"]) if row.get("project_id") else None),
        surface=row.get("surface") or "runtime",
        workflow_id=(str(row["workflow_id"]) if row.get("workflow_id") else None),
        endpoint_slug=row.get("endpoint_slug"),
        served_version=row.get("served_version"),
        execution_mode=row.get("execution_mode"),
        experiment_id=(str(row["experiment_id"]) if row.get("experiment_id") else None),
        variant_name=row.get("variant_name"),
        duration_ms=row.get("duration_ms"),
        strategy_id=(str(row["strategy_id"]) if row.get("strategy_id") else None),
        strategy_fingerprint=row.get("strategy_fingerprint"),
        facts=facts,
    )


def get_attempt(
    org_id: str,
    attempt_ref: str,
    *,
    attempt_source: str = "workflow_run",
) -> Optional[Attempt]:
    """
    Fetch one attempt, org-scoped. None when it does not exist for this org.

    This is the lookup an outcome-attachment endpoint uses to validate that a
    caller-supplied attempt id really belongs to their organization.

    Identifier note: a direct-inference attempt is addressed by its
    customer-visible ``chatcmpl-...`` id, which is NOT a UUID and therefore
    cannot be ``api_request_log.id``. Callers frequently do not know which kind
    of id they hold, so the shape of the reference is used to route the lookup
    and a mislabelled ``attempt_source`` still resolves.
    """
    ref = (attempt_ref or "").strip()
    if not ref:
        return None

    looks_direct = ref.startswith(_DIRECT_ID_PREFIX)
    if looks_direct or attempt_source == "direct_inference":
        found = _get_direct_attempt(org_id, ref)
        if found is not None or looks_direct:
            return found

    if attempt_source in ("workflow_run", "external", "none"):
        try:
            resp = (
                supabase.table("workflow_runs")
                .select(_RUN_COLS)
                .eq("id", ref)
                .eq("org_id", org_id)
                .limit(1)
                .execute()
            )
            rows = getattr(resp, "data", None) or []
            if rows:
                return attempt_from_workflow_run(rows[0])
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("get_attempt(workflow_run) failed: %s", type(exc).__name__)
            return None

    if attempt_source == "api_request":
        # api_request_log has no node_results; it records the request envelope.
        # For a direct-inference row the attempts view is the richer source, so
        # try that first; otherwise fall back to the log row itself.
        direct = _get_direct_attempt(org_id, ref, by_log_id=True)
        if direct is not None:
            return direct
        try:
            resp = (
                supabase.table("api_request_log")
                .select(
                    "id, org_id, endpoint_slug, served_version, experiment_id, "
                    "variant_name, total_latency_ms, total_cost, workflow_run_id, success"
                )
                .eq("id", ref)
                .eq("org_id", org_id)
                .limit(1)
                .execute()
            )
            rows = getattr(resp, "data", None) or []
            if not rows:
                return None
            row = rows[0]
            if row.get("workflow_run_id"):
                inner = get_attempt(
                    org_id, str(row["workflow_run_id"]), attempt_source="workflow_run"
                )
                if inner is not None:
                    return inner
            # No execution detail available: report that plainly rather than
            # returning zeros that would read as measurements.
            facts = AttemptFacts(unparseable=True, cost_measured=False)
            return Attempt(
                attempt_id=str(row["id"]),
                attempt_source="api_request",
                org_id=org_id,
                surface="direct_inference",
                endpoint_slug=row.get("endpoint_slug"),
                served_version=row.get("served_version"),
                experiment_id=(str(row["experiment_id"]) if row.get("experiment_id") else None),
                variant_name=row.get("variant_name"),
                duration_ms=row.get("total_latency_ms"),
                facts=facts,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("get_attempt(api_request) failed: %s", type(exc).__name__)
            return None

    return None


def _get_direct_attempt(
    org_id: str, ref: str, *, by_log_id: bool = False
) -> Optional[Attempt]:
    """Resolve a direct-inference attempt through the attempts view."""
    column = "attempt_id" if by_log_id else "external_attempt_id"
    try:
        resp = (
            supabase.table("attempts")
            .select(ATTEMPT_VIEW_COLS)
            .eq("org_id", org_id)
            .eq("attempt_source", "direct_inference")
            .eq(column, ref)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return attempt_from_view_row(rows[0]) if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("_get_direct_attempt failed: %s", type(exc).__name__)
        return None


def record_direct_inference_link(
    org_id: str,
    *,
    attempt_id: str,
    workload_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    strategy_fingerprint: Optional[str] = None,
    executor_id: Optional[str] = None,
    occurred_at: Optional[str] = None,
) -> Optional[dict]:
    """
    Attribute one direct-inference attempt to its workload and baseline strategy.

    This writes ONLY the two facts the attempts view cannot recover from
    `api_request_log`: the resolved workload UUID (the log carries a string
    identity ref, which a rename would break) and the strategy fingerprint
    (derived partly from the system prompt, which the log deliberately does not
    store). Cost, latency, tokens and status are NOT written here — they live on
    the log row, and a second copy would be a second source of truth.

    Idempotent on (org_id, attempt_id): a retry updates the existing row rather
    than creating a duplicate attempt.
    """
    key = (attempt_id or "").strip()
    if not key:
        return None

    row = {
        "org_id": org_id,
        "attempt_id": key,
        "workload_id": workload_id,
        "strategy_id": strategy_id,
        "strategy_fingerprint": strategy_fingerprint,
        "executor_id": executor_id,
    }
    if occurred_at:
        row["occurred_at"] = occurred_at

    try:
        existing = (
            supabase.table("direct_inference_attempt_links")
            .select("id")
            .eq("org_id", org_id)
            .eq("attempt_id", key)
            .limit(1)
            .execute()
        )
        if getattr(existing, "data", None):
            updated = (
                supabase.table("direct_inference_attempt_links")
                .update({k: v for k, v in row.items() if k not in ("org_id", "attempt_id")})
                .eq("id", existing.data[0]["id"])
                .eq("org_id", org_id)
                .execute()
            )
            return (updated.data or [None])[0]

        inserted = supabase.table("direct_inference_attempt_links").insert(row).execute()
        return (inserted.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("record_direct_inference_link failed: %s", type(exc).__name__)
        return None


def list_attempts(
    org_id: str,
    *,
    workload_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
    execution_mode: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 200,
) -> list[Attempt]:
    """Attempts for an org, newest first. Always re-filtered by org_id."""
    try:
        q = supabase.table("attempts").select(ATTEMPT_VIEW_COLS).eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        if endpoint_slug:
            q = q.eq("endpoint_slug", endpoint_slug)
        if execution_mode:
            q = q.eq("execution_mode", execution_mode)
        if since is not None:
            q = q.gte("occurred_at", _iso(since))
        resp = q.order("occurred_at", desc=True).limit(max(1, min(limit, 1000))).execute()
        return [attempt_from_view_row(r) for r in (getattr(resp, "data", None) or [])]
    except Exception as exc:  # pragma: no cover
        logger.warning("list_attempts failed: %s", type(exc).__name__)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Cost events — cost that is not (only) token cost
# ═══════════════════════════════════════════════════════════════════════════

#: Attempt sources whose INFERENCE cost is already carried by the execution
#: record and therefore already surfaced by the `attempts` view.
#:
#: This is the surface-neutral form of a rule that was previously written as
#: "inference cost always lives on workflow_runs". That assumption stopped
#: holding when Direct Inference landed: there, the execution record is
#: `api_request_log` and the cost is `api_request_log.total_cost`. The rule was
#: never really about workflow_runs — it is about not counting a cost twice that
#: the attempts view already reports.
_INFERENCE_COST_ON_EXECUTION_RECORD = ("workflow_run", "api_request", "direct_inference")

COST_BASIS_MEASURED = "measured"
COST_BASIS_ESTIMATED = "estimated"
COST_BASES = (COST_BASIS_MEASURED, COST_BASIS_ESTIMATED)


class DoubleCountedCost(ValueError):
    """Raised when a cost event would double-count against the attempts view."""


def record_cost_event(
    org_id: str,
    *,
    cost_type: str,
    amount: float,
    unit: str = "usd",
    idempotency_key: Optional[str] = None,
    workload_id: Optional[str] = None,
    executor_id: Optional[str] = None,
    attempt_ref: Optional[str] = None,
    external_attempt_ref: Optional[str] = None,
    attempt_source: str = "workflow_run",
    amount_usd: Optional[float] = None,
    quantity: Optional[float] = None,
    quantity_unit: Optional[str] = None,
    occurred_at: Optional[datetime] = None,
    metadata: Optional[dict] = None,
    basis: str = COST_BASIS_MEASURED,
) -> Optional[dict]:
    """
    Record one economic cost event, idempotently.

    ── The inference rule, stated surface-neutrally ────────────────────────────
    Inference cost that the `attempts` view ALREADY SURFACES must not be written
    here, because it would then be counted twice. That applies to both surfaces:
    `workflow_runs.total_cost` for runtime, `api_request_log.total_cost` for
    direct inference. This is now ENFORCED, not merely documented — passing
    ``cost_type='inference'`` with ``basis='measured'`` for such an attempt
    raises `DoubleCountedCost`.

    There is one legitimate inference cost event: an ESTIMATED one. When OptiML
    has no real price for a model, `api_request_log.total_cost` is NULL — the
    cost genuinely has no home on the execution record. Recording it with
    ``basis='estimated'`` keeps the information without laundering a guess into
    measured spend: `amount_usd` is forced to None, so `total_economic_cost`
    reports it in a separate `estimated_not_counted` bucket and never in the
    total.

    Everything else — agent credits, ACUs, SaaS usage, seat cost, human review
    minutes, infrastructure — is what this table is for.

    `amount_usd` stays None unless a real conversion rate is known. A caller
    that does not know the dollar value must leave it None rather than invent
    one; aggregations report how much of a total was unconvertible.
    """
    if basis not in COST_BASES:
        raise ValueError(f"basis must be one of {COST_BASES}.")

    if cost_type == "inference":
        if basis == COST_BASIS_MEASURED and attempt_source in _INFERENCE_COST_ON_EXECUTION_RECORD:
            raise DoubleCountedCost(
                "Measured inference cost for attempt_source="
                f"'{attempt_source}' is already carried by the execution record and "
                "surfaced by the attempts view; writing it here would double-count "
                "it. Record it with basis='estimated' only when the execution "
                "record has no cost (estimated pricing)."
            )
        if basis == COST_BASIS_ESTIMATED and amount_usd is not None:
            # An estimate must never reach a measured-dollars aggregate.
            amount_usd = None

    key = idempotency_key or str(_uuid.uuid4())
    meta = dict(metadata or {})
    meta.setdefault("cost_basis", basis)

    row = {
        "org_id": org_id,
        "workload_id": workload_id,
        "executor_id": executor_id,
        "attempt_ref": attempt_ref,
        "external_attempt_ref": external_attempt_ref,
        "attempt_source": attempt_source,
        "cost_type": cost_type,
        "amount": amount,
        "unit": unit,
        "amount_usd": amount_usd,
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "occurred_at": _iso(occurred_at or _utc_now()),
        "recorded_at": _iso(_utc_now()),
        "metadata": meta,
        "idempotency_key": key,
    }
    try:
        existing = (
            supabase.table("cost_events")
            .select(COST_EVENT_COLS)
            .eq("org_id", org_id)
            .eq("idempotency_key", key)
            .limit(1)
            .execute()
        )
        if existing.data:
            return existing.data[0]
        inserted = supabase.table("cost_events").insert(row).execute()
        return (inserted.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("record_cost_event failed: %s", type(exc).__name__)
        return None


def total_economic_cost(
    org_id: str,
    *,
    workload_id: Optional[str] = None,
    since: Optional[datetime] = None,
) -> dict:
    """
    Total economic cost for a workload: inference cost from the execution
    record (either surface) PLUS non-inference cost_events.

    Returns measured components and a `coverage` block naming what could not be
    counted. Nothing unconvertible or estimated is silently folded into the
    total — an "$0.004 per task" claim that quietly absorbs 12 unpriced agent
    credits, or a guessed model price, is a lie.
    """
    inference_usd = 0.0
    inference_attempts = 0
    unparseable_attempts = 0
    estimated_only_attempts = 0
    estimated_usd = 0.0

    for attempt in list_attempts(org_id, workload_id=workload_id, since=since, limit=1000):
        inference_attempts += 1
        if attempt.facts.unparseable:
            unparseable_attempts += 1
            continue
        measured = attempt.facts.measured_cost_usd
        if measured is None:
            estimated_only_attempts += 1
            if attempt.facts.cost_estimated_usd is not None:
                estimated_usd += attempt.facts.cost_estimated_usd
            continue
        inference_usd += measured

    other_usd = 0.0
    unconverted: dict[str, dict] = {}
    estimated_events = 0
    try:
        q = supabase.table("cost_events").select(COST_EVENT_COLS).eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        if since is not None:
            q = q.gte("occurred_at", _iso(since))
        rows = getattr(q.limit(2000).execute(), "data", None) or []
    except Exception:  # pragma: no cover
        rows = []

    for r in rows:
        if r.get("cost_type") == "inference":
            # Inference cost is taken from the execution record above. An
            # estimated inference event is counted separately, never in total.
            if (r.get("metadata") or {}).get("cost_basis") == COST_BASIS_ESTIMATED:
                estimated_events += 1
                estimated_usd += float(r.get("amount") or 0)
            continue
        if r.get("amount_usd") is not None:
            other_usd += float(r["amount_usd"])
        else:
            unit = r.get("unit") or "unknown"
            bucket = unconverted.setdefault(unit, {"unit": unit, "amount": 0.0, "events": 0})
            bucket["amount"] += float(r.get("amount") or 0)
            bucket["events"] += 1

    return {
        "inference_cost_usd": round(inference_usd, 8) if inference_attempts else None,
        "other_cost_usd": round(other_usd, 8) if rows else None,
        "total_cost_usd": (
            round(inference_usd + other_usd, 8) if (inference_attempts or rows) else None
        ),
        "estimated_not_counted": {
            "amount_usd": round(estimated_usd, 8) if (estimated_only_attempts or estimated_events) else None,
            "attempts": estimated_only_attempts,
            "cost_events": estimated_events,
            "note": (
                "Cost derived from estimated pricing rather than a known rate. "
                "Excluded from total_cost_usd so a guess is never presented as spend."
            ),
        },
        "coverage": {
            "attempts": inference_attempts,
            "attempts_with_unparseable_steps": unparseable_attempts,
            "attempts_with_estimated_cost_only": estimated_only_attempts,
            "cost_events": len(rows),
            "unconverted_units": list(unconverted.values()),
            "note": (
                "total_cost_usd EXCLUDES cost events whose USD value is unknown "
                "(listed in unconverted_units) and any estimated cost (see "
                "estimated_not_counted). Inference cost comes from the execution "
                "record — workflow_runs for runtime, api_request_log for direct "
                "inference — never from cost_events, to avoid double counting."
            ),
        },
    }
