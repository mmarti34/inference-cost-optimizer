"""
The ONE call site where Direct Inference feeds the optimization domain model.

Every direct-inference request becomes an **Attempt** in the
Workload → ExecutionStrategy → Executor → Attempt → Outcome model, attributed
with cost, latency, executor and strategy.  On the Studio surface an Attempt is
derived from ``workflow_runs`` through the ``public.attempts`` view.  Direct
Inference has no workflow run to derive from, so the record is handed to the
domain layer explicitly — and this module is the only place that happens.

WHAT IS CONSUMED FROM ``optimization/`` (nothing is reimplemented here)
----------------------------------------------------------------------
* ``workloads.direct_inference_identity`` — the structural identity of the
  request. Pure, no I/O, safe on the hot path.
* ``workloads.resolve_workload(surface="direct_inference", ...)`` — persistence,
  explicit-vs-structural precedence, and the identity levels.
* ``strategy.from_direct_inference_request`` — the baseline Strategy for a
  request with no deployment behind it.

There is deliberately no local copy of workload identity, cost aggregation or
strategy construction: one implementation per concept.

HOW THE ATTEMPT IS PERSISTED
----------------------------
``public.attempts`` now UNIONs two execution surfaces
(``migration_optimization_v6_direct_inference_attempts.sql``): ``workflow_runs``
for runtime, and ``api_request_log`` for direct inference.  ``api_request_log``
ALREADY IS this surface's execution record — it carries status, latency, measured
cost, and, in ``custom_metrics``, the provider, model, token counts and workload
ref — so nothing here copies those.  There is no second execution table.

Exactly two facts cannot be recovered from that log row, and only those are
written by :func:`_persist_attempt`:

  1. the resolved ``workloads.id`` — the log carries a string identity *ref*, and
     a join on it silently breaks if a workload is renamed;
  2. the baseline **strategy fingerprint** — direct-inference identity is derived
     partly from the SYSTEM PROMPT, which ``api_request_log`` deliberately does
     not store, so the fingerprint is permanently unrecoverable from it.

They go to ``direct_inference_attempt_links``, which has no cost, latency, token
or status column on purpose.

ORDERING: this module and the request logger race — on the streaming path this
runs first, on the non-streaming path both are scheduled concurrently — and
neither can wait for the other without adding latency to a customer's request.
The view therefore drives from ``api_request_log`` and LEFT JOINs the link, so an
execution is never invisible merely because attribution failed.

COST is NOT written as a cost event: ``api_request_log.total_cost`` is already
surfaced by the view, and ``attempts.record_cost_event`` now ENFORCES that rule
for every surface rather than only documenting it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SURFACE = "direct_inference"
ATTEMPT_SOURCE = "direct_inference"


@dataclass
class DirectInferenceAttempt:
    """
    One execution of work on the Direct Inference surface.

    Field names follow the ``public.attempts`` view and the ``workloads``
    columns so the domain layer needs no translation table.
    """

    # ── identity ────────────────────────────────────────────────────────────
    attempt_id: str  # the OptiML request id, echoed to the caller
    org_id: str
    occurred_at: str  # ISO-8601 UTC
    surface: str = SURFACE
    attempt_source: str = ATTEMPT_SOURCE

    # ── executor: who actually ran it ──────────────────────────────────────
    executor_kind: str = "model"
    provider: str = ""
    model: str = ""  # provider-native model id
    requested_model: str = ""  # exactly what the caller sent

    # ── request shape, used to derive workload identity and strategy ───────
    system_prompt: Optional[str] = None
    tools: Any = None
    response_format: Any = None
    explicit_workload: Optional[str] = None  # metadata.optiml.workload
    params: dict[str, Any] = field(default_factory=dict)

    # ── filled in by record_direct_inference_attempt ───────────────────────
    workload: dict[str, Any] = field(default_factory=dict)
    workload_id: Optional[str] = None
    strategy: dict[str, Any] = field(default_factory=dict)

    # ── measurement ────────────────────────────────────────────────────────
    duration_ms: Optional[int] = None
    provider_latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    #: Measured spend. None when tokens were unknown OR pricing is estimated —
    #: an estimate is never presented as measured cost.
    inference_cost_usd: Optional[float] = None
    #: The estimate, when there is one. Always paired with cost_estimated=True.
    estimated_cost_usd: Optional[float] = None
    cost_estimated: bool = False
    pricing_source: str = "default"
    tokens_known: bool = False

    # ── outcome of the attempt itself (not a business Outcome) ─────────────
    success: bool = True
    http_status: int = 200
    finish_reason: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # ── customer-supplied context ──────────────────────────────────────────
    end_user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    experiment_tags: list[str] = field(default_factory=list)
    streamed: bool = False
    tool_call_count: int = 0

    def to_domain_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # The prompt text and tool schemas are inputs to identity, not part of
        # the record; identity has already been derived by the time this runs.
        d.pop("system_prompt", None)
        d.pop("tools", None)
        return d

    def summary(self) -> str:
        cost = (
            f"{self.inference_cost_usd:.6f}"
            if self.inference_cost_usd is not None
            else (f"~{self.estimated_cost_usd:.6f}" if self.estimated_cost_usd is not None else "?")
        )
        return (
            f"attempt={self.attempt_id} org={self.org_id} "
            f"workload={self.workload.get('identity_ref')}"
            f"({self.workload.get('identity_level')}) "
            f"executor={self.provider}/{self.model} cost_usd={cost} "
            f"latency_ms={self.duration_ms} status={self.http_status}"
        )


def describe_workload(attempt: DirectInferenceAttempt) -> dict[str, Any]:
    """
    Resolve this request's workload identity WITHOUT touching the database.

    Used by the router so the request log and response carry the workload even
    when the durable resolution below is deferred off the hot path. Delegates to
    ``optimization.workloads`` — there is no second identity implementation.
    """
    from optimization import workloads

    if attempt.explicit_workload:
        key = attempt.explicit_workload.strip()[:200]
        return {
            "identity_level": "explicit",
            "identity_kind": "explicit",
            "identity_ref": key,
            "external_key": key,
            "name": key,
            "surface": SURFACE,
        }

    identity = workloads.direct_inference_identity(
        provider=attempt.provider,
        model=attempt.model,
        messages=(
            [{"role": "system", "content": attempt.system_prompt}]
            if attempt.system_prompt
            else []
        ),
        tools=attempt.tools,
        response_format=attempt.response_format,
    )
    return {
        "identity_level": "structural",
        "identity_kind": "model_endpoint",
        "identity_ref": identity["model_target"],
        "external_key": None,
        "name": identity["name"],
        "surface": SURFACE,
        "components": identity["components"],
    }


def _resolve_workload_row(attempt: DirectInferenceAttempt) -> Optional[dict]:
    """Persist/lookup the workload through the domain layer. Does I/O."""
    from optimization import workloads

    return workloads.resolve_workload(
        attempt.org_id,
        external_key=attempt.explicit_workload or None,
        surface=SURFACE,
        model_target=(
            None if attempt.explicit_workload else attempt.workload.get("identity_ref")
        ),
        name=attempt.workload.get("name"),
    )


def _build_strategy(attempt: DirectInferenceAttempt) -> Any:
    """
    The customer's own configuration, as the baseline to beat.

    ``temperature``, ``max_tokens`` and ``top_p`` ARE passed as genuine strategy
    configuration on this surface.  They are dropped by ``workflow_runtime``, so
    the domain refuses them for workflow graphs — but the OpenAI dialect forwards
    them in the body and the Anthropic translation maps them explicitly, so here
    they really do reach the provider and a change in them is really
    benchmarkable.  Applicability is scoped per surface in
    ``optimization.strategy.SURFACE_APPLICABLE_DIMENSIONS``.

    Returns the ``Strategy`` object; the caller keeps its dict form on the
    attempt and its fingerprint on the link row.
    """
    from optimization import strategy as strategy_mod

    params = attempt.params or {}
    max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")

    return strategy_mod.from_direct_inference_request(
        model=attempt.model,
        provider=attempt.provider,
        system_prompt=attempt.system_prompt,
        temperature=params.get("temperature"),
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        top_p=params.get("top_p"),
        workload_id=attempt.workload_id,
    )


def _persist_attempt(attempt: DirectInferenceAttempt, strategy: Any) -> bool:
    """
    Persist this attempt's attribution.

    Writes the baseline strategy (deduped on fingerprint — one row per distinct
    configuration, NOT one per request) and the narrow link row binding the
    customer-visible attempt id to its workload and strategy.

    Deliberately does NOT write cost, latency, tokens or status: those are on
    ``api_request_log`` and the attempts view reads them from there.  Writing
    them again would create a second source of truth for the numbers the whole
    product rests on.

    Returns True when the link row was written.
    """
    from optimization import attempts as attempts_mod
    from optimization import service

    strategy_id = None
    fingerprint = None
    if strategy is not None:
        fingerprint = strategy.fingerprint()
        row = service.upsert_strategy(
            attempt.org_id,
            strategy,
            workload_id=attempt.workload_id,
            kind="baseline",
            name=f"Direct inference: {attempt.provider}/{attempt.model}",
            description=(
                "The customer's own current configuration on the direct-inference "
                "surface. This is the baseline a candidate must beat."
            ),
        )
        if row:
            strategy_id = str(row.get("id")) if row.get("id") else None

    link = attempts_mod.record_direct_inference_link(
        attempt.org_id,
        attempt_id=attempt.attempt_id,
        workload_id=attempt.workload_id,
        strategy_id=strategy_id,
        strategy_fingerprint=fingerprint,
        occurred_at=attempt.occurred_at,
    )
    return link is not None


def record_direct_inference_attempt(attempt: DirectInferenceAttempt) -> None:
    """
    THE Direct Inference → optimization-domain integration call site.

    Called exactly once per direct-inference request; the streaming and
    non-streaming paths converge here. Does database I/O (workload resolution),
    so callers run it OFF the request hot path.

    Never raises: observability must not be able to fail a customer's
    production request.
    """
    try:
        if not attempt.workload:
            attempt.workload = describe_workload(attempt)

        row = _resolve_workload_row(attempt)
        if row:
            attempt.workload_id = str(row.get("id")) if row.get("id") else None
            attempt.workload.update(
                {
                    "id": attempt.workload_id,
                    "identity_kind": row.get("identity_kind") or attempt.workload.get("identity_kind"),
                    "identity_level": row.get("identity_level") or attempt.workload.get("identity_level"),
                    "name": row.get("name") or attempt.workload.get("name"),
                }
            )

        strategy = _build_strategy(attempt)
        attempt.strategy = strategy.to_dict() if strategy is not None else {}

        if _persist_attempt(attempt, strategy):
            return

        # Attribution could not be written. The execution itself is still on
        # api_request_log and still appears in the attempts view (the view falls
        # back to an identity-ref join), so nothing is lost silently — but the
        # strategy pin is missing, and that is worth a log line.
        logger.warning(
            "direct_inference.attempt.unattributed %s", attempt.summary()
        )
        logger.debug(
            "direct_inference.attempt.full %s",
            json.dumps(attempt.to_domain_dict(), default=str),
        )
    except Exception:
        logger.warning(
            "failed to record direct-inference attempt %s", attempt.attempt_id, exc_info=True
        )
