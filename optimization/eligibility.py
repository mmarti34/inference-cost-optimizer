"""
Candidate ELIGIBILITY PREFLIGHT — the gate between hypothesis and spend.

Candidate generation is HYPOTHESIS generation. Before this module existed, a
generated candidate became a benchmark arm automatically, and the first real
production run paid for that twice:

  * `o1-mini` ran 140 replay cases at a 100% error rate and produced zero usable
    data. The o-series refuses `temperature` and wants `max_completion_tokens`
    instead of `max_tokens`. Nothing asked before dispatch, so the
    incompatibility was discovered through provider errors, 140 times.

  * `GPT-5` was benchmarked under a `cost` objective and measured +321.8%. It
    consumed $0.503 — 69% of the entire run's provider spend.

So a candidate now has to EARN a benchmark arm. Every check below runs against
real data — the org's credentials, `shared/providers.json`, the policy in force,
or the request shape the strategy would actually send — and every exclusion
carries a reason code plus the facts behind it.

THE THREE RULES
---------------
1. NO PROVIDER REQUEST FOR AN INELIGIBLE CANDIDATE. This module runs before the
   replay loop and removes candidates from it. Exclusion is not an arm with
   NULL metrics; it is the absence of an arm.

2. AN EXCLUSION IS AN OPPORTUNITY, NOT A FAILURE. Excluded candidates keep
   structured consideration evidence and stay in the funnel. Nothing here may
   reduce benchmark coverage: coverage is about whether the WORKLOAD reached a
   determination, and a candidate that was never worth dispatching does not
   change that.

3. UNKNOWN NEVER EXCLUDES. A check with no data to run on returns
   `not_assessed`, and `not_assessed` is not `fail`. An inflated funnel with a
   fabricated `incompatible` bucket would be a worse product than no funnel.

VENDOR PRICING IS SCREENING EVIDENCE, NEVER VERIFICATION EVIDENCE — see
`screen_cost_objective`, which is the only place a price is allowed to remove a
candidate, and which is deliberately biased against doing so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from optimization import capabilities as caps_mod
from optimization import domain, executors, strategy as strategy_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check vocabulary
# ---------------------------------------------------------------------------

STATUS_PASS = "pass"
STATUS_ADAPTED = "adapted"
STATUS_FAIL = "fail"
#: The check could not run — no data, or the dimension is not expressible on
#: this surface. Explicitly NOT a failure and explicitly NOT a pass.
STATUS_NOT_ASSESSED = "not_assessed"

DIM_PROVIDER_CONFIGURED = "provider_configured"
DIM_MODEL_AVAILABLE = "model_available"
DIM_SURFACE_COMPATIBLE = "surface_compatible"
DIM_INPUT_MODALITY = "input_modality"
DIM_OUTPUT_MODALITY = "output_modality"
DIM_CONTEXT_WINDOW = "context_window"
DIM_SYSTEM_MESSAGE = "system_message_support"
DIM_TOOL_CALLING = "tool_calling"
DIM_STRUCTURED_OUTPUT = "structured_output"
DIM_STREAMING = "streaming"
DIM_REQUEST_PARAMS = "request_parameters"
DIM_POLICY = "policy_restrictions"
DIM_PRICING = "pricing_provenance"
DIM_OBJECTIVE = "objective_feasibility"

#: Every dimension a candidate is asked about, in evaluation order. Emitted in
#: full on every candidate — including the ones that came back `not_assessed` —
#: so a reader can tell "we checked and it was fine" from "we cannot check this
#: yet". The second is the honest description of most modality checks today.
ELIGIBILITY_DIMENSIONS = (
    DIM_PROVIDER_CONFIGURED,
    DIM_MODEL_AVAILABLE,
    DIM_SURFACE_COMPATIBLE,
    DIM_POLICY,
    DIM_REQUEST_PARAMS,
    DIM_SYSTEM_MESSAGE,
    DIM_TOOL_CALLING,
    DIM_STRUCTURED_OUTPUT,
    DIM_STREAMING,
    DIM_INPUT_MODALITY,
    DIM_OUTPUT_MODALITY,
    DIM_CONTEXT_WINDOW,
    DIM_PRICING,
    DIM_OBJECTIVE,
)


def check(dimension: str, status: str, *, code: Optional[str] = None, **facts) -> dict:
    """One structured check result. Codes and facts; never a sentence."""
    if code is not None and code not in domain.REASON_CODES:
        raise ValueError(f"Unknown reason code '{code}' from eligibility check.")
    out: dict[str, Any] = {"dimension": dimension, "status": status, "code": code}
    out.update({k: v for k, v in facts.items() if v is not None})
    return out


@dataclass
class CandidateEligibility:
    """Structured eligibility evidence for exactly one candidate."""

    label: Optional[str]
    generator: Optional[str]
    strategy_fingerprint: Optional[str]
    executor_refs: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    adaptations: list[dict] = field(default_factory=list)
    #: Facts worth persisting that are NOT exclusions (e.g. "this family bills
    #: reasoning tokens"). Kept apart from checks so a note can never be read as
    #: a verdict.
    notes: list[dict] = field(default_factory=list)
    eligible: bool = True
    code: Optional[str] = None
    disposition: Optional[str] = None

    @property
    def failed(self) -> Optional[dict]:
        return next((c for c in self.checks if c["status"] == STATUS_FAIL), None)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "generator": self.generator,
            "strategy_fingerprint": self.strategy_fingerprint,
            "executor_refs": list(self.executor_refs),
            "eligible": self.eligible,
            "code": self.code,
            "disposition": self.disposition,
            "checks": list(self.checks),
            "adaptations": list(self.adaptations),
            "notes": list(self.notes),
            "dimensions_assessed": sorted(
                c["dimension"] for c in self.checks
                if c["status"] != STATUS_NOT_ASSESSED
            ),
            "dimensions_not_assessed": sorted(
                c["dimension"] for c in self.checks
                if c["status"] == STATUS_NOT_ASSESSED
            ),
        }


@dataclass
class PreflightResult:
    """What survived, what did not, and why — in a shape the funnel can read."""

    eligible: list = field(default_factory=list)
    #: Exclusion records in `generate_candidates`' `dropped` shape, so
    #: `benchmark._dispositions` needs no special case for them.
    excluded: list[dict] = field(default_factory=list)
    #: TIER 2, same contract as the existing provider_not_configured items.
    opportunities: list[dict] = field(default_factory=list)
    #: Full eligibility evidence for EVERY candidate, eligible ones included.
    evaluations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "considered": len(self.evaluations),
            "eligible": len(self.eligible),
            "excluded": len(self.excluded),
            "opportunities": len(self.opportunities),
            "evaluations": list(self.evaluations),
        }


# ---------------------------------------------------------------------------
# Objective-aware screening — the ONE place a price sheet may remove an arm
# ---------------------------------------------------------------------------
#
# THE ASYMMETRY, stated once and honoured everywhere below:
#
#   A list price may be used to say "this is not worth paying to measure".
#   A list price may NEVER be used to say "this saves money".
#
# Those are not two ends of one scale. The second is a claim about the
# customer's workload and requires measurement; the first is a claim about how
# to spend a benchmark budget and requires only that we not be obviously wrong.
#
# The live run is the proof that list price does not predict measured cost:
#
#   gpt-5-mini   list blended ~84% below gpt-4o   ->   MEASURED  -41.8%
#   gpt-5        list blended ~21% below gpt-4o   ->   MEASURED +321.8%
#
# In both cases the measured result was WORSE than the price sheet implied,
# because token behaviour differs between models — reasoning models in
# particular bill reasoning output the price sheet says nothing about. Note what
# that means for screening: it is evidence that the price sheet is OPTIMISTIC,
# and an optimistic input to a screen that removes candidates is the dangerous
# direction. Hence the safeguards:
#
#   * The screen only ever fires on the OPTIMISTIC bound for the candidate, not
#     on the point estimate: the candidate is allowed to be substantially more
#     token-efficient than the baseline and must STILL fail materiality.
#   * When the input/output ratio is assumed rather than measured, the
#     allowance widens further, because the input itself is weaker.
#   * When either price is unknown, the screen does not fire at all.
#   * Screening is NOT applied to `emits_billed_reasoning_tokens`. It is
#     tempting — it is exactly what made GPT-5 expensive — but gpt-5-mini is in
#     the same family and is the only verified win this product has ever
#     produced. A screen that would have deleted the win is not a screen, it is
#     a bug. The fact is recorded as a NOTE and nothing acts on it.

#: How much more token-efficient than the baseline a candidate is ALLOWED to be
#: before the screen may fire. 0.5 means "assume it could do the job on half the
#: billed tokens". Deliberately generous and deliberately one-directional: it is
#: only ever used to keep a candidate IN, never to project a saving.
OPTIMISTIC_TOKEN_FACTOR_MEASURED_RATIO = 0.5

#: Weaker input, wider allowance. Today no code path supplies a measured
#: input/output ratio (`observed_production_traffic` does not compute one), so
#: this is the value in force in production. Said out loud rather than hidden in
#: a default argument.
OPTIMISTIC_TOKEN_FACTOR_ASSUMED_RATIO = 0.35

#: The ratio assumed when nothing measured one. Same constant the candidate
#: generator uses, so screening and generation cannot disagree.
DEFAULT_IO_RATIO = 3.0


def _measured_io_ratio(history: Optional[dict]) -> Optional[float]:
    traffic = (history or {}).get("traffic") or {}
    try:
        ratio = traffic.get("io_token_ratio")
        return float(ratio) if ratio else None
    except (TypeError, ValueError):
        return None


def _monthly_baseline_spend(history: Optional[dict]) -> Optional[float]:
    """Measured monthly spend on this workload, or None. Never assumed."""
    traffic = (history or {}).get("traffic") or {}
    total = traffic.get("total_cost_usd")
    window = (history or {}).get("lookback_days")
    if total is None or not window:
        return None
    try:
        return float(total) * 30.0 / float(window)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _cost_thresholds(materiality: Optional[dict]) -> tuple[Optional[float], Optional[float]]:
    """(relative threshold as a ratio, absolute threshold in USD/month)."""
    relative = absolute = None
    for t in ((materiality or {}).get("thresholds") or []):
        if t.get("metric") != "cost":
            continue
        if t.get("comparator") == "relative_decrease_at_least":
            relative = float(t.get("value")) if t.get("value") is not None else None
        elif t.get("comparator") == "absolute_decrease_at_least":
            absolute = float(t.get("value")) if t.get("value") is not None else None
    return relative, absolute


def screen_cost_objective(
    *,
    baseline_ref: dict,
    candidate_ref: dict,
    materiality: Optional[dict],
    history: Optional[dict],
) -> dict:
    """
    Is there a PLAUSIBLE path from known pricing to the cost materiality
    threshold? Returns a check result; only `STATUS_FAIL` removes an arm.

    Screening, not verification. The output never becomes a saving, is never
    written to a measured field, and is not a claim that the candidate IS more
    expensive — only that known information gives it no route to being
    materially cheaper.
    """
    measured_ratio = _measured_io_ratio(history)
    io_ratio = measured_ratio or DEFAULT_IO_RATIO
    ratio_source = "measured" if measured_ratio else "assumed_default"
    factor = (
        OPTIMISTIC_TOKEN_FACTOR_MEASURED_RATIO if measured_ratio
        else OPTIMISTIC_TOKEN_FACTOR_ASSUMED_RATIO
    )

    # A price that came from `utils.pricing.DEFAULT_PRICING` is a GUESS, and
    # `blended_vendor_price` cannot tell the caller which it got — it returns a
    # number either way. Screening on a guessed price is the manufactured-saving
    # bug pointed the other way: it would delete a candidate on the strength of
    # a fallback constant. So provenance is resolved first and an estimated
    # price is treated as no price at all.
    provenance = executors.pricing_provenance([baseline_ref, candidate_ref])
    facts_pricing_basis = provenance["basis"]

    base_price = executors.blended_vendor_price(
        baseline_ref.get("vendor") or "", baseline_ref.get("external_id") or "",
        input_output_ratio=io_ratio,
    )
    cand_price = executors.blended_vendor_price(
        candidate_ref.get("vendor") or "", candidate_ref.get("external_id") or "",
        input_output_ratio=io_ratio,
    )
    if facts_pricing_basis != executors.COST_BASIS_MEASURED:
        base_price = cand_price = None

    facts = {
        "objective": "cost",
        "baseline_model": baseline_ref.get("external_id"),
        "candidate_model": candidate_ref.get("external_id"),
        "baseline_blended_price_per_1k": base_price,
        "candidate_blended_price_per_1k": cand_price,
        "io_ratio": io_ratio,
        "io_ratio_source": ratio_source,
        "optimistic_token_factor": factor,
        "price_source": "vendor_list_price:shared/providers.json",
        "pricing_basis": facts_pricing_basis,
        "estimated_models": provenance["estimated_models"] or None,
        "evidence_class": "screening_only",
    }

    if base_price is None or cand_price is None or base_price <= 0:
        # One side has no known price. A screen needs both, and guessing one
        # would be the manufactured-saving bug in reverse.
        return check(
            DIM_OBJECTIVE, STATUS_NOT_ASSESSED,
            reason="pricing_incomplete_for_screen", **facts,
        )

    list_relative = (base_price - cand_price) / base_price
    # The OPTIMISTIC bound: the candidate is allowed to bill only `factor` of
    # the tokens the baseline bills, and must still miss materiality.
    best_case_relative = 1.0 - (cand_price / base_price) * factor
    facts["list_price_relative_delta"] = round(list_relative, 6)
    facts["best_case_relative_saving"] = round(best_case_relative, 6)

    relative_threshold, absolute_threshold = _cost_thresholds(materiality)
    facts["materiality_relative_threshold"] = relative_threshold
    facts["materiality_absolute_threshold_usd_per_month"] = absolute_threshold

    if relative_threshold is None:
        return check(
            DIM_OBJECTIVE, STATUS_NOT_ASSESSED,
            reason="no_relative_cost_threshold_declared", **facts,
        )

    if best_case_relative >= relative_threshold:
        return check(DIM_OBJECTIVE, STATUS_PASS, **facts)

    # The relative route is closed. Before screening, check the ABSOLUTE route:
    # a policy combining thresholds with `any` is satisfied by a large enough
    # dollar saving at a small enough percentage, so a sub-threshold percentage
    # is not on its own a dead end.
    combine = (materiality or {}).get("combine") or "any"
    if absolute_threshold is not None and combine == "any":
        monthly = _monthly_baseline_spend(history)
        facts["measured_monthly_baseline_spend_usd"] = monthly
        if monthly is None:
            # Volume is unmeasured, so the absolute route cannot be ruled out —
            # UNLESS the candidate cannot save anything at all at any volume.
            if best_case_relative > 0:
                return check(
                    DIM_OBJECTIVE, STATUS_NOT_ASSESSED,
                    reason="absolute_threshold_unevaluable_without_measured_volume",
                    **facts,
                )
        else:
            best_case_absolute = best_case_relative * monthly
            facts["best_case_absolute_saving_usd_per_month"] = round(best_case_absolute, 6)
            if best_case_absolute >= absolute_threshold:
                return check(DIM_OBJECTIVE, STATUS_PASS, **facts)

    return check(
        DIM_OBJECTIVE, STATUS_FAIL, code="economically_dominated",
        detail=(
            "Known list pricing leaves no path to the cost materiality "
            "threshold even allowing the candidate to bill a fraction "
            f"({factor}) of the baseline's tokens."
        ),
        **facts,
    )


# ---------------------------------------------------------------------------
# The per-candidate checks
# ---------------------------------------------------------------------------

def _known_vendors(catalog_index: dict) -> set:
    """Vendors `shared/providers.json` describes well enough to dispatch to."""
    return {vendor for (vendor, _model) in (catalog_index or {})}


def _model_steps(strategy) -> list:
    return [
        s for s in (getattr(strategy, "steps", None) or [])
        if getattr(s, "executor_type", "model") == "model"
        and (getattr(s, "executor_ref", None) or {}).get("external_id")
    ]


def _refs(strategy) -> list[dict]:
    return [
        dict(s.executor_ref) for s in (getattr(strategy, "steps", None) or [])
        if getattr(s, "executor_ref", None)
    ]


def _vendors(strategy) -> set[str]:
    out = set()
    for ref in _refs(strategy):
        vendor = str(ref.get("vendor") or "").strip().lower()
        # 'optiml' names an internal step target, not an external provider.
        if vendor and vendor != "optiml":
            out.add(vendor)
    return out


def _check_provider_configured(strategy, configured: Optional[set]) -> dict:
    needed = sorted(_vendors(strategy))
    if not configured:
        # Empty set means the credential lookup did not run or failed. Refusing
        # every candidate because a read errored would be worse than letting the
        # arms report their own failures — same rule candidates.py already holds.
        return check(
            DIM_PROVIDER_CONFIGURED, STATUS_NOT_ASSESSED,
            reason="configured_providers_unknown", providers_required=needed,
        )
    missing = sorted(set(needed) - set(configured))
    if missing:
        return check(
            DIM_PROVIDER_CONFIGURED, STATUS_FAIL, code="provider_not_configured",
            providers_required=needed, providers_missing=missing,
            providers_configured=sorted(configured),
        )
    return check(
        DIM_PROVIDER_CONFIGURED, STATUS_PASS, providers_required=needed,
    )


def _check_model_available(profiles: list, known_vendors: set) -> dict:
    """
    Can a request be CONSTRUCTED for this model at all?

    The only availability fact OptiML actually holds is membership of
    `shared/providers.json`, and that file is a PRICE SHEET, not an availability
    oracle. A model absent from it is routinely still dispatchable — the routers
    forward whatever model id they are given, which is how custom and
    newly-released ids work today. So absence means UNKNOWN, and unknown does
    not exclude.

    What genuinely blocks dispatch is an unknown VENDOR: without a provider
    entry there is no api_base and no api_style, so no request exists to send.
    That is the only failing branch, and it is a real one.
    """
    unknown_vendor = [
        p for p in profiles
        if str(p.vendor or "").strip().lower() not in known_vendors
    ]
    if unknown_vendor:
        return check(
            DIM_MODEL_AVAILABLE, STATUS_FAIL, code="model_not_available",
            models=[f"{p.vendor}/{p.model_id}" for p in unknown_vendor],
            reason="vendor_absent_from_provider_registry",
            catalog="shared/providers.json",
        )
    uncatalogued = [p for p in profiles if not p.known_to_catalog]
    if uncatalogued:
        return check(
            DIM_MODEL_AVAILABLE, STATUS_NOT_ASSESSED,
            reason="model_absent_from_catalog_but_vendor_dispatchable",
            models=[f"{p.vendor}/{p.model_id}" for p in uncatalogued],
            catalog="shared/providers.json",
        )
    return check(
        DIM_MODEL_AVAILABLE, STATUS_PASS,
        models=[f"{p.vendor}/{p.model_id}" for p in profiles],
        catalog="shared/providers.json",
    )


def _check_surface(candidate) -> dict:
    surface = getattr(candidate.strategy, "surface", strategy_mod.SURFACE_RUNTIME)
    unapplicable = strategy_mod.unapplicable_dimensions(
        list(candidate.dimensions or []), surface
    )
    if unapplicable:
        return check(
            DIM_SURFACE_COMPATIBLE, STATUS_FAIL, code="strategy_not_applicable",
            surface=surface, dimensions=sorted(unapplicable),
            detail=unapplicable,
        )
    return check(
        DIM_SURFACE_COMPATIBLE, STATUS_PASS, surface=surface,
        dimensions=sorted(candidate.dimensions or []) or None,
    )


def _check_policy(strategy, policy: Optional[dict]) -> dict:
    from optimization import policies as policies_mod

    constraints = policies_mod.constraints_of(policy)
    vendors = _vendors(strategy)
    if not constraints:
        return check(DIM_POLICY, STATUS_NOT_ASSESSED, reason="no_policy_constraints")

    allowed = constraints.get("allowed_vendors")
    blocked = constraints.get("blocked_vendors")
    if not (isinstance(allowed, list) and allowed) and not (
        isinstance(blocked, list) and blocked
    ):
        return check(DIM_POLICY, STATUS_NOT_ASSESSED, reason="no_vendor_constraints")

    if not vendors:
        return check(
            DIM_POLICY, STATUS_NOT_ASSESSED, reason="vendors_undeterminable",
        )

    if isinstance(allowed, list) and allowed:
        allowed_set = {str(v).strip().lower() for v in allowed}
        bad = sorted(vendors - allowed_set)
        if bad:
            return check(
                DIM_POLICY, STATUS_FAIL, code="provider_not_permitted",
                constraint="allowed_vendors", required=sorted(allowed_set),
                observed=bad, policy_version=(policy or {}).get("version"),
            )

    if isinstance(blocked, list) and blocked:
        blocked_set = {str(v).strip().lower() for v in blocked}
        bad = sorted(vendors & blocked_set)
        if bad:
            return check(
                DIM_POLICY, STATUS_FAIL, code="provider_not_permitted",
                constraint="blocked_vendors", required=sorted(blocked_set),
                observed=bad, policy_version=(policy or {}).get("version"),
            )

    return check(DIM_POLICY, STATUS_PASS, vendors=sorted(vendors))


def _check_request_params(candidate, profiles_by_step: dict) -> tuple[dict, list[dict]]:
    """
    THE o1-mini CHECK, expressed structurally.

    Model/runtime request differences are represented explicitly here: the
    request shape the strategy would send is compared against the family's
    DECLARED parameter rules, and the answer is known before dispatch. A
    declared, lossless adapter (`max_tokens` -> `max_completion_tokens`) makes
    the candidate executable and is recorded. A rejected parameter with no
    lossless adapter (`temperature` on the o-series) makes it ineligible.
    """
    blockers: list[dict] = []
    adaptations: list[dict] = []
    unknown_params: list[str] = []

    for step in _model_steps(candidate.strategy):
        profile = profiles_by_step.get(step.step_id)
        if profile is None:
            continue
        request = caps_mod.request_params_of_step(step)
        result = caps_mod.adapt_request(profile, request)
        for b in result.blockers:
            blockers.append({**b, "step_id": step.step_id})
        for a in result.adaptations:
            adaptations.append({**a, "step_id": step.step_id})
        unknown_params.extend(result.unknown_params)

    if blockers:
        return check(
            DIM_REQUEST_PARAMS, STATUS_FAIL, code="request_shape_incompatible",
            blockers=blockers,
            detail=(
                "Determined from declared family capabilities before dispatch. "
                "No provider request was made."
            ),
        ), adaptations

    if adaptations:
        return check(
            DIM_REQUEST_PARAMS, STATUS_ADAPTED, code="request_adapted",
            adaptations=adaptations,
            undeclared_params=sorted(set(unknown_params)) or None,
        ), adaptations

    return check(
        DIM_REQUEST_PARAMS, STATUS_PASS,
        undeclared_params=sorted(set(unknown_params)) or None,
    ), adaptations


#: What a SURFACE unconditionally puts in every request it builds, regardless of
#: what the strategy configures.
#
# This is the other half of "represent model/runtime request differences
# explicitly". A request has two authors: the customer's configuration, and the
# execution surface itself. The runtime surface writes a system-role message on
# every prompt it sends — `workflow_runtime._execute_model_node` calls
# `openai_router.handle_prompt`, which hardcodes
# `{"role": "system", "content": ...}` before the user turn. The customer never
# asked for it and cannot see it in their graph, but the provider does.
#
# THAT is what actually killed the o1-mini arm. The o-series does not accept a
# system-role message, so every one of the 140 replay cases was refused before
# it could produce a single usable datum. Writing the surface's own contribution
# down here is what makes it knowable before dispatch instead of 140 times
# afterwards.
SURFACE_REQUEST_REQUIREMENTS: dict[str, dict] = {
    strategy_mod.SURFACE_RUNTIME: {
        "system_message": {
            "required": True,
            "source": (
                "workflow_runtime._execute_model_node -> "
                "routers.openai_router.handle_prompt, which prepends a "
                "system-role message to every runtime prompt."
            ),
        },
    },
    # On direct inference the customer's own request is the whole request, so
    # every requirement is read from the strategy config below. The surface adds
    # nothing of its own.
    strategy_mod.SURFACE_DIRECT_INFERENCE: {},
}

#: Which strategy config key expresses a requirement for each capability.
CAPABILITY_CONFIG_KEY = {
    "system_message": "system_instructions",
    "tool_calling": "tools",
    "structured_output": "response_format",
    "streaming": "stream",
}


def _requirement(candidate, capability: str) -> tuple[bool, Optional[str]]:
    """
    Does this (surface, request) pair actually need the capability, and why?

    Returns (required, source). Only a genuine requirement can turn a declared
    `not_supported` into an exclusion — a model that cannot stream is perfectly
    eligible for a workload that never streams.
    """
    surface = getattr(candidate.strategy, "surface", strategy_mod.SURFACE_RUNTIME)
    surface_rule = (SURFACE_REQUEST_REQUIREMENTS.get(surface) or {}).get(capability)
    if surface_rule and surface_rule.get("required"):
        return True, surface_rule.get("source")

    config_key = CAPABILITY_CONFIG_KEY.get(capability)
    if config_key:
        for step in _model_steps(candidate.strategy):
            if (getattr(step, "config", None) or {}).get(config_key) is not None:
                return True, f"strategy config sets '{config_key}'"
    return False, None


def _check_declared_capability(
    dimension: str,
    capability: str,
    requirement: tuple,
    profiles: list,
) -> dict:
    """
    A capability check that can only fail on an EXPLICIT `not_supported`.

    `unknown` is the answer for almost every model today, and it returns
    `not_assessed`. That is the honest reading, and it keeps the funnel from
    growing a bucket nothing actually measured.
    """
    required, requirement_source = requirement
    if not required:
        return check(
            dimension, STATUS_NOT_ASSESSED, reason="not_required_by_request",
            capability=capability,
        )

    refusing = [
        p for p in profiles
        if p.capability(capability) == caps_mod.SUPPORT_NO
    ]
    if refusing:
        return check(
            dimension, STATUS_FAIL, code="required_capability_missing",
            capability=capability,
            models=[f"{p.vendor}/{p.model_id}" for p in refusing],
            declared_by=sorted({f for p in refusing for f in p.families}) or None,
            required_because=requirement_source,
        )
    if all(p.capability(capability) == caps_mod.SUPPORT_UNKNOWN for p in profiles):
        return check(
            dimension, STATUS_NOT_ASSESSED, reason="capability_undeclared",
            capability=capability, required_because=requirement_source,
        )
    return check(
        dimension, STATUS_PASS, capability=capability,
        required_because=requirement_source,
    )


def _check_context_window(candidate, profiles: list, history: Optional[dict]) -> dict:
    """
    Published context window vs the workload's MEASURED input requirement.

    Nothing measures the second today — `observed_production_traffic` records
    cost, latency and errors, not token counts — so this comes back
    `not_assessed` with the reason named. It is wired rather than faked so that
    the day a token measurement exists, one function supplies it.
    """
    required = (history or {}).get("max_observed_input_tokens")
    if required is None:
        return check(
            DIM_CONTEXT_WINDOW, STATUS_NOT_ASSESSED,
            reason="workload_input_tokens_not_measured",
            windows={f"{p.vendor}/{p.model_id}": p.context_window for p in profiles},
        )
    too_small = [
        p for p in profiles
        if p.context_window is not None and p.context_window < float(required)
    ]
    if too_small:
        return check(
            DIM_CONTEXT_WINDOW, STATUS_FAIL, code="context_window_insufficient",
            required_tokens=required,
            models={f"{p.vendor}/{p.model_id}": p.context_window for p in too_small},
        )
    return check(DIM_CONTEXT_WINDOW, STATUS_PASS, required_tokens=required)


#: Should an arm with no known vendor price be refused a benchmark under a cost
#: objective? OFF, deliberately.
#:
#: The temptation is real — its cost can only come from
#: `utils.pricing.DEFAULT_PRICING`, a guess. But the loop ALREADY handles that
#: correctly downstream: `_execute_arm` leaves `mean_cost_usd` NULL, keeps the
#: guess in a separate clearly-labelled field, and the verdict refuses to bank
#: the fictitious saving with `cost_pricing_estimated`. Excluding the arm would
#: throw away its real quality and latency measurements to prevent a bug that is
#: already prevented. The switch exists so the choice is visible rather than
#: absent; nothing in the benchmark path turns it on.
EXCLUDE_UNPRICED_UNDER_COST_OBJECTIVE = False


def _check_pricing(candidate, objective: str) -> dict:
    """
    Does a REAL vendor price exist for every model this arm would run?

    A FACT, recorded on every candidate. Not a gate — see the constant above.
    """
    provenance = executors.pricing_provenance(_refs(candidate.strategy))
    if provenance["basis"] == executors.COST_BASIS_MEASURED:
        return check(DIM_PRICING, STATUS_PASS, basis=provenance["basis"])
    if objective in ("cost", "balanced") and EXCLUDE_UNPRICED_UNDER_COST_OBJECTIVE:
        return check(
            DIM_PRICING, STATUS_FAIL, code="pricing_unknown",
            basis=provenance["basis"],
            estimated_models=provenance["estimated_models"],
        )
    return check(
        DIM_PRICING, STATUS_NOT_ASSESSED, reason="pricing_estimated",
        basis=provenance["basis"], estimated_models=provenance["estimated_models"],
        detail=(
            "Cost for this arm can only be a guess. The arm still runs: the "
            "benchmark records the guess separately and refuses to report it as "
            "a measured cost."
        ),
    )


def _check_objective(candidate, baseline, objective: str, materiality, history) -> dict:
    if objective not in ("cost", "balanced"):
        return check(
            DIM_OBJECTIVE, STATUS_NOT_ASSESSED,
            reason="no_screening_rule_for_objective", objective=objective,
        )

    base_steps = _model_steps(baseline)
    cand_steps = {s.step_id: s for s in _model_steps(candidate.strategy)}
    # Screen only the clean single-model-substitution case. A multi-step
    # strategy's blended cost is not the sum of two list prices, and screening
    # on a number that does not describe the arm would be the fabricated-figure
    # bug in a new costume.
    if len(base_steps) != 1 or len(cand_steps) != 1:
        return check(
            DIM_OBJECTIVE, STATUS_NOT_ASSESSED,
            reason="multi_step_strategy_not_screenable",
            baseline_model_steps=len(base_steps),
            candidate_model_steps=len(cand_steps),
        )

    base_step = base_steps[0]
    cand_step = cand_steps.get(base_step.step_id) or list(cand_steps.values())[0]
    return screen_cost_objective(
        baseline_ref=base_step.executor_ref or {},
        candidate_ref=cand_step.executor_ref or {},
        materiality=materiality,
        history=history,
    )


# ---------------------------------------------------------------------------
# The funnel-facing entry point
# ---------------------------------------------------------------------------

#: Which funnel stage each exclusion code exits at. Every code here is emitted
#: by a check that ran against real data.
CODE_TO_DISPOSITION = {
    "provider_not_configured": domain.DISPOSITION_PROVIDER_NOT_CONFIGURED,
    "provider_not_permitted": domain.DISPOSITION_POLICY_BLOCKED,
    "policy_blocked": domain.DISPOSITION_POLICY_BLOCKED,
    "economically_dominated": domain.DISPOSITION_ECONOMICALLY_DOMINATED,
    "model_not_available": domain.DISPOSITION_INCOMPATIBLE,
    "strategy_not_applicable": domain.DISPOSITION_INCOMPATIBLE,
    "request_shape_incompatible": domain.DISPOSITION_INCOMPATIBLE,
    "required_capability_missing": domain.DISPOSITION_INCOMPATIBLE,
    "context_window_insufficient": domain.DISPOSITION_INCOMPATIBLE,
    "pricing_unknown": domain.DISPOSITION_INCOMPATIBLE,
}


def evaluate_candidate(
    candidate,
    *,
    baseline,
    objective: str = "cost",
    policy: Optional[dict] = None,
    materiality: Optional[dict] = None,
    history: Optional[dict] = None,
    configured_providers: Optional[set] = None,
    catalog_index: Optional[dict] = None,
) -> CandidateEligibility:
    """
    Run every eligibility dimension against one candidate. PURE — no I/O beyond
    reading the vendor catalog, and no provider request under any branch.

    Checks run in order and the FIRST failure decides the disposition, but every
    remaining dimension is still recorded (as `not_assessed` where a prior
    failure makes it moot) so the evidence is complete rather than truncated at
    the first bad news.
    """
    profiles_by_step: dict[str, caps_mod.ModelProfile] = {}
    index = catalog_index if catalog_index is not None else caps_mod._catalog_index()
    for step in _model_steps(candidate.strategy):
        ref = step.executor_ref or {}
        profiles_by_step[step.step_id] = caps_mod.resolve_profile(
            ref.get("vendor") or "", ref.get("external_id") or "",
            catalog_index=index,
        )
    profiles = list(profiles_by_step.values())

    ev = CandidateEligibility(
        label=getattr(candidate, "title", None),
        generator=getattr(candidate, "generator", None),
        strategy_fingerprint=getattr(candidate, "fingerprint", None),
        executor_refs=_refs(candidate.strategy),
    )

    # Facts that are not verdicts. `emits_billed_reasoning_tokens` is the
    # clearest example: it explains GPT-5's +321.8%, and it deliberately does
    # NOT screen, because gpt-5-mini shares the declaration and is the only
    # verified win this product has produced.
    for p in profiles:
        if p.capability("emits_billed_reasoning_tokens") == caps_mod.SUPPORT_YES:
            ev.notes.append({
                "code": "cost_risk_billed_reasoning_tokens",
                "model": f"{p.vendor}/{p.model_id}",
                "families": list(p.families),
                "detail": (
                    "This family bills reasoning output the price sheet does not "
                    "describe, so its measured cost may exceed its list price. "
                    "RECORDED ONLY — this never excludes a candidate."
                ),
            })

    params_check, adaptations = _check_request_params(candidate, profiles_by_step)
    ev.adaptations = adaptations

    ordered = [
        _check_provider_configured(candidate.strategy, configured_providers),
        _check_model_available(profiles, _known_vendors(index)),
        _check_surface(candidate),
        _check_policy(candidate.strategy, policy),
        params_check,
        _check_declared_capability(
            DIM_SYSTEM_MESSAGE, "system_message",
            _requirement(candidate, "system_message"), profiles,
        ),
        _check_declared_capability(
            DIM_TOOL_CALLING, "tool_calling",
            _requirement(candidate, "tool_calling"), profiles,
        ),
        _check_declared_capability(
            DIM_STRUCTURED_OUTPUT, "structured_output",
            _requirement(candidate, "structured_output"), profiles,
        ),
        _check_declared_capability(
            DIM_STREAMING, "streaming",
            _requirement(candidate, "streaming"), profiles,
        ),
        # Modality is not expressible today: nothing records what a workload's
        # inputs or outputs ARE, and providers.json declares no modality. Both
        # come back `not_assessed` with the reason named, rather than a
        # fabricated pass.
        check(DIM_INPUT_MODALITY, STATUS_NOT_ASSESSED,
              reason="workload_modality_not_recorded"),
        check(DIM_OUTPUT_MODALITY, STATUS_NOT_ASSESSED,
              reason="workload_modality_not_recorded"),
        _check_context_window(candidate, profiles, history),
        _check_pricing(candidate, objective),
        _check_objective(candidate, baseline, objective, materiality, history),
    ]
    ev.checks = ordered

    failed = next((c for c in ordered if c["status"] == STATUS_FAIL), None)
    if failed is not None:
        ev.eligible = False
        ev.code = failed["code"]
        ev.disposition = CODE_TO_DISPOSITION.get(
            failed["code"], domain.DISPOSITION_INCOMPATIBLE
        )
    return ev


def preflight(
    candidates: list,
    *,
    baseline,
    objective: str = "cost",
    policy: Optional[dict] = None,
    materiality: Optional[dict] = None,
    history: Optional[dict] = None,
    configured_providers: Optional[set] = None,
) -> PreflightResult:
    """
    THE GATE. Returns only candidates that may be dispatched, plus complete
    structured evidence for every one that may not.

    A candidate excluded here never reaches `_execute_arm`, so no external
    provider request occurs for it. That is the whole point of the module and
    the property `test_eligibility.py` proves by counting runtime calls.
    """
    result = PreflightResult()
    index = caps_mod._catalog_index()

    for cand in candidates or []:
        try:
            ev = evaluate_candidate(
                cand, baseline=baseline, objective=objective, policy=policy,
                materiality=materiality, history=history,
                configured_providers=configured_providers, catalog_index=index,
            )
        except Exception as exc:  # pragma: no cover - defensive
            # A preflight bug must not silently let an unchecked candidate
            # through to spend money. It is recorded and excluded.
            logger.warning(
                "eligibility preflight raised for %s: %s",
                getattr(cand, "title", "?"), type(exc).__name__,
            )
            result.excluded.append({
                "generator": getattr(cand, "generator", None),
                "title": getattr(cand, "title", None),
                "code": "generator_error",
                "detail": type(exc).__name__,
                "stage": "eligibility_preflight",
            })
            continue

        evidence = ev.to_dict()
        result.evaluations.append(evidence)

        if ev.eligible:
            # Attached to the candidate itself so the arm it becomes carries its
            # own preflight record. "We checked every dimension and it passed"
            # belongs in the audit trail just as much as an exclusion does.
            try:
                cand.eligibility = evidence
            except Exception:  # pragma: no cover - non-dataclass candidate
                pass
            result.eligible.append(cand)
            continue

        if ev.code == "provider_not_configured":
            # SAME TIER-2 TREATMENT the generator already gives: not
            # benchmarked, not forgotten. Extended here so a caller-supplied
            # candidate for an unconnected provider gets it too.
            missing = next(
                (c.get("providers_missing") for c in ev.checks
                 if c["dimension"] == DIM_PROVIDER_CONFIGURED),
                [],
            )
            result.opportunities.append({
                "generator": ev.generator,
                "label": ev.label,
                "tier": domain.TIER_OPPORTUNITY,
                "code": "provider_not_configured",
                "providers": list(missing or []),
                "dimensions": list(getattr(cand, "dimensions", None) or []),
                "strategy_fingerprint": ev.strategy_fingerprint,
                "executor_refs": list(ev.executor_refs),
                "evidence_source": "none",
                "evidence_strength": domain.evidence_strength("none"),
                "verified": False,
                "next_action": "connect_provider",
                "projected_savings_usd": getattr(cand, "projected_savings_usd", None),
                "projection_basis": getattr(cand, "projection_basis", None),
                "measured_quality": None,
                "measured_cost_usd": None,
                "eligibility": ev.to_dict(),
                "detail": (
                    "No provider credential is configured for this organization, "
                    "so this candidate was NOT executed and nothing about it has "
                    "been measured."
                ),
            })
            continue

        failed = ev.failed or {}
        result.excluded.append({
            "generator": ev.generator,
            "title": ev.label,
            "code": ev.code,
            "stage": "eligibility_preflight",
            "disposition": ev.disposition,
            "dimension": failed.get("dimension"),
            "facts": {k: v for k, v in failed.items()
                      if k not in ("dimension", "status", "code")},
            "eligibility": ev.to_dict(),
        })

    return result
