"""
Shared vocabulary for the optimization layer. Pure functions, no I/O.

Everything here exists to stop two specific failure modes:

  1. Treating an assertion as a measurement.  `rationale` is prose; only an
     optimization_benchmarks row or a production outcome is evidence.
  2. Treating weak evidence as strong.  "We watched A happen" is not "we ran A
     and B on the same inputs", which is not "B served real traffic".
     EVIDENCE_STRENGTH and OUTCOME_PROVENANCE_RANK encode that ordering, and
     compute_evidence_maturity() refuses to produce a strong-looking number
     from a thin sample or a weak signal.

Nothing in this module ever invents a value. Every function that cannot compute
its result returns None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Surfaces and executors
# ---------------------------------------------------------------------------

SURFACES = ("runtime", "direct_inference", "workforce")

#: How a workload came to be identified. Customers are NEVER required to name a
#: workload — structural discovery works without one — but naming one wins.
IDENTITY_LEVELS = ("explicit", "structural", "learned")

#: 'learned' identity (clustering repeated semantically-similar work) is a
#: documented extension point. Nothing produces it today.
UNBUILT_IDENTITY_LEVELS = ("learned",)

# A model is ONE kind of executor, not the centre of the model.
EXECUTOR_TYPES = ("model", "agent", "software", "human")

#: 'human' is permitted by the schema so that human execution can be
#: represented later without a rewrite. NOTHING is built for it: there is no
#: human routing, queueing, assignment or payout anywhere in this package.
UNBUILT_EXECUTOR_TYPES = ("human",)

WORKLOAD_GRAINS = ("invocation", "task_class", "workflow", "endpoint", "cluster")

IDENTITY_KINDS = (
    "explicit",
    "endpoint",
    "workflow",
    "prompt_template",
    "model_endpoint",
    "manual",
    "inferred",
)


# ---------------------------------------------------------------------------
# Objectives — the optimizer must NOT hardcode "minimize dollars"
# ---------------------------------------------------------------------------

OBJECTIVES = ("cost", "quality", "latency", "balanced", "custom")
DEFAULT_OBJECTIVE = "cost"

#: Which measured metric an objective is primarily judged on. 'custom' is
#: deliberately absent: a custom objective is defined by objective_config and
#: is evaluated by whatever registers it, not by a hardcoded branch here.
OBJECTIVE_PRIMARY_METRIC = {
    "cost": "cost",
    "quality": "quality",
    "latency": "latency_p95_ms",
    "balanced": None,  # requires all three; no single primary metric
}


def is_valid_objective(objective: Optional[str]) -> bool:
    return (objective or "") in OBJECTIVES


# ---------------------------------------------------------------------------
# Optimization dimensions
# ---------------------------------------------------------------------------

#: The full vocabulary. Storing a dimension here does NOT mean the system can
#: apply it — see strategy.APPLICABLE_DIMENSIONS for what Runtime can actually
#: change in a workflow graph today.
#: NOTE: whether a dimension can actually be APPLIED is a property of the
#: EXECUTION SURFACE, not a global fact — see strategy.SURFACE_APPLICABLE_DIMENSIONS.
#: `temperature` is dropped by workflow_runtime but genuinely reaches the
#: provider on direct inference.
DIMENSIONS = (
    "model",
    "provider",
    "prompt",
    "context_length",
    "reasoning_effort",
    "temperature",
    "max_tokens",
    "top_p",
    "retrieval",
    "reranking",
    "caching",
    "fallback_chain",
    "llm_call_count",
    "workflow_structure",
    "tool_selection",
    "deterministic_code",
)


# ---------------------------------------------------------------------------
# Evidence: counterfactual strength, not merely presence
# ---------------------------------------------------------------------------

EVIDENCE_SOURCES = (
    "none",
    "observational",
    "replay",
    "shadow",
    "ab_test",
    "canary",
    "production",
)

#: Deprecated values still accepted by the DB CHECK, mapped forward.
EVIDENCE_SOURCE_ALIASES = {
    "historical_analysis": "observational",
    "online_experiment": "ab_test",
}

#: How strong the counterfactual is. Observing A tells you nothing about B.
EVIDENCE_STRENGTH = {
    "none": 0,
    "observational": 10,
    "replay": 30,
    "shadow": 40,
    "ab_test": 60,
    "canary": 70,
    "production": 80,
}


def normalize_evidence_source(source: Optional[str]) -> str:
    s = (source or "none").strip().lower()
    s = EVIDENCE_SOURCE_ALIASES.get(s, s)
    return s if s in EVIDENCE_STRENGTH else "none"


def evidence_strength(source: Optional[str]) -> int:
    return EVIDENCE_STRENGTH[normalize_evidence_source(source)]


#: Evidence at or above this strength involves an actual counterfactual: the
#: candidate was executed and compared, not merely observed. Below it, a
#: recommendation must not be verified.
MIN_EVIDENCE_STRENGTH_FOR_VERIFICATION = EVIDENCE_STRENGTH["replay"]


# ---------------------------------------------------------------------------
# Outcome provenance — signal quality precedence
# ---------------------------------------------------------------------------

#: DEFAULT signal-quality ordering. Mirrors public.outcome_provenance_rank().
#: Keep the two in sync.
#:
#: *** THIS IS A DEFAULT, NOT A LAW. ***
#: It applies only when no optimization_policies.success_signal names the
#: deciding signal for the workload. A global hierarchy would be wrong for real
#: workloads: for one, JSON-schema validity genuinely IS the hard requirement;
#: for another it is conversion rate. Use resolve_success_signal() rather than
#: reaching for this table directly.
DEFAULT_OUTCOME_PROVENANCE_RANK = {
    "business_outcome": 80,
    "deterministic": 70,
    "human": 60,
    "user_feedback": 50,
    "automated_test": 40,
    "schema": 40,
    "llm_judge": 30,
    "implicit": 20,
    "heuristic": 20,
    "unknown": 10,
}

#: Backwards-compatible alias. Prefer DEFAULT_OUTCOME_PROVENANCE_RANK, whose
#: name says that it is a default.
OUTCOME_PROVENANCE_RANK = DEFAULT_OUTCOME_PROVENANCE_RANK

OUTCOME_PROVENANCES = tuple(DEFAULT_OUTCOME_PROVENANCE_RANK.keys())

#: `outcome_type` is an OPEN vocabulary, named per workload: 'thumbs_up',
#: 'ticket_resolved', 'escalation', 'reopened_7d', 'pr_merged', 'pr_reverted'.
#: MANY named outcomes attach to one attempt and arrive at different times.
#: This tuple is the set of coarse CATEGORIES for grouping only — it never
#: constrains outcome_type and never decides success.
OUTCOME_CATEGORIES = (
    "task_success",
    "quality_score",
    "user_feedback",
    "business_value",
    "error",
    "escalation",
    "human_intervention",
    "custom",
)


def is_valid_outcome_type(outcome_type: Optional[str]) -> bool:
    """Open vocabulary: any non-empty name up to 120 chars is valid."""
    t = (outcome_type or "").strip()
    return 1 <= len(t) <= 120

OUTCOME_SOURCES = ("api", "human", "system", "judge", "connector", "import")

ATTEMPT_SOURCES = (
    "workflow_run",
    "api_request",
    "direct_inference",
    "external",
    "none",
)


def provenance_rank(provenance: Optional[str]) -> int:
    """DEFAULT rank for a provenance. See DEFAULT_OUTCOME_PROVENANCE_RANK."""
    return DEFAULT_OUTCOME_PROVENANCE_RANK.get((provenance or "unknown").strip().lower(), 10)


def signal_strength(outcome: dict) -> float:
    """
    Strength of one outcome signal on 0..1.

    Prefers the reporter's explicit `signal_strength` when present — a verified
    refund event and a heuristically-inferred one may share
    provenance='business_outcome' yet differ in how much they should be
    trusted. Falls back to the DEFAULT provenance rank.
    """
    explicit = outcome.get("signal_strength")
    if explicit is not None:
        try:
            return max(0.0, min(1.0, float(explicit)))
        except (TypeError, ValueError):
            pass
    return provenance_rank(outcome.get("provenance")) / 80.0


def group_outcomes_by_provenance(outcomes: Iterable[dict]) -> dict[str, list[dict]]:
    """
    Bucket outcomes by provenance tier.

    Aggregation MUST happen within a tier and be reported with that tier.
    Averaging an LLM-judge score together with a measured business result
    produces a number that means nothing, and this function exists so callers
    are structurally pushed away from doing it.
    """
    buckets: dict[str, list[dict]] = {}
    for o in outcomes:
        p = (o.get("provenance") or "unknown").strip().lower()
        buckets.setdefault(p, []).append(o)
    return buckets


def strongest_provenance(outcomes: Iterable[dict]) -> Optional[str]:
    """Highest-ranked provenance present, or None when there are no outcomes."""
    best: Optional[str] = None
    best_rank = -1
    for o in outcomes:
        p = (o.get("provenance") or "unknown").strip().lower()
        r = provenance_rank(p)
        if r > best_rank:
            best_rank, best = r, p
    return best


# ---------------------------------------------------------------------------
# Quality provenance (shared vocabulary with outcomes)
# ---------------------------------------------------------------------------

QUALITY_PROVENANCES = OUTCOME_PROVENANCES

#: Below this rank a quality signal is too weak to satisfy a min_quality
#: policy on its own. llm_judge (30) sits below it deliberately: an LLM saying
#: the output looked fine is not verification.
MIN_QUALITY_PROVENANCE_RANK_FOR_CONSTRAINT = DEFAULT_OUTCOME_PROVENANCE_RANK["automated_test"]


# ---------------------------------------------------------------------------
# Recommendation lifecycle
# ---------------------------------------------------------------------------

STATUS_DISCOVERED = "discovered"
STATUS_BENCHMARKING = "benchmarking"
STATUS_VERIFIED = "verified"
STATUS_AWAITING_APPROVAL = "awaiting_approval"
STATUS_SHADOWING = "shadowing"
STATUS_CANARY = "canary"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_FAILED = "failed"
STATUS_SUPERSEDED = "superseded"
STATUS_ROLLED_BACK = "rolled_back"

#: Deprecated alias retained in the DB CHECK for rows written by the first
#: migration. New code never emits it.
STATUS_APPLIED_DEPRECATED = "applied"

STATUSES = (
    STATUS_DISCOVERED,
    STATUS_BENCHMARKING,
    STATUS_VERIFIED,
    STATUS_AWAITING_APPROVAL,
    STATUS_SHADOWING,
    STATUS_CANARY,
    STATUS_PROMOTED,
    STATUS_REJECTED,
    STATUS_INCONCLUSIVE,
    STATUS_FAILED,
    STATUS_SUPERSEDED,
    STATUS_ROLLED_BACK,
)

#: Happy path: discovered -> benchmarking -> verified -> awaiting_approval ->
#: (shadowing) -> canary -> promoted.
#:
#: HUMAN APPROVAL IS THE DEFAULT. There is no edge from `verified` straight to
#: `canary`: production is never changed without passing through
#: awaiting_approval, unless an optimization_policies.automation flag has been
#: explicitly enabled by the org (see policies.approval_required).
LEGAL_TRANSITIONS: dict[str, tuple[str, ...]] = {
    STATUS_DISCOVERED: (STATUS_BENCHMARKING, STATUS_REJECTED, STATUS_SUPERSEDED),
    STATUS_BENCHMARKING: (
        STATUS_VERIFIED,
        STATUS_REJECTED,
        STATUS_INCONCLUSIVE,
        STATUS_FAILED,
    ),
    STATUS_VERIFIED: (
        STATUS_AWAITING_APPROVAL,
        STATUS_BENCHMARKING,
        STATUS_REJECTED,
        STATUS_SUPERSEDED,
    ),
    STATUS_AWAITING_APPROVAL: (
        STATUS_SHADOWING,
        STATUS_CANARY,
        STATUS_REJECTED,
        STATUS_SUPERSEDED,
    ),
    STATUS_SHADOWING: (
        STATUS_CANARY,
        STATUS_REJECTED,
        STATUS_INCONCLUSIVE,
        STATUS_FAILED,
    ),
    STATUS_CANARY: (STATUS_PROMOTED, STATUS_ROLLED_BACK, STATUS_REJECTED, STATUS_FAILED),
    STATUS_PROMOTED: (STATUS_ROLLED_BACK, STATUS_SUPERSEDED),
    STATUS_INCONCLUSIVE: (STATUS_BENCHMARKING, STATUS_REJECTED, STATUS_SUPERSEDED),
    STATUS_FAILED: (STATUS_BENCHMARKING, STATUS_REJECTED),
    # Terminal
    STATUS_REJECTED: (),
    STATUS_SUPERSEDED: (),
    STATUS_ROLLED_BACK: (),
}

TERMINAL_STATUSES = tuple(s for s, nxt in LEGAL_TRANSITIONS.items() if not nxt)

#: Statuses at which the candidate is, or has been, affecting production.
LIVE_STATUSES = (STATUS_CANARY, STATUS_PROMOTED)

#: Statuses that represent an open opportunity for the user to act on.
OPEN_STATUSES = (
    STATUS_DISCOVERED,
    STATUS_BENCHMARKING,
    STATUS_VERIFIED,
    STATUS_AWAITING_APPROVAL,
    STATUS_SHADOWING,
)


class IllegalTransition(ValueError):
    """Raised when a lifecycle transition is not permitted."""


def assert_transition(from_status: Optional[str], to_status: str) -> None:
    frm = (from_status or STATUS_DISCOVERED).strip().lower()
    if frm == STATUS_APPLIED_DEPRECATED:
        frm = STATUS_PROMOTED
    to = (to_status or "").strip().lower()
    if to not in STATUSES:
        raise IllegalTransition(f"Unknown status '{to_status}'.")
    allowed = LEGAL_TRANSITIONS.get(frm)
    if allowed is None:
        raise IllegalTransition(f"Unknown current status '{from_status}'.")
    if to not in allowed:
        raise IllegalTransition(
            f"Illegal transition {frm} -> {to}. Allowed from {frm}: "
            f"{', '.join(allowed) if allowed else '(terminal)'}."
        )


MONITORING_STATUSES = ("not_started", "monitoring", "healthy", "degraded", "stopped")


# ---------------------------------------------------------------------------
# Savings vocabulary — three different things that must never be conflated
# ---------------------------------------------------------------------------

SAVINGS_KINDS = ("projected", "verified", "realized")

SAVINGS_SEMANTICS = {
    # extrapolated: measured-or-priced per-call delta x observed traffic volume
    "projected": "projected_savings_usd",
    # measured inside a benchmark or canary, over the sample only
    "verified": "verified_savings_usd",
    # observed in production after promotion
    "realized": "realized_savings_usd",
}


def savings_column(kind: str) -> str:
    """
    Map a savings kind to its column. Raises on anything else so that a caller
    cannot accidentally write a projection into the verified column.
    """
    k = (kind or "").strip().lower()
    if k not in SAVINGS_SEMANTICS:
        raise ValueError(
            f"Unknown savings kind '{kind}'. Must be one of {SAVINGS_KINDS}. "
            "projected = extrapolated; verified = measured in benchmark/canary; "
            "realized = observed in production after promotion."
        )
    return SAVINGS_SEMANTICS[k]


# ---------------------------------------------------------------------------
# Statistics helpers — all return None rather than a made-up number
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> Optional[float]:
    """Nearest-rank percentile. None for an empty sample."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    k = max(0, min(len(vals) - 1, int(math.ceil(p / 100.0 * len(vals))) - 1))
    return float(vals[k])


def mean(values: list[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def coefficient_of_variation(values: list[float]) -> Optional[float]:
    """
    Relative standard deviation. High CV means the sample is noisy and the
    measured delta deserves less confidence. None when undefined.
    """
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    if m == 0:
        return None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var) / abs(m)


# ---------------------------------------------------------------------------
# Evidence maturity — first class, INTERNAL, and deliberately hard to inflate
# ---------------------------------------------------------------------------
#
# This index used to be called `confidence`, and the name was wrong in a way
# that misled the product. It is NOT a statistical confidence, NOT a p-value,
# and NOT the probability that a verdict is correct. It is a blended maturity
# index over five unrelated things — sample size, counterfactual evidence
# class, quality-signal provenance, observed variance, historical consistency —
# and no probabilistic statement can be recovered from it.
#
# The concrete harm: a run reported `confidence 0.188 (band: low)` immediately
# beside a quality-safety verdict that was ESTABLISHED — 140 paired
# evaluations, non-inferiority within 2pp at 95%, discordant_b=1,
# discordant_c=2. Read together, 0.188 says "an 18% chance the verdict holds".
# It says nothing of the kind. The two numbers are not on the same axis and
# nothing in the payload said so.
#
# So this index is now INTERNAL ONLY. It is genuinely useful for ranking and
# prioritising candidates and for deciding whether more data would change a
# conclusion, and it keeps doing that. It does not appear in any
# customer-facing payload. The three things a customer needs are already
# reported, separately and unmixed, and must never be collapsed into one score:
#
#   1. SAFETY VERDICT   quality_safety: established, allowed_regression,
#                       confidence_level, n_pairs, discordant_b/discordant_c.
#                       A real statistical statement with its assumptions.
#   2. EVIDENCE STAGE   evidence_source: observational -> replay -> shadow ->
#                       ab_test -> canary -> production. How the claim was
#                       obtained.
#   3. PRODUCTION STATUS the recommendation status and rollout block. Whether
#                       it is actually serving traffic.
#
# The STORED value is unchanged and the formula below is unchanged, so a
# historical row's number still means exactly what it meant when written. See
# migration_optimization_v10_evidence_maturity_semantics.sql.

#: Sample size at which the sample-size term saturates. 14 replay examples and
#: 180,000 production outcomes must not look alike.
_SAMPLE_SATURATION = 1000


def compute_evidence_maturity(
    *,
    sample_size: Optional[int],
    evidence_source: Optional[str],
    quality_provenance: Optional[str],
    variation: Optional[float] = None,
    historical_consistency: Optional[float] = None,
    production_confirmed: bool = False,
) -> Optional[float]:
    """
    How MATURE the evidence behind a measured claim is, on 0..1. Returns None
    when there is nothing to score.

    NOT a probability and not a confidence level — see the module section
    above. Internal use only: ranking, prioritisation, and deciding whether
    more data could change a conclusion.

    Terms, all multiplicative so a weakness anywhere caps the whole:

      sample     log-scaled sample size, saturating at _SAMPLE_SATURATION.
                 Sub-floor samples are handled by the caller (the benchmark
                 refuses to run) — here they simply score very low.
      counter    counterfactual strength of evidence_source (see
                 EVIDENCE_STRENGTH). Observational evidence is capped hard.
      signal     strength of the quality signal, from outcome provenance rank.
                 'unknown' provenance means we measured cost but not quality,
                 which caps the score rather than zeroing it.
      stability  1 - coefficient of variation, clamped. Noisy samples score low.
      history    optional 0..1 agreement with previous measurements.

    Production confirmation adds a bounded bonus, never pushing past 0.99 —
    there is no such thing as certainty here and the number should not pretend.
    """
    if not sample_size or sample_size <= 0:
        return None
    src = normalize_evidence_source(evidence_source)
    if src == "none":
        return None

    n = float(sample_size)
    sample_term = min(1.0, math.log10(n + 1.0) / math.log10(_SAMPLE_SATURATION + 1.0))

    counter_term = evidence_strength(src) / float(EVIDENCE_STRENGTH["production"])

    rank = provenance_rank(quality_provenance)
    # 'unknown' (rank 10) -> 0.5; deterministic (70) -> ~0.89; business (80) -> 1.0
    signal_term = 0.5 + 0.5 * ((rank - 10) / 70.0)
    signal_term = max(0.0, min(1.0, signal_term))

    stability_term = 1.0
    if variation is not None:
        stability_term = max(0.25, min(1.0, 1.0 - float(variation)))

    history_term = 1.0
    if historical_consistency is not None:
        history_term = max(0.0, min(1.0, float(historical_consistency)))

    score = sample_term * counter_term * signal_term * stability_term * history_term

    if production_confirmed:
        score = score + (1.0 - score) * 0.25

    return round(max(0.0, min(0.99, score)), 3)


# ---------------------------------------------------------------------------
# Savings attribution — never double-count
# ---------------------------------------------------------------------------

def attributable_savings(
    recommendations: Iterable[dict],
    kind: str,
) -> tuple[Optional[float], dict[str, Any]]:
    """
    Sum a savings column across recommendations WITHOUT double-counting.

    If recommendation B optimizes a configuration that recommendation A already
    produced, B's baseline is A's result. Naively summing A + B claims A's
    savings twice. The rule applied here: within a chain (linked by
    parent_recommendation_id / baseline_reference.derived_from_recommendation_id)
    only the DEEPEST descendant that is still counted contributes, because its
    savings are measured against its own parent's result, and its ancestors'
    savings are already embedded in the current production baseline.

    Within a bundle_id, only the bundled candidate is counted if present;
    otherwise its individual members are, since a bundle and its parts describe
    overlapping changes to the same baseline.

    Returns (total_or_None, coverage) where coverage explains what was
    excluded and why. Returns None — never 0.0 — when nothing is measurable.
    """
    col = savings_column(kind)
    recs = list(recommendations)

    by_id = {str(r.get("id")): r for r in recs if r.get("id")}
    superseded_by_child: set[str] = set()
    for r in recs:
        parent = r.get("parent_recommendation_id") or (
            (r.get("baseline_reference") or {}).get("derived_from_recommendation_id")
            if isinstance(r.get("baseline_reference"), dict)
            else None
        )
        if parent and str(parent) in by_id:
            superseded_by_child.add(str(parent))
        sup = r.get("supersedes_id")
        if sup and str(sup) in by_id:
            superseded_by_child.add(str(sup))

    # Bundle handling: prefer the widest candidate in each bundle.
    bundle_winner: dict[str, str] = {}
    for r in recs:
        b = r.get("bundle_id")
        if not b:
            continue
        b = str(b)
        cur = bundle_winner.get(b)
        dims = len(r.get("dimensions") or [])
        if cur is None or dims > len(by_id[cur].get("dimensions") or []):
            bundle_winner[b] = str(r.get("id"))

    total = 0.0
    counted = 0
    excluded_chain = 0
    excluded_bundle = 0
    missing = 0

    for r in recs:
        rid = str(r.get("id"))
        if rid in superseded_by_child:
            excluded_chain += 1
            continue
        b = r.get("bundle_id")
        if b and bundle_winner.get(str(b)) != rid:
            excluded_bundle += 1
            continue
        val = r.get(col)
        if val is None:
            missing += 1
            continue
        try:
            total += float(val)
            counted += 1
        except (TypeError, ValueError):
            missing += 1

    coverage = {
        "kind": kind,
        "column": col,
        "recommendations_considered": len(recs),
        "counted": counted,
        "excluded_superseded_by_descendant": excluded_chain,
        "excluded_bundle_member": excluded_bundle,
        "unmeasured": missing,
        "note": (
            "Chained recommendations contribute only at the deepest counted "
            "descendant; ancestors' savings are already embedded in the current "
            "baseline. Bundle members are represented by the widest candidate."
        ),
    }
    if counted == 0:
        return None, coverage
    return round(total, 6), coverage


# ---------------------------------------------------------------------------
# Success signal — the POLICY decides, not a global constant
# ---------------------------------------------------------------------------

@dataclass
class SuccessSignal:
    """
    Which measured signal decides whether a strategy is acceptable for a
    workload, and how it was chosen.

    `resolved_from` is 'policy' when an optimization_policies.success_signal
    named it, and 'default' when the default provenance ordering picked the
    strongest available signal. It is snapshotted onto the benchmark and the
    recommendation so a later policy change cannot silently rewrite what an old
    verdict meant.
    """

    outcome_type: Optional[str] = None
    provenance: Optional[str] = None
    aggregate: str = "mean"           # 'mean' | 'rate'
    direction: str = "higher_is_better"
    min_value: Optional[float] = None
    min_sample: Optional[int] = None
    resolved_from: str = "default"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "outcome_type": self.outcome_type,
            "provenance": self.provenance,
            "aggregate": self.aggregate,
            "direction": self.direction,
            "min_value": self.min_value,
            "min_sample": self.min_sample,
            "resolved_from": self.resolved_from,
            "reason": self.reason,
        }


def resolve_success_signal(
    policy_success_signal: Optional[dict],
    available_outcomes: Optional[Iterable[dict]] = None,
) -> SuccessSignal:
    """
    Decide which signal judges this workload.

    Order of authority:
      1. The policy. If it names an outcome_type, that is the deciding signal,
         full stop — even if a "stronger" signal by the default ranking is
         present. For a workload whose hard requirement is schema validity,
         a business-outcome signal must NOT override it.
      2. A policy fallback_outcome_types entry that is actually present.
      3. The DEFAULT provenance ordering over whatever outcomes exist.
      4. Nothing. Returns a SuccessSignal with outcome_type=None, meaning no
         quality claim can be made. That is a legitimate answer, not an error.
    """
    outcomes = list(available_outcomes or [])
    cfg = policy_success_signal if isinstance(policy_success_signal, dict) else {}

    def _mk(outcome_type, provenance, resolved_from, reason):
        return SuccessSignal(
            outcome_type=outcome_type,
            provenance=provenance,
            aggregate=(cfg.get("aggregate") or "mean"),
            direction=(cfg.get("direction") or "higher_is_better"),
            min_value=cfg.get("min_value"),
            min_sample=cfg.get("min_sample"),
            resolved_from=resolved_from,
            reason=reason,
        )

    named = (cfg.get("outcome_type") or "").strip()
    if named:
        return _mk(
            named,
            cfg.get("provenance"),
            "policy",
            "Named by the workload's optimization policy.",
        )

    present_types = {(o.get("outcome_type") or "").strip() for o in outcomes}
    for fallback in (cfg.get("fallback_outcome_types") or []):
        if str(fallback).strip() in present_types:
            return _mk(
                str(fallback).strip(),
                None,
                "policy",
                "Policy fallback outcome type, present in the observed data.",
            )

    if outcomes:
        best = max(outcomes, key=signal_strength)
        return _mk(
            (best.get("outcome_type") or "").strip() or None,
            best.get("provenance"),
            "default",
            (
                "No policy named a deciding signal; selected the strongest "
                "available signal by the DEFAULT provenance ordering."
            ),
        )

    return SuccessSignal(
        resolved_from="default",
        reason="No policy signal and no outcomes recorded; no quality claim is possible.",
    )


# ---------------------------------------------------------------------------
# Materiality — policy-owned, never hardcoded
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Benchmark conclusions — EVIDENCE, explicitly stated
# ---------------------------------------------------------------------------

CONCLUSION_SAFE_IMPROVEMENT = "safe_improvement_found"
CONCLUSION_NO_MATERIAL_IMPROVEMENT = "no_material_improvement"
CONCLUSION_CANDIDATES_FAILED_POLICY = "candidates_failed_policy"
CONCLUSION_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CONCLUSION_BENCHMARK_FAILED = "benchmark_failed"

#: A candidate that is cheaper, satisfies every hard policy constraint, and
#: shows no disqualifying observed regression — but whose evidence is not yet
#: strong enough to RULE OUT a material quality regression against the measured
#: baseline. See optimization/noninferiority.py.
#:
#: This exists because collapsing it into `safe_improvement_found` is exactly
#: the bug this vocabulary was extended to fix: a candidate 10 percentage points
#: below baseline on 30 cases cleared an absolute floor of 0.90 by a margin of
#: zero and was written as a VERIFIED recommendation. Collapsing it into
#: `insufficient_evidence` instead would be almost as bad in the other
#: direction — it would discard the fact that we found something worth pursuing
#: and would tell the customer nothing actionable. "Promising, run about N more
#: evaluations" is a different product state from "we could not measure this".
CONCLUSION_PROMISING_UNVERIFIED = "promising_candidate_unverified"

CONCLUSIONS = (
    CONCLUSION_SAFE_IMPROVEMENT,
    CONCLUSION_NO_MATERIAL_IMPROVEMENT,
    CONCLUSION_CANDIDATES_FAILED_POLICY,
    CONCLUSION_PROMISING_UNVERIFIED,
    CONCLUSION_INSUFFICIENT_EVIDENCE,
    CONCLUSION_BENCHMARK_FAILED,
)

#: Conclusions that represent KNOWLEDGE about the workload.
KNOWLEDGE_CONCLUSIONS = (
    CONCLUSION_SAFE_IMPROVEMENT,
    CONCLUSION_NO_MATERIAL_IMPROVEMENT,
    CONCLUSION_CANDIDATES_FAILED_POLICY,
)

#: Conclusions that represent IGNORANCE. Structurally separated so that no
#: aggregate can count them as a finding.
#:
#: `promising_candidate_unverified` belongs here and not in KNOWLEDGE: we did
#: measure something, but we did NOT reach a determination about whether it is
#: safe to adopt. Counting it as coverage would let "we found something we
#: cannot yet vouch for" be reported as "this workload has been assessed".
IGNORANCE_CONCLUSIONS = (
    CONCLUSION_PROMISING_UNVERIFIED,
    CONCLUSION_INSUFFICIENT_EVIDENCE,
    CONCLUSION_BENCHMARK_FAILED,
)

#: THE ONLY conclusion that means "this workload looks efficient".
#: `insufficient_evidence` is NOT evidence that the current configuration is
#: optimal, and counting it as one would be exactly the dishonesty this product
#: exists to prevent. Any summary that counts "workloads with no opportunity"
#: must use this tuple, and must report ignorance conclusions in a separate
#: "not yet assessable" bucket.
NO_OPPORTUNITY_CONCLUSIONS = (CONCLUSION_NO_MATERIAL_IMPROVEMENT,)

NOT_YET_ASSESSABLE_CONCLUSIONS = IGNORANCE_CONCLUSIONS


def is_efficiency_finding(conclusion: Optional[str]) -> bool:
    """
    May this conclusion be rendered as "your configuration looks efficient"?

    Only `no_material_improvement` may. Enforced here so that no caller can
    reach the wrong answer by writing `conclusion != 'safe_improvement_found'`.
    """
    return (conclusion or "") in NO_OPPORTUNITY_CONCLUSIONS


def is_assessable(conclusion: Optional[str]) -> bool:
    """False when the run produced ignorance rather than knowledge."""
    return (conclusion or "") in KNOWLEDGE_CONCLUSIONS


#: Conclusion -> recommendation lifecycle state, applied ONLY when the
#: benchmark is attached to a recommendation. The benchmark conclusion is the
#: EVIDENCE; the recommendation status is the DECISION. Never conflated.
#:
#: 'no_material_improvement' -> 'rejected' (not 'inconclusive'): reaching it
#: requires that the sample cleared the floor, the candidates satisfied policy,
#: and the measured improvement was genuinely below the policy's materiality
#: threshold. We know the answer — not worth changing. 'inconclusive' stays
#: reserved for the case where we could not tell.
#: 'promising_candidate_unverified' -> 'inconclusive' (never 'verified'): the
#: candidate is real and worth pursuing, but the evidence does not yet rule out
#: a material quality regression. `inconclusive` is the existing state for "we
#: could not tell", and the lifecycle allows inconclusive -> benchmarking, which
#: is exactly the next action (run more cases and re-measure). There is no edge
#: from 'inconclusive' to 'verified' without passing through 'benchmarking'
#: again, so a promising candidate structurally cannot become a verified
#: recommendation without new evidence.
CONCLUSION_TO_STATUS = {
    CONCLUSION_SAFE_IMPROVEMENT: STATUS_VERIFIED,
    CONCLUSION_CANDIDATES_FAILED_POLICY: STATUS_REJECTED,
    CONCLUSION_NO_MATERIAL_IMPROVEMENT: STATUS_REJECTED,
    CONCLUSION_PROMISING_UNVERIFIED: STATUS_INCONCLUSIVE,
    CONCLUSION_INSUFFICIENT_EVIDENCE: STATUS_INCONCLUSIVE,
    CONCLUSION_BENCHMARK_FAILED: STATUS_FAILED,
}

MORE_DATA_YES = "yes"
MORE_DATA_NO = "no"
MORE_DATA_UNKNOWN = "unknown"
MORE_DATA_VALUES = (MORE_DATA_YES, MORE_DATA_NO, MORE_DATA_UNKNOWN)


# ---------------------------------------------------------------------------
# Reason codes — the API contract is CODES AND FACTS, never wording
# ---------------------------------------------------------------------------
#
# The backend must not return customer-facing prose. Coupling the API to
# wording would make rephrasing a sentence an API behaviour change. A caller
# receives a stable `code` plus the FACTS behind it (observed, required, which
# constraint, unit) and derives all wording itself.
#
# Adding a code is a contract addition; changing a code's meaning is a breaking
# change. Treat this dict as the documented vocabulary.

REASON_CODES: dict[str, str] = {
    # Evidence adequacy
    "sample_size_below_threshold": "Fewer comparable cases than the floor required to conclude.",
    "insufficient_comparable_cases": "Cases could not be compared like-for-like across arms.",
    "missing_primary_outcome": "The outcome type the policy designates as deciding was never recorded.",
    "outcome_signal_too_weak": "The only available quality signal is too weak to satisfy a hard constraint.",
    "quality_not_measured": "No quality signal was produced by this run.",
    "coverage_gap": "Part of the workload's traffic is not represented in the evidence.",
    "cost_not_measured": "An arm's cost could not be measured, so no cost comparison is possible.",
    "cost_pricing_estimated": (
        "An arm's cost was computed from an estimated fallback price rather than a "
        "known vendor price, so it is a guess and is not reported as measured."
    ),
    # Workload selection
    "workload_volume_below_threshold": "Observed production volume or spend is below the floor worth benchmarking.",
    "no_replay_cases": "The workload has no golden inputs to replay, so no controlled comparison can be run.",
    "workload_not_registered": "Observed traffic has no registered workload; discovery has not been run for it.",
    # Constraint violations
    "quality_below_threshold": "Measured quality is below the policy minimum.",
    "latency_above_threshold": "Measured latency exceeds the policy maximum.",
    "error_rate_above_threshold": "Measured error rate exceeds the policy maximum.",
    "cost_above_threshold": "Measured cost per task exceeds the policy maximum.",
    "provider_not_permitted": "A candidate uses a vendor the policy does not permit.",
    "constraint_unenforceable": "A declared constraint cannot be verified by OptiML today.",
    # Quality safety — RELATIVE to the measured baseline, not an absolute floor
    "quality_regression_above_threshold": (
        "Measured quality is below the baseline by more than the policy's "
        "max_quality_regression. This is a comparison against what the customer "
        "runs today, not against an absolute floor."
    ),
    "non_inferiority_not_established": (
        "The evidence does not rule out, at the policy's confidence level, that "
        "the candidate is worse than the baseline by more than the allowed "
        "regression. Not a claim that the candidate IS worse."
    ),
    "candidate_evaluation_stopped_early": (
        "Evaluation of this candidate was stopped before the full case set "
        "because the evidence already gathered determined its verdict: no "
        "outcome on the remaining cases could bring its quality regression back "
        "within the policy's max_quality_regression. The record carries the "
        "stage, the cases run, and the bound. This is a settled verdict, not a "
        "shortage of evidence."
    ),
    "quality_non_inferiority_established": (
        "The evidence rules out, at the policy's confidence level, a quality "
        "regression larger than the policy allows."
    ),
    "paired_cases_unavailable": (
        "Per-case results could not be paired across the baseline and candidate "
        "arms, so no paired comparison was possible."
    ),
    "baseline_quality_not_measured": (
        "The baseline arm produced no quality measurement, so a relative "
        "quality comparison has no reference point."
    ),
    # Candidate ELIGIBILITY PREFLIGHT — decided BEFORE any provider request.
    #
    # Every code here is emitted only from a check that actually ran against
    # real data (the org's credentials, shared/providers.json, the policy in
    # force, or the request shape the strategy would send). A candidate carrying
    # one of these was NOT benchmarked and NOT failed — it was never dispatched,
    # which is the point: the alternative is discovering the same fact through
    # 140 provider errors.
    "model_not_available": (
        "The model is not present in the executor catalog OptiML can dispatch "
        "to, so no request could be constructed for it."
    ),
    "request_shape_incompatible": (
        "The model family explicitly refuses a request parameter this workload's "
        "configuration sets, and no lossless adapter is declared for it. "
        "Determined from declared capabilities before dispatch, not from a "
        "provider error."
    ),
    "required_capability_missing": (
        "The request requires a capability the model family is DECLARED not to "
        "support. An undeclared capability is 'unknown' and never produces this "
        "code."
    ),
    "context_window_insufficient": (
        "The model's published context window is smaller than the measured "
        "input this workload requires."
    ),
    "policy_blocked": (
        "A policy constraint excludes this candidate before execution."
    ),
    "pricing_unknown": (
        "No vendor list price exists for this model, so its cost could only be "
        "computed from a fallback guess — which cannot be compared against a "
        "known price without manufacturing a saving."
    ),
    "economically_dominated": (
        "Under a cost objective, known list pricing leaves no plausible path to "
        "the materiality threshold even after a deliberately generous allowance "
        "for the candidate being more token-efficient than the baseline. "
        "SCREENING evidence only: this says the candidate was not worth paying "
        "to measure. It is never evidence that a saving exists."
    ),
    "request_adapted": (
        "A declared, lossless adapter rewrote the request so this family could "
        "execute the customer's configuration unchanged in meaning."
    ),
    # Candidate consideration funnel
    "provider_not_configured": (
        "The organization holds no credential for this candidate's provider, so "
        "it could not be benchmarked. Retained as an UNVERIFIED opportunity."
    ),
    "eliminated_by_historical_evidence": (
        "A prior measurement on this workload already showed this executor to be "
        "worse, so it was not re-benchmarked."
    ),
    "duplicate_strategy": "The candidate is identical to one already considered.",
    "generator_error": "A candidate generator raised while proposing candidates.",
    # Materiality
    "improvement_below_materiality": "The measured improvement is below the policy's materiality threshold.",
    # Operational
    "execution_error": "A technical failure prevented a valid comparison.",
    "no_candidates_generated": "No candidate strategy could be generated for this workload.",
    "strategy_not_applicable": "A candidate changes a dimension this runtime cannot apply.",
    "baseline_unavailable": "The current configuration could not be resolved to measure against.",
    # Evidence semantics
    "evidence_maturity_internal_only": (
        "The evidence-maturity index is computed and stored but is not part of "
        "any customer-facing payload. It blends sample size, counterfactual "
        "evidence class, signal provenance, variance and historical "
        "consistency into one 0..1 number; it is NOT a probability and NOT a "
        "confidence level, and printing it beside a non-inferiority verdict "
        "invited exactly that misreading. Safety verdict, evidence stage and "
        "production status are reported separately instead."
    ),
}


def reason(code: str, **facts: Any) -> dict:
    """
    Build one structured reason.

    Raises on an unknown code: an ad-hoc code would silently become part of the
    API contract without being documented.
    """
    if code not in REASON_CODES:
        raise ValueError(
            f"Unknown reason code '{code}'. Add it to REASON_CODES with a description first."
        )
    out = {"code": code}
    out.update({k: v for k, v in facts.items() if v is not None})
    return out


# ---------------------------------------------------------------------------
# Candidate consideration — TWO TIERS, and an auditable funnel
# ---------------------------------------------------------------------------
#
# A customer must not have to configure every provider before OptiML can
# discover opportunities. There are two genuinely different kinds of finding and
# the product needs both, kept strictly apart:
#
#   TIER 1 — EXECUTABLE. The provider is configured, the benchmark actually ran,
#   the numbers are MEASURED. This tier may reach `verified`.
#
#   TIER 2 — OPPORTUNITY, UNVERIFIED. A model from a provider the org has not
#   connected. Derived from a vendor price sheet (evidence_source='none') or, at
#   best, from observation of comparable work ('observational'). OptiML has NOT
#   run it. It carries no measured quality figure, no measured cost, no verified
#   saving, and it can never reach `verified`. Its next action is "connect this
#   provider so it can be benchmarked".
#
# Previously these were DROPPED. Not benchmarking them was right — an arm for a
# provider with no credential is a 100%-error arm, a measurement of nothing
# wearing the costume of a failed candidate. Forgetting them was wrong: a
# customer who has only connected OpenAI still deserves to be told that a model
# elsewhere is worth evaluating.

TIER_EXECUTABLE = "executable"
TIER_OPPORTUNITY = "opportunity"
CANDIDATE_TIERS = (TIER_EXECUTABLE, TIER_OPPORTUNITY)

#: Evidence sources a TIER 2 item may ever carry. 'replay' and stronger require
#: an execution that by definition did not happen.
TIER_OPPORTUNITY_MAX_EVIDENCE_STRENGTH = EVIDENCE_STRENGTH["observational"]

#: Where a candidate left the consideration funnel. Ordered from widest to
#: narrowest; every model that enters consideration leaves with exactly one of
#: these plus a reason code carrying the facts.
DISPOSITION_CONSIDERED = "considered"
DISPOSITION_INCOMPATIBLE = "incompatible"
DISPOSITION_POLICY_BLOCKED = "policy_blocked"
DISPOSITION_PROVIDER_NOT_CONFIGURED = "provider_not_configured"
DISPOSITION_ECONOMICALLY_DOMINATED = "economically_dominated"
DISPOSITION_ELIMINATED_BY_HISTORY = "eliminated_by_historical_evidence"
DISPOSITION_DUPLICATE = "duplicate"
DISPOSITION_GENERATOR_ERROR = "generator_error"
DISPOSITION_BENCHMARKED = "benchmarked"
DISPOSITION_NOT_MEASURED = "not_measured"
DISPOSITION_FAILED_POLICY = "failed_policy"
DISPOSITION_PROMISING = "promising"
DISPOSITION_QUALITY_SAFE = "quality_safe"

#: The per-candidate dispositions, in a stable render order. Each is DISJOINT:
#: a candidate carries exactly one, and it names where that candidate STOPPED.
#: These codes are the contract the UI reads off each candidate and they are
#: NOT changing. What changed is the funnel COUNTS built from them — see
#: FUNNEL_STAGES below.
DISPOSITIONS_ORDERED = (
    DISPOSITION_CONSIDERED,
    DISPOSITION_INCOMPATIBLE,
    DISPOSITION_POLICY_BLOCKED,
    DISPOSITION_PROVIDER_NOT_CONFIGURED,
    DISPOSITION_ECONOMICALLY_DOMINATED,
    DISPOSITION_ELIMINATED_BY_HISTORY,
    DISPOSITION_DUPLICATE,
    DISPOSITION_GENERATOR_ERROR,
    DISPOSITION_BENCHMARKED,
    DISPOSITION_NOT_MEASURED,
    DISPOSITION_FAILED_POLICY,
    DISPOSITION_PROMISING,
    DISPOSITION_QUALITY_SAFE,
)

#: Dispositions NOTHING emits today. Kept in the vocabulary so the shape is
#: stable as discovery grows, and listed here so a zero is never mistaken for
#: "we checked and found none". Same convention as UNBUILT_EXECUTOR_TYPES.
UNBUILT_DISPOSITIONS = (DISPOSITION_ELIMINATED_BY_HISTORY,)

#: Dispositions that mean the candidate was NEVER DISPATCHED — it left the
#: funnel before any provider request was made. Everything else means at least
#: one arm ran.
EXCLUSION_DISPOSITIONS = (
    DISPOSITION_INCOMPATIBLE,
    DISPOSITION_POLICY_BLOCKED,
    DISPOSITION_PROVIDER_NOT_CONFIGURED,
    DISPOSITION_ECONOMICALLY_DOMINATED,
    DISPOSITION_ELIMINATED_BY_HISTORY,
    DISPOSITION_DUPLICATE,
    DISPOSITION_GENERATOR_ERROR,
)

#: Every reason code that can exclude a candidate before dispatch, paired with
#: the disposition it exits at, in render order.
#:
#: Reported at CODE grain and not at DISPOSITION grain on purpose. Four
#: genuinely different pre-dispatch findings — request_shape_incompatible,
#: required_capability_missing, context_window_insufficient, pricing_unknown —
#: all exit at DISPOSITION_INCOMPATIBLE, and collapsing them into one number
#: erases exactly the evidence that the search was thorough. Each of these is a
#: check that RAN against real data; it is a finding, not an error.
EXCLUSION_CODE_TO_DISPOSITION = {
    "provider_not_configured": DISPOSITION_PROVIDER_NOT_CONFIGURED,
    "model_not_available": DISPOSITION_INCOMPATIBLE,
    "request_shape_incompatible": DISPOSITION_INCOMPATIBLE,
    "required_capability_missing": DISPOSITION_INCOMPATIBLE,
    "context_window_insufficient": DISPOSITION_INCOMPATIBLE,
    "policy_blocked": DISPOSITION_POLICY_BLOCKED,
    "provider_not_permitted": DISPOSITION_POLICY_BLOCKED,
    "pricing_unknown": DISPOSITION_INCOMPATIBLE,
    "economically_dominated": DISPOSITION_ECONOMICALLY_DOMINATED,
    "eliminated_by_historical_evidence": DISPOSITION_ELIMINATED_BY_HISTORY,
    "duplicate_strategy": DISPOSITION_DUPLICATE,
    "strategy_not_applicable": DISPOSITION_INCOMPATIBLE,
    "generator_error": DISPOSITION_GENERATOR_ERROR,
}
EXCLUSION_CODES = tuple(EXCLUSION_CODE_TO_DISPOSITION)

#: Exclusion codes nothing emits today. Same `emitted: false` convention as
#: UNBUILT_DISPOSITIONS: a zero here is "not built", not "we checked".
UNBUILT_EXCLUSION_CODES = ("eliminated_by_historical_evidence",)


# --- The funnel proper -----------------------------------------------------
#
# A disposition says where a candidate STOPPED. Counting only that answers
# "where did each one end up?" and cannot answer "how many did we actually
# test?", because a candidate that entered replay and was then stopped on a
# quality regression increments `failed_policy` and nothing else. A real run
# reported `benchmarked: 0` while three arms had demonstrably executed.
#
# The funnel below counts STAGES REACHED instead: a candidate is counted at
# every stage it got to, not only at the one it stopped in. Each stage is
# derived from the one above it (see stages_reached), so the counts are
# monotonically non-increasing BY CONSTRUCTION rather than by coincidence of
# the data.

STAGE_CONSIDERED = "considered"
STAGE_EXECUTABLE = "executable"
STAGE_ENTERED_REPLAY = "entered_replay"
STAGE_STOPPED_EARLY = "stopped_early"
STAGE_COMPLETED_VERIFICATION = "completed_verification"
STAGE_REPLAY_VERIFIED_IMPROVEMENT = "replay_verified_improvement"

#: Render order. `stopped_early` sits where it happened in the pipeline.
FUNNEL_STAGES = (
    STAGE_CONSIDERED,
    STAGE_EXECUTABLE,
    STAGE_ENTERED_REPLAY,
    STAGE_STOPPED_EARLY,
    STAGE_COMPLETED_VERIFICATION,
    STAGE_REPLAY_VERIFIED_IMPROVEMENT,
)

#: The cumulative SPINE: every one of these is "reached at least this far", so
#: each is a subset of the one before it and the counts never increase going
#: down. This is the invariant the API guarantees.
#:
#: `stopped_early` is deliberately NOT in it. It is a DISJOINT exit count
#: (`entered_replay` minus `completed_verification`), reported inline because
#: the product needs it visible, and flagged `cumulative: false` so nobody
#: reads it as a subset of the stage below. Asserting monotonicity across it
#: would be asserting something untrue: a run where one candidate is stopped
#: early and three finish gives 4, 1, 3, which is not non-increasing and is
#: also not wrong.
CUMULATIVE_STAGES = (
    STAGE_CONSIDERED,
    STAGE_EXECUTABLE,
    STAGE_ENTERED_REPLAY,
    STAGE_COMPLETED_VERIFICATION,
    STAGE_REPLAY_VERIFIED_IMPROVEMENT,
)


def stages_reached(disposition: dict) -> set:
    """
    Which cumulative stages ONE candidate reached.

    Each membership test is gated on the previous stage, so the returned set is
    always a prefix of CUMULATIVE_STAGES. That is what makes the funnel's
    monotonicity structural: a candidate cannot appear at a later stage without
    appearing at every earlier one, whatever the caller passes in.
    """
    d = disposition or {}
    reached = {STAGE_CONSIDERED}

    if d.get("disposition") in EXCLUSION_DISPOSITIONS:
        return reached
    reached.add(STAGE_EXECUTABLE)

    if not d.get("entered_replay"):
        return reached
    reached.add(STAGE_ENTERED_REPLAY)

    if d.get("stopped_early"):
        # A SETTLED verdict, but not a completed verification: the remaining
        # cases were deliberately not run because no outcome on them could
        # change the answer. Counting it as verification completed would claim
        # evidence that was never gathered.
        return reached
    reached.add(STAGE_COMPLETED_VERIFICATION)

    if (
        d.get("disposition") == DISPOSITION_QUALITY_SAFE
        and d.get("objective_improved") is True
    ):
        reached.add(STAGE_REPLAY_VERIFIED_IMPROVEMENT)
    return reached


def build_funnel(dispositions: Iterable[dict]) -> dict:
    """
    Turn per-candidate dispositions into the auditable consideration funnel.

    Each disposition: {"label", "disposition", "code", "tier", "entered_replay",
    "stopped_early", "objective_improved", ...facts}.

    Returns three ordered, self-describing lists plus the un-narrated
    per-candidate records:

      stages       CUMULATIVE stage counts — how far the search actually got.
                   Entries carry `cumulative`, and the `cumulative: true` ones
                   are monotonically non-increasing in list order.
      exclusions   why candidates never reached dispatch, at REASON CODE grain.
      outcomes     the disjoint per-disposition counts (where each candidate
                   stopped). Unchanged in meaning; they are evidence of a
                   thorough search, not a list of errors.

    Every entry carries `emitted`. `emitted: false` means nothing in the system
    can populate that row yet, so its zero must not be read as "we checked and
    found none". No prose anywhere: the frontend owns the wording.
    """
    items = [d for d in (dispositions or []) if isinstance(d, dict)]
    reached = [stages_reached(d) for d in items]

    counts = {stage: 0 for stage in FUNNEL_STAGES}
    for stage in CUMULATIVE_STAGES:
        counts[stage] = sum(1 for r in reached if stage in r)
    counts[STAGE_STOPPED_EARLY] = sum(
        1 for d, r in zip(items, reached)
        if STAGE_ENTERED_REPLAY in r and d.get("stopped_early")
    )

    outcome_counts = {stage: 0 for stage in DISPOSITIONS_ORDERED}
    for d in items:
        stage = d.get("disposition")
        if stage in outcome_counts:
            outcome_counts[stage] += 1
    outcome_counts[DISPOSITION_CONSIDERED] = len(items)

    code_counts = {code: 0 for code in EXCLUSION_CODES}
    for d in items:
        if d.get("disposition") not in EXCLUSION_DISPOSITIONS:
            continue
        code = d.get("code")
        if code in code_counts:
            code_counts[code] += 1

    return {
        "considered": len(items),
        "stages": [
            {
                "stage": stage,
                "count": counts[stage],
                "emitted": True,
                "cumulative": stage in CUMULATIVE_STAGES,
            }
            for stage in FUNNEL_STAGES
        ],
        "exclusions": [
            {
                "code": code,
                "count": code_counts[code],
                "emitted": code not in UNBUILT_EXCLUSION_CODES,
                "disposition": EXCLUSION_CODE_TO_DISPOSITION[code],
            }
            for code in EXCLUSION_CODES
        ],
        "outcomes": [
            {
                "stage": stage,
                "count": outcome_counts[stage],
                "emitted": stage not in UNBUILT_DISPOSITIONS,
            }
            for stage in DISPOSITIONS_ORDERED
        ],
        "by_tier": {
            TIER_EXECUTABLE: sum(
                1 for d in items if d.get("tier", TIER_EXECUTABLE) == TIER_EXECUTABLE
            ),
            TIER_OPPORTUNITY: sum(
                1 for d in items if d.get("tier") == TIER_OPPORTUNITY
            ),
        },
        "dispositions": items,
    }


#: Band values are ('low', 'medium', 'high') and STAY that way: they are what
#: the `confidence_band` CHECK constraint accepts and what the preserved
#: historical rows contain. Renaming the values would either break the
#: constraint or require rewriting rows that must stay as written.
EVIDENCE_MATURITY_BANDS = ("low", "medium", "high")


def evidence_maturity_band(score: Optional[float]) -> Optional[str]:
    """
    Coarse band over compute_evidence_maturity(). Internal, like the score:
    "low" means the evidence is EARLY, never that a verdict is unlikely.
    """
    if score is None:
        return None
    c = float(score)
    if c < 0.34:
        return "low"
    if c < 0.67:
        return "medium"
    return "high"


def conclusion_payload(conclusion: Optional[str], *, reasons: Optional[list[dict]] = None) -> dict:
    """
    The API shape for a conclusion: a stable code, structured facts, and the
    conclusion's epistemic class. NO customer-facing sentence is included, by
    design — see REASON_CODES.

    The evidence-maturity index is deliberately ABSENT. It is not a
    probability, and printing a 0..1 number here put it on the same visual axis
    as the quality-safety verdict's confidence level, where readers merged the
    two. The three axes a caller needs are carried separately and unmixed:
    `quality_safety` (safety verdict), `evidence_source` (evidence stage) and
    the recommendation status (production status). The absence is named rather
    than silent so a client that used to read the field can tell this was a
    decision, not a regression.
    """
    c = conclusion or None
    return {
        "conclusion": c,
        "reasons": reasons or [],
        "evidence_maturity_absent_reason": "evidence_maturity_internal_only",
        "is_efficiency_finding": is_efficiency_finding(c),
        "is_assessable": is_assessable(c),
        "coverage_class": coverage_class(c),
        "maps_to_recommendation_status": CONCLUSION_TO_STATUS.get(c or ""),
    }


# ---------------------------------------------------------------------------
# Optimization Coverage — ONE classifier, not scattered predicates
# ---------------------------------------------------------------------------
#
# Coverage is "the percentage of eligible workload spend/volume for which
# OptiML has sufficient evidence to make an optimization determination."

COVERAGE_COVERED = "covered"
COVERAGE_NOT_COVERED = "not_covered"

#: Conclusions that mean we reached a determination.
COVERED_CONCLUSIONS = (
    CONCLUSION_SAFE_IMPROVEMENT,
    CONCLUSION_NO_MATERIAL_IMPROVEMENT,
    CONCLUSION_CANDIDATES_FAILED_POLICY,
)

#: Everything else, including a technical failure and anything still running.
#: "We couldn't run the experiment" must never look like "we evaluated this
#: workload", and neither must "we're still measuring".
NOT_COVERED_CONCLUSIONS = (
    CONCLUSION_PROMISING_UNVERIFIED,
    CONCLUSION_INSUFFICIENT_EVIDENCE,
    CONCLUSION_BENCHMARK_FAILED,
)

IN_PROGRESS_BENCHMARK_STATUSES = ("pending", "running")


def coverage_class(conclusion: Optional[str], benchmark_status: Optional[str] = None) -> str:
    """
    THE coverage classifier. Every coverage figure in the product must call
    this rather than re-deriving the rule.

    A workload is COVERED only when a benchmark reached one of the three
    determination conclusions. Not-yet-run, in-progress, failed and
    insufficient-evidence are all NOT covered.
    """
    if benchmark_status and benchmark_status in IN_PROGRESS_BENCHMARK_STATUSES:
        return COVERAGE_NOT_COVERED
    return COVERAGE_COVERED if (conclusion or "") in COVERED_CONCLUSIONS else COVERAGE_NOT_COVERED


def compute_coverage(
    entries: Iterable[dict],
    *,
    objective: str = "cost",
) -> dict:
    """
    Workload, spend and volume coverage from per-workload entries.

    Each entry: {workload_id, conclusion, benchmark_status, spend_usd, volume}.
    `spend_usd` / `volume` may be None (never measured), in which case that
    workload is excluded from that denominator and counted in `unmeasured`.

    Returns all three figures plus `primary`, which names SPEND coverage when
    the objective is cost. The reason is concrete: "8 of 10 workloads assessed"
    sounds excellent right up until the two unassessed ones are 78% of spend.
    """
    total_wl = covered_wl = 0
    total_spend = covered_spend = 0.0
    total_vol = covered_vol = 0.0
    spend_unmeasured = vol_unmeasured = 0

    for e in entries:
        covered = coverage_class(e.get("conclusion"), e.get("benchmark_status")) == COVERAGE_COVERED
        total_wl += 1
        if covered:
            covered_wl += 1

        spend = e.get("spend_usd")
        if spend is None:
            spend_unmeasured += 1
        else:
            total_spend += float(spend)
            if covered:
                covered_spend += float(spend)

        vol = e.get("volume")
        if vol is None:
            vol_unmeasured += 1
        else:
            total_vol += float(vol)
            if covered:
                covered_vol += float(vol)

    def _pct(num: float, den: float) -> Optional[float]:
        return round(num / den, 4) if den > 0 else None

    return {
        "workload_coverage": {
            "covered": covered_wl,
            "total": total_wl,
            "ratio": _pct(covered_wl, total_wl),
        },
        "spend_coverage": {
            "assessable_usd": round(covered_spend, 6) if total_spend else None,
            "awaiting_evidence_usd": (
                round(total_spend - covered_spend, 6) if total_spend else None
            ),
            "total_usd": round(total_spend, 6) if total_spend else None,
            "ratio": _pct(covered_spend, total_spend),
            "workloads_with_unmeasured_spend": spend_unmeasured,
        },
        "volume_coverage": {
            "assessable": covered_vol or None,
            "total": total_vol or None,
            "ratio": _pct(covered_vol, total_vol),
            "workloads_with_unmeasured_volume": vol_unmeasured,
        },
        "primary": "spend_coverage" if objective == "cost" else "workload_coverage",
        "definition": (
            "Share of eligible workload spend/volume for which OptiML has sufficient "
            "evidence to make an optimization determination. Covered: "
            "safe_improvement_found, no_material_improvement, candidates_failed_policy. "
            "Not covered: insufficient_evidence, benchmark_failed, in-progress, never run."
        ),
    }


# ---------------------------------------------------------------------------
# Materiality, objective-aware
# ---------------------------------------------------------------------------
#
# "5% or $5/month" only makes sense for a cost objective. A latency objective
# needs "150ms p95"; a resolution-rate objective needs "1.5 percentage points".
# A threshold is therefore metric + comparator + value + unit, OR/AND-combined.

MATERIALITY_METRICS = ("cost", "latency_p95_ms", "quality", "outcome_rate", "error_rate", "custom")

MATERIALITY_COMPARATORS = (
    "relative_decrease_at_least",
    "relative_increase_at_least",
    "absolute_decrease_at_least",
    "absolute_increase_at_least",
)

MATERIALITY_UNITS = ("ratio", "usd_per_month", "usd_per_task", "ms", "percentage_points", "score")

#: Default thresholds per objective. Each is a genuine judgement about the
#: metric's own units, not a savings rule reused everywhere.
DEFAULT_MATERIALITY_BY_OBJECTIVE = {
    # Below ~5% the measured delta is commonly inside replay noise at realistic
    # sample sizes; and a change saving under a few dollars a month does not
    # repay the risk and review cost of touching production.
    "cost": {
        "thresholds": [
            {"metric": "cost", "comparator": "relative_decrease_at_least",
             "value": 0.05, "unit": "ratio"},
            {"metric": "cost", "comparator": "absolute_decrease_at_least",
             "value": 5.0, "unit": "usd_per_month"},
        ],
        "combine": "any",
    },
    # 150ms at p95 is roughly the point at which a latency change is perceptible
    # in an interactive product.
    "latency": {
        "thresholds": [
            {"metric": "latency_p95_ms", "comparator": "absolute_decrease_at_least",
             "value": 150.0, "unit": "ms"},
        ],
        "combine": "any",
    },
    # 1.5 percentage points on an outcome rate is a change worth acting on and
    # large enough to separate from sampling noise at practical volumes.
    "quality": {
        "thresholds": [
            {"metric": "outcome_rate", "comparator": "absolute_increase_at_least",
             "value": 0.015, "unit": "percentage_points"},
        ],
        "combine": "any",
    },
}
DEFAULT_MATERIALITY_BY_OBJECTIVE["balanced"] = DEFAULT_MATERIALITY_BY_OBJECTIVE["cost"]
DEFAULT_MATERIALITY_BY_OBJECTIVE["custom"] = {"thresholds": [], "combine": "any"}

#: Legacy savings-shaped keys, accepted as SUGAR and normalised into cost
#: thresholds. They are not the domain model.
_SUGAR_KEYS = ("min_relative_improvement", "min_absolute_monthly_savings_usd")


def normalize_materiality(raw: Optional[dict], objective: str = "cost") -> dict:
    """
    Normalise a policy's materiality into {thresholds: [...], combine}.

    Accepts the generic form, the legacy savings sugar, or nothing (defaults per
    objective). Always stamps `source` so a benchmark can record which threshold
    it actually applied.
    """
    obj = objective if objective in DEFAULT_MATERIALITY_BY_OBJECTIVE else "cost"

    if not isinstance(raw, dict) or not raw:
        return {**copy_default_materiality(obj), "source": "default", "objective": obj}

    if raw.get("thresholds"):
        return {
            "thresholds": list(raw["thresholds"]),
            "combine": (raw.get("combine") or "any").lower(),
            "source": "policy",
            "objective": obj,
        }

    if any(k in raw for k in _SUGAR_KEYS):
        thresholds = []
        if raw.get("min_relative_improvement") is not None:
            thresholds.append({
                "metric": "cost",
                "comparator": "relative_decrease_at_least",
                "value": float(raw["min_relative_improvement"]),
                "unit": "ratio",
            })
        if raw.get("min_absolute_monthly_savings_usd") is not None:
            thresholds.append({
                "metric": "cost",
                "comparator": "absolute_decrease_at_least",
                "value": float(raw["min_absolute_monthly_savings_usd"]),
                "unit": "usd_per_month",
            })
        return {
            "thresholds": thresholds,
            "combine": (raw.get("require") or raw.get("combine") or "any").lower(),
            "source": "policy_sugar",
            "objective": obj,
        }

    return {**copy_default_materiality(obj), "source": "default", "objective": obj}


def copy_default_materiality(objective: str) -> dict:
    base = DEFAULT_MATERIALITY_BY_OBJECTIVE.get(objective, DEFAULT_MATERIALITY_BY_OBJECTIVE["cost"])
    return {"thresholds": [dict(t) for t in base["thresholds"]], "combine": base["combine"]}


def evaluate_materiality(measurements: dict, materiality: dict) -> tuple[bool, dict]:
    """
    Is a measured change material under this threshold set?

    `measurements` maps a materiality metric to {"relative": float|None,
    "absolute": float|None, "unit": str}. Improvements are expressed as POSITIVE
    numbers in the direction the comparator names.

    A threshold whose measurement is missing is NOT satisfied — it is recorded
    as unmeasurable. An unknown can never be quietly rounded up into a
    recommendation.
    """
    thresholds = materiality.get("thresholds") or []
    combine = (materiality.get("combine") or "any").lower()
    results: list[dict] = []

    for t in thresholds:
        metric = t.get("metric")
        comparator = t.get("comparator")
        required = t.get("value")
        m = measurements.get(metric) or {}
        kind = "relative" if str(comparator).startswith("relative_") else "absolute"
        observed = m.get(kind)

        if observed is None or required is None:
            results.append({
                **t, "observed": None, "met": False, "unmeasurable": True,
            })
            continue

        met = float(observed) >= float(required)
        results.append({**t, "observed": float(observed), "met": met, "unmeasurable": False})

    if not results:
        # No threshold declared: nothing can be judged material. Say so rather
        # than defaulting to "yes".
        return False, {
            "applied": materiality,
            "thresholds": [],
            "met": False,
            "reason": "no_thresholds_declared",
        }

    met = all(r["met"] for r in results) if combine == "all" else any(r["met"] for r in results)
    return met, {
        "applied": materiality,
        "thresholds": results,
        "combine": combine,
        "met": met,
        "unmeasurable_count": sum(1 for r in results if r["unmeasurable"]),
    }
