"""
Candidate generators — recommendation engine stages 1 and 2.

A generator proposes a candidate STRATEGY for a workload. A proposal is a
HYPOTHESIS, never a finding: every candidate leaves here with
evidence_source='none' or 'observational', and only optimization/benchmark.py
can upgrade that by actually measuring.

Engine staging (architected for 1-6, built for 1-2):
  1 rules / obvious opportunities   BUILT  — AlternateModelGenerator
  2 replay evidence                 BUILT  — CheaperMeasuredModelGenerator feeds
                                             candidates; benchmark.py measures
                                    BUILT  — ContextReductionGenerator: the same
                                             model carrying less context. Same
                                             provider, same tools, same output
                                             contract; only the context
                                             representation changes.
  3 production experiment           WIRED  — the existing canary/experiment
                                             infrastructure carries this; no new
                                             machinery is added here
  4 historical workload learning    EXTENSION POINT (documented stub)
  5 prediction                      EXTENSION POINT (documented stub)
  6 allocation optimization         EXTENSION POINT (allocation.py records the
                                             decisions a future engine reads)

The two boundaries this module holds:

  VENDOR vs EMPIRICAL. AlternateModelGenerator reads a PRICE SHEET
  (optimization.executors) and says "this looks cheaper, go measure it".
  CheaperMeasuredModelGenerator reads MEASUREMENTS (optimization.evidence) and
  says "this WAS cheaper here". They are separate classes because they are
  separate epistemic claims, and the second is worth far more than the first.

  APPLICABLE vs ASPIRATIONAL. A generator may only emit dimensions
  optimization.strategy can actually apply. Anything else is rejected before it
  can reach a benchmark and produce a measurement of nothing.

  PROPOSED vs PROVEN. An LLM may PROPOSE a shorter prompt. An LLM may never be
  the EVIDENCE that the shorter prompt works. A model-authored candidate enters
  the same paired replay against the same recorded production inputs, is
  measured for real provider cost and latency, and clears the same paired
  non-inferiority test at the same margin — or it is not a finding. See
  ContextReductionGenerator.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from optimization import domain, evidence, executors, strategy as strategy_mod

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """
    A proposed alternative strategy. A hypothesis, not a finding.

    `projected_savings_usd` is populated ONLY when it can be computed from
    measured traffic; it is an extrapolation and is never written into a
    verified or realized column. `rationale` is prose and is explicitly not
    evidence.
    """

    title: str
    strategy: strategy_mod.Strategy
    dimensions: list[str]
    generator: str
    rationale: str
    evidence_source: str = "none"
    projected_savings_usd: Optional[float] = None
    projection_basis: Optional[dict] = None
    measured_basis: Optional[dict] = None
    notes: list[dict] = field(default_factory=list)
    #: Structured eligibility evidence, attached by optimization.eligibility's
    #: preflight. Present on EVERY candidate that reached preflight, including
    #: the ones that passed — "we checked fourteen dimensions and all were fine"
    #: is as much a part of the audit trail as an exclusion is.
    eligibility: Optional[dict] = None

    @property
    def fingerprint(self) -> str:
        return self.strategy.fingerprint()

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "fingerprint": self.fingerprint,
            "dimensions": self.dimensions,
            "generator": self.generator,
            "rationale": self.rationale,
            "evidence_source": self.evidence_source,
            "evidence_strength": domain.evidence_strength(self.evidence_source),
            "projected_savings_usd": self.projected_savings_usd,
            "projection_basis": self.projection_basis,
            "measured_basis": self.measured_basis,
            "notes": self.notes,
            "eligibility": self.eligibility,
            "strategy": self.strategy.to_dict(),
        }


class CandidateGenerator(Protocol):
    """Pluggable generator contract."""

    name: str

    def generate(
        self,
        workload: dict,
        baseline: strategy_mod.Strategy,
        history: dict,
    ) -> list[Candidate]:
        """
        Propose candidates.

        `history` is the empirical context assembled by `build_history`:
        measured model stats, observed traffic and the baseline's own measured
        cost. A generator that needs something not in there must not invent it.
        """
        ...


def build_history(
    org_id: str,
    workload: dict,
    *,
    workflow_id: Optional[str] = None,
    lookback_days: int = 30,
) -> dict:
    """
    Assemble the EMPIRICAL context a generator may reason over.

    Everything in here is measured. Vendor data is fetched separately by the
    generator that needs it, so a generator cannot accidentally treat a price
    sheet as an observation.
    """
    endpoint_slug = (
        workload.get("identity_ref") if workload.get("identity_kind") == "endpoint" else None
    )
    return {
        "org_id": org_id,
        "workload_id": str(workload.get("id")) if workload.get("id") else None,
        "workflow_id": workflow_id,
        "endpoint_slug": endpoint_slug,
        "model_stats": evidence.models_used_by_workload(
            org_id,
            workflow_id=workflow_id,
            endpoint_slug=endpoint_slug,
            lookback_days=lookback_days,
        ),
        "traffic": evidence.observed_production_traffic(
            org_id,
            endpoint_slug=endpoint_slug,
            workflow_id=workflow_id,
            lookback_days=lookback_days,
        ),
        "lookback_days": lookback_days,
    }


def _primary_model_steps(baseline: strategy_mod.Strategy) -> list[strategy_mod.StrategyStep]:
    return [s for s in baseline.steps if s.executor_type == "model" and s.executor_ref.get("external_id")]


def _swap_model(
    baseline: strategy_mod.Strategy,
    step_id: str,
    vendor: str,
    external_id: str,
) -> strategy_mod.Strategy:
    """Clone the baseline with one step's model/provider replaced."""
    steps: list[strategy_mod.StrategyStep] = []
    for s in baseline.steps:
        if s.step_id == step_id:
            steps.append(
                strategy_mod.StrategyStep(
                    step_id=s.step_id,
                    order=s.order,
                    executor_type=s.executor_type,
                    executor_ref={
                        **(s.executor_ref or {}),
                        "executor_type": "model",
                        "vendor": vendor,
                        "external_id": external_id,
                    },
                    executor_id=None,
                    role=s.role,
                    config=dict(s.config or {}),
                    on_failure=s.on_failure,
                )
            )
        else:
            steps.append(s)
    return strategy_mod.Strategy(
        steps=steps,
        surface=baseline.surface,
        surface_binding=dict(baseline.surface_binding or {}),
        dimensions=[],
    )


def _monthly_projection(
    per_call_saving_usd: Optional[float],
    traffic: dict,
    lookback_days: int,
) -> tuple[Optional[float], dict]:
    """
    Extrapolate a per-call saving to a monthly figure using MEASURED volume.

    Returns (None, basis) when volume was not measured. A projection built on an
    assumed volume would be a fabricated number wearing a dollar sign.
    """
    run_count = traffic.get("run_count")
    basis: dict[str, Any] = {
        "kind": "projection",
        "per_call_saving_usd": per_call_saving_usd,
        "observed_run_count": run_count,
        "observed_window_days": lookback_days,
        "coverage": traffic.get("coverage"),
    }
    if per_call_saving_usd is None or not run_count:
        basis["result"] = "not_projectable"
        basis["reason"] = (
            "No measured production volume for this workload in the window; "
            "a monthly figure would be invented."
        )
        return None, basis

    calls_per_day = run_count / float(max(1, lookback_days))
    monthly = per_call_saving_usd * calls_per_day * 30.0
    basis["result"] = "projected"
    basis["calls_per_day"] = round(calls_per_day, 4)
    basis["formula"] = "per_call_saving_usd * (observed_run_count / window_days) * 30"
    return round(monthly, 6), basis


# ---------------------------------------------------------------------------
# Stage 1 — rules over the VENDOR price sheet
# ---------------------------------------------------------------------------

class AlternateModelGenerator:
    """
    Propose cheaper models the workload has NOT yet tried.

    Evidence class: NONE. This reads a vendor price sheet
    (shared/providers.json). A published price is a reason to run a benchmark;
    it is not a finding, and every candidate from here carries
    evidence_source='none'.

    A candidate is emitted only when:
      * the alternative's blended list price is meaningfully below the
        baseline's (a rounding-error difference is not worth a benchmark), and
      * the workload has NOT already measured that model — models it HAS tried
        belong to CheaperMeasuredModelGenerator, which can say something
        stronger about them.
    """

    name = "alternate_model"

    #: Minimum blended list-price advantage before proposing. Below this, the
    #: benchmark cost is unlikely to be repaid.
    MIN_PRICE_ADVANTAGE = 0.15

    #: Blended price assumes this many input tokens per output token when the
    #: workload's own ratio is unknown. An ASSUMPTION used only for ordering
    #: candidates, never for stating a saving.
    DEFAULT_IO_RATIO = 3.0

    def __init__(self, max_candidates: int = 3):
        self.max_candidates = max_candidates

    def generate(self, workload, baseline, history) -> list[Candidate]:
        steps = _primary_model_steps(baseline)
        if not steps:
            return []

        model_stats = history.get("model_stats") or {}
        already_tried = {m.lower() for m in model_stats}
        catalog = executors.vendor_catalog()
        if not catalog:
            return []

        io_ratio = self._measured_io_ratio(history) or self.DEFAULT_IO_RATIO
        # Rank only within providers this org can actually execute. Filtering
        # after ranking is not equivalent: the globally-cheapest models crowd
        # out an executable substitution, and the org is left with no candidate
        # at all rather than the cheaper model it could actually have run.
        # Empty set means "unknown" -> do not constrain.
        allowed = history.get("configured_providers") or set()
        out: list[Candidate] = []

        for step in steps:
            base_model = step.executor_ref.get("external_id")
            base_vendor = (
                step.executor_ref.get("vendor")
                or evidence.infer_provider(base_model or "")
            )
            base_price = executors.blended_vendor_price(
                base_vendor, base_model, input_output_ratio=io_ratio
            )
            if base_price is None or base_price <= 0:
                continue

            ranked = []
            #: Models from providers this org has NOT connected. Ranked
            #: SEPARATELY and appended after the executable ones, never merged
            #: into the same list: the globally-cheapest models are frequently
            #: at unconnected vendors, and one merged ranking would crowd out
            #: the substitution the org could actually run today. They are
            #: emitted so downstream can retain them as TIER 2 opportunities —
            #: "connect this provider to evaluate it" — instead of the org being
            #: shown only the vendors it already pays.
            ranked_unconfigured = []
            for entry in catalog:
                if entry["external_id"].lower() in already_tried:
                    continue
                if entry["external_id"] == base_model:
                    continue
                price = executors.blended_vendor_price(
                    entry["vendor"], entry["external_id"], input_output_ratio=io_ratio
                )
                if price is None or price <= 0:
                    continue
                advantage = (base_price - price) / base_price
                if advantage < self.MIN_PRICE_ADVANTAGE:
                    continue
                executable = (
                    not allowed or str(entry["vendor"]).strip().lower() in allowed
                )
                (ranked if executable else ranked_unconfigured).append(
                    (advantage, price, entry)
                )

            ranked.sort(key=lambda t: -t[0])
            ranked_unconfigured.sort(key=lambda t: -t[0])

            for advantage, price, entry in (
                ranked[: self.max_candidates]
                + ranked_unconfigured[: self.max_candidates]
            ):
                cand_strategy = _swap_model(
                    baseline, step.step_id, entry["vendor"], entry["external_id"]
                )
                dims = strategy_mod.diff_dimensions(baseline, cand_strategy)
                if not dims:
                    continue

                measured_cost = (model_stats.get(base_model) or {}).get("avg_cost")
                per_call_saving = (
                    measured_cost * advantage if measured_cost is not None else None
                )
                projected, basis = _monthly_projection(
                    per_call_saving, history.get("traffic") or {}, history.get("lookback_days", 30)
                )
                basis["price_advantage_ratio"] = round(advantage, 4)
                basis["io_ratio_source"] = (
                    "measured" if self._measured_io_ratio(history) else "assumed_default"
                )
                basis["baseline_measured_cost_per_call_usd"] = measured_cost

                out.append(Candidate(
                    title=f"Try {entry['display_name']} on step {step.step_id}",
                    strategy=cand_strategy,
                    dimensions=dims,
                    generator=self.name,
                    rationale=(
                        f"{entry['display_name']} lists at roughly "
                        f"{advantage * 100:.0f}% below {base_model} on blended token "
                        f"price. This workload has not run it. This is a VENDOR LIST "
                        f"PRICE comparison, not a measurement — it is a reason to "
                        f"benchmark, not a reason to switch."
                    ),
                    evidence_source="none",
                    projected_savings_usd=projected,
                    projection_basis=basis,
                    measured_basis=None,
                    notes=[{
                        "code": "vendor_price_only",
                        "detail": (
                            "Derived from shared/providers.json list prices. No "
                            "quality, latency or reliability claim is implied."
                        ),
                    }],
                ))

        return out

    @staticmethod
    def _measured_io_ratio(history: dict) -> Optional[float]:
        """Input:output token ratio measured for this workload, if available."""
        traffic = history.get("traffic") or {}
        ratio = traffic.get("io_token_ratio")
        try:
            return float(ratio) if ratio else None
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Stage 2 — rules over MEASURED history
# ---------------------------------------------------------------------------

class CheaperMeasuredModelGenerator:
    """
    Propose models this workload HAS run that measured cheaper.

    Evidence class: OBSERVATIONAL. This is genuinely stronger than a price
    sheet — the model really did run on this work — but it is still NOT a
    counterfactual: those runs happened on different inputs at different times.
    Candidates carry evidence_source='observational' and still require a replay
    benchmark before anything is verified.

    Reuses the measured-history logic that used to live in workflow_runtime as
    `_get_model_performance_history` / `_select_optimal_model`; that code now
    lives in optimization.evidence and workflow_runtime imports it back, so
    there is exactly one implementation.
    """

    name = "cheaper_measured_model"

    #: Minimum measured runs before a model's average is worth proposing on.
    MIN_OBSERVED_RUNS = 5

    #: Same error-rate ceiling the original selector used.
    MAX_ERROR_RATE = 0.2

    #: Minimum measured cost advantage.
    MIN_COST_ADVANTAGE = 0.10

    def __init__(self, max_candidates: int = 3):
        self.max_candidates = max_candidates

    def generate(self, workload, baseline, history) -> list[Candidate]:
        model_stats: dict = history.get("model_stats") or {}
        if not model_stats:
            return []

        steps = _primary_model_steps(baseline)
        out: list[Candidate] = []

        for step in steps:
            base_model = step.executor_ref.get("external_id")
            base = model_stats.get(base_model)
            if not base or base.get("avg_cost") is None:
                continue
            base_cost = float(base["avg_cost"])
            if base_cost <= 0:
                continue

            ranked = []
            for model, stats in model_stats.items():
                if model == base_model:
                    continue
                if (stats.get("runs") or 0) < self.MIN_OBSERVED_RUNS:
                    continue
                if stats.get("avg_cost") is None:
                    continue
                err = stats.get("error_rate")
                if err is not None and err > self.MAX_ERROR_RATE:
                    continue
                advantage = (base_cost - float(stats["avg_cost"])) / base_cost
                if advantage < self.MIN_COST_ADVANTAGE:
                    continue
                ranked.append((advantage, stats))

            ranked.sort(key=lambda t: -t[0])

            for advantage, stats in ranked[: self.max_candidates]:
                cand_strategy = _swap_model(
                    baseline, step.step_id, stats["provider"], stats["model"]
                )
                dims = strategy_mod.diff_dimensions(baseline, cand_strategy)
                if not dims:
                    continue

                per_call_saving = base_cost * advantage
                projected, basis = _monthly_projection(
                    per_call_saving, history.get("traffic") or {}, history.get("lookback_days", 30)
                )
                basis["measured_cost_advantage_ratio"] = round(advantage, 4)

                out.append(Candidate(
                    title=f"Switch step {step.step_id} to {stats['model']}",
                    strategy=cand_strategy,
                    dimensions=dims,
                    generator=self.name,
                    rationale=(
                        f"{stats['model']} has run on this workload "
                        f"{stats['runs']} times and averaged "
                        f"{advantage * 100:.0f}% cheaper than {base_model}. Those "
                        f"runs were on DIFFERENT inputs, so this is an observation, "
                        f"not a controlled comparison — a replay is still required."
                    ),
                    evidence_source="observational",
                    projected_savings_usd=projected,
                    projection_basis=basis,
                    measured_basis={
                        "model": stats["model"],
                        "runs": stats["runs"],
                        "avg_cost_usd": stats["avg_cost"],
                        "avg_latency_ms": stats["avg_latency"],
                        "error_rate": stats["error_rate"],
                        "cost_variation": stats.get("cost_variation"),
                        "baseline_model": base_model,
                        "baseline_avg_cost_usd": base_cost,
                        "baseline_runs": base.get("runs"),
                        "source": "workflow_runs.node_results",
                    },
                    notes=[{
                        "code": "observational_not_counterfactual",
                        "detail": (
                            "Measured on this workload but not against the same "
                            "inputs as the baseline. Confidence is capped "
                            "accordingly until a replay runs."
                        ),
                    }],
                ))

        return out


# ---------------------------------------------------------------------------
# Stage 2 — CONTEXT EFFICIENCY. Internally `context_reduction`.
# ---------------------------------------------------------------------------
#
# WHAT THIS DIMENSION CLAIMS, and it claims nothing wider:
#
#     Same model. Same provider. Same sampling parameters. Same tools. Same
#     output contract. ONLY the prompt/context representation changes.
#
# That is deliberately narrower than "prompt optimization", which is not what
# this is and is not what it will be called. Prompt optimization invites
# rewriting prompts for style, tone and cleverness; this removes tokens that a
# MEASUREMENT says are there and a REPLAY says were not load-bearing.
#
# THE GOVERNING RULE
# ------------------
# An LLM may PROPOSE a shorter prompt. An LLM may never be the EVIDENCE that
# the shorter prompt works. Replay is the evidence.
#
# Concretely: a model-authored candidate is a candidate and nothing more. It
# enters the same paired replay against the same recorded production inputs, is
# measured for real provider cost and latency, and must clear the same paired
# non-inferiority test at the same policy margin as a model swap. A candidate
# that cannot be measured is not a finding — it is excluded, with a code.
#
# THE FUNNEL, and where each stage lives
# --------------------------------------
#   original strategy                       optimization.strategy.from_graph_json
#   find reducible context                  optimization.context_accounting
#   generate conservative variants          _variants below
#   STATIC compatibility checks             _static_checks below — no provider
#                                           request exists yet
#   small replay stage                      optimization.staging (stage 1)
#   quality futility stopping               optimization.staging.early_stop_assessment
#   cost / latency measurement              optimization.benchmark._execute_arm
#   full paired verification                optimization.noninferiority
#
# Only the middle three are new. Everything below the static checks is the
# benchmark machinery that already carries every other candidate, reused
# unchanged and on purpose: the same models, the same recorded cases, the same
# quality checks, the same paired non-inferiority test.
#
# WHAT REMAINS TRUE ABOUT THE ORIGINAL BLOCKERS
# ---------------------------------------------
# Blocker 1 (no per-component token breakdown) is answered by
# optimization.context_accounting, which MEASURES tokens per component and
# returns None — never an estimate — where it cannot.
#
# Blocker 2 (a compressed prompt still looks plausible and degrades on the tail)
# is the constraint this generator is shaped around, and it is NOT waved away:
#
#   * The benchmark's quality signal is already deterministic. `_run_quality_checks`
#     scores `deterministic`, `structural` and `format` checks and explicitly
#     refuses `model_graded` — so the evidence path cannot be an LLM judge even
#     if someone wanted it to be.
#   * This generator refuses to propose at all for a workload that has no such
#     check (`context_reduction_quality_signal_insufficient`), because for that
#     workload the failure mode genuinely is undetectable.
#   * It refuses on a case set too small to contain a tail
#     (`context_reduction_case_count_insufficient`).
#   * Variants are CONSERVATIVE by construction, and every one of them is
#     checked statically for the things whose loss a plausible-looking output
#     would hide: an un-interpolated placeholder, a dropped tool reference, a
#     dropped output-format instruction.
#
# What is still NOT established is that a passing eval suite COVERS the tail.
# That is a property of the customer's eval suite, not of this module, and it is
# why nothing here promotes anything: human approval stays the default.
#
# Blocker 3 (a model-written prompt is not evidence) is resolved by the
# governing rule above, and structurally: a proposer is an injected dependency,
# its output is marked `proposed_rewrite`, and it takes exactly the same path as
# a variant produced by deterministic string surgery.


#: Eval-check types that produce a MEASURABLE quality verdict. Mirrors
#: `benchmark._QUALITY_CHECK_PROVENANCE`; `model_graded` is absent from BOTH,
#: which is the whole reason this dimension can exist honestly.
MEASURABLE_QUALITY_CHECK_TYPES = ("deterministic", "structural", "format")


def _load_replay_cases(org_id: str, workflow_id: Optional[str]) -> list[dict]:
    """
    The workload's recorded replay cases, read through the benchmark's own
    loader so there is exactly one definition of "the case set".

    Imported lazily: optimization.benchmark imports this module, and a
    module-level import would close the cycle.
    """
    if not workflow_id:
        return []
    try:
        from optimization import benchmark as benchmark_mod
        return benchmark_mod._load_golden_inputs(org_id, workflow_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("replay case load failed: %s", type(exc).__name__)
        return []


def _load_quality_checks(org_id: str, workflow_id: Optional[str]) -> list[dict]:
    """The workload's own eval-suite checks, via the benchmark's loader."""
    if not workflow_id:
        return []
    try:
        from optimization import benchmark as benchmark_mod
        return benchmark_mod._load_eval_checks(org_id, workflow_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("quality check load failed: %s", type(exc).__name__)
        return []


def measurable_quality_checks(checks: list[dict], cases: list[dict]) -> list[dict]:
    """
    Which of a workload's eval checks would actually PRODUCE a quality number.

    Mirrors the skip conditions in `benchmark._run_quality_checks` exactly, and
    is evaluated against the real case set: a `deterministic` check with no
    expected output anywhere in the case set scores nothing, and a suite made
    only of those cannot detect a regression however many checks it declares.
    """
    has_expected = any(
        (c.get("expected_output") or "").strip() for c in (cases or [])
        if isinstance(c, dict)
    )
    usable: list[dict] = []
    for ch in (checks or []):
        if not isinstance(ch, dict):
            continue
        ctype = (ch.get("type") or "deterministic").lower()
        cfg = ch.get("config") or {}
        if ctype == "deterministic" and has_expected:
            usable.append(ch)
        elif ctype == "structural" and cfg.get("expect_json"):
            usable.append(ch)
        elif ctype == "format" and (cfg.get("pattern") or "").strip():
            usable.append(ch)
    return usable


def _context_accounting():
    """
    The measured per-component token breakdown, or None if it is not present.

    Imported defensively and by name so this module states its dependency
    without requiring it: with no accounting module there is no measurement, and
    with no measurement there is no candidate — which is the correct outcome,
    not a degraded one.
    """
    try:
        from optimization import context_accounting  # type: ignore
        return context_accounting
    except Exception:
        return None


_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
_JSON_KEY_RE = re.compile(r'"(\w+)"\s*:')

#: Lines that state an OUTPUT CONTRACT. Deliberately broad: a false positive
#: costs one refused variant, a false negative costs a customer malformed output
#: in production. The asymmetry decides the tuning.
_OUTPUT_CONTRACT_RE = re.compile(
    r"(?i)\b("
    r"json|yaml|xml|csv|markdown|schema|output\s+format|format\s*:|"
    r"respond\s+(?:only\s+)?(?:with|in)|return\s+only|reply\s+only|"
    r"must\s+(?:return|output|respond|be)|valid\s+json|code\s*fence|"
    r"do\s+not\s+include|no\s+preamble|fields?\s*:|keys?\s*:"
    r")\b"
)


def _norm_line(line: str) -> str:
    return " ".join(str(line).split())


def output_contract_markers(text: Optional[str], declared: Optional[dict] = None) -> set[str]:
    """
    The output-contract commitments a prompt makes, as comparable markers.

    Three sources, all read off real data:
      * whitespace-normalised lines that state a format or schema instruction,
      * JSON-ish keys appearing in an inline example or schema block,
      * property names from a schema DECLARED on the node
        (`strategy._node_invariants`), which must survive even when the prompt
        never spelled them out.

    Comparison is by normalised line, so reformatting whitespace preserves a
    marker and REWORDING a format instruction does not. That is intentional: a
    reworded output contract is a changed output contract until a replay says
    otherwise, and this check runs before any replay exists.
    """
    markers: set[str] = set()
    if isinstance(text, str) and text:
        for line in text.splitlines():
            norm = _norm_line(line)
            if norm and _OUTPUT_CONTRACT_RE.search(norm):
                markers.add(f"line:{norm.lower()}")
        for key in _JSON_KEY_RE.findall(text):
            markers.add(f"key:{key}")
    for value in (declared or {}).values():
        markers.update(f"key:{k}" for k in _declared_property_names(value))
    return markers


def _declared_property_names(value: Any, _depth: int = 0) -> set[str]:
    """Property names inside a declared schema, at any nesting depth."""
    names: set[str] = set()
    if _depth > 6 or not isinstance(value, dict):
        return names
    props = value.get("properties")
    if isinstance(props, dict):
        for key, sub in props.items():
            names.add(str(key))
            names |= _declared_property_names(sub, _depth + 1)
    for key in ("json_schema", "schema", "items"):
        names |= _declared_property_names(value.get(key), _depth + 1)
    return names


def remove_duplicate_blocks(text: Optional[str]) -> Optional[str]:
    """
    Drop later EXACT repeats of a block, keeping the first occurrence.

    Deterministic, no model involved, and the most conservative text reduction
    there is: the surviving prompt is a subsequence of the original, so it can
    contain no instruction the original did not, and every instruction that
    appeared at least once still appears.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    seen: set[str] = set()
    kept: list[str] = []
    for block in _BLOCK_SPLIT_RE.split(text):
        key = _norm_line(block).lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(block.strip())
    return "\n\n".join(kept)


def normalize_prompt_whitespace(text: Optional[str]) -> Optional[str]:
    """
    Strip trailing whitespace and collapse runs of blank lines to one.

    Lossless with respect to every check in this module: markers, placeholders
    and tool references are all compared on whitespace-normalised text, so
    nothing this can remove is something anything else depends on.
    """
    if not isinstance(text, str) or not text:
        return text
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip()


@dataclass(frozen=True)
class ContextVariant:
    """
    One proposed reduction, before any check has run.

    `origin` records WHO wrote it — 'deterministic' or 'model' — and travels all
    the way into the audit trail. A model-authored variant is given no
    additional credit anywhere; the field exists so a reader can see which
    findings came from a proposal and which from string surgery.
    """

    kind: str
    step_id: str
    strategy: strategy_mod.Strategy
    origin: str = "deterministic"
    baseline_text: Optional[str] = None
    candidate_text: Optional[str] = None
    facts: dict = field(default_factory=dict)

    @property
    def is_text_rewrite(self) -> bool:
        return self.candidate_text is not None


class ContextReductionGenerator:
    """
    Propose the SAME work, carrying less context.

    Evidence class: NONE on emission. A measured token reduction is a
    measurement of the PROMPT, not of cost and not of quality; it says the
    candidate is worth benchmarking and nothing else. Only
    optimization/benchmark.py can upgrade the evidence, by running the arm.

    Dependencies are injected so the module can be exercised without a database
    and without a provider:

      `accounting`  the module implementing `profile_workload_context`. Defaults
                    to `optimization.context_accounting` if importable.
      `proposer`    optional callable proposing a shorter text. It may be an
                    LLM. Its output is a CANDIDATE and never evidence.
    """

    name = "context_reduction"

    #: Fraction of the effective character budget each budget variant proposes.
    #: Two only: this is a search for a defensible reduction, not a sweep.
    BUDGET_FRACTIONS = (0.75, 0.5)

    #: A reduction must clear BOTH floors to be worth a benchmark. The absolute
    #: floor stops a five-token win from consuming a replay budget; the relative
    #: floor keeps the reduction meaningful against the workload's own size.
    MIN_TOKEN_REDUCTION = 25
    MIN_TOKEN_REDUCTION_RATIO = 0.05

    #: Fewest recorded cases at which a tail can plausibly be present. Matches
    #: the first stage of the default staged-evaluation schedule
    #: (`staging.DEFAULT_STAGE_SIZES[0]`) — below it the first replay stage
    #: cannot even be filled.
    MIN_CASES = 30

    def __init__(
        self,
        max_candidates: int = 3,
        *,
        accounting: Any = None,
        proposer: Any = None,
    ):
        self.max_candidates = max_candidates
        self._accounting = accounting
        self.proposer = proposer

    # -- plumbing ---------------------------------------------------------

    @property
    def accounting(self):
        return self._accounting if self._accounting is not None else _context_accounting()

    def generate(self, workload, baseline, history) -> list[Candidate]:
        """Candidates only. See `generate_with_report` for the exclusions."""
        return self.generate_with_report(workload, baseline, history)[0]

    # -- the funnel -------------------------------------------------------

    def generate_with_report(
        self, workload, baseline, history
    ) -> tuple[list[Candidate], list[dict]]:
        """
        Returns (candidates, exclusions).

        An exclusion is in `generate_candidates`' `dropped` shape, so it lands
        in the consideration funnel at reason-code grain beside every other
        pre-dispatch refusal. A refused variant is a finding about a thorough
        search; it is not an error and it is not silence.
        """
        excluded: list[dict] = []
        steps = _primary_model_steps(baseline)
        if not steps:
            return [], excluded

        org_id = history.get("org_id")
        workflow_id = history.get("workflow_id")

        cases = history.get("replay_cases")
        if cases is None:
            cases = _load_replay_cases(org_id, workflow_id)
        checks = history.get("quality_checks")
        if checks is None:
            checks = _load_quality_checks(org_id, workflow_id)

        # ── Blocker 2, enforced before anything else is even proposed. ──────
        # These two refusals are about the WORKLOAD, not about a variant, so
        # exactly one record is emitted for each rather than one per variant.
        usable_checks = measurable_quality_checks(checks, cases)
        if not usable_checks:
            return [], [{
                "title": None,
                "code": "context_reduction_quality_signal_insufficient",
                "checks_declared": len(checks or []),
                "checks_measurable": 0,
                "measurable_types": list(MEASURABLE_QUALITY_CHECK_TYPES),
                "cases": len(cases or []),
            }]
        if len(cases or []) < self.MIN_CASES:
            return [], [{
                "title": None,
                "code": "context_reduction_case_count_insufficient",
                "observed": len(cases or []),
                "required": self.MIN_CASES,
                "unit": "cases",
            }]

        acct = self.accounting
        out: list[Candidate] = []

        for step in steps:
            profile = self._profile(acct, org_id, workload, cases, step.step_id)
            if profile is None:
                if self._has_reducible_material(step):
                    excluded.append({
                        "title": f"Reduce context on step {step.step_id}",
                        "code": "context_reduction_unmeasured",
                        "step_id": step.step_id,
                        "stage": "profile",
                        "detail_code": (
                            "accounting_unavailable" if acct is None
                            else "profile_not_produced"
                        ),
                    })
                continue

            reducible = getattr(profile, "reducible_tokens", None)
            if reducible is None:
                excluded.append({
                    "title": f"Reduce context on step {step.step_id}",
                    "code": "context_reduction_unmeasured",
                    "step_id": step.step_id,
                    "stage": "reducible_tokens",
                    "tokenizer": getattr(profile, "tokenizer", None),
                    "coverage": getattr(profile, "coverage", None),
                })
                continue
            if reducible <= 0:
                excluded.append({
                    "title": f"Reduce context on step {step.step_id}",
                    "code": "context_reduction_immaterial",
                    "step_id": step.step_id,
                    "reducible_tokens": reducible,
                    "stage": "reducible_tokens",
                })
                continue

            cands, step_excluded = self._for_step(
                acct, org_id, workload, cases, baseline, step, profile, history,
            )
            out.extend(cands)
            excluded.extend(step_excluded)

        selected, overflow = self._select(out)
        return selected, excluded + overflow

    def _select(self, cands: list[Candidate]) -> tuple[list[Candidate], list[dict]]:
        """
        Cap the search at `max_candidates`, VISIBLY.

        A replay arm costs real money, so the number of them is bounded. Two
        properties matter about how:

          * DIVERSITY. One variant of each kind is taken first, in kind order,
            so a step with two budget variants cannot crowd out the text
            variant entirely — kinds fail for different reasons, and measuring
            two flavours of the same idea learns less than measuring two ideas.
          * VISIBILITY. What the cap set aside is REPORTED, with a code. A
            silent slice would make a bounded search indistinguishable from an
            exhaustive one that found nothing more.
        """
        if len(cands) <= self.max_candidates:
            return cands, []

        def kind(c: Candidate) -> str:
            return (c.measured_basis or {}).get("variant_kind") or ""

        chosen: list[int] = []
        seen_kinds: set[str] = set()
        for k in domain.CONTEXT_VARIANT_KINDS:
            for i, cand in enumerate(cands):
                if kind(cand) == k and k not in seen_kinds:
                    chosen.append(i)
                    seen_kinds.add(k)
                    break
        for i in range(len(cands)):
            if i not in chosen:
                chosen.append(i)

        keep = set(chosen[: self.max_candidates])
        selected = [c for i, c in enumerate(cands) if i in keep]
        overflow = [{
            "title": c.title,
            "code": "context_reduction_variant_not_selected",
            "variant_kind": kind(c),
            "variant_origin": (c.measured_basis or {}).get("variant_origin"),
            "step_id": (c.measured_basis or {}).get("step_id"),
            "tokens_reduced": (c.measured_basis or {}).get("tokens_reduced"),
            "max_candidates": self.max_candidates,
        } for i, c in enumerate(cands) if i not in keep]
        return selected, overflow

    # -- per step ---------------------------------------------------------

    def _for_step(
        self, acct, org_id, workload, cases, baseline, step, profile, history,
    ) -> tuple[list[Candidate], list[dict]]:
        out: list[Candidate] = []
        excluded: list[dict] = []

        for variant in self._variants(baseline, step, profile):
            verdict = self._static_checks(baseline, step, variant, profile)
            if not verdict["ok"]:
                excluded.append({
                    "title": self._title(variant, step),
                    "code": verdict["code"],
                    "step_id": step.step_id,
                    "variant_kind": variant.kind,
                    "variant_origin": variant.origin,
                    **verdict["facts"],
                })
                continue

            measurement = self._measure(
                acct, org_id, workload, cases, variant, profile,
            )
            if measurement["code"]:
                excluded.append({
                    "title": self._title(variant, step),
                    "code": measurement["code"],
                    "step_id": step.step_id,
                    "variant_kind": variant.kind,
                    "variant_origin": variant.origin,
                    **measurement["facts"],
                })
                continue

            out.append(self._candidate(
                baseline, step, variant, profile, measurement, history,
            ))

        return out, excluded

    @staticmethod
    def _has_reducible_material(step) -> bool:
        cfg = step.config or {}
        ctx = cfg.get("context") or {}
        return bool(
            (ctx.get("enabled") and ctx.get("source_count"))
            or (cfg.get("task_description") or "").strip()
            or (cfg.get("system_instructions") or "").strip()
        )

    @staticmethod
    def _title(variant: "ContextVariant", step) -> str:
        return f"Reduce context on step {step.step_id} ({variant.kind})"

    def _profile(self, acct, org_id, workload, cases, step_id):
        if acct is None or not hasattr(acct, "profile_workload_context"):
            return None
        try:
            return acct.profile_workload_context(
                org_id, workload, cases=cases, step_id=step_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "context profiling failed for step %s: %s", step_id, type(exc).__name__,
            )
            return None

    # -- variant generation ----------------------------------------------

    def _variants(self, baseline, step, profile) -> list[ContextVariant]:
        """
        Conservative variants, cheapest and most defensible FIRST.

        A budget change rewrites nothing at all — it lowers a character budget
        the runtime already enforces — so it leads. Text surgery follows, and a
        model-authored proposal comes last and last for a reason.
        """
        variants: list[ContextVariant] = []
        variants.extend(self._budget_variants(baseline, step, profile))
        variants.extend(self._text_variants(baseline, step))
        variants.extend(self._proposed_variants(baseline, step, profile))
        return variants

    def _budget_variants(self, baseline, step, profile) -> list[ContextVariant]:
        ctx = (step.config or {}).get("context") or {}
        if not ctx.get("enabled"):
            return []

        sizes = _measured_source_chars(profile)
        labels = list(ctx.get("source_labels") or [])
        required = list(ctx.get("source_required") or [])
        measured_total = sum(sizes.get(lbl, 0) for lbl in labels)
        required_chars = sum(
            sizes.get(lbl, 0)
            for lbl, req in zip(labels, required + [False] * len(labels))
            if req
        )

        effective = ctx.get("packaging_max_chars")
        budget_basis = "declared_packaging_max_chars"
        if not effective:
            # No declared budget: the effective one is what the workload
            # MEASURABLY assembles today. Not an estimate — it is the measured
            # size of the context this step really packages.
            effective = measured_total
            budget_basis = "measured_assembled_chars"
        if not effective:
            return []

        out: list[ContextVariant] = []
        for fraction in self.BUDGET_FRACTIONS:
            budget = int(effective * fraction)
            if budget <= 0:
                continue
            out.append(ContextVariant(
                kind=domain.CONTEXT_VARIANT_PACKAGING_BUDGET,
                step_id=step.step_id,
                strategy=strategy_mod.with_context_budget(
                    baseline, step.step_id, packaging_max_chars=budget,
                ),
                facts={
                    "packaging_max_chars": budget,
                    "baseline_packaging_max_chars": ctx.get("packaging_max_chars"),
                    "effective_baseline_chars": effective,
                    "budget_basis": budget_basis,
                    "budget_fraction": fraction,
                    "measured_required_source_chars": required_chars,
                },
            ))

        source_variant = self._largest_optional_source_variant(
            baseline, step, ctx, sizes,
        )
        if source_variant is not None:
            out.append(source_variant)
        return out

    def _largest_optional_source_variant(self, baseline, step, ctx, sizes):
        """
        Halve the budget of the largest NON-REQUIRED source.

        "Reducing irrelevant retrieved context", stated precisely: the workload
        itself declared this source optional, and the accounting module measured
        it as the biggest thing on the step.
        """
        labels = list(ctx.get("source_labels") or [])
        budgets = list(ctx.get("source_max_chars") or [])
        required = list(ctx.get("source_required") or [])
        if not labels or len(budgets) != len(labels):
            return None
        required += [False] * (len(labels) - len(required))

        ranked = sorted(
            (
                (sizes.get(lbl, 0), i, lbl)
                for i, lbl in enumerate(labels)
                if not required[i] and sizes.get(lbl, 0) > 0
            ),
            reverse=True,
        )
        if not ranked:
            return None
        chars, index, label = ranked[0]
        new_budgets = list(budgets)
        new_budgets[index] = int(chars * 0.5)
        if new_budgets[index] <= 0 or new_budgets[index] == budgets[index]:
            return None
        return ContextVariant(
            kind=domain.CONTEXT_VARIANT_SOURCE_BUDGET,
            step_id=step.step_id,
            strategy=strategy_mod.with_context_budget(
                baseline, step.step_id, source_max_chars=new_budgets,
            ),
            facts={
                "source_index": index,
                "source_label": label,
                "source_required": False,
                "measured_source_chars": chars,
                "source_max_chars": new_budgets[index],
                "baseline_source_max_chars": budgets[index],
            },
        )

    def _text_variants(self, baseline, step) -> list[ContextVariant]:
        cfg = step.config or {}
        task = cfg.get("task_description")
        system = cfg.get("system_instructions")
        out: list[ContextVariant] = []

        for kind, fn in (
            (domain.CONTEXT_VARIANT_DUPLICATE_BLOCK_REMOVAL, remove_duplicate_blocks),
            (domain.CONTEXT_VARIANT_WHITESPACE_NORMALIZATION, normalize_prompt_whitespace),
        ):
            new_task = fn(task) if isinstance(task, str) else None
            new_system = fn(system) if isinstance(system, str) else None
            if (new_task or "") == (task or "") and (new_system or "") == (system or ""):
                continue
            out.append(ContextVariant(
                kind=kind,
                step_id=step.step_id,
                strategy=strategy_mod.with_prompt_text(
                    baseline, step.step_id,
                    task_description=new_task if new_task != task else None,
                    system_instructions=new_system if new_system != system else None,
                ),
                baseline_text=_joined_prompt(task, system),
                candidate_text=_joined_prompt(new_task, new_system),
                facts={
                    "baseline_chars": len(_joined_prompt(task, system)),
                    "candidate_chars": len(_joined_prompt(new_task, new_system)),
                },
            ))
        return out

    def _proposed_variants(self, baseline, step, profile) -> list[ContextVariant]:
        """
        A model-authored proposal, if a proposer was injected.

        It is generated LAST, marked `origin='model'`, and given no shortcut of
        any kind: the same static checks, the same measurement requirement, the
        same replay, the same non-inferiority margin. A proposal that cannot be
        measured produces no finding, exactly as if a human had written it on a
        napkin.
        """
        if self.proposer is None:
            return []
        cfg = step.config or {}
        task = cfg.get("task_description")
        system = cfg.get("system_instructions")
        try:
            proposed = self.proposer(
                task_description=task,
                system_instructions=system,
                profile=profile,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("context proposer failed: %s", type(exc).__name__)
            return []
        if not isinstance(proposed, dict):
            return []

        new_task = proposed.get("task_description")
        new_system = proposed.get("system_instructions")
        if new_task is None and new_system is None:
            return []
        if (new_task or task or "") == (task or "") and (new_system or system or "") == (system or ""):
            return []

        return [ContextVariant(
            kind=domain.CONTEXT_VARIANT_PROPOSED_REWRITE,
            step_id=step.step_id,
            origin="model",
            strategy=strategy_mod.with_prompt_text(
                baseline, step.step_id,
                task_description=new_task,
                system_instructions=new_system,
            ),
            baseline_text=_joined_prompt(task, system),
            candidate_text=_joined_prompt(
                new_task if new_task is not None else task,
                new_system if new_system is not None else system,
            ),
            facts={
                "proposer": getattr(self.proposer, "__name__", type(self.proposer).__name__),
                "proposer_model": proposed.get("model"),
            },
        )]

    # -- static compatibility checks, BEFORE any provider call ------------

    def _static_checks(self, baseline, step, variant: ContextVariant, profile) -> dict:
        """
        Every check here runs against REAL data and produces its OWN reason
        code. None of them makes a provider request; a variant that fails one
        was never dispatched, and that is the point.
        """
        def fail(code, **facts):
            return {"ok": False, "code": code, "facts": facts}

        # 1. Nothing but prompt text and context budget may differ. This is the
        #    definition of the dimension, so it is asserted rather than assumed.
        scope = strategy_mod.context_only_change(baseline, variant.strategy)
        if not scope["ok"]:
            return fail(
                "context_reduction_changed_other_dimension",
                changed_dimensions=scope["changed"],
                unexpected_dimensions=scope["unexpected"],
                allowed_dimensions=scope["allowed"],
            )

        invariants = (step.config or {}).get(strategy_mod.CONFIG_KEY_INVARIANTS) or {}

        if variant.is_text_rewrite:
            before, after = variant.baseline_text or "", variant.candidate_text or ""

            # 2. Every placeholder the runtime interpolates survives.
            lost = sorted(
                strategy_mod.placeholders_in(before) - strategy_mod.placeholders_in(after)
            )
            if lost:
                return fail(
                    "context_placeholder_dropped",
                    dropped_placeholders=lost,
                    baseline_placeholders=sorted(strategy_mod.placeholders_in(before)),
                )

            # 3. Every tool the step has bound and the prompt referenced
            #    survives. A tool the baseline never mentioned is not made a
            #    requirement by this check.
            dropped_tools = sorted(
                name for name in (invariants.get("tool_names") or [])
                if name and name in before and name not in after
            )
            if dropped_tools:
                return fail(
                    "context_tool_reference_dropped",
                    dropped_tools=dropped_tools,
                    bound_tools=list(invariants.get("tool_names") or []),
                )

            # 4. Every output-schema / output-format commitment survives.
            declared = invariants.get("declared_output_contract") or {}
            lost_markers = sorted(
                output_contract_markers(before, declared)
                - output_contract_markers(after, declared)
            )
            if lost_markers:
                return fail(
                    "context_output_contract_dropped",
                    dropped_markers=lost_markers[:20],
                    dropped_marker_count=len(lost_markers),
                )

        # 5. The variant still fits what the workload REQUIRES, measured.
        budget_fail = self._budget_requirement_check(step, variant, profile)
        if budget_fail is not None:
            return budget_fail

        return {"ok": True, "code": None, "facts": {}}

    def _budget_requirement_check(self, step, variant: ContextVariant, profile):
        """
        A budget below the MEASURED size of required context would truncate
        content the workload declares it must have. Measured, from the
        accounting module — not inferred from the config.
        """
        ctx = (variant.strategy.step(step.step_id).config or {}).get("context") or {}
        base_ctx = (step.config or {}).get("context") or {}
        if not ctx.get("enabled"):
            return None
        if (
            ctx.get("packaging_max_chars") == base_ctx.get("packaging_max_chars")
            and (ctx.get("source_max_chars") or []) == (base_ctx.get("source_max_chars") or [])
        ):
            # This variant proposes no budget at all, so it cannot be the reason
            # a budget is too small. Reporting a baseline condition as a
            # variant's failure would blame the candidate for what the workload
            # already does.
            return None
        sizes = _measured_source_chars(profile)
        labels = list(ctx.get("source_labels") or [])
        required = list(ctx.get("source_required") or [])
        required += [False] * (len(labels) - len(required))
        required_chars = sum(
            sizes.get(lbl, 0) for lbl, req in zip(labels, required) if req
        )

        packaging = ctx.get("packaging_max_chars")
        if packaging is not None and required_chars and packaging < required_chars:
            return {
                "ok": False,
                "code": "context_budget_below_requirement",
                "facts": {
                    "scope": "packaging",
                    "proposed_max_chars": packaging,
                    "measured_required_chars": required_chars,
                    "required_sources": [
                        lbl for lbl, req in zip(labels, required) if req
                    ],
                },
            }

        budgets = list(ctx.get("source_max_chars") or [])
        for i, label in enumerate(labels):
            if i >= len(budgets) or budgets[i] is None or not required[i]:
                continue
            measured = sizes.get(label, 0)
            if measured and budgets[i] < measured:
                return {
                    "ok": False,
                    "code": "context_budget_below_requirement",
                    "facts": {
                        "scope": "source",
                        "source_index": i,
                        "source_label": label,
                        "proposed_max_chars": budgets[i],
                        "measured_required_chars": measured,
                    },
                }
        return None

    # -- measurement -------------------------------------------------------

    def _measure(self, acct, org_id, workload, cases, variant, profile) -> dict:
        """
        MEASURE the reduction, or refuse the variant.

        Two measurement paths, both real:

          text rewrites   the exact resulting text is in hand, so
                          `count_tokens` on before and after is a direct
                          measurement of the artifact that will be sent.
          budget changes  the resulting text is assembled by the runtime and is
                          NOT in hand, so the candidate strategy is re-profiled
                          through the same accounting function that produced the
                          baseline profile. Same function, same tokenizer,
                          comparable numbers.

        Where neither path is available the result is `None`, and None means NOT
        MEASURED. It is never filled in from character counts: chars-per-token
        is a ratio that varies with exactly the content being cut, so a
        character-derived token saving would be an estimate dressed as a
        measurement, and would be the one number a customer would quote back.
        """
        facts: dict[str, Any] = {"tokenizer": getattr(profile, "tokenizer", None)}
        before = after = None

        if variant.is_text_rewrite:
            counter = getattr(acct, "count_tokens", None) if acct else None
            if counter is None:
                return {
                    "code": "context_reduction_unmeasured",
                    "facts": {**facts, "stage": "count_tokens",
                              "detail_code": "count_tokens_unavailable"},
                }
            try:
                before = counter(
                    variant.baseline_text or "",
                    tokenizer=getattr(profile, "tokenizer", None),
                )
                after = counter(
                    variant.candidate_text or "",
                    tokenizer=getattr(profile, "tokenizer", None),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("token count failed: %s", type(exc).__name__)
                return {
                    "code": "context_reduction_unmeasured",
                    "facts": {**facts, "stage": "count_tokens",
                              "detail_code": "count_tokens_raised"},
                }
            facts["measurement_path"] = "count_tokens"
        else:
            profiler = getattr(acct, "profile_strategy_context", None) if acct else None
            if profiler is None:
                return {
                    "code": "context_reduction_unmeasured",
                    "facts": {**facts, "stage": "profile_strategy_context",
                              "detail_code": "profile_strategy_context_unavailable"},
                }
            try:
                cand_profile = profiler(
                    org_id, workload, variant.strategy,
                    cases=cases, step_id=variant.step_id,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("candidate profiling failed: %s", type(exc).__name__)
                cand_profile = None
            if cand_profile is None:
                return {
                    "code": "context_reduction_unmeasured",
                    "facts": {**facts, "stage": "profile_strategy_context",
                              "detail_code": "candidate_profile_not_produced"},
                }
            if getattr(cand_profile, "tokenizer", None) != getattr(profile, "tokenizer", None):
                # Two token counts from two different tokenizers are not a
                # difference; subtracting them would manufacture one.
                return {
                    "code": "context_reduction_unmeasured",
                    "facts": {
                        **facts, "stage": "tokenizer_mismatch",
                        "candidate_tokenizer": getattr(cand_profile, "tokenizer", None),
                    },
                }
            before = getattr(profile, "total_tokens", None)
            after = getattr(cand_profile, "total_tokens", None)
            facts["measurement_path"] = "profile_strategy_context"
            facts["candidate_coverage"] = getattr(cand_profile, "coverage", None)

        if before is None or after is None:
            return {
                "code": "context_reduction_unmeasured",
                "facts": {**facts, "stage": "token_totals",
                          "tokens_before": before, "tokens_after": after},
            }

        reduction = int(before) - int(after)
        total = getattr(profile, "total_tokens", None)
        ratio_basis = "workload_total_tokens" if total else "measured_artifact_tokens"
        denominator = total if total else before
        ratio = (reduction / float(denominator)) if denominator else None

        facts.update({
            "tokens_before": int(before),
            "tokens_after": int(after),
            "tokens_reduced": reduction,
            "reduction_ratio": round(ratio, 6) if ratio is not None else None,
            "reduction_ratio_basis": ratio_basis,
            "workload_total_tokens": total,
            "coverage": getattr(profile, "coverage", None),
            "n_cases": getattr(profile, "n_cases", None),
        })

        if reduction < self.MIN_TOKEN_REDUCTION or (
            ratio is not None and ratio < self.MIN_TOKEN_REDUCTION_RATIO
        ):
            return {
                "code": "context_reduction_immaterial",
                "facts": {
                    **facts,
                    "min_tokens_reduced": self.MIN_TOKEN_REDUCTION,
                    "min_reduction_ratio": self.MIN_TOKEN_REDUCTION_RATIO,
                },
            }

        return {"code": None, "facts": facts}

    # -- candidate assembly ------------------------------------------------

    def _candidate(self, baseline, step, variant, profile, measurement, history) -> Candidate:
        facts = measurement["facts"]
        dims = strategy_mod.diff_dimensions(baseline, variant.strategy)

        per_call, pricing = _input_token_saving_usd(
            step.executor_ref or {}, facts["tokens_reduced"],
        )
        projected, basis = _monthly_projection(
            per_call, history.get("traffic") or {}, history.get("lookback_days", 30),
        )
        basis["input_token_pricing"] = pricing
        basis["tokens_reduced"] = facts["tokens_reduced"]
        basis["basis_note_code"] = "vendor_input_price_times_measured_token_reduction"

        notes = [{
            "code": "context_token_measurement_only",
            "detail": (
                "The token reduction is MEASURED on the prompt. It is not a "
                "measured cost, latency or quality result, and it becomes one "
                "only when the paired replay runs."
            ),
        }, {
            "code": "same_model_same_provider",
            "detail": (
                "Model, provider, sampling parameters, tools and output "
                "contract are unchanged; only the context representation "
                "differs."
            ),
        }]
        if variant.origin == "model":
            notes.append({
                "code": "model_authored_proposal",
                "detail": (
                    "This text was written by a model. That makes it a "
                    "candidate and nothing else: it clears the same paired "
                    "non-inferiority test at the same margin, or it is not a "
                    "finding."
                ),
            })

        return Candidate(
            title=self._title(variant, step),
            strategy=variant.strategy,
            dimensions=dims,
            generator=self.name,
            rationale=(
                f"Measured context on step {step.step_id} is "
                f"{facts.get('workload_total_tokens')} tokens; this variant "
                f"({variant.kind}) measures {facts['tokens_reduced']} tokens "
                f"smaller on the same recorded cases. The model, provider and "
                f"output contract are unchanged. A measured token reduction is "
                f"a reason to replay, not a saving."
            ),
            evidence_source="none",
            projected_savings_usd=projected,
            projection_basis=basis,
            measured_basis={
                "kind": "context_token_measurement",
                "is_outcome_evidence": False,
                "variant_kind": variant.kind,
                "variant_origin": variant.origin,
                "step_id": step.step_id,
                "source": "optimization.context_accounting",
                **facts,
                **variant.facts,
            },
            notes=notes,
        )


def _joined_prompt(task: Optional[str], system: Optional[str]) -> str:
    """
    The static prompt text of a step, in the order the runtime assembles it.

    `workflow_runtime` prepends the system instructions to the task, so the
    checks operate on the same concatenation the provider will see.
    """
    return "\n\n".join(p for p in (system or "", task or "") if p)


def _measured_source_chars(profile) -> dict[str, int]:
    """
    MEASURED characters per context source, keyed by the source's label.

    Reads `ContextProfile.components`, whose context entries are labelled
    'context_source:<label>' — the same label `contextConfig.sources[].label`
    carries, which is what lets a budget be compared against the size of the
    content it would truncate.
    """
    out: dict[str, int] = {}
    for component in (getattr(profile, "components", None) or ()):
        name = getattr(component, "component", "") or ""
        if not name.startswith("context_source:"):
            continue
        label = name.split(":", 1)[1]
        try:
            out[label] = int(getattr(component, "chars", 0) or 0)
        except (TypeError, ValueError):
            continue
    return out


def _input_token_saving_usd(
    executor_ref: dict, tokens_reduced: int,
) -> tuple[Optional[float], dict]:
    """
    Per-call dollar value of a measured input-token reduction, at LIST PRICE.

    A projection, and labelled as one everywhere it appears. It multiplies a
    MEASURED token reduction by a PUBLISHED price; the token count is real and
    the price is a price sheet, so the product is a hypothesis about money and
    is never written to a verified or realized field.

    Returns (None, basis) when the price is a fallback guess rather than a
    published one — guessing the price of a real token saving would produce a
    fabricated dollar figure just as surely as guessing the tokens.
    """
    provenance = executors.pricing_provenance([executor_ref])
    basis: dict[str, Any] = {
        "pricing_basis": provenance["basis"],
        "price_source": "vendor_list_price:shared/providers.json",
        "unit": "usd_per_1k_tokens",
        "applies_to": "input_tokens",
    }
    if provenance["basis"] != executors.COST_BASIS_MEASURED:
        basis["result"] = "not_priceable"
        basis["estimated_models"] = provenance["estimated_models"]
        return None, basis

    cost_model = executors.vendor_cost_model(
        executor_ref.get("vendor") or "", executor_ref.get("external_id") or "",
    )
    input_price = cost_model.get("input")
    basis["input_price_per_1k"] = input_price
    if input_price is None or tokens_reduced <= 0:
        basis["result"] = "not_priceable"
        return None, basis
    basis["result"] = "priced"
    return (tokens_reduced / 1000.0) * float(input_price), basis


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Generators that can be run today, each backed by real data.
CANDIDATE_GENERATORS: dict[str, CandidateGenerator] = {
    AlternateModelGenerator.name: AlternateModelGenerator(),
    CheaperMeasuredModelGenerator.name: CheaperMeasuredModelGenerator(),
    ContextReductionGenerator.name: ContextReductionGenerator(),
}


def _configured_providers(org_id: str) -> set[str]:
    """
    Providers this org holds a credential for, lowercased.

    An empty set means "unknown" (the lookup failed), and callers must then
    filter nothing — refusing every candidate because a read errored would be
    worse than letting the arms report their own failures.
    """
    try:
        from supabase_client import supabase
        res = (
            supabase.table("api_keys")
            .select("provider")
            .eq("org_id", org_id)
            .execute()
        )
        return {
            str(r.get("provider") or "").strip().lower()
            for r in (res.data or [])
            if r.get("provider")
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "provider credential lookup failed for org %s: %s",
            org_id, type(exc).__name__,
        )
        return set()


def _unconfigured_providers(cand_strategy, configured: set[str]) -> set[str]:
    """Providers named by this strategy's steps that the org cannot execute."""
    if not configured:
        return set()  # unknown -> do not filter
    needed: set[str] = set()
    for step in (getattr(cand_strategy, "steps", None) or []):
        ref = getattr(step, "executor_ref", None) or {}
        vendor = str(ref.get("vendor") or "").strip().lower()
        # 'optiml' is an internal step target, not an external provider.
        if vendor and vendor != "optiml":
            needed.add(vendor)
    return needed - configured


def generate_candidates(
    org_id: str,
    workload: dict,
    baseline: strategy_mod.Strategy,
    *,
    workflow_id: Optional[str] = None,
    generators: Optional[list[str]] = None,
    lookback_days: int = 30,
) -> tuple[list[Candidate], dict]:
    """
    Run the registered generators and return deduped candidates.

    Returns (executable candidates, metadata). The metadata carries TWO further
    lists that must not be conflated:

      `dropped`        candidates that cannot be run and are not opportunities
                       either — inapplicable dimensions, duplicates, generator
                       errors. Each carries a code.
      `opportunities`  TIER 2. Real alternatives from providers this org has not
                       connected. Not benchmarked (the arm would measure
                       nothing) but RETAINED, so the product can say "connect
                       Google to evaluate this" instead of silently narrowing
                       the customer's options to the vendors they already pay.

    Candidates whose dimensions cannot actually be applied to the runtime graph
    are dropped here, with a reason, rather than being benchmarked into a
    measurement of nothing.
    """
    history = build_history(
        org_id, workload, workflow_id=workflow_id, lookback_days=lookback_days
    )
    names = generators or list(CANDIDATE_GENERATORS)
    seen: set[str] = {baseline.fingerprint()}
    out: list[Candidate] = []
    dropped: list[dict] = []
    #: TIER 2 — retained, never executed. See the provider_not_configured branch.
    opportunities: list[dict] = []
    configured = _configured_providers(org_id)
    # Generators rank within executable providers; the post-filter below is
    # defence in depth for generators that ignore this.
    history["configured_providers"] = configured

    for name in names:
        gen = CANDIDATE_GENERATORS.get(name)
        if gen is None:
            continue
        try:
            # A generator may report its OWN pre-dispatch exclusions. A static
            # check that refused a variant is a finding about a thorough search
            # and belongs in the funnel at reason-code grain, not in a log line;
            # `generate` alone cannot express that, so generators that run such
            # checks implement `generate_with_report` and the rest are
            # unaffected.
            reporter = getattr(gen, "generate_with_report", None)
            if callable(reporter):
                produced, gen_excluded = reporter(workload, baseline, history)
            else:
                produced, gen_excluded = gen.generate(workload, baseline, history), []
        except Exception as exc:  # pragma: no cover
            logger.warning("generator %s failed: %s", name, type(exc).__name__)
            dropped.append({"generator": name, "code": "generator_error",
                            "detail": type(exc).__name__})
            continue

        for record in (gen_excluded or []):
            if not isinstance(record, dict) or not record.get("code"):
                continue
            if record["code"] not in domain.REASON_CODES:
                # An undocumented code would enter the API contract silently.
                logger.warning(
                    "generator %s emitted unknown exclusion code %r",
                    name, record["code"],
                )
                continue
            dropped.append({"generator": name, **record})

        for cand in produced:
            # Applicability is per surface: temperature is unapplicable on a
            # workflow graph but genuinely benchmarkable on direct inference.
            unapplicable = strategy_mod.unapplicable_dimensions(
                cand.dimensions, cand.strategy.surface
            )
            if unapplicable:
                dropped.append({
                    "generator": name,
                    "title": cand.title,
                    "code": "strategy_not_applicable",
                    "surface": cand.strategy.surface,
                    "dimensions": sorted(unapplicable),
                    "detail": unapplicable,
                })
                continue
            # A provider the org holds no credential for cannot execute. Running
            # it anyway yields a 100%-error arm with every metric NULL, which is
            # a coverage gap wearing the costume of a policy failure: it reads as
            # "the alternative was tested and lost" when nothing was tested at
            # all. Same principle as strategy_not_applicable above — refuse
            # before producing a measurement of nothing.
            missing = _unconfigured_providers(cand.strategy, configured)
            if missing:
                # NOT BENCHMARKED, NOT FORGOTTEN.
                #
                # Running this arm would produce a 100%-error result with every
                # metric NULL — a measurement of nothing wearing the costume of
                # "the alternative was tested and lost". So it still does not
                # run. But discarding it was the other half of the mistake: a
                # customer who has only connected OpenAI still deserves to be
                # told that a model elsewhere looks worth evaluating.
                #
                # It is retained as a TIER 2 opportunity: a hypothesis from a
                # vendor price sheet, explicitly UNVERIFIED, whose next action
                # is connecting the provider. It carries no measured number, and
                # optimization/benchmark.py never lets it win, never gives it a
                # verified saving and never lets it reach `verified` — it is not
                # in `measured` at all, so there is no path by which it could.
                opportunities.append({
                    "generator": name,
                    "label": cand.title,
                    "tier": domain.TIER_OPPORTUNITY,
                    "code": "provider_not_configured",
                    "providers": sorted(missing),
                    "dimensions": list(cand.dimensions or []),
                    "strategy_fingerprint": cand.fingerprint,
                    "executor_refs": [
                        st.executor_ref for st in cand.strategy.steps if st.executor_ref
                    ],
                    # A price-sheet hypothesis is the WEAKEST evidence class
                    # there is. Asserted here rather than inherited, so a tier-2
                    # item can never arrive carrying 'replay'.
                    "evidence_source": "none",
                    "evidence_strength": domain.evidence_strength("none"),
                    "verified": False,
                    "next_action": "connect_provider",
                    # Vendor-price extrapolation, and labelled as such. It is a
                    # reason to benchmark, never a saving.
                    "projected_savings_usd": cand.projected_savings_usd,
                    "projection_basis": cand.projection_basis,
                    "measured_quality": None,
                    "measured_cost_usd": None,
                    "detail": (
                        "No provider credential is configured for this "
                        "organization, so this candidate was NOT executed and "
                        "nothing about it has been measured."
                    ),
                })
                continue
            fp = cand.fingerprint
            if fp in seen:
                dropped.append({
                    "generator": name, "title": cand.title, "code": "duplicate_strategy",
                })
                continue
            seen.add(fp)
            out.append(cand)

    return out, {
        "history": {
            "model_stats_count": len(history.get("model_stats") or {}),
            "traffic": history.get("traffic"),
            "lookback_days": lookback_days,
        },
        "dropped": dropped,
        "opportunities": opportunities,
        "generators_run": names,
        "configured_providers": sorted(configured),
        # Everything that entered consideration, benchmarkable or not. The
        # funnel is assembled from this by optimization.domain.build_funnel.
        "considered": len(out) + len(dropped) + len(opportunities),
    }


# ---------------------------------------------------------------------------
# Documented extension points — stubs that REFUSE rather than fabricate
# ---------------------------------------------------------------------------

# NOTE — the `prompt_compression` stub that stood here is GONE, and the name
# went with it. It has been replaced by `ContextReductionGenerator` above, which
# is registered and runs. The rename is not cosmetic: "prompt compression"
# invites rewriting prompts for style, and the dimension that could be built
# honestly is narrower than that — same model, same provider, same tools, same
# output contract, only the context representation changes.
#
# Of the three blockers that stub named:
#
#   1. per-component token breakdown — ANSWERED by optimization.context_accounting.
#   2. a quality signal able to detect tail degradation — ADDRESSED, not
#      dissolved. The benchmark's quality signal was already deterministic
#      (`_run_quality_checks` refuses `model_graded`), and the generator now
#      refuses to propose for a workload with no deterministic/structural/format
#      check and for a case set too small to contain a tail. Whether a given
#      customer's eval suite COVERS their tail is still their property, not
#      ours, which is why nothing here promotes anything.
#   3. a model-written prompt is not evidence — RESOLVED BY DESIGN. A proposer
#      is an injected dependency, its output is marked `proposed_rewrite`, and
#      it takes the identical path to a variant made by string surgery.


class CallCountReductionGenerator:
    """
    EXTENSION POINT — stage 4/6, not implemented.

    Would propose collapsing two model calls into one, or replacing a model step
    with deterministic software (an `executor_type='software'` step, which the
    schema already supports).

    Required before this can exist honestly:
      1. A model of what each step CONTRIBUTES to the final output. `llm_call_count`
         and `workflow_structure` are both marked unapplicable in
         optimization.strategy precisely because this module cannot verify that
         a restructured graph preserves the workflow's contract.
      2. Per-step outcome attribution: which step's output the deciding outcome
         actually depended on. `outcomes` supports this shape but nothing
         populates per-step outcomes today.
      3. For software substitution, an executable software executor. Software
         executors can be REGISTERED but cannot yet be EXECUTED in the runtime
         graph.

    A graph rewrite that silently changes the contract would not show up as a
    cost regression — it would show up as a customer incident.
    """

    name = "call_count_reduction"

    def generate(self, workload, baseline, history) -> list[Candidate]:
        raise NotImplementedError(
            "Call-count reduction requires per-step outcome attribution and an "
            "executable software executor. See the docstring."
        )


#: Registered separately so a caller cannot accidentally run one. They are
#: listed for discoverability and to keep their required-evidence docstrings
#: attached to the codebase rather than to a planning document.
PLANNED_GENERATORS: dict[str, type] = {
    CallCountReductionGenerator.name: CallCountReductionGenerator,
}
