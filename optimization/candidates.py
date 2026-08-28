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
"""
from __future__ import annotations

import logging
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
# Registry
# ---------------------------------------------------------------------------

#: Generators that can be run today, each backed by real data.
CANDIDATE_GENERATORS: dict[str, CandidateGenerator] = {
    AlternateModelGenerator.name: AlternateModelGenerator(),
    CheaperMeasuredModelGenerator.name: CheaperMeasuredModelGenerator(),
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
            produced = gen.generate(workload, baseline, history)
        except Exception as exc:  # pragma: no cover
            logger.warning("generator %s failed: %s", name, type(exc).__name__)
            dropped.append({"generator": name, "code": "generator_error",
                            "detail": type(exc).__name__})
            continue

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

class PromptCompressionGenerator:
    """
    EXTENSION POINT — stage 2, not implemented.

    Would propose a shorter prompt or a tighter context budget for the same
    task. The `prompt` and `context_length` dimensions are both APPLICABLE by
    optimization.strategy, so the blocker is not the mechanism — it is the
    evidence.

    Required before this can exist honestly:
      1. A measured token breakdown per step attributable to the static prompt
         vs the injected context vs the user input. `node_results` records
         totals, not that split, so nothing today can say which part to cut.
      2. A quality signal strong enough to detect the failure mode that matters:
         a compressed prompt usually still produces plausible output, and only
         degrades on the harder tail of inputs. A deterministic or business
         outcome signal is needed; an LLM judge would rate the degraded output
         fine.
      3. A rewriting step whose output is itself benchmarked, because a
         "compressed" prompt written by a model is exactly the kind of
         LLM-said-so change this product exists to refuse.

    Emitting a plausible shorter prompt without those would be generating
    confident-looking candidates with no way to tell the good ones from the
    quietly damaging ones.
    """

    name = "prompt_compression"

    def generate(self, workload, baseline, history) -> list[Candidate]:
        raise NotImplementedError(
            "Prompt compression requires a per-component token breakdown and a "
            "quality signal able to detect tail degradation. See the docstring."
        )


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
    PromptCompressionGenerator.name: PromptCompressionGenerator,
    CallCountReductionGenerator.name: CallCountReductionGenerator,
}
