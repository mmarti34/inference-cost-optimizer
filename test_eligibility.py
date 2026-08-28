"""
Candidate eligibility preflight, and objective-aware screening.

These tests exist because of a specific, expensive production run. 140 live
cases against a real workload produced one genuine verified win
(gpt-4o -> gpt-5-mini, -41.8% cost, quality 0.9714 -> 0.9786) and two defects
that cost money to learn:

  * `o1-mini` ran all 140 cases at a 100% error rate and produced ZERO usable
    data. Nothing checked before dispatch, so the incompatibility was
    discovered through provider errors, one case at a time, 140 times.

  * `GPT-5` was benchmarked under a `cost` objective and measured +321.8%.
    It consumed $0.503 — 69% of the whole run's provider spend.

The load-bearing assertion in most of these tests is a COUNT OF ZERO: the
excluded candidate must appear in the funnel with a reason code and must not
appear in the runtime's call log at all. An arm that was refused is not an arm
that failed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

if "Crypto" not in sys.modules:  # same shim the sibling suites use
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for _sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        sys.modules["Crypto." + _sub] = types.ModuleType("Crypto." + _sub)
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from optimization import candidates as candidates_mod  # noqa: E402
from optimization import capabilities as caps_mod  # noqa: E402
from optimization import domain  # noqa: E402
from optimization import eligibility as el  # noqa: E402
from optimization import strategy as strategy_mod  # noqa: E402

from test_optimization_loop import (  # noqa: E402
    BASELINE_MODEL,
    CHEAP_MODEL,
    WORKFLOW_ID,
    FakeRuntime,
    FakeSupabase,
    _candidate,
    _graph,
    _run_loop,
    _seed,
)

#: The model whose 140-case, 100%-error arm is the reason this module exists.
INCOMPATIBLE_MODEL = "o1-mini"
#: Genuinely more expensive than the gpt-4o baseline on the published sheet.
EXPENSIVE_MODEL = "gpt-4-turbo"
#: The model that actually won the live run. Nothing here may screen it out.
LIVE_WINNER_MODEL = "gpt-5-mini"


@pytest.fixture
def db():
    return FakeSupabase()


def _baseline_strategy():
    return strategy_mod.from_graph_json(_graph(), workflow_id=WORKFLOW_ID)


def _run(db, runtime, *, configured=("openai",), **kwargs):
    """`_run_loop` with the org's configured providers pinned rather than read."""
    with patch.object(
        candidates_mod, "_configured_providers", return_value=set(configured)
    ):
        return _run_loop(db, runtime, **kwargs)


def _models_called(runtime):
    return {c["model"] for c in runtime.calls}


def _outcome(result, disposition):
    """Count in the DISJOINT terminal bucket — where a candidate stopped."""
    return next(
        s for s in result["consideration"]["outcomes"] if s["stage"] == disposition
    )["count"]


def _stage(result, stage):
    """Count in the CUMULATIVE funnel — how far candidates got."""
    return next(
        s for s in result["consideration"]["stages"] if s["stage"] == stage
    )["count"]


def _exclusion(result, code):
    return next(
        e for e in result["consideration"]["exclusions"] if e["code"] == code
    )["count"]


def _disposition_for(result, label_fragment):
    return next(
        d for d in result["consideration"]["dispositions"]
        if label_fragment in (d.get("label") or "")
    )


# ===========================================================================
# 1. No external provider request may occur for an ineligible candidate
# ===========================================================================

def test_an_incompatible_model_incurs_zero_provider_executions(db):
    """
    THE o1-mini REGRESSION TEST.

    The old behaviour dispatched 140 cases and collected 140 provider errors.
    The new behaviour dispatches nothing: the count that matters is zero.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL), _candidate(INCOMPATIBLE_MODEL)],
    )

    # ZERO executions. Not a failed arm, not an arm with NULL metrics — no arm.
    assert INCOMPATIBLE_MODEL not in _models_called(runtime)
    assert [c for c in runtime.calls if c["model"] == INCOMPATIBLE_MODEL] == []
    # And no candidate result row was written for it either.
    labels = {r["label"] for r in db.rows("benchmark_candidate_results")}
    assert not any(INCOMPATIBLE_MODEL in label for label in labels)

    # The compatible candidate was measured exactly as before.
    assert CHEAP_MODEL in _models_called(runtime)
    assert BASELINE_MODEL in _models_called(runtime)


def test_an_unconfigured_provider_incurs_zero_provider_executions(db):
    """
    An arm for a provider the org holds no credential for is a 100%-error arm:
    a measurement of nothing wearing the costume of "the alternative lost".

    The generator already refused these. The preflight extends the same refusal
    to CALLER-SUPPLIED candidates, which previously bypassed the check entirely.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(
        db, runtime, configured=("openai",),
        candidates=[
            _candidate(CHEAP_MODEL),
            _candidate("claude-haiku-4-5-20251001", provider="anthropic"),
        ],
    )

    assert "claude-haiku-4-5-20251001" not in _models_called(runtime)

    # NOT BENCHMARKED, NOT FORGOTTEN — it is retained as a tier-2 opportunity.
    opportunities = result["frontier"]["unverified_opportunities"]
    assert len(opportunities) == 1
    opp = opportunities[0]
    assert opp["tier"] == domain.TIER_OPPORTUNITY
    assert opp["code"] == "provider_not_configured"
    assert opp["providers"] == ["anthropic"]
    assert opp["next_action"] == "connect_provider"
    assert opp["verified"] is False
    assert opp["measured_cost_usd"] is None
    assert opp["measured_quality"] is None
    assert _outcome(result, domain.DISPOSITION_PROVIDER_NOT_CONFIGURED) == 1


# ===========================================================================
# 2. The o1 family, decided by a DECLARATION rather than a name check
# ===========================================================================

def test_the_o1_family_is_ineligible_before_dispatch_via_declared_capability(db):
    """
    The exclusion must come from the capability model, not from
    `model.startswith("o1")`.

    On the runtime surface the request has TWO authors: the customer's graph,
    and the surface itself. `workflow_runtime._execute_model_node` calls
    `openai_router.handle_prompt`, which prepends a system-role message to every
    prompt. The o-series does not accept one. The customer cannot see that in
    their graph — but the provider does, and it is what refused all 140 cases.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(db, runtime, candidates=[_candidate(INCOMPATIBLE_MODEL)])

    # Not one provider request — the baseline arm is not run either, because
    # there is no longer anything to compare it against.
    assert runtime.calls == []
    assert INCOMPATIBLE_MODEL not in _models_called(runtime)

    d = _disposition_for(result, INCOMPATIBLE_MODEL)
    assert d["disposition"] == domain.DISPOSITION_INCOMPATIBLE
    assert d["code"] == "required_capability_missing"
    facts = d["facts"]["facts"]
    assert facts["capability"] == "system_message"
    # Traceable to the DECLARATION that produced it, not to a name test.
    assert facts["declared_by"] == ["openai_reasoning_o_series"]
    assert "openai_router.handle_prompt" in facts["required_because"]


def test_the_o1_verdict_is_data_and_moves_when_the_declaration_moves():
    """
    Proof that no name check is involved: change the DATA and the verdict
    changes with it. A hardcoded `startswith("o1")` could not be moved this way.
    """
    baseline = _baseline_strategy()
    cand = _candidate(INCOMPATIBLE_MODEL)

    def verdict(declarations):
        with patch.object(caps_mod, "FAMILY_DECLARATIONS", declarations):
            return el.evaluate_candidate(
                cand, baseline=baseline, objective="quality",
                configured_providers={"openai"},
            )

    # As declared today: refused.
    assert verdict(caps_mod.FAMILY_DECLARATIONS).eligible is False

    # With the family declaration removed, nothing is declared about the model,
    # and UNKNOWN NEVER EXCLUDES — it becomes eligible again.
    without_o_series = [
        d for d in caps_mod.FAMILY_DECLARATIONS
        if d["family_id"] != "openai_reasoning_o_series"
    ]
    ev = verdict(without_o_series)
    assert ev.eligible is True
    system_check = next(
        c for c in ev.checks if c["dimension"] == el.DIM_SYSTEM_MESSAGE
    )
    assert system_check["status"] == el.STATUS_NOT_ASSESSED
    assert system_check["reason"] == "capability_undeclared"

    # And the same rule generalises to a family that has nothing to do with o1:
    # an invented vendor declaring the same refusal is excluded identically.
    invented = list(caps_mod.FAMILY_DECLARATIONS) + [{
        "family_id": "acme_no_system_role",
        "match": {"vendor": "openai", "id_prefixes": ["gpt-4o-mini"]},
        "params": {},
        "capabilities": {"system_message": caps_mod.SUPPORT_NO},
        "provenance": "test",
    }]
    with patch.object(caps_mod, "FAMILY_DECLARATIONS", invented):
        other = el.evaluate_candidate(
            _candidate(CHEAP_MODEL), baseline=baseline, objective="quality",
            configured_providers={"openai"},
        )
    assert other.eligible is False
    assert other.code == "required_capability_missing"


def test_a_lossless_adapter_makes_an_incompatible_request_executable():
    """
    `max_tokens` -> `max_completion_tokens` is a RENAME: it preserves the
    customer's intent exactly, so the adapter may apply it and the candidate
    stays eligible. `temperature` on the same family has no lossless adapter,
    so it does not.

    Eligibility is a property of the (model, request) PAIR. The same model is
    executable for a request that never sets `temperature`.
    """
    baseline = strategy_mod.from_direct_inference_request(
        model=BASELINE_MODEL, provider="openai", temperature=0.0, max_tokens=512,
    )

    def direct(model, **params):
        st = strategy_mod.from_direct_inference_request(
            model=model, provider="openai", **params
        )
        return candidates_mod.Candidate(
            title=f"Try {model}", strategy=st, dimensions=["model"],
            generator="test_supplied", rationale="Supplied by the test.",
        )

    kwargs = dict(
        baseline=baseline, objective="quality", configured_providers={"openai"},
    )

    # max_tokens only -> adapted, and executable.
    adapted = el.evaluate_candidate(direct(INCOMPATIBLE_MODEL, max_tokens=512), **kwargs)
    assert adapted.eligible is True
    assert adapted.adaptations[0]["adapter"] == caps_mod.ADAPTER_RENAME
    assert adapted.adaptations[0]["renamed_to"] == "max_completion_tokens"
    assert adapted.adaptations[0]["lossless"] is True
    params_check = next(
        c for c in adapted.checks if c["dimension"] == el.DIM_REQUEST_PARAMS
    )
    assert params_check["status"] == el.STATUS_ADAPTED
    assert params_check["code"] == "request_adapted"

    # temperature as well -> no lossless route, so INELIGIBLE rather than a
    # quietly different experiment with the temperature dropped.
    blocked = el.evaluate_candidate(
        direct(INCOMPATIBLE_MODEL, temperature=0.0, max_tokens=512), **kwargs
    )
    assert blocked.eligible is False
    assert blocked.code == "request_shape_incompatible"
    blocker = blocked.failed["blockers"][0]
    assert blocker["param"] == "temperature"
    assert blocker["declared_adapter"] is None


def test_the_adapter_refuses_a_lossy_transformation_by_default():
    """
    A dropped parameter is not an adaptation, it is a different experiment. The
    lossy adapter exists and is declarable; `adapt_request` will not apply it
    unless a caller explicitly opts in, and no benchmark caller does.
    """
    profile = caps_mod.ModelProfile(
        vendor="acme", model_id="x1",
        params={"seed": {"support": caps_mod.PARAM_REJECTED,
                         "adapter": caps_mod.ADAPTER_DROP}},
    )
    default = caps_mod.adapt_request(profile, {"seed": 7})
    assert default.executable is False
    assert default.blockers[0]["param"] == "seed"
    assert default.blockers[0]["lossless"] is False

    opted_in = caps_mod.adapt_request(profile, {"seed": 7}, allow_lossy=True)
    assert opted_in.executable is True
    assert "seed" not in opted_in.adapted_params


# ===========================================================================
# 3. Objective-aware screening — and the safeguard against over-screening
# ===========================================================================

def test_a_more_expensive_candidate_under_a_cost_objective_is_screened(db):
    """
    Known list pricing already rules this one out, so it does not get to spend
    the customer's money proving it.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL), _candidate(EXPENSIVE_MODEL)],
    )

    assert EXPENSIVE_MODEL not in _models_called(runtime)
    assert _outcome(result, domain.DISPOSITION_ECONOMICALLY_DOMINATED) == 1

    d = _disposition_for(result, EXPENSIVE_MODEL)
    assert d["code"] == "economically_dominated"
    facts = d["facts"]["facts"]
    # SCREENING evidence, labelled as such, with the arithmetic attached.
    assert facts["evidence_class"] == "screening_only"
    assert facts["price_source"] == "vendor_list_price:shared/providers.json"
    assert facts["best_case_relative_saving"] < facts["materiality_relative_threshold"]
    assert facts["io_ratio_source"] == "assumed_default"


def test_screening_never_removes_a_candidate_that_could_win_on_measured_cost(db):
    """
    THE ASYMMETRY SAFEGUARD.

    gpt-5-mini's list price is ~84% below gpt-4o; its MEASURED delta in the live
    run was -41.8%. List price does not predict measured cost, so the screen is
    only ever allowed to fire on the optimistic bound. A screen that deleted the
    only verified win this product has produced would be a bug, not a feature.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(db, runtime, candidates=[_candidate(LIVE_WINNER_MODEL)])

    assert LIVE_WINNER_MODEL in _models_called(runtime)
    assert _outcome(result, domain.DISPOSITION_ECONOMICALLY_DOMINATED) == 0


def test_the_screen_does_not_fire_on_a_marginal_price_difference():
    """
    A candidate whose list price is only slightly worse than the baseline's may
    still win on measured tokens, so it is NOT screened: the candidate is
    allowed to bill a fraction of the baseline's tokens and must still miss
    materiality before the screen fires.
    """
    materiality = domain.copy_default_materiality("cost")
    # 12% above the baseline on list price. Under the assumed-ratio allowance
    # (0.35) the optimistic bound is still a large saving, so it survives.
    marginal = el.screen_cost_objective(
        baseline_ref={"vendor": "openai", "external_id": "gpt-4o"},
        candidate_ref={"vendor": "openai", "external_id": "gpt-4.1"},
        materiality=materiality, history=None,
    )
    assert marginal["status"] == el.STATUS_PASS

    # 8x the baseline on list price. No token efficiency plausibly recovers it.
    dominated = el.screen_cost_objective(
        baseline_ref={"vendor": "openai", "external_id": "gpt-4o"},
        candidate_ref={"vendor": "openai", "external_id": "gpt-4"},
        materiality=materiality, history=None,
    )
    assert dominated["status"] == el.STATUS_FAIL
    assert dominated["code"] == "economically_dominated"


def test_screening_declines_to_run_when_a_price_is_unknown():
    """
    A screen needs both prices. Guessing one is the manufactured-saving bug in
    reverse, so the screen simply does not run.
    """
    out = el.screen_cost_objective(
        baseline_ref={"vendor": "openai", "external_id": "gpt-4o"},
        candidate_ref={"vendor": "openai", "external_id": "acme-internal-v3"},
        materiality=domain.copy_default_materiality("cost"), history=None,
    )
    assert out["status"] == el.STATUS_NOT_ASSESSED
    assert out["reason"] == "pricing_incomplete_for_screen"


def test_screening_is_not_applied_to_other_objectives():
    """
    Cost pricing is not a reason to refuse a latency or quality benchmark.
    """
    baseline = _baseline_strategy()
    ev = el.evaluate_candidate(
        _candidate(EXPENSIVE_MODEL), baseline=baseline, objective="latency",
        materiality=domain.copy_default_materiality("latency"),
        configured_providers={"openai"},
    )
    assert ev.eligible is True
    objective_check = next(
        c for c in ev.checks if c["dimension"] == el.DIM_OBJECTIVE
    )
    assert objective_check["status"] == el.STATUS_NOT_ASSESSED
    assert objective_check["reason"] == "no_screening_rule_for_objective"


def test_billed_reasoning_tokens_are_recorded_as_a_fact_and_never_screen():
    """
    Reasoning-token billing is what made GPT-5 measure +321.8%, and screening on
    it is tempting. It is refused: gpt-5-mini carries the SAME declaration and
    is the only verified win this product has produced. The fact is recorded;
    nothing acts on it.
    """
    baseline = _baseline_strategy()
    for model in ("gpt-5", LIVE_WINNER_MODEL):
        ev = el.evaluate_candidate(
            _candidate(model), baseline=baseline, objective="cost",
            materiality=domain.copy_default_materiality("cost"),
            configured_providers={"openai"},
        )
        assert ev.eligible is True, model
        codes = {n["code"] for n in ev.notes}
        assert "cost_risk_billed_reasoning_tokens" in codes
        # A note is not a verdict: it appears in `notes`, never in `checks`.
        assert all(c["status"] != el.STATUS_FAIL for c in ev.checks)


# ===========================================================================
# 4. Policy restrictions, decided before the money is spent
# ===========================================================================

def test_a_policy_blocked_vendor_is_refused_before_dispatch(db):
    """
    `blocked_vendors` used to be evaluated only AFTER the arm had run, so a
    forbidden vendor was still paid for. It is now a preflight check.
    """
    _seed(
        db, golden_inputs=60, production_runs=90,
        constraints={"min_quality": 0.90, "allowed_vendors": ["anthropic"]},
    )
    runtime = FakeRuntime(n_cases=60)

    result = _run(db, runtime, candidates=[_candidate(CHEAP_MODEL)])

    assert CHEAP_MODEL not in _models_called(runtime)
    assert _outcome(result, domain.DISPOSITION_POLICY_BLOCKED) == 1
    d = _disposition_for(result, CHEAP_MODEL)
    assert d["code"] == "provider_not_permitted"
    assert d["facts"]["facts"]["constraint"] == "allowed_vendors"


# ===========================================================================
# 5. Excluded candidates keep their evidence, and the funnel stays honest
# ===========================================================================

def test_an_excluded_candidate_retains_structured_consideration_evidence(db):
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(
        db, runtime,
        candidates=[
            _candidate(CHEAP_MODEL),
            _candidate(INCOMPATIBLE_MODEL),
            _candidate(EXPENSIVE_MODEL),
        ],
    )

    for model in (INCOMPATIBLE_MODEL, EXPENSIVE_MODEL):
        d = _disposition_for(result, model)
        assert d["code"] in domain.REASON_CODES
        evidence = d["facts"]["eligibility"]
        # Every dimension was asked, and the ones that could not be answered say
        # so rather than reporting a fabricated pass.
        dims = {c["dimension"] for c in evidence["checks"]}
        assert dims == set(el.ELIGIBILITY_DIMENSIONS)
        assert evidence["eligible"] is False
        assert evidence["code"] == d["code"]
        assert evidence["strategy_fingerprint"]
        assert evidence["executor_refs"]
        # Facts, not prose: every check carries a machine-readable status.
        assert all(
            c["status"] in (
                el.STATUS_PASS, el.STATUS_FAIL, el.STATUS_ADAPTED,
                el.STATUS_NOT_ASSESSED,
            )
            for c in evidence["checks"]
        )


def test_unassessable_dimensions_are_reported_as_such_not_as_passes():
    """
    Nothing records a workload's modality and nothing measures its input token
    requirement, so those dimensions come back `not_assessed` with the reason
    named. An inflated funnel with a fabricated bucket is exactly the failure
    this product exists to prevent.
    """
    ev = el.evaluate_candidate(
        _candidate(CHEAP_MODEL), baseline=_baseline_strategy(), objective="cost",
        materiality=domain.copy_default_materiality("cost"),
        configured_providers={"openai"},
    )
    by_dim = {c["dimension"]: c for c in ev.checks}
    for dim, reason in (
        (el.DIM_INPUT_MODALITY, "workload_modality_not_recorded"),
        (el.DIM_OUTPUT_MODALITY, "workload_modality_not_recorded"),
        (el.DIM_CONTEXT_WINDOW, "workload_input_tokens_not_measured"),
    ):
        assert by_dim[dim]["status"] == el.STATUS_NOT_ASSESSED
        assert by_dim[dim]["reason"] == reason
        assert by_dim[dim]["code"] is None
    assert set(ev.to_dict()["dimensions_not_assessed"]) >= {
        el.DIM_INPUT_MODALITY, el.DIM_OUTPUT_MODALITY, el.DIM_CONTEXT_WINDOW,
    }


def test_the_funnel_counts_every_candidate_and_exclusions_do_not_reduce_coverage(db):
    """
    A candidate being excluded is not a benchmark failure. The workload's
    conclusion, its coverage class and the arms that DID run must be identical
    whether or not ineligible candidates were also proposed.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime_alone = FakeRuntime(n_cases=60)
    alone = _run(db, runtime_alone, candidates=[_candidate(CHEAP_MODEL)])

    db2 = FakeSupabase()
    _seed(db2, golden_inputs=60, production_runs=90)
    runtime_mixed = FakeRuntime(n_cases=60)
    mixed = _run(
        db2, runtime_mixed,
        candidates=[
            _candidate(CHEAP_MODEL),
            _candidate(INCOMPATIBLE_MODEL),
            _candidate(EXPENSIVE_MODEL),
        ],
    )

    # Same determination, same coverage, same measured arms.
    assert mixed["conclusion"] == alone["conclusion"]
    assert domain.coverage_class(mixed["conclusion"]) == domain.coverage_class(
        alone["conclusion"]
    )
    assert _models_called(runtime_mixed) == _models_called(runtime_alone)

    # The funnel is WIDER, and every exit is accounted for exactly once.
    funnel = mixed["consideration"]
    assert funnel["considered"] == 3
    assert _outcome(mixed, domain.DISPOSITION_INCOMPATIBLE) == 1
    assert _outcome(mixed, domain.DISPOSITION_ECONOMICALLY_DOMINATED) == 1
    # The surviving candidate exits exactly where it did on its own: the
    # exclusions changed the funnel's width, not the verdict on the arm that ran.
    survivor_alone = _disposition_for(alone, CHEAP_MODEL)
    survivor_mixed = _disposition_for(mixed, CHEAP_MODEL)
    assert survivor_mixed["disposition"] == survivor_alone["disposition"]
    assert survivor_mixed["code"] == survivor_alone["code"]
    exits = [
        d["disposition"] for d in funnel["dispositions"]
    ]
    assert len(exits) == 3
    assert len(set(exits)) == 3  # one exit each, none double-counted

    # And the narrower run's funnel counts only what it considered.
    assert alone["consideration"]["considered"] == 1


def test_every_funnel_stage_is_either_populated_or_declared_unbuilt(db):
    """
    A stage nothing can populate must say so, so a zero is never read as
    "we checked and found none".
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)
    result = _run(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL), _candidate(INCOMPATIBLE_MODEL)],
    )

    outcomes = {s["stage"]: s for s in result["consideration"]["outcomes"]}
    # Newly real: something now actually populates these.
    assert outcomes[domain.DISPOSITION_INCOMPATIBLE]["emitted"] is True
    assert outcomes[domain.DISPOSITION_ECONOMICALLY_DOMINATED]["emitted"] is True
    assert outcomes[domain.DISPOSITION_POLICY_BLOCKED]["emitted"] is True
    # Still nothing populates historical elimination, and it says so.
    assert outcomes[domain.DISPOSITION_ELIMINATED_BY_HISTORY]["emitted"] is False

    # Same convention at reason-code grain, which is the grain the exclusion
    # evidence is rendered at.
    exclusions = {e["code"]: e for e in result["consideration"]["exclusions"]}
    assert exclusions["eliminated_by_historical_evidence"]["emitted"] is False
    for code in (
        "provider_not_configured", "request_shape_incompatible",
        "required_capability_missing", "context_window_insufficient",
        "policy_blocked", "pricing_unknown", "economically_dominated",
    ):
        # The exclusion codes the UI renders as evidence of a thorough search
        # must stay individually distinguishable, not collapsed into
        # `incompatible`.
        assert exclusions[code]["emitted"] is True
        assert exclusions[code]["disposition"] in domain.DISPOSITIONS_ORDERED

    # Every exclusion code the preflight can emit maps to a real disposition.
    for code, stage in el.CODE_TO_DISPOSITION.items():
        assert code in domain.REASON_CODES
        assert stage in domain.DISPOSITIONS_ORDERED
        assert code in domain.EXCLUSION_CODES


def test_the_funnel_stages_are_cumulative_and_never_increase(db):
    """
    THE invariant. A candidate counts at every stage it REACHED, not only at
    the one it stopped in, so the cumulative spine can only narrow going down.

    The bug this replaces: counting each candidate solely in its terminal
    bucket reported `benchmarked: 0` on a run where arms had demonstrably
    executed and then been stopped, because they incremented `failed_policy`
    instead. "How many did we actually test?" was unanswerable.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)
    result = _run(
        db, runtime,
        candidates=[
            _candidate(CHEAP_MODEL),
            _candidate(INCOMPATIBLE_MODEL),
            _candidate(EXPENSIVE_MODEL),
        ],
    )

    stages = result["consideration"]["stages"]
    assert [s["stage"] for s in stages] == list(domain.FUNNEL_STAGES)

    spine = [s for s in stages if s["cumulative"]]
    counts = [s["count"] for s in spine]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == result["consideration"]["considered"] == 3

    # Two were excluded before dispatch; one was executable and did run.
    assert _stage(result, domain.STAGE_EXECUTABLE) == 1
    assert _stage(result, domain.STAGE_ENTERED_REPLAY) == 1

    # The disjoint outcomes still account for every candidate exactly once.
    assert sum(
        o["count"] for o in result["consideration"]["outcomes"]
        if o["stage"] != domain.DISPOSITION_CONSIDERED
    ) == 3

    # `stopped_early` is reported inline but is an EXIT count, not part of the
    # cumulative spine — it is never asserted to nest inside the stage below it.
    stopped = next(s for s in stages if s["stage"] == domain.STAGE_STOPPED_EARLY)
    assert stopped["cumulative"] is False
    assert stopped["count"] <= _stage(result, domain.STAGE_ENTERED_REPLAY)


def test_a_fully_screened_run_still_emits_the_funnel(db):
    """
    "We considered two and neither was eligible, here is the code for each" is a
    finding. It must not degrade to silence just because nothing ran.
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(
        db, runtime,
        candidates=[_candidate(INCOMPATIBLE_MODEL), _candidate(EXPENSIVE_MODEL)],
    )

    assert _models_called(runtime) == set()  # not even the baseline arm ran
    assert result["conclusion"] == domain.CONCLUSION_INSUFFICIENT_EVIDENCE
    assert result["consideration"]["considered"] == 2
    assert _outcome(result, domain.DISPOSITION_INCOMPATIBLE) == 1
    assert _outcome(result, domain.DISPOSITION_ECONOMICALLY_DOMINATED) == 1
    # Nothing was measured, so nothing is covered. That is the honest reading,
    # and it is not what "the candidates failed policy" would mean.
    assert domain.coverage_class(result["conclusion"]) == domain.COVERAGE_NOT_COVERED


# ===========================================================================
# 6. Contract hygiene
# ===========================================================================

def test_every_code_the_preflight_emits_is_a_documented_one():
    baseline = _baseline_strategy()
    for model, provider in (
        (CHEAP_MODEL, "openai"),
        (INCOMPATIBLE_MODEL, "openai"),
        (EXPENSIVE_MODEL, "openai"),
        ("claude-3-haiku-20240307", "anthropic"),
        ("acme-internal-v3", "openai"),
    ):
        ev = el.evaluate_candidate(
            _candidate(model, provider=provider), baseline=baseline,
            objective="cost", materiality=domain.copy_default_materiality("cost"),
            configured_providers={"openai"},
        )
        for c in ev.checks:
            assert c["code"] is None or c["code"] in domain.REASON_CODES
        if ev.code is not None:
            assert ev.code in domain.REASON_CODES


def test_an_unknown_vendor_is_the_only_availability_failure():
    """
    `shared/providers.json` is a PRICE SHEET, not an availability oracle. A model
    absent from it is routinely still dispatchable, so absence is UNKNOWN. What
    genuinely blocks dispatch is an unknown vendor: no api_base, no request.
    """
    baseline = _baseline_strategy()
    kwargs = dict(baseline=baseline, objective="quality",
                  configured_providers={"openai", "acme"})

    uncatalogued = el.evaluate_candidate(
        _candidate("acme-internal-v3", provider="openai"), **kwargs
    )
    check = next(
        c for c in uncatalogued.checks if c["dimension"] == el.DIM_MODEL_AVAILABLE
    )
    assert check["status"] == el.STATUS_NOT_ASSESSED
    assert uncatalogued.eligible is True

    unknown_vendor = el.evaluate_candidate(
        _candidate("some-model", provider="acme"), **kwargs
    )
    assert unknown_vendor.eligible is False
    assert unknown_vendor.code == "model_not_available"


def test_a_failed_credential_lookup_refuses_nothing():
    """
    An empty configured-provider set means the lookup did not run, not that the
    org has no credentials. Refusing every candidate because a read errored
    would be worse than letting the arms report their own failures.
    """
    ev = el.evaluate_candidate(
        _candidate("claude-3-haiku-20240307", provider="anthropic"),
        baseline=_baseline_strategy(), objective="quality",
        configured_providers=set(),
    )
    check = next(
        c for c in ev.checks if c["dimension"] == el.DIM_PROVIDER_CONFIGURED
    )
    assert check["status"] == el.STATUS_NOT_ASSESSED
    assert check["reason"] == "configured_providers_unknown"


def test_the_preflight_makes_no_provider_request_under_any_branch():
    """
    The whole module is a decision, not an execution. Nothing in it may reach a
    provider — proven by executing every branch with the runtime patched to
    explode if it is called at all.
    """
    exploding = MagicMock(side_effect=AssertionError("a provider was called"))
    baseline = _baseline_strategy()
    with patch("workflow_runtime.execute_workflow", exploding):
        result = el.preflight(
            [
                _candidate(CHEAP_MODEL),
                _candidate(INCOMPATIBLE_MODEL),
                _candidate(EXPENSIVE_MODEL),
                _candidate("claude-3-haiku-20240307", provider="anthropic"),
            ],
            baseline=baseline, objective="cost",
            materiality=domain.copy_default_materiality("cost"),
            configured_providers={"openai"},
        )
    exploding.assert_not_called()
    assert len(result.evaluations) == 4
    assert [c.title for c in result.eligible] == [f"Switch step n1 to {CHEAP_MODEL}"]
    assert len(result.excluded) == 2
    assert len(result.opportunities) == 1


def test_an_admitted_candidate_also_carries_its_preflight_record(db):
    """
    The funnel must answer "why was this one allowed to spend money?" with the
    same structure it answers "why was that one refused?".
    """
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL), _candidate(INCOMPATIBLE_MODEL)],
    )

    admitted = _disposition_for(result, CHEAP_MODEL)
    evidence = admitted["eligibility"]
    assert evidence is not None
    assert evidence["eligible"] is True
    assert evidence["code"] is None
    assert {c["dimension"] for c in evidence["checks"]} == set(
        el.ELIGIBILITY_DIMENSIONS
    )
    # It really was assessed on the dimensions that can be assessed today.
    assert el.DIM_PROVIDER_CONFIGURED in evidence["dimensions_assessed"]
    assert el.DIM_OBJECTIVE in evidence["dimensions_assessed"]
