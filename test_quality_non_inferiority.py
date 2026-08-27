"""
Quality safety: relative, uncertainty-aware, and impossible to round up.

These tests exist because of a specific production failure. The first live
benchmark ran 30 authored replay cases with deterministic exact-match grading
and produced:

    baseline   gpt-4o        $0.000857   quality 1.0000
    candidate  gpt-4.1-mini  $0.000137   quality 0.9000   -84% cost
    candidate  gpt-5-mini    $0.000433   quality 1.0000   -49% cost
    candidate  gpt-4o-mini   $0.000051   quality 0.7667   -94% cost

The policy carried `min_quality: 0.90`. gpt-4.1-mini scored EXACTLY 0.9000,
cleared the absolute floor by a margin of zero, won on cost, and was written to
the customer as a recommendation at status `verified` with confidence 0.171.
A 10 percentage-point observed regression against the customer's own baseline
was presented as safe. Meanwhile gpt-5-mini matched baseline exactly at half the
cost and was passed over entirely.

Two separate bugs, and the tests below hold both closed:

  * The constraint was ABSOLUTE when the promise is RELATIVE. A floor knows
    nothing about the baseline, so it cannot express "do not make my workload
    worse than it is today".
  * The evidence was a POINT ESTIMATE. Even a candidate that ties baseline on 30
    cases has not established that it is not materially worse; 30/30 and 30/30
    are compatible with a real deficit.

The pure statistics live in `test_noninferiority_math`; the loop-level
consequences live below it.
"""
import sys
import types
import uuid
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

from optimization import benchmark as benchmark_mod  # noqa: E402
from optimization import domain  # noqa: E402
from optimization import noninferiority as ni  # noqa: E402
from optimization import policies as policies_mod  # noqa: E402

# Reuse the in-memory database, the priced fake runtime and the seeding helper
# from the end-to-end suite. Duplicating them would let the two suites drift and
# would mean these tests were exercising a different loop from the real one.
from test_optimization_loop import (  # noqa: E402
    BASELINE_MODEL,
    CHEAP_MODEL,
    ORG_ID,
    WORKLOAD_ID,
    FakeRuntime,
    FakeSupabase,
    _candidate,
    _run_loop,
    _seed,
)


@pytest.fixture
def db():
    return FakeSupabase()


def _per_case(passes: int, total: int, *, prefix="gi"):
    """Per-case rows for an arm where the first `passes` of `total` cases pass."""
    return [
        {
            "case_id": f"{prefix}-{i}",
            "quality_checks_ran": 1,
            "quality_checks_passed": 1 if i < passes else 0,
            "case_passed": i < passes,
            "error": False,
        }
        for i in range(total)
    ]


# ===========================================================================
# 1. The statistics, hand-checkable
# ===========================================================================

def test_constrained_mle_reduces_to_mcnemar_at_a_zero_margin():
    """
    At delta = 0 the constrained MLE of the discordant cell must collapse to
    (b + c) / 2n — the McNemar null. If that identity does not hold, the
    quadratic is wrong and every interval built on it is wrong.
    """
    for n, b, c in [(30, 3, 1), (100, 12, 7), (52, 0, 0), (40, 5, 5)]:
        assert ni.constrained_q(n, b, c, 0.0) == pytest.approx((b + c) / (2.0 * n))


def test_score_statistic_matches_the_closed_form_for_a_perfect_tie():
    """
    With zero discordant pairs the algebra collapses to Z = sqrt(n * m / (1 - m)).
    Hand-worked: n = 30, m = 0.05 -> sqrt(30 * 0.05 / 0.95) = sqrt(1.5789) =
    1.2566, which is BELOW the 1.6449 needed at 95% one-sided. Thirty identical
    passes do not establish non-inferiority at a 5pp margin.
    """
    import math

    for n, m in [(30, 0.05), (52, 0.05), (20, 0.05), (133, 0.02)]:
        assert ni.constrained_q(n, 0, 0, -m) == pytest.approx(m)
        assert ni.score_z(n, 0, 0, -m) == pytest.approx(math.sqrt(n * m / (1 - m)))

    assert ni.score_z(30, 0, 0, -0.05) == pytest.approx(1.256562, abs=1e-5)
    assert ni.score_z(52, 0, 0, -0.05) == pytest.approx(1.654340, abs=1e-5)


def test_discordant_pairs_are_counted_from_the_paired_cases_not_the_arm_means():
    """
    Hand-worked pairing. Ten cases:
      cases 0-5   both arms pass                  -> concordant_pass = 6
      cases 6-7   baseline passes, candidate fails -> b = 2
      case  8     baseline fails, candidate passes -> c = 1
      case  9     both fail                        -> concordant_fail = 1

    Arm means are baseline 8/10 and candidate 7/10; the naive difference is
    -0.1. The paired difference is (c - b)/n = (1 - 2)/10 = -0.1 as well, but it
    is derived from the pairs, and only the 3 discordant pairs carry information
    about it. That distinction is the whole reason for the paired test.
    """
    base = [
        {"case_id": f"c{i}", "quality_checks_ran": 1,
         "quality_checks_passed": 1 if i in (0, 1, 2, 3, 4, 5, 6, 7) else 0}
        for i in range(10)
    ]
    cand = [
        {"case_id": f"c{i}", "quality_checks_ran": 1,
         "quality_checks_passed": 1 if i in (0, 1, 2, 3, 4, 5, 8) else 0}
        for i in range(10)
    ]
    counts = ni.paired_counts(base, cand)
    assert counts["n_pairs"] == 10
    assert counts["concordant_pass"] == 6
    assert counts["concordant_fail"] == 1
    assert counts["discordant_b"] == 2
    assert counts["discordant_c"] == 1
    assert counts["baseline_quality_paired"] == pytest.approx(0.8)
    assert counts["candidate_quality_paired"] == pytest.approx(0.7)


def test_a_case_no_check_ran_on_is_unusable_not_a_failure():
    """"No check ran" and "the check failed" are different facts."""
    base = [{"case_id": "a", "quality_checks_ran": 1, "quality_checks_passed": 1},
            {"case_id": "b", "quality_checks_ran": 0, "quality_checks_passed": 0}]
    cand = [{"case_id": "a", "quality_checks_ran": 1, "quality_checks_passed": 0},
            {"case_id": "b", "quality_checks_ran": 1, "quality_checks_passed": 1},
            {"case_id": "c", "quality_checks_ran": 1, "quality_checks_passed": 1}]
    counts = ni.paired_counts(base, cand)
    assert counts["n_pairs"] == 1          # only case 'a' is a usable pair
    assert counts["discordant_b"] == 1
    assert counts["unusable_pairs"] == 2   # 'b' unmeasured on baseline, 'c' unpaired


def test_an_errored_case_never_counts_as_a_quality_failure():
    base = [{"case_id": "a", "quality_checks_ran": 1, "quality_checks_passed": 1}]
    cand = [{"case_id": "a", "error": True, "quality_checks_ran": 0,
             "quality_checks_passed": 0}]
    assert ni.paired_counts(base, cand)["n_pairs"] == 0


def test_the_live_failure_is_not_non_inferior_at_the_default_margin():
    """
    gpt-4.1-mini, as measured: 27/30 against a baseline of 30/30, so b = 3 and
    c = 0. Under the conservative default (5pp, 95% one-sided) the score
    statistic is NEGATIVE — the data are on the wrong side of the margin — and
    no sample size fixes a point estimate already beyond it.
    """
    assessment = ni.assess(
        baseline_per_case=_per_case(30, 30),
        candidate_per_case=_per_case(27, 30),
        margin=0.05, confidence_level=0.95,
        baseline_quality=1.0, candidate_quality=0.9,
    )
    assert assessment["established"] is False
    assert assessment["discordant_b"] == 3 and assessment["discordant_c"] == 0
    assert assessment["observed_regression"] == pytest.approx(0.10)
    assert assessment["test_statistic_z"] < 0
    assert assessment["reason_code"] == ni.NOT_ESTABLISHED_REGRESSION_EXCEEDS_MARGIN
    assert assessment["additional_cases_required"] is None
    assert assessment["additional_cases_reason"] == (
        ni.MORE_CASES_NOT_DERIVABLE_REGRESSION
    )


def test_a_perfect_tie_on_thirty_cases_is_promising_and_says_how_many_more():
    """
    gpt-5-mini, as measured: 30/30 against 30/30 — ZERO discordant pairs in
    either direction. That is genuinely uninformative: it is equally consistent
    with "identical" and with "slightly worse, got lucky". No paired test can
    establish non-inferiority from it at 30 cases, so the honest output is a
    DERIVED sample-size target rather than a verdict.
    """
    assessment = ni.assess(
        baseline_per_case=_per_case(30, 30),
        candidate_per_case=_per_case(30, 30),
        margin=0.05, confidence_level=0.95,
        baseline_quality=1.0, candidate_quality=1.0,
    )
    assert assessment["established"] is False
    assert assessment["observed_regression"] == pytest.approx(0.0)
    assert assessment["reason_code"] == ni.NOT_ESTABLISHED_EVIDENCE_INSUFFICIENT
    # 52 total, i.e. 22 more, derived by solving the same test for n.
    assert assessment["required_total_cases"] == 52
    assert assessment["additional_cases_required"] == 22
    # And it really does pass at exactly that sample size, not one case earlier.
    assert ni.assess(
        baseline_per_case=_per_case(52, 52), candidate_per_case=_per_case(52, 52),
        margin=0.05, confidence_level=0.95,
    )["established"] is True
    assert ni.assess(
        baseline_per_case=_per_case(51, 51), candidate_per_case=_per_case(51, 51),
        margin=0.05, confidence_level=0.95,
    )["established"] is False


def test_a_tighter_margin_needs_a_much_larger_sample_and_says_so():
    """
    The N figure is driven by the MARGIN, not by the data, when the data show no
    discordance at all. Demanding proof of near-exact equality (2pp) costs 133
    cases; 5pp costs 52. Reported plainly rather than hidden, because it is the
    argument for having a margin at all.
    """
    at_2pp = ni.assess(
        baseline_per_case=_per_case(30, 30), candidate_per_case=_per_case(30, 30),
        margin=0.02, confidence_level=0.95,
    )
    assert at_2pp["required_total_cases"] == 133
    assert at_2pp["additional_cases_required"] == 103


def test_a_sample_below_the_asymptotic_floor_is_refused_with_a_reason():
    assessment = ni.assess(
        baseline_per_case=_per_case(10, 10), candidate_per_case=_per_case(10, 10),
        margin=0.05, confidence_level=0.95,
    )
    assert assessment["established"] is False
    assert assessment["reason_code"] == ni.NOT_ESTABLISHED_SAMPLE_TOO_SMALL


def test_no_paired_cases_is_reported_not_silently_treated_as_a_pass():
    assessment = ni.assess(
        baseline_per_case=[], candidate_per_case=[],
        margin=0.05, confidence_level=0.95,
    )
    assert assessment["established"] is False
    assert assessment["reason_code"] == ni.NOT_ESTABLISHED_PAIRED_CASES_UNAVAILABLE
    assert assessment["additional_cases_required"] is None
    assert assessment["additional_cases_reason"] == ni.MORE_CASES_NOT_DERIVABLE_NO_PAIRS


def test_a_candidate_that_is_better_establishes_non_inferiority_easily():
    assessment = ni.assess(
        baseline_per_case=_per_case(24, 30), candidate_per_case=_per_case(30, 30),
        margin=0.05, confidence_level=0.95,
        baseline_quality=0.8, candidate_quality=1.0,
    )
    assert assessment["established"] is True
    assert assessment["discordant_c"] == 6 and assessment["discordant_b"] == 0
    assert assessment["lower_confidence_bound"] > -0.05
    assert assessment["additional_cases_required"] == 0


# ===========================================================================
# 2. The policy layer
# ===========================================================================

def test_the_default_policy_is_conservative_without_being_invented():
    """
    OptiML defaults the RELATIVE ceiling because a customer who configured
    nothing still expects not to be handed a regression. It does NOT default the
    ABSOLUTE floor, because inventing a quality bar for someone else's workload
    would be a fabrication. Both facts are stamped with their source.
    """
    safety = policies_mod.quality_safety_of(None)
    assert safety["max_quality_regression"] == 0.05
    assert safety["max_quality_regression_source"] == "default"
    assert safety["require_quality_non_inferiority"] is True
    assert safety["quality_confidence_level"] == 0.95
    assert safety["min_quality"] is None
    assert safety["min_quality_source"] == "unset"


def test_min_quality_and_max_quality_regression_are_both_required():
    """They answer different questions and are ANDed, never substituted."""
    policy = {"constraints": {"min_quality": 0.90, "max_quality_regression": 0.05}}

    # Ties the absolute floor exactly, 10pp under baseline: floor satisfied,
    # relative constraint violated. This is the live failure, isolated.
    ev = policies_mod.evaluate(
        policy, measured={"quality": 0.90}, quality_provenance="deterministic",
        baseline={"quality": 1.00},
    )
    assert ev["eligible"] is False
    assert "min_quality" in ev["satisfied"]
    assert [v["constraint"] for v in ev["violated"]] == ["max_quality_regression"]

    # Matches baseline but both are under the customer's absolute floor: the
    # relative constraint is satisfied and the floor is not.
    ev = policies_mod.evaluate(
        policy, measured={"quality": 0.85}, quality_provenance="deterministic",
        baseline={"quality": 0.85},
    )
    assert ev["eligible"] is False
    assert "max_quality_regression" in ev["satisfied"]
    assert [v["constraint"] for v in ev["violated"]] == ["min_quality"]


def test_an_unmeasured_baseline_makes_the_relative_constraint_uncheckable():
    """Not a weaker check — an uncheckable one. Never reported as satisfied."""
    ev = policies_mod.evaluate(
        None, measured={"quality": 0.95}, quality_provenance="deterministic",
        baseline={"quality": None},
    )
    assert ev["eligible"] is False
    assert "max_quality_regression" not in ev["satisfied"]
    assert [u["constraint"] for u in ev["unmeasured"]] == ["max_quality_regression"]


def test_a_weak_quality_signal_cannot_satisfy_the_relative_constraint_either():
    ev = policies_mod.evaluate(
        None, measured={"quality": 1.0}, quality_provenance="llm_judge",
        baseline={"quality": 1.0},
    )
    assert "max_quality_regression" not in ev["satisfied"]
    assert any(u["constraint"] == "max_quality_regression" for u in ev["unmeasured"])


def test_the_enforceable_list_names_the_new_constraint():
    assert "max_quality_regression" in policies_mod.ENFORCEABLE_CONSTRAINTS


# ===========================================================================
# 3. The loop: five states that must stay distinguishable
# ===========================================================================

def test_a_candidate_tying_the_floor_while_regressing_is_rejected_not_verified(db):
    """
    (c) POLICY FAILURE. The live case, driven through the real loop: policy
    floor at 0.90, candidate measures exactly 0.90 against a baseline of 1.00,
    and it is much cheaper. It must not be a recommendation at any status.
    """
    _seed(db, golden_inputs=30, constraints={"min_quality": 0.90}, production_runs=60)
    runtime = FakeRuntime(quality_for={CHEAP_MODEL: 0.90}, n_cases=30)

    result = _run_loop(
        db, runtime, candidates=[_candidate(CHEAP_MODEL)], create_recommendation=True,
    )

    assert result["conclusion"] == domain.CONCLUSION_CANDIDATES_FAILED_POLICY
    assert result["recommendation_created"] is False
    assert db.rows("optimization_recommendations") == []

    codes = {r["code"] for r in result["reasons"]}
    assert "quality_regression_above_threshold" in codes
    assert "quality_below_threshold" not in codes  # the floor was satisfied

    # The evidence survives the adverse verdict, cost delta and all.
    arm = next(a for a in db.rows("benchmark_candidate_results") if a["arm"] == "candidate")
    assert arm["quality"] == pytest.approx(0.90)
    assert arm["cost_delta_pct"] < 0
    assert arm["outcome_metrics"]["paired_vs_baseline"]["discordant_b"] == 3


def test_a_tie_on_a_small_sample_is_promising_not_verified(db):
    """
    (b) PROMISING BUT SHORT OF EVIDENCE. gpt-5-mini's situation: matches
    baseline exactly, saves real money, 30 cases. Must NOT be `verified`, must
    NOT be discarded, and must come back with a derived sample-size target.
    """
    _seed(db, golden_inputs=30, constraints={"min_quality": 0.90}, production_runs=60)
    runtime = FakeRuntime(n_cases=30)

    result = _run_loop(
        db, runtime, candidates=[_candidate(CHEAP_MODEL)], create_recommendation=True,
    )

    assert result["conclusion"] == domain.CONCLUSION_PROMISING_UNVERIFIED
    # Distinguishable from every other state.
    assert result["conclusion"] != domain.CONCLUSION_SAFE_IMPROVEMENT
    assert result["conclusion"] != domain.CONCLUSION_INSUFFICIENT_EVIDENCE
    assert result["conclusion"] != domain.CONCLUSION_CANDIDATES_FAILED_POLICY

    # Nothing proposed, and no verified saving anywhere.
    assert result["recommendation_created"] is False
    assert db.rows("optimization_recommendations") == []

    # It is not counted as an assessment of the workload, and never as
    # "your configuration looks efficient".
    assert result["is_assessable"] is False
    assert result["is_efficiency_finding"] is False
    assert result["coverage_class"] == domain.COVERAGE_NOT_COVERED
    assert result["maps_to_recommendation_status"] == domain.STATUS_INCONCLUSIVE

    # The actionable part: "run approximately N more evaluations", derived.
    qs = result["quality_safety"]
    assert qs["established"] is False
    assert qs["n_pairs"] == 30
    assert qs["additional_cases_required"] == 22
    assert qs["required_total_cases"] == 52
    more = next(
        r for r in result["more_data_reasons"]
        if r["code"] == "sample_size_below_threshold"
    )
    assert more["observed"] == 30
    assert more["required"] == 52
    assert more["additional_cases_required"] == 22
    assert more["derived_from"] == ni.METHOD
    assert result["more_data_changes_conclusion"] == domain.MORE_DATA_YES

    # The candidate is still identified, by result id, so the UI can name it.
    conclusion_row = db.rows("benchmark_conclusions")[0]
    assert conclusion_row["conclusion"] == domain.CONCLUSION_PROMISING_UNVERIFIED
    assert conclusion_row["selected_candidate_result_id"] is not None
    assert conclusion_row["quality_safety"]["established"] is False


def test_matching_baseline_with_enough_cases_is_evidentially_safe(db):
    """
    (d) EVIDENTIALLY SAFE, and (e) the VERIFIED recommendation that follows from
    it. The same tie as above with 60 cases instead of 30 clears the derived
    threshold of 52.
    """
    _seed(db, golden_inputs=60, constraints={"min_quality": 0.90}, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    result = _run_loop(
        db, runtime, candidates=[_candidate(CHEAP_MODEL)], create_recommendation=True,
    )

    assert result["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    assert result["is_assessable"] is True
    qs = result["quality_safety"]
    assert qs["established"] is True
    assert qs["n_pairs"] == 60
    assert qs["lower_confidence_bound"] > -qs["allowed_regression"]

    recs = db.rows("optimization_recommendations")
    assert len(recs) == 1
    assert recs[0]["status"] == domain.STATUS_VERIFIED
    # Human approval is still required: evidence of safety is not permission.
    assert recs[0]["approval_required"] is True
    assert recs[0]["quality_safety"]["established"] is True
    assert recs[0]["verified_savings_usd"] > 0


def test_the_lifecycle_keeps_all_five_states_apart():
    """
    (a) discovered, (b) promising, (c) policy failure, (d) evidentially safe,
    (e) verified recommendation. Two axes: the benchmark CONCLUSION is the
    evidence, the recommendation STATUS is the decision.
    """
    m = domain.CONCLUSION_TO_STATUS
    assert m[domain.CONCLUSION_SAFE_IMPROVEMENT] == domain.STATUS_VERIFIED
    assert m[domain.CONCLUSION_PROMISING_UNVERIFIED] == domain.STATUS_INCONCLUSIVE
    assert m[domain.CONCLUSION_CANDIDATES_FAILED_POLICY] == domain.STATUS_REJECTED
    assert m[domain.CONCLUSION_INSUFFICIENT_EVIDENCE] == domain.STATUS_INCONCLUSIVE

    # Every state is a distinct value; none collapses into another.
    assert len({
        domain.STATUS_DISCOVERED,
        m[domain.CONCLUSION_PROMISING_UNVERIFIED],
        m[domain.CONCLUSION_CANDIDATES_FAILED_POLICY],
        domain.CONCLUSION_SAFE_IMPROVEMENT,
        domain.STATUS_VERIFIED,
    }) == 5

    # A promising candidate cannot jump to verified: the only route out of
    # 'inconclusive' towards it goes back through 'benchmarking', i.e. through
    # taking more measurements.
    assert domain.STATUS_VERIFIED not in domain.LEGAL_TRANSITIONS[
        domain.STATUS_INCONCLUSIVE
    ]
    assert domain.STATUS_BENCHMARKING in domain.LEGAL_TRANSITIONS[
        domain.STATUS_INCONCLUSIVE
    ]

    # Promising is IGNORANCE, not knowledge: it must never be counted as
    # coverage or rendered as "your configuration looks efficient".
    assert domain.CONCLUSION_PROMISING_UNVERIFIED in domain.IGNORANCE_CONCLUSIONS
    assert domain.CONCLUSION_PROMISING_UNVERIFIED not in domain.KNOWLEDGE_CONCLUSIONS
    assert domain.CONCLUSION_PROMISING_UNVERIFIED not in domain.NO_OPPORTUNITY_CONCLUSIONS
    assert domain.is_efficiency_finding(domain.CONCLUSION_PROMISING_UNVERIFIED) is False
    assert domain.coverage_class(domain.CONCLUSION_PROMISING_UNVERIFIED) == (
        domain.COVERAGE_NOT_COVERED
    )


def test_ranking_prefers_the_safe_candidate_over_the_merely_cheaper_one(db):
    """
    THE ORIGINAL BUG, END TO END. Two candidates: one much cheaper but 10pp
    below baseline, one matching baseline exactly at a smaller (still material)
    saving. Cost-first ranking picked the first. Correct ranking excludes it at
    the policy stage and never sees it again.
    """
    _seed(db, golden_inputs=60, constraints={"min_quality": 0.90}, production_runs=90)
    # gpt-4o-mini is the cheapest model in the sheet and is made to regress;
    # the mid-priced candidate ties the baseline.
    runtime = FakeRuntime(quality_for={"gpt-4o-mini": 0.90}, n_cases=60)

    result = _run_loop(
        db, runtime,
        candidates=[_candidate("gpt-4o-mini"), _candidate("gpt-4.1-mini")],
        create_recommendation=True,
    )

    assert result["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    rec = db.rows("optimization_recommendations")[0]
    assert "gpt-4.1-mini" in rec["title"]
    assert "gpt-4o-mini" not in rec["title"]

    # And the cheaper, rejected arm is not hidden — it leads the frontier, with
    # the reason it is not adoptable attached.
    frontier = result["frontier"]
    assert "gpt-4o-mini" in frontier["largest_observed_savings"]["label"]
    assert "quality_regression_above_threshold" in (
        frontier["largest_observed_savings"]["reason_codes"]
    )
    assert frontier["lowest_cost_rejected"]["label"] == (
        frontier["largest_observed_savings"]["label"]
    )
    assert "gpt-4.1-mini" in frontier["selected"]["label"]
    assert "gpt-4.1-mini" in frontier["quality_preserving"]["label"]


def test_a_policy_may_switch_the_evidence_gate_off_and_it_is_recorded(db):
    """
    Point estimates alone are the customer's right, not OptiML's default. When
    they choose it, the verdict must say so rather than implying evidence that
    was never gathered.
    """
    _seed(
        db, golden_inputs=30,
        constraints={"min_quality": 0.90, "require_quality_non_inferiority": False},
        production_runs=60,
    )
    runtime = FakeRuntime(n_cases=30)

    result = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])

    assert result["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    detail = next(
        r for r in result["reasons"] if r["code"] == "non_inferiority_not_established"
    )
    assert detail["detail_code"] == "non_inferiority_check_disabled_by_policy"
    assert result["quality_safety"]["established"] is False
    assert result["quality_safety_policy"]["require_quality_non_inferiority"] is False
    assert result["quality_safety_policy"]["require_quality_non_inferiority_source"] == (
        "policy"
    )


def test_the_stored_verdict_records_the_regime_that_produced_it(db):
    """A conclusion must stay reproducible as (evidence + policy version)."""
    _seed(db, golden_inputs=30, production_runs=60)
    runtime = FakeRuntime(n_cases=30)
    result = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])

    row = db.rows("benchmark_conclusions")[0]
    regime = row["quality_safety_policy"]
    assert regime["max_quality_regression"] == 0.05
    assert regime["max_quality_regression_source"] == "default"
    assert regime["quality_confidence_level"] == 0.95
    assert row["quality_safety"]["method"] == ni.METHOD
    assert result["quality_safety"]["confidence_level"] == 0.95


def test_the_frontier_names_the_alternatives_and_why_each_was_or_was_not_eligible(db):
    _seed(db, golden_inputs=60, constraints={"min_quality": 0.90}, production_runs=90)
    runtime = FakeRuntime(quality_for={"gpt-4o-mini": 0.70}, n_cases=60)

    result = _run_loop(
        db, runtime,
        candidates=[_candidate("gpt-4o-mini"), _candidate("gpt-4.1-mini")],
    )

    frontier = result["frontier"]
    assert frontier["baseline"]["quality"] == pytest.approx(1.0)
    assert len(frontier["entries"]) == 2
    for entry in frontier["entries"]:
        # Codes and facts only. No sentence the frontend would have to parse.
        assert entry["reason_codes"]
        assert all(c in domain.REASON_CODES for c in entry["reason_codes"])
        assert entry["tier"] == domain.TIER_EXECUTABLE
        assert entry["mean_cost_usd"] is not None
        assert entry["quality"] is not None

    rejected = next(e for e in frontier["entries"] if "gpt-4o-mini" in e["label"])
    assert rejected["status"] == domain.DISPOSITION_FAILED_POLICY
    assert rejected["quality_delta"] == pytest.approx(-0.30)


def test_the_consideration_funnel_accounts_for_every_candidate(db):
    _seed(db, golden_inputs=60, constraints={"min_quality": 0.90}, production_runs=90)
    runtime = FakeRuntime(quality_for={"gpt-4o-mini": 0.70}, n_cases=60)

    result = _run_loop(
        db, runtime,
        candidates=[_candidate("gpt-4o-mini"), _candidate("gpt-4.1-mini")],
    )

    funnel = result["consideration"]
    stages = {s["stage"]: s for s in funnel["stages"]}
    assert funnel["considered"] == 2
    assert stages[domain.DISPOSITION_QUALITY_SAFE]["count"] == 1
    assert stages[domain.DISPOSITION_FAILED_POLICY]["count"] == 1
    # A stage nothing populates yet is marked as such, so a zero is never read
    # as "we checked and found none".
    assert stages[domain.DISPOSITION_ELIMINATED_BY_HISTORY]["emitted"] is False
    assert funnel["by_tier"][domain.TIER_EXECUTABLE] == 2


# ===========================================================================
# 4. Tier 2 — opportunities the org cannot yet run
# ===========================================================================

def test_an_unconfigured_provider_becomes_an_unverified_opportunity_not_a_drop():
    """
    Not benchmarked (the arm would measure nothing) but RETAINED. A customer who
    has connected only OpenAI still deserves to be told a model elsewhere is
    worth evaluating.
    """
    from optimization import candidates as candidates_mod
    from optimization import strategy as strategy_mod
    from test_optimization_loop import WORKFLOW_ID, _graph

    baseline = strategy_mod.from_graph_json(_graph(), workflow_id=WORKFLOW_ID)
    cand = candidates_mod.Candidate(
        title="Try a model at a provider we have no key for",
        strategy=candidates_mod._swap_model(
            baseline, "n1", "anthropic", "claude-haiku-4-5"
        ),
        dimensions=["model", "provider"],
        generator="alternate_model",
        rationale="Vendor list price only.",
        evidence_source="none",
        projected_savings_usd=12.5,
    )

    with patch.object(
        candidates_mod, "_configured_providers", return_value={"openai"}
    ), patch.object(
        candidates_mod, "build_history", return_value={"model_stats": {}, "traffic": {}}
    ), patch.dict(
        candidates_mod.CANDIDATE_GENERATORS, {}, clear=True
    ):
        candidates_mod.CANDIDATE_GENERATORS["stub"] = type(
            "Stub", (), {"name": "stub", "generate": lambda self, w, b, h: [cand]}
        )()
        out, meta = candidates_mod.generate_candidates(
            ORG_ID, {"id": WORKLOAD_ID}, baseline, workflow_id=WORKFLOW_ID
        )

    assert out == []                                   # never benchmarked
    assert meta["dropped"] == []                       # and never forgotten
    assert len(meta["opportunities"]) == 1
    opp = meta["opportunities"][0]
    assert opp["tier"] == domain.TIER_OPPORTUNITY
    assert opp["code"] == "provider_not_configured"
    assert opp["providers"] == ["anthropic"]
    assert opp["next_action"] == "connect_provider"
    assert opp["verified"] is False
    # A hypothesis from a price sheet is the weakest evidence class there is.
    assert opp["evidence_source"] == "none"
    assert opp["evidence_strength"] == 0
    # And it carries NO measurement of any kind.
    assert opp["measured_quality"] is None
    assert opp["measured_cost_usd"] is None
    assert opp["projected_savings_usd"] == 12.5   # extrapolated, clearly labelled


def test_a_tier_two_opportunity_can_never_be_verified_or_win(db):
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)

    verdict = benchmark_mod.evaluate_conclusion(
        baseline={"quality": 1.0, "mean_cost_usd": 0.001, "per_case": [],
                  "quality_provenance": "deterministic"},
        measured=[],
        policy=None,
        materiality=domain.copy_default_materiality("cost"),
        objective="cost",
        signal=domain.SuccessSignal(),
        sample_size=60,
        opportunities=[{
            "label": "Try claude-haiku-4-5", "tier": domain.TIER_OPPORTUNITY,
            "code": "provider_not_configured", "providers": ["anthropic"],
            "evidence_source": "none", "verified": False,
            "next_action": "connect_provider",
        }],
        generation={"dropped": []},
    )

    # It appears in the frontier and in the funnel...
    assert len(verdict["frontier"]["unverified_opportunities"]) == 1
    stages = {s["stage"]: s for s in verdict["consideration"]["stages"]}
    assert stages[domain.DISPOSITION_PROVIDER_NOT_CONFIGURED]["count"] == 1
    assert verdict["consideration"]["by_tier"][domain.TIER_OPPORTUNITY] == 1

    # ...and nowhere near the verdict. It was never measured, so there is
    # nothing for it to win with.
    assert verdict["winner"] is None
    assert verdict["conclusion"] == domain.CONCLUSION_INSUFFICIENT_EVIDENCE
    assert verdict["frontier"]["selected"] is None
    for opp in verdict["frontier"]["unverified_opportunities"]:
        assert opp["verified"] is False
        assert domain.evidence_strength(opp["evidence_source"]) <= (
            domain.TIER_OPPORTUNITY_MAX_EVIDENCE_STRENGTH
        )


# ===========================================================================
# 5. Contract hygiene
# ===========================================================================

def test_every_emitted_reason_code_is_a_documented_one(db):
    _seed(db, golden_inputs=30, constraints={"min_quality": 0.90}, production_runs=60)
    runtime = FakeRuntime(n_cases=30)
    result = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])
    for bucket in ("reasons", "more_data_reasons"):
        for r in result[bucket]:
            assert r["code"] in domain.REASON_CODES


def test_the_new_conclusion_is_in_the_documented_vocabulary():
    assert domain.CONCLUSION_PROMISING_UNVERIFIED in domain.CONCLUSIONS
