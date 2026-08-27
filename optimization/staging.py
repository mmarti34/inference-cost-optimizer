"""
Staged candidate evaluation, and the bound that makes stopping early SOUND.

WHY THIS MODULE EXISTS
----------------------
The quality-safety default is a 2 percentage-point regression margin at 95%
one-sided confidence. A candidate that ties the baseline perfectly needs

    n >= z^2 * (1 - m) / m   =   1.644854^2 * 0.98 / 0.02   =   132.6  ->  133

paired cases before non-inferiority can be established at all (see
`n_for_perfect_tie` below, which derives that number rather than asserting it).

Running EVERY candidate over all 133 cases is the naive way to get there, and it
spends the customer's provider budget on candidates whose verdict is already
settled. The live example: gpt-4o-mini came back at 0.7667 against a baseline of
1.0000 with b=7, c=0 inside the first 30 cases. Nothing that could happen on the
remaining 103 cases would bring it back inside a 2pp margin — but it was run to
completion anyway.

WHAT "ALREADY DETERMINED" HAS TO MEAN
-------------------------------------
"It looks bad" is not a reason to stop; it is a guess dressed as a decision.
The only defensible early stop is one where the remaining cases CANNOT change
the verdict, whatever they contain. That is a bound, and it has to be derived.

Two things must both be impossible-to-recover before we drop a candidate,
because two different gates decide its verdict:

  (1) THE POLICY GATE, `max_quality_regression`, compares arm-level quality
      (checks passed / checks run) against the baseline's. This gate is what
      actually produces the `failed_policy` verdict.
  (2) THE EVIDENCE GATE, paired non-inferiority, works on per-case verdicts.
      This is what the statistics are built on and is the stricter of the two.

WHY THE BASELINE RUNS TO COMPLETION FIRST
-----------------------------------------
This is the load-bearing design choice, and it is what makes any bound tight
enough to ever fire.

If the baseline were staged alongside the candidates, then at stage 1 the
baseline's verdicts on the remaining r cases would be UNKNOWN. The candidate
could, in principle, pass every remaining case while the baseline failed every
one of them, so the best possible final difference would be

    (c - b + r) / (n + r)

which is non-negative whenever r >= b — i.e. essentially always. No candidate
could ever be dropped, and staging would buy nothing.

The baseline is never a candidate for elimination: it is the reference, it is
run over the full case set in every design, and it costs the same either way.
So it is executed FIRST, over all N cases, in a fixed order. Every candidate is
then staged against a baseline whose per-case verdicts on the remaining cases
are already MEASURED. `remaining_baseline_failed` below is a count of real
observations, not an assumption, and that is what makes the bound tight.

PAIRING IS PRESERVED EXACTLY
----------------------------
Every arm consumes the same ordered case list. A candidate stopped after k cases
is compared against the baseline restricted to those SAME first k cases — never
against the baseline's full-run figures, which would be a comparison over two
different case sets and would not be a paired statistic at all. The alignment is
verified positionally by `case_id` before any bound is computed; a mismatch
returns "cannot stop" rather than a bound derived from mismatched rows.

THE BOUND, DERIVED
------------------
A candidate has run the first n_run cases. Let the paired counts over that
prefix be b (baseline passed, candidate failed) and c (baseline failed,
candidate passed) over n usable pairs. Of the r cases not yet run, the BASELINE
has already been measured: r_pairable of them yield a usable baseline verdict,
and r_fail of those are cases the baseline FAILED.

  Paired bound.  The final difference is delta = D / n_final where
  D = c_final - b_final.

    * D is largest when the candidate passes every remaining case: on a case the
      baseline passed that is a concordant pass (D unchanged); on a case the
      baseline failed it is a new c (D + 1). So  D_max = c - b + r_fail.
      A candidate can never IMPROVE D on a case the baseline passed.
    * If D_max >= 0 the candidate could still finish at or above the baseline,
      so no stop is possible. Report the bound as D_max / n (the smallest
      denominator, which maximises a non-negative ratio).
    * If D_max < 0, delta is negative, and a NEGATIVE ratio is maximised by the
      LARGEST denominator: every remaining pairable case being usable, so
      n_final_max = n + r_pairable. Hence

          best_case_final_paired_delta = (c - b + r_fail) / (n + r_pairable)

      Both extremes are simultaneously achievable (candidate passes everything,
      every remaining pairable case usable), so the bound is attained and is not
      merely an over-estimate.

  Arm bound.  Arm quality is passed/ran over checks, not cases. The candidate
  has `passed` of `ran` so far. The number of checks that will run on each
  remaining case is NOT a guess: `_run_quality_checks` skips a check only for
  reasons that depend on the check's own configuration and the case's expected
  output, never on the arm's output, so the baseline's own measured
  `quality_checks_ran` on those cases is the exact ceiling (the candidate can
  only run FEWER, by erroring). With K = that measured total,

          best_case_final_quality = (passed + K) / (ran + K)

      which is (p + q)/(r + q) increasing in q, so K is the maximising choice.
      The bound on the final regression is baseline_quality_full minus that.

STOP CONDITION
--------------
All three must hold, and each has a distinct job:

  observed_regression_prefix      > margin   the record written NOW is a real,
                                             like-for-like policy violation over
                                             the shared prefix
  best_case_final_regression      > margin   the policy gate can never be
                                             satisfied by finishing the run
  best_case_final_paired_delta    < -margin  the evidence gate can never be
                                             satisfied by finishing the run

Anything softer is a guess and this module will not stop on it.

WHAT THIS MODULE DOES NOT CLAIM
-------------------------------
* It never stops a candidate that could still recover. The bound is the whole
  point; "looks bad on 30 cases" is not a reason.
* It cannot MEASURE the dollars not spent, because the avoided cases were never
  executed and therefore never priced. `spend_avoided` reports the exactly
  measured counts and leaves the measured dollar field NULL with a reason code;
  a projection, when the arm's own cost was measured, lands in a separate
  explicitly-projected field carrying the measured inputs it came from. That is
  the same measured/estimated split the cost path already uses.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Optional

from optimization import noninferiority as ni

#: Identifier persisted with every staged run, so a stored decision says which
#: bound produced it and can be re-derived later.
BOUND_METHOD = "best_case_completion_paired_and_arm"

#: DEFAULT STAGE SCHEDULE, as CUMULATIVE case counts.
#:
#: Every number here is derived, not chosen for roundness:
#:   30   the smallest replay set this product has run, and comfortably above
#:        `noninferiority.MIN_PAIRS_FOR_ASYMPTOTIC` (20), so the paired counts
#:        at the first checkpoint are already meaningful rather than noise.
#:   60   a doubling. A second checkpoint exists so a candidate that only
#:        becomes unrecoverable later is still caught before the full run.
#:   133  `n_for_perfect_tie(0.02, 0.95)` — the smallest sample at which a
#:        candidate that ties the baseline PERFECTLY can establish
#:        non-inferiority at the default margin. Past this point, more cases
#:        cannot change a tied candidate's verdict.
#:
#: The schedule is CLAMPED to the cases actually available, so a workload with
#: 45 golden inputs runs stages [30, 45] and one with 200 runs [30, 60, 133,
#: 200]. The final stage is always the whole case set: staging never silently
#: shortens a run.
DEFAULT_STAGE_SIZES: tuple[int, ...] = (30, 60, 133)

#: Reason codes. Machine codes only; the frontend owns all wording.
STOP_REGRESSION_UNRECOVERABLE = "regression_unrecoverable_on_remaining_cases"

CONTINUE_NO_CASES_REMAINING = "no_cases_remaining"
CONTINUE_CASE_ALIGNMENT_UNVERIFIED = "case_alignment_unverified"
CONTINUE_QUALITY_NOT_MEASURED = "quality_not_measured"
CONTINUE_OBSERVED_WITHIN_MARGIN = "observed_regression_within_margin"
CONTINUE_REMAINING_COULD_RECOVER = "remaining_cases_could_recover_candidate"

#: Why the dollars avoided are not a measurement.
SPEND_AVOIDED_NOT_MEASURABLE = "avoided_cases_never_executed_so_never_priced"
SPEND_AVOIDED_PROJECTION_BASIS = "projected_from_measured_mean_cost_per_case"
SPEND_AVOIDED_PROJECTION_UNAVAILABLE = "candidate_cost_not_measured"


# ---------------------------------------------------------------------------
# The sample size the margin implies
# ---------------------------------------------------------------------------

def n_for_perfect_tie(margin: float, confidence_level: float) -> Optional[int]:
    """
    Smallest paired sample at which a PERFECT tie (b = c = 0) establishes
    non-inferiority at `margin`.

    From the closed form in `noninferiority` (asserted in its tests): at
    b = c = 0 and delta = -m the score statistic is Z = sqrt(n * m / (1 - m)).
    Setting Z >= z_(1-alpha) and solving gives n >= z^2 (1 - m) / m.

    This is the ONLY place the "133" in the default schedule comes from.
    """
    m = abs(float(margin))
    if m <= 0 or m >= 1:
        return None
    z = NormalDist().inv_cdf(float(confidence_level))
    return int(math.ceil((z * z) * (1.0 - m) / m))


# ---------------------------------------------------------------------------
# The stage plan
# ---------------------------------------------------------------------------

def resolve_stages(
    total_cases: int, stage_sizes: Optional[list] = None
) -> list[dict]:
    """
    Turn a cumulative stage schedule into concrete [start, end) slices.

    Sizes are cumulative case counts. They are de-duplicated, sorted, clamped to
    `total_cases`, and the full case set is always appended as the final stage —
    so no configuration can make a benchmark run fewer cases than it holds.
    """
    total = max(0, int(total_cases or 0))
    if total == 0:
        return []

    raw = DEFAULT_STAGE_SIZES if stage_sizes is None else stage_sizes
    cuts: list[int] = []
    for s in (raw or ()):
        try:
            v = int(s)
        except (TypeError, ValueError):
            continue
        if 0 < v < total:
            cuts.append(v)
    cuts = sorted(set(cuts))
    cuts.append(total)

    stages: list[dict] = []
    start = 0
    for i, end in enumerate(cuts):
        stages.append({
            "stage_index": i + 1,
            "start": start,
            "end": end,
            "size": end - start,
            "cases_cumulative": end,
        })
        start = end
    return stages


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------

def _check_totals(rows) -> tuple[int, int]:
    """(checks_passed, checks_ran) summed over per-case rows."""
    passed = ran = 0
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        try:
            cr = int(r.get("quality_checks_ran") or 0)
            cp = int(r.get("quality_checks_passed") or 0)
        except (TypeError, ValueError):
            continue
        if cr > 0:
            ran += cr
            passed += cp
    return passed, ran


def _aligned_prefix_length(baseline_rows: list, candidate_rows: list) -> Optional[int]:
    """
    Confirm the candidate ran exactly a PREFIX of the baseline's case order.

    Returns the prefix length, or None when the two arms do not line up — in
    which case no bound may be computed, because the counts would be over two
    different case sets.
    """
    n = len(candidate_rows or [])
    if n == 0 or n > len(baseline_rows or []):
        return None
    for i in range(n):
        b_id = str((baseline_rows[i] or {}).get("case_id") or "")
        c_id = str((candidate_rows[i] or {}).get("case_id") or "")
        if not b_id or b_id != c_id:
            return None
    return n


def early_stop_assessment(
    *,
    margin: float,
    baseline_per_case: Optional[list],
    candidate_per_case: Optional[list],
) -> dict:
    """
    Can this candidate's verdict still change if we finish the run?

    `baseline_per_case` is the baseline arm over the FULL ordered case set;
    `candidate_per_case` is the candidate over a PREFIX of that same order.

    Returns the decision AND every quantity it was derived from, so a stored
    stop is checkable years later without re-running anything.
    """
    m = abs(float(margin))
    base_rows = list(baseline_per_case or [])
    cand_rows = list(candidate_per_case or [])

    out: dict[str, Any] = {
        "stop": False,
        "reason_code": None,
        "bound_method": BOUND_METHOD,
        "margin": round(m, 6),
        "cases_run": len(cand_rows),
        "cases_total": len(base_rows),
        "cases_remaining": max(0, len(base_rows) - len(cand_rows)),
        "alignment_verified": False,
        "observed_regression_prefix": None,
        "observed_paired_delta_prefix": None,
        "best_case_final_paired_delta": None,
        "best_case_final_quality": None,
        "best_case_final_regression": None,
        "remaining_baseline_pairable": None,
        "remaining_baseline_failed": None,
        "remaining_baseline_checks": None,
    }

    n_prefix = _aligned_prefix_length(base_rows, cand_rows)
    if n_prefix is None:
        out["reason_code"] = CONTINUE_CASE_ALIGNMENT_UNVERIFIED
        return out
    out["alignment_verified"] = True

    remaining = base_rows[n_prefix:]
    if not remaining:
        out["reason_code"] = CONTINUE_NO_CASES_REMAINING
        return out

    # ── Arm-level figures, each over an explicit case set.
    b_passed_pre, b_ran_pre = _check_totals(base_rows[:n_prefix])
    b_passed_all, b_ran_all = _check_totals(base_rows)
    c_passed, c_ran = _check_totals(cand_rows)
    if b_ran_pre <= 0 or c_ran <= 0 or b_ran_all <= 0:
        out["reason_code"] = CONTINUE_QUALITY_NOT_MEASURED
        return out

    baseline_quality_prefix = b_passed_pre / b_ran_pre
    baseline_quality_full = b_passed_all / b_ran_all
    candidate_quality_prefix = c_passed / c_ran
    observed_regression = baseline_quality_prefix - candidate_quality_prefix
    out["observed_regression_prefix"] = round(observed_regression, 6)

    # ── Paired counts over the SHARED prefix, and only the shared prefix.
    counts = ni.paired_counts(base_rows[:n_prefix], cand_rows)
    n = counts["n_pairs"]
    b = counts["discordant_b"]
    c = counts["discordant_c"]
    out["paired"] = {
        "n_pairs": n,
        "discordant_b": b,
        "discordant_c": c,
        "concordant_pass": counts["concordant_pass"],
        "concordant_fail": counts["concordant_fail"],
    }
    if n > 0:
        out["observed_paired_delta_prefix"] = round((c - b) / float(n), 6)

    # ── What the baseline already MEASURED on the cases still to run.
    r_pairable = 0
    r_fail = 0
    for row in remaining:
        v = ni.case_verdict(row)
        if v is None:
            continue
        r_pairable += 1
        if v is False:
            r_fail += 1
    _, r_checks = _check_totals(remaining)
    out["remaining_baseline_pairable"] = r_pairable
    out["remaining_baseline_failed"] = r_fail
    out["remaining_baseline_checks"] = r_checks

    if n <= 0:
        # Checks ran but nothing paired: no paired statement is derivable, so
        # nothing may be concluded and nothing may be stopped.
        out["reason_code"] = CONTINUE_QUALITY_NOT_MEASURED
        return out

    # ── Paired bound. D_max = c - b + r_fail: the candidate can gain a
    # discordant pair only where the BASELINE failed, and can never gain one
    # where the baseline passed.
    d_max = c - b + r_fail
    if d_max >= 0:
        # Still reachable at or above the baseline. Smallest denominator
        # maximises a non-negative ratio.
        best_delta = d_max / float(n)
    else:
        # A negative ratio is maximised by the largest denominator.
        best_delta = d_max / float(n + r_pairable)
    out["best_case_final_paired_delta"] = round(best_delta, 6)

    # ── Arm bound. The candidate can run at most as many checks on the
    # remaining cases as the baseline MEASURED itself running on them.
    best_quality = (c_passed + r_checks) / float(c_ran + r_checks)
    out["best_case_final_quality"] = round(best_quality, 6)
    out["best_case_final_regression"] = round(baseline_quality_full - best_quality, 6)

    # ── The three conditions. Every one is required.
    if observed_regression <= m:
        out["reason_code"] = CONTINUE_OBSERVED_WITHIN_MARGIN
        return out
    if not (out["best_case_final_regression"] > m and best_delta < -m):
        out["reason_code"] = CONTINUE_REMAINING_COULD_RECOVER
        return out

    out["stop"] = True
    out["reason_code"] = STOP_REGRESSION_UNRECOVERABLE
    return out


# ---------------------------------------------------------------------------
# Provider spend not incurred
# ---------------------------------------------------------------------------

def spend_avoided(
    *,
    cases_not_run: int,
    mean_cost_usd: Optional[float],
    cases_priced: Optional[int],
) -> dict:
    """
    What early stopping did NOT spend.

    `cases_not_run` and the executions they represent are exact counts of
    something that demonstrably did not happen — those are measured.

    The DOLLARS are not, and cannot be: the avoided cases were never executed,
    so no provider ever priced them. `spend_avoided_usd` is therefore always
    NULL with a reason code. A projection from the arm's own measured mean cost
    per case is carried in a separately named field alongside the measured
    inputs it came from, exactly as the cost path already separates
    `mean_cost_usd` from `mean_cost_estimated_usd`.
    """
    skipped = max(0, int(cases_not_run or 0))
    out: dict[str, Any] = {
        "cases_not_run": skipped,
        "workflow_executions_avoided": skipped,
        "spend_avoided_usd": None,
        "spend_avoided_reason": SPEND_AVOIDED_NOT_MEASURABLE,
        "spend_avoided_projected_usd": None,
        "spend_avoided_projection_basis": None,
        "projected_from_mean_cost_usd": None,
        "projected_from_cases_measured": None,
    }
    if skipped == 0:
        return out
    if mean_cost_usd is None or not cases_priced:
        out["spend_avoided_projection_basis"] = SPEND_AVOIDED_PROJECTION_UNAVAILABLE
        return out

    out["spend_avoided_projected_usd"] = round(float(mean_cost_usd) * skipped, 10)
    out["spend_avoided_projection_basis"] = SPEND_AVOIDED_PROJECTION_BASIS
    out["projected_from_mean_cost_usd"] = mean_cost_usd
    out["projected_from_cases_measured"] = int(cases_priced)
    return out


def rollup(stage_records: list[dict]) -> dict:
    """
    Run-level summary over every candidate's staging record.

    Pure aggregation of measured counts. The dollar total stays NULL for the
    same reason the per-candidate one does; the projected total is the sum of
    the per-candidate projections that were derivable, and states how many
    candidates it covers so a partial total is never read as a complete one.
    """
    stopped = [s for s in stage_records if s.get("stopped_early")]
    cases_not_run = sum(int(s.get("cases_not_run") or 0) for s in stage_records)

    projections = [
        s.get("spend_avoided_projected_usd")
        for s in stage_records
        if s.get("spend_avoided_projected_usd") is not None
    ]
    return {
        "candidates_evaluated": len(stage_records),
        "candidates_stopped_early": len(stopped),
        "cases_not_run": cases_not_run,
        "workflow_executions_avoided": cases_not_run,
        "spend_avoided_usd": None,
        "spend_avoided_reason": SPEND_AVOIDED_NOT_MEASURABLE,
        "spend_avoided_projected_usd": (
            round(sum(projections), 10) if projections else None
        ),
        "spend_avoided_projection_basis": (
            SPEND_AVOIDED_PROJECTION_BASIS if projections else None
        ),
        "projection_covers_candidates": len(projections),
        "bound_method": BOUND_METHOD,
    }
