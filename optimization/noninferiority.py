"""
Paired non-inferiority evidence for quality.

WHY THIS MODULE EXISTS
----------------------
OptiML's promise is not "find the cheapest candidate above an absolute floor".
It is "find a cheaper candidate and establish that it does not materially
degrade the customer's actual workload". Those two things diverge, and the
divergence is not hypothetical: a live benchmark produced a candidate scoring
exactly 0.9000 against a baseline of 1.0000 on 30 cases. It cleared a
`min_quality: 0.90` floor by a margin of zero, won on cost, and was written as
a VERIFIED recommendation. A 10 percentage-point observed regression was
presented to a customer as safe.

An absolute floor cannot catch that, because the floor knows nothing about the
baseline. A point estimate cannot catch it either, because 27/30 and 30/30 are
also compatible with a much larger true gap. What is needed is a statement of
the form "we can rule out, at a stated confidence, that this candidate is worse
than what you run today by more than X". That is a NON-INFERIORITY test.

WHY A PAIRED TEST, AND WHY TANGO'S SCORE INTERVAL
-------------------------------------------------
The baseline arm and every candidate arm are replayed over the SAME golden
cases. The data is therefore PAIRED: for each case we know whether the baseline
passed and whether the candidate passed. Treating the two arm means as
independent samples throws that pairing away, inflates the standard error, and
is simply the wrong model for the data we hold. The paired structure collapses
to four counts:

    concordant_pass  both passed          (uninformative about the difference)
    concordant_fail  both failed          (uninformative about the difference)
    b = n01          baseline passed, candidate FAILED   <- discordant
    c = n10          baseline failed, candidate PASSED   <- discordant

The difference in pass rates is exactly (c - b) / n. Only the discordant pairs
carry information about the difference, which is the McNemar insight.

We do NOT use exact conditional McNemar, for two reasons that matter here:

  1. McNemar tests the null "b = c" — EQUALITY. We are not asking whether the
     candidate differs from baseline; we are asking whether it is worse by more
     than a stated margin. A margin-based question needs a margin-based test.

  2. The conditional exact test conditions on the number of discordant pairs.
     When b = c = 0 — which is precisely what a candidate that ties a perfect
     baseline produces — there are zero discordant pairs, the conditional
     distribution is degenerate, and the exact test yields p = 1 and no
     information at all. That is a real limitation of the data, but the
     unconditional score test can still say something useful about it: with
     n = 30 and no observed discordance, a true 5pp deficit remains plausible,
     and the score test quantifies exactly how implausible it is.

So we use TANGO'S SCORE-BASED CONFIDENCE INTERVAL for the difference of paired
proportions (Tango 1998), which is the method Newcombe's comparative work and
subsequent practice recommend for this design. It is an asymptotic score test,
but it is well behaved at the boundaries (p = 1.0, zero discordant pairs) where
Wald intervals collapse to zero width and would declare a perfect tie
infinitely precise.

DERIVATION (so the constants below are checkable, not folklore)
---------------------------------------------------------------
Let delta = p_candidate - p_baseline, and let q = P(candidate fails, baseline
passes). Then P(candidate passes, baseline fails) = q + delta. Profiling the
multinomial likelihood over the two concordant cells at fixed delta gives a
constrained MLE q~ solving

    2n*q^2  +  [ 2*delta*(n - c) - (b + c)*(1 - delta) ] * q  -  b*delta*(1 - delta) = 0

(positive root), and the score statistic

    Z(delta) = (c - b - n*delta) / sqrt( n * (2*q~ + delta*(1 - delta)) )

Sanity checks that fall out of the algebra, both asserted in the tests:
  * delta = 0 gives q~ = (b + c) / 2n, i.e. the McNemar null.
  * b = c = 0 and delta = -m gives q~ = m and Z = sqrt(n * m / (1 - m)).

Non-inferiority at margin m is ESTABLISHED when Z(-m) >= z_(1-alpha), i.e. when
the one-sided lower confidence bound on delta lies above -m.

HONESTY ABOUT WHAT THIS DOES NOT DO
-----------------------------------
* It is asymptotic. At very small n the stated confidence level is nominal
  rather than exact. `MIN_PAIRS_FOR_ASYMPTOTIC` records where we stop claiming
  the approximation is reasonable, and the assessment carries the assumption
  explicitly rather than burying it.
* It assumes the replay cases are representative of the workload. Nothing in a
  benchmark can establish that; only the customer's case selection can.
* It says nothing about quality dimensions the eval suite does not check. A
  deterministic exact-match suite measures exact-match, not "goodness".
* Every arm's per-case verdict is binary. A workload whose eval suite runs
  several checks per case is scored strictly: a case counts as a pass only when
  every check that ran on it passed. That is stated in the output as
  `case_pass_rule` rather than left implicit, because it makes the paired pass
  rate potentially STRICTER than the arm-level check-ratio quality figure.

Nothing here ever invents a number. Every function returns None with a reason
code when the quantity is not derivable from the data in hand.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Iterable, Optional

#: The method identifier persisted alongside every assessment, so a stored
#: verdict says which test produced it and can be re-derived years later.
METHOD = "tango_score_paired_noninferiority"

#: Below this many usable pairs the normal approximation underlying the score
#: test is not trustworthy enough to call anything established. This does not
#: silently downgrade the result — it produces an explicit reason code.
MIN_PAIRS_FOR_ASYMPTOTIC = 20

#: Hard ceiling on the "how many more cases" search. Past this the honest answer
#: is "more than we can sensibly ask you to author", not a number.
MAX_PROJECTED_SAMPLE = 200_000

#: Reason codes explaining why non-inferiority was NOT established. These are
#: machine codes; the frontend owns all wording.
NOT_ESTABLISHED_PAIRED_CASES_UNAVAILABLE = "paired_cases_unavailable"
NOT_ESTABLISHED_SAMPLE_TOO_SMALL = "paired_sample_below_asymptotic_floor"
NOT_ESTABLISHED_EVIDENCE_INSUFFICIENT = "non_inferiority_not_established"
NOT_ESTABLISHED_REGRESSION_EXCEEDS_MARGIN = "observed_regression_exceeds_margin"

#: Reason codes for why an "N more cases" figure could not be derived.
MORE_CASES_NOT_DERIVABLE_NO_PAIRS = "no_paired_cases_to_extrapolate_from"
MORE_CASES_NOT_DERIVABLE_REGRESSION = "observed_regression_exceeds_margin"
MORE_CASES_NOT_DERIVABLE_TOO_MANY = "required_sample_exceeds_practical_limit"


# ---------------------------------------------------------------------------
# Per-case verdicts and pairing
# ---------------------------------------------------------------------------

def case_verdict(per_case_row: dict) -> Optional[bool]:
    """
    Did this one replay case pass?

    True/False when at least one quality check actually ran on it; None when
    nothing measurable ran, or the case errored. None means UNUSABLE for
    pairing — never a silent False, because "no check ran" and "the check
    failed" are different facts and conflating them would manufacture a
    regression out of a coverage gap.
    """
    if not isinstance(per_case_row, dict):
        return None
    if per_case_row.get("error"):
        return None
    explicit = per_case_row.get("case_passed")
    if isinstance(explicit, bool):
        return explicit
    ran = per_case_row.get("quality_checks_ran")
    passed = per_case_row.get("quality_checks_passed")
    if ran is None or passed is None:
        return None
    try:
        ran_i, passed_i = int(ran), int(passed)
    except (TypeError, ValueError):
        return None
    if ran_i <= 0:
        return None
    # Strict: a case passes only when every check that ran on it passed.
    return passed_i >= ran_i


def paired_counts(
    baseline_per_case: Optional[Iterable[dict]],
    candidate_per_case: Optional[Iterable[dict]],
) -> dict:
    """
    Join two arms' per-case records on case_id and count the four cells.

    Cases present in only one arm, or unusable in either, are EXCLUDED from the
    pairing and counted in `unusable_pairs`. Comparing a candidate case against
    a baseline case that never ran would not be a pair.
    """
    base_by_case: dict[str, Optional[bool]] = {}
    for row in (baseline_per_case or []):
        cid = str((row or {}).get("case_id") or "")
        if cid:
            base_by_case[cid] = case_verdict(row)

    b = c = concordant_pass = concordant_fail = 0
    unusable = 0
    seen: set[str] = set()

    for row in (candidate_per_case or []):
        cid = str((row or {}).get("case_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if cid not in base_by_case:
            unusable += 1
            continue
        base_ok = base_by_case[cid]
        cand_ok = case_verdict(row)
        if base_ok is None or cand_ok is None:
            unusable += 1
            continue
        if base_ok and cand_ok:
            concordant_pass += 1
        elif (not base_ok) and (not cand_ok):
            concordant_fail += 1
        elif base_ok and not cand_ok:
            b += 1
        else:
            c += 1

    # Baseline cases the candidate arm never reached are unusable too.
    unusable += len([cid for cid in base_by_case if cid not in seen])

    n = concordant_pass + concordant_fail + b + c
    return {
        "n_pairs": n,
        "discordant_b": b,       # baseline passed, candidate failed
        "discordant_c": c,       # baseline failed, candidate passed
        "concordant_pass": concordant_pass,
        "concordant_fail": concordant_fail,
        "unusable_pairs": unusable,
        "baseline_quality_paired": (
            round((concordant_pass + b) / n, 6) if n else None
        ),
        "candidate_quality_paired": (
            round((concordant_pass + c) / n, 6) if n else None
        ),
        "case_pass_rule": "all_checks_that_ran_on_the_case_passed",
    }


# ---------------------------------------------------------------------------
# Tango's score test
# ---------------------------------------------------------------------------

def constrained_q(n: int, b: int, c: int, delta: float) -> Optional[float]:
    """
    Constrained MLE of q = P(candidate fails, baseline passes) under the null
    that p_candidate - p_baseline = `delta`. Positive root of

        2n q^2 + [2 delta (n - c) - (b + c)(1 - delta)] q - b delta (1 - delta) = 0

    See the module docstring for the derivation. Returns None when n <= 0.
    """
    if n <= 0:
        return None
    A = 2.0 * n
    B = 2.0 * delta * (n - c) - (b + c) * (1.0 - delta)
    C = -b * delta * (1.0 - delta)
    disc = B * B - 4.0 * A * C
    if disc < 0:
        # Numerically possible only from float error at the boundary.
        disc = 0.0
    q = (-B + math.sqrt(disc)) / (2.0 * A)
    return max(0.0, q)


def score_z(n: int, b: int, c: int, delta: float) -> Optional[float]:
    """
    Tango score statistic for H0: p_candidate - p_baseline = `delta`.

    Positive Z means the observed data sit ABOVE the null difference, i.e. the
    candidate looks better than the null allows. Returns None when the variance
    under the null is not positive (degenerate; happens only at delta = 0 with
    no discordant pairs, which is never a non-inferiority null).
    """
    if n <= 0:
        return None
    q = constrained_q(n, b, c, delta)
    if q is None:
        return None
    var_term = 2.0 * q + delta * (1.0 - delta)
    if var_term <= 0:
        return None
    return (c - b - n * delta) / math.sqrt(n * var_term)


def lower_confidence_bound(
    n: int, b: int, c: int, confidence_level: float
) -> Optional[float]:
    """
    One-sided lower confidence bound on delta = p_candidate - p_baseline.

    Found by bisecting Z(delta) = z, which is monotonically decreasing in delta.
    A bound of -0.04 at 95% reads: "we can rule out, at 95% one-sided
    confidence, that the candidate is more than 4 percentage points worse".
    """
    if n <= 0:
        return None
    z = NormalDist().inv_cdf(confidence_level)
    delta_hat = (c - b) / float(n)

    lo, hi = -1.0 + 1e-9, delta_hat
    # Z(hi) is 0 at delta_hat and must exceed z at lo for a root to exist.
    z_lo = score_z(n, b, c, lo)
    if z_lo is None or z_lo < z:
        return -1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        z_mid = score_z(n, b, c, mid)
        if z_mid is None:
            lo = mid
            continue
        if z_mid >= z:
            lo = mid
        else:
            hi = mid
    return round(lo, 6)


def _establishes(n: int, b: int, c: int, margin: float, confidence_level: float) -> bool:
    z_crit = NormalDist().inv_cdf(confidence_level)
    z = score_z(n, b, c, -abs(margin))
    return z is not None and z >= z_crit


# ---------------------------------------------------------------------------
# "How many more cases would it take?"
# ---------------------------------------------------------------------------

def additional_cases_required(
    *,
    n: int,
    b: int,
    c: int,
    margin: float,
    confidence_level: float,
    max_total: int = MAX_PROJECTED_SAMPLE,
) -> tuple[Optional[int], Optional[str], Optional[int]]:
    """
    How many MORE replay cases would be needed to establish non-inferiority,
    holding the observed discordance RATES constant.

    Derived from the same test that gates the verdict — not a rule of thumb.
    The projection is explicitly conditional: it assumes the additional cases
    behave like the ones already measured. That assumption is stated in the
    output rather than hidden, because if the candidate is genuinely worse, more
    cases will confirm that instead.

    Returns (additional_cases, reason_code, required_total). `additional_cases`
    is None whenever it cannot be derived, and the reason code says why.
    """
    if n <= 0:
        return None, MORE_CASES_NOT_DERIVABLE_NO_PAIRS, None

    m = abs(margin)
    rate_b = b / float(n)
    rate_c = c / float(n)
    delta_hat = rate_c - rate_b

    # No sample size rescues a point estimate at or beyond the margin: scaling
    # the observed rates keeps delta_hat fixed, and the bound converges to it.
    if delta_hat <= -m:
        return None, MORE_CASES_NOT_DERIVABLE_REGRESSION, None

    def passes(total: int) -> bool:
        if total < MIN_PAIRS_FOR_ASYMPTOTIC:
            return False
        return _establishes(
            total, int(round(rate_b * total)), int(round(rate_c * total)),
            m, confidence_level,
        )

    if passes(n):
        return 0, None, n
    if not passes(max_total):
        return None, MORE_CASES_NOT_DERIVABLE_TOO_MANY, None

    lo, hi = n, max_total
    while lo < hi:
        mid = (lo + hi) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid + 1
    # Rounding of the scaled counts can make `passes` non-monotone by a case or
    # two; walk forward until it holds stably rather than trusting the bisect.
    total = lo
    while total <= max_total and not passes(total):
        total += 1
    if total > max_total:
        return None, MORE_CASES_NOT_DERIVABLE_TOO_MANY, None
    return max(0, total - n), None, total


# ---------------------------------------------------------------------------
# The assessment the rest of the system stores and ranks on
# ---------------------------------------------------------------------------

def assess(
    *,
    baseline_per_case: Optional[Iterable[dict]],
    candidate_per_case: Optional[Iterable[dict]],
    margin: float,
    confidence_level: float,
    baseline_quality: Optional[float] = None,
    candidate_quality: Optional[float] = None,
) -> dict:
    """
    Full, explainable non-inferiority evidence for one candidate arm.

    This is what gets persisted. It deliberately does NOT collapse into the
    generic `confidence` field: `confidence` answers "how much do we trust this
    measurement in general", and this answers "can we rule out a material
    quality regression". Overloading one number with both questions is how a
    -10pp regression came to be shipped with confidence 0.171 attached.
    """
    m = abs(float(margin))
    counts = paired_counts(baseline_per_case, candidate_per_case)
    n = counts["n_pairs"]
    b = counts["discordant_b"]
    c = counts["discordant_c"]

    out: dict[str, Any] = {
        "method": METHOD,
        "established": False,
        "reason_code": None,
        "allowed_regression": round(m, 6),
        "confidence_level": round(float(confidence_level), 4),
        "critical_z": round(NormalDist().inv_cdf(confidence_level), 6),
        # Arm-level figures, carried so the stored evidence is self-contained.
        "baseline_quality": baseline_quality,
        "candidate_quality": candidate_quality,
        "observed_regression": (
            round(float(baseline_quality) - float(candidate_quality), 6)
            if baseline_quality is not None and candidate_quality is not None
            else None
        ),
        "test_statistic_z": None,
        "p_value": None,
        "lower_confidence_bound": None,
        "additional_cases_required": None,
        "additional_cases_reason": None,
        "required_total_cases": None,
        "assumptions": [
            "replay_cases_representative_of_workload",
            "asymptotic_normal_approximation",
            "additional_cases_behave_like_measured_cases",
        ],
    }
    out.update(counts)

    if n <= 0:
        out["reason_code"] = NOT_ESTABLISHED_PAIRED_CASES_UNAVAILABLE
        out["additional_cases_reason"] = MORE_CASES_NOT_DERIVABLE_NO_PAIRS
        return out

    # Paired figures are the ones the test actually uses. Report both, so a
    # difference between them is visible rather than silently reconciled.
    out["observed_regression_paired"] = round(
        (counts["baseline_quality_paired"] or 0.0)
        - (counts["candidate_quality_paired"] or 0.0),
        6,
    )

    z = score_z(n, b, c, -m)
    z_crit = out["critical_z"]
    if z is not None:
        out["test_statistic_z"] = round(z, 6)
        out["p_value"] = round(1.0 - NormalDist().cdf(z), 8)
    out["lower_confidence_bound"] = lower_confidence_bound(n, b, c, confidence_level)

    extra, extra_reason, required_total = additional_cases_required(
        n=n, b=b, c=c, margin=m, confidence_level=confidence_level,
    )
    out["additional_cases_required"] = extra
    out["additional_cases_reason"] = extra_reason
    out["required_total_cases"] = required_total

    if n < MIN_PAIRS_FOR_ASYMPTOTIC:
        out["reason_code"] = NOT_ESTABLISHED_SAMPLE_TOO_SMALL
        return out

    if z is not None and z >= z_crit:
        out["established"] = True
        return out

    delta_hat = (c - b) / float(n)
    out["reason_code"] = (
        NOT_ESTABLISHED_REGRESSION_EXCEEDS_MARGIN
        if delta_hat <= -m
        else NOT_ESTABLISHED_EVIDENCE_INSUFFICIENT
    )
    return out
