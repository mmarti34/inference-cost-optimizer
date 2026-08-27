-- ============================================================================
-- Migration: OptiML optimization layer v7 — quality safety is RELATIVE and
--            UNCERTAINTY-AWARE.
--
-- RUN AFTER: migration_optimization.sql
--            migration_optimization_work_graph.sql
--            migration_optimization_v3_outcomes.sql
--            migration_optimization_v4_benchmark_conclusions.sql
--            migration_optimization_v5_evidence_model.sql
--            migration_optimization_v6_direct_inference_attempts.sql
--
-- Idempotent. ADDITIVE ONLY — no column, constraint value or row is dropped.
--
-- *** NOT AUTOMATICALLY APPLIED. *** Reviewed and run by a human against
-- production. Nothing in this file rewrites history: existing benchmarks,
-- conclusions and recommendations keep every value they hold. In particular
-- benchmark 88813bfb-5581-45a0-abf6-884732a0b19b, its conclusion, and
-- recommendation 41d79006-bc14-4180-9d3d-97e71d6a2809 remain intact as the
-- historical record of what was concluded under policy v1. New semantics arrive
-- as a NEW policy version and a NEW conclusion row; see section 6.
--
--
-- ── WHAT WENT WRONG ─────────────────────────────────────────────────────────
--
-- The first live benchmark ran 30 authored replay cases with deterministic
-- exact-match grading:
--
--     baseline   gpt-4o        $0.000857  quality 1.0000
--     candidate  gpt-4.1-mini  $0.000137  quality 0.9000   -84% cost
--     candidate  gpt-5-mini    $0.000433  quality 1.0000   -49% cost
--     candidate  gpt-4o-mini   $0.000051  quality 0.7667   -94% cost
--
-- The policy carried min_quality = 0.90. gpt-4.1-mini scored EXACTLY 0.9000,
-- cleared the floor by a margin of zero, won on cost, and was written to the
-- customer as a recommendation at status 'verified' with confidence 0.171. A 10
-- percentage-point observed regression against the customer's own baseline was
-- presented as safe. gpt-5-mini matched the baseline exactly at half the cost
-- and was passed over.
--
-- Two independent defects:
--
--   1. THE CONSTRAINT WAS ABSOLUTE WHEN THE PROMISE IS RELATIVE. A floor knows
--      nothing about the baseline, so it cannot express "do not make my
--      workload worse than it is today". Fixed by `max_quality_regression`,
--      which does NOT replace min_quality — the two answer different questions
--      and are ANDed.
--
--   2. THE EVIDENCE WAS A POINT ESTIMATE. Even a candidate that TIES the
--      baseline on 30 cases has not established that it is not materially
--      worse. Fixed by a paired non-inferiority test over the per-case
--      pass/fail data both arms already produce.
--
--
-- ── THE FOUR THINGS THIS MIGRATION ADDS ─────────────────────────────────────
--
-- 1. STRUCTURED QUALITY-SAFETY EVIDENCE, NOT AN OVERLOADED `confidence`.
--    `confidence` answers "how much do we trust this measurement overall".
--    Whether a quality regression was ruled out is a DIFFERENT question, and
--    folding it into one number is how a -10pp regression shipped with 0.171
--    attached and no field anywhere recording that safety had never been
--    established. benchmark_conclusions.quality_safety holds the paired
--    discordant counts, the margin, the confidence level, the interval and an
--    explicit `established` boolean.
--
-- 2. A CONCLUSION FOR "PROMISING BUT SHORT OF EVIDENCE".
--    Collapsing that state into 'safe_improvement_found' is the bug. Collapsing
--    it into 'insufficient_evidence' would discard a real finding and tell the
--    customer nothing actionable. It is its own conclusion, it maps to
--    recommendation status 'inconclusive', and it counts as NOT COVERED.
--
-- 3. THE FRONTIER AND THE CONSIDERATION FUNNEL.
--    A customer shown only "we recommend X" cannot tell whether anything else
--    was looked at. The frontier names the largest observed saving, the
--    quality-preserving candidate, the cheapest rejected one and the reason
--    each was or was not eligible. The funnel accounts for every model that
--    entered consideration, including TIER 2 items for providers the org has
--    not connected — retained as unverified opportunities rather than dropped.
--
-- 4. NOTHING NEW ON THE EVIDENCE ROWS EXCEPT MEASUREMENTS.
--    benchmark_candidate_results gains no column. Paired discordant counts are
--    a MEASUREMENT (they depend only on the two arms' per-case verdicts) and go
--    into the existing outcome_metrics JSONB; the non-inferiority VERDICT is an
--    INTERPRETATION, is policy-versioned, and lives only on the conclusion.
--    That separation is what keeps `reevaluate` honest.
-- ============================================================================


-- ============================================================================
-- 1. benchmark_conclusions — structured, explainable quality-safety evidence.
-- ============================================================================
ALTER TABLE public.benchmark_conclusions
    ADD COLUMN IF NOT EXISTS quality_safety        JSONB,
    ADD COLUMN IF NOT EXISTS quality_safety_policy JSONB,
    ADD COLUMN IF NOT EXISTS frontier              JSONB,
    ADD COLUMN IF NOT EXISTS consideration         JSONB;

COMMENT ON COLUMN public.benchmark_conclusions.quality_safety IS
  'Paired non-inferiority evidence for the leading candidate, produced by optimization/noninferiority.py. Keys: method, established (bool), reason_code, n_pairs, discordant_b (baseline passed / candidate failed), discordant_c (baseline failed / candidate passed), concordant_pass, concordant_fail, baseline_quality, candidate_quality, observed_regression, allowed_regression, confidence_level, critical_z, test_statistic_z, p_value, lower_confidence_bound, additional_cases_required, required_total_cases, assumptions. NULL means no candidate reached assessment — never that safety was established. DELIBERATELY SEPARATE from `confidence`: that column answers "how much do we trust this measurement", this one answers "can we rule out a material quality regression", and overloading one number with both is what let a -10pp regression ship as verified.';

COMMENT ON COLUMN public.benchmark_conclusions.quality_safety_policy IS
  'The quality-safety REGIME in force when this verdict was produced: max_quality_regression, quality_confidence_level, require_quality_non_inferiority, min_quality, and a _source marker on each ("policy" or "default"). Recorded on every conclusion, including ones where no candidate was assessed, so a historical verdict stays reproducible as (evidence + policy version + regime + objective).';

COMMENT ON COLUMN public.benchmark_conclusions.frontier IS
  'The whole consideration set with the reason each option was or was not eligible: largest_observed_savings, quality_preserving, lowest_cost_rejected, selected, entries[], unverified_opportunities[] (TIER 2). Structured codes and measured facts ONLY — no customer-facing prose; the frontend owns all wording. Figures are measured or NULL, never estimated into a measured field.';

COMMENT ON COLUMN public.benchmark_conclusions.consideration IS
  'Candidate discovery funnel: ordered stage counts plus a per-candidate disposition with the code explaining where it exited. Stages: considered, incompatible, policy_blocked, provider_not_configured, eliminated_by_historical_evidence, duplicate, generator_error, benchmarked, not_measured, failed_policy, promising, quality_safe. A stage nothing populates yet carries emitted=false so a zero is never read as "we checked and found none".';


-- ============================================================================
-- 2. The new conclusion value.
--
-- ADDITIVE: every existing value is retained, so no stored conclusion becomes
-- invalid and no historical row needs rewriting.
-- ============================================================================
ALTER TABLE public.benchmark_conclusions
    DROP CONSTRAINT IF EXISTS benchmark_conclusions_conclusion_check;
ALTER TABLE public.benchmark_conclusions
    ADD CONSTRAINT benchmark_conclusions_conclusion_check
    CHECK (conclusion IN ('safe_improvement_found',
                          'no_material_improvement',
                          'candidates_failed_policy',
                          'promising_candidate_unverified',
                          'insufficient_evidence',
                          'benchmark_failed'));

COMMENT ON COLUMN public.benchmark_conclusions.conclusion IS
  'What the evidence supports, under this policy version and objective. safe_improvement_found = cheaper AND quality non-inferiority ESTABLISHED against the measured baseline. promising_candidate_unverified = cheaper, satisfies every hard constraint, no disqualifying observed regression, but the evidence does not yet RULE OUT a material regression — a real finding whose next action is "run about N more evaluations", NOT a recommendation. candidates_failed_policy = a hard constraint was violated. no_material_improvement = the only conclusion that may be rendered as "your configuration looks efficient". insufficient_evidence / benchmark_failed = ignorance, never a finding. COVERAGE counts only the first, second and third of these; promising_candidate_unverified is NOT coverage, because finding something we cannot vouch for is not the same as having assessed the workload.';

-- The denormalised mirror on optimization_benchmarks must accept it too.
ALTER TABLE public.optimization_benchmarks
    DROP CONSTRAINT IF EXISTS optimization_benchmarks_conclusion_check;
ALTER TABLE public.optimization_benchmarks
    ADD CONSTRAINT optimization_benchmarks_conclusion_check
    CHECK (conclusion IS NULL OR conclusion IN ('safe_improvement_found',
                                                'no_material_improvement',
                                                'candidates_failed_policy',
                                                'promising_candidate_unverified',
                                                'insufficient_evidence',
                                                'benchmark_failed'));


-- ============================================================================
-- 3. optimization_recommendations — the safety evidence travels with the
--    proposal.
--
-- Without this a reader of the recommendations table sees status='verified' and
-- candidate_quality=0.90 and has no way to tell whether a regression against
-- baseline was ever ruled out. That is exactly how the original failure reached
-- a customer.
-- ============================================================================
ALTER TABLE public.optimization_recommendations
    ADD COLUMN IF NOT EXISTS quality_safety JSONB;

COMMENT ON COLUMN public.optimization_recommendations.quality_safety IS
  'The paired non-inferiority evidence that justified this recommendation, copied from the benchmark conclusion that produced it. Same shape as benchmark_conclusions.quality_safety. NULL on recommendations created before v7, and on any recommendation not derived from a replay benchmark — NULL means NOT ESTABLISHED, never "fine".';

-- Recommendations that predate v7 have no such evidence. They are NOT
-- backfilled: inventing a `quality_safety` blob for a verdict that never ran
-- the test would be a fabrication, and a fabricated safety claim is worse than
-- an absent one. A reader distinguishes them by quality_safety IS NULL.


-- ============================================================================
-- 4. Documenting the two quality constraints on optimization_policies.
--
-- `constraints` is JSONB, so no DDL is needed to accept the new keys. What IS
-- needed is for the column comment to state that they are different questions
-- and are ANDed — the single misunderstanding that caused the incident.
-- ============================================================================
COMMENT ON COLUMN public.optimization_policies.constraints IS
  'Hard constraints that make a strategy INVALID, not merely worse. Enforceable today: min_quality, max_quality_regression, max_error_rate, max_latency_p95_ms, max_cost_per_task_usd, allowed_vendors, blocked_vendors, require_human_approval, min_sample_size. Quality-evidence keys: quality_confidence_level (one-sided, default 0.95), require_quality_non_inferiority (default true).

TWO QUALITY CONSTRAINTS, ANDed, ANSWERING DIFFERENT QUESTIONS:
  min_quality             ABSOLUTE floor. "Never run anything below X, whatever the baseline does." Customer-owned; OptiML declares NO default, because inventing a quality bar for someone else''s workload would be a fabrication.
  max_quality_regression  RELATIVE ceiling on degradation vs the MEASURED baseline arm. "Do not make my workload materially worse than it is today." OptiML DOES default this (0.05), because a customer who configured nothing still expects not to be handed a regression.
A candidate must satisfy BOTH. A candidate that ties the absolute floor exactly while sitting 10 percentage points under baseline satisfies the first and fails the second; before max_quality_regression existed it was recommended as verified.

Accepted, stored, reported, but NOT verifiable today (never reported as satisfied): require_zero_data_retention, allow_prompt_storage, data_region, require_certifications. See optimization/policies.py UNENFORCEABLE_CONSTRAINTS.';


-- ============================================================================
-- 5. Indexes.
-- ============================================================================
-- Current promising candidates per workload: the "worth pursuing, needs more
-- cases" queue the product surfaces.
CREATE INDEX IF NOT EXISTS idx_benchmark_conclusions_promising
    ON public.benchmark_conclusions (org_id, workload_id, created_at DESC)
    WHERE is_current AND conclusion = 'promising_candidate_unverified';

-- Verified recommendations whose safety evidence is missing — i.e. everything
-- decided before v7. An operational query, and the audit trail for the
-- incident.
CREATE INDEX IF NOT EXISTS idx_recommendations_missing_quality_safety
    ON public.optimization_recommendations (org_id, created_at DESC)
    WHERE quality_safety IS NULL AND status = 'verified';

-- No new RLS policies: both tables already have org-scoped SELECT policies from
-- v5 and v3, and adding columns does not change row visibility. Nothing here
-- weakens auth, RLS or org isolation.


-- ============================================================================
-- 6. RE-ASSESSING THE EXISTING BENCHMARK — WHAT TO DO, AND WHAT NOT TO DO.
--
-- DO NOT UPDATE benchmark 88813bfb-5581-45a0-abf6-884732a0b19b, its conclusion,
-- or recommendation 41d79006-bc14-4180-9d3d-97e71d6a2809. That verdict was
-- CORRECT under the policy in force at the time; it is the evidence that the
-- policy was wrong, and editing it would destroy the record of the incident.
--
-- The supported path, all of it additive and none of it in this file:
--
--   a. Insert a NEW policy VERSION (optimization/policies.update_policy does
--      this by INSERT, never in place) carrying max_quality_regression. The v1
--      row is retained with is_current = false.
--
--   b. Call optimization.benchmark.reevaluate() for the benchmark. It re-reads
--      the RETAINED benchmark_candidate_results — including the per-case
--      pass/fail data the paired test needs — runs ZERO model calls, and
--      INSERTS a new benchmark_conclusions row against the new policy version,
--      flipping is_current on the old one. Both verdicts then coexist, each
--      bound to the policy that produced it.
--
--   c. The v1 recommendation is transitioned through the normal lifecycle by a
--      human (to 'rejected' or 'superseded'). It is not deleted and not
--      rewritten.
--
-- Under a conservative default (max_quality_regression = 0.05, 95% one-sided)
-- the re-read of that benchmark reaches:
--     gpt-4.1-mini  0.9000 vs 1.0000, b=3 c=0  -> FAILS max_quality_regression
--     gpt-4o-mini   0.7667 vs 1.0000, b=7 c=0  -> FAILS max_quality_regression
--     gpt-5-mini    1.0000 vs 1.0000, b=0 c=0  -> promising_candidate_unverified;
--                                                 52 paired cases are needed at a
--                                                 5pp margin, so 22 more.
-- i.e. 'candidates_failed_policy' for the two regressions and a promising,
-- explicitly UNVERIFIED finding for the model that actually matched baseline.
-- ============================================================================


-- ============================================================================
-- Verify.
-- ============================================================================
-- -- Verified recommendations with no established quality safety. Before v7
-- -- every row qualifies; after it, any row here is one to look at.
-- SELECT r.id, r.title, r.status, r.baseline_quality, r.candidate_quality,
--        (r.quality_safety ->> 'established')::boolean AS non_inferiority_established,
--        r.quality_safety ->> 'reason_code'           AS why_not,
--        (r.quality_safety ->> 'observed_regression')::numeric AS observed_regression,
--        (r.quality_safety ->> 'allowed_regression')::numeric  AS allowed_regression
--   FROM public.optimization_recommendations r
--  WHERE r.org_id = '00000000-0000-0000-0000-000000000000'
--    AND r.status = 'verified'
--    AND COALESCE((r.quality_safety ->> 'established')::boolean, false) IS NOT TRUE
--  ORDER BY r.created_at DESC;
--
-- -- The promising queue, with the derived "N more cases" figure.
-- SELECT c.workload_id,
--        c.quality_safety ->> 'method'                              AS method,
--        (c.quality_safety ->> 'n_pairs')::int                      AS cases_so_far,
--        (c.quality_safety ->> 'additional_cases_required')::int     AS run_this_many_more,
--        (c.quality_safety ->> 'required_total_cases')::int          AS to_reach_total,
--        (c.quality_safety ->> 'lower_confidence_bound')::numeric    AS lower_bound,
--        (c.quality_safety_policy ->> 'max_quality_regression')::numeric AS margin,
--        c.quality_safety_policy ->> 'max_quality_regression_source' AS margin_source
--   FROM public.benchmark_conclusions c
--  WHERE c.org_id = '00000000-0000-0000-0000-000000000000'
--    AND c.is_current
--    AND c.conclusion = 'promising_candidate_unverified'
--  ORDER BY c.created_at DESC;
