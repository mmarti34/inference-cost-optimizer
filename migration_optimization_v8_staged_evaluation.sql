-- ============================================================================
-- Migration: OptiML optimization layer v8 — the 2pp default margin, and staged
--            candidate evaluation.
--
-- RUN AFTER: migration_optimization_v7_quality_non_inferiority.sql
--
-- *** NOT AUTOMATICALLY APPLIED, AND NOT RUN BY THE AGENT THAT WROTE IT. ***
--
-- DOCUMENTATION ONLY. Every statement in this file is a COMMENT ON. There is no
-- DDL, no DML, no index, no policy, no grant. It creates nothing, drops
-- nothing, rewrites nothing, and changes no row. COMMENT ON is idempotent by
-- construction — re-running it replaces the same descriptions with the same
-- text — so this file is safe to run more than once and safe never to run at
-- all: the application behaves identically either way.
--
-- Nothing here weakens auth, RLS or org isolation; comments are metadata and
-- carry no visibility semantics.
--
-- Benchmark 88813bfb-5581-45a0-abf6-884732a0b19b, its conclusion, and
-- recommendation 41d79006-bc14-4180-9d3d-97e71d6a2809 are untouched and remain
-- the historical record of what was concluded under policy v1.
--
--
-- ── WHY THIS FILE EXISTS ────────────────────────────────────────────────────
--
-- 1. v7's column comment states that OptiML defaults max_quality_regression to
--    0.05. That default is now 0.02. v7 is being applied to production as
--    written and must not be edited, so the correction arrives here. Leaving it
--    would leave the database documenting a default the code no longer applies.
--
-- 2. Staged candidate evaluation stores a new key inside an EXISTING JSONB
--    column. It needs no schema change — which is exactly why its shape needs
--    to be written down somewhere a reader of the schema will find it.
--
--
-- ── 1. THE DEFAULT MARGIN IS 0.02 ───────────────────────────────────────────
--
-- The margin represents acceptable quality degradation, not an
-- evaluation-budget knob. Sample size should adapt to the safety requirement,
-- not the other way around.
--
-- At 0.02 with 95% one-sided confidence a candidate that ties the baseline
-- PERFECTLY needs n >= z^2 (1 - m) / m = 1.644854^2 * 0.98 / 0.02 = 132.6, i.e.
-- 133 paired cases. That is the cost of a 2pp safety claim, and the answer to
-- it is staged evaluation (section 2), not a looser margin.
--
-- 0.05 remains a SUPPORTED per-workload override — one key in `constraints`,
-- source-stamped as the customer's choice — appropriate where the eval suite is
-- a coarse proxy rather than a decisive correctness test and a 2pp difference
-- sits below the suite's own resolution. It is no longer the default and
-- nothing prefers it.
-- ============================================================================
COMMENT ON COLUMN public.optimization_policies.constraints IS
  'Hard constraints that make a strategy INVALID, not merely worse. Enforceable today: min_quality, max_quality_regression, max_error_rate, max_latency_p95_ms, max_cost_per_task_usd, allowed_vendors, blocked_vendors, require_human_approval, min_sample_size. Quality-evidence keys: quality_confidence_level (one-sided, default 0.95), require_quality_non_inferiority (default true). Evaluation-schedule keys: staged_evaluation_enabled (default true), evaluation_stage_sizes (cumulative case counts, default [30, 60, 133]).

TWO QUALITY CONSTRAINTS, ANDed, ANSWERING DIFFERENT QUESTIONS:
  min_quality             ABSOLUTE floor. "Never run anything below X, whatever the baseline does." Customer-owned; OptiML declares NO default, because inventing a quality bar for someone else''s workload would be a fabrication.
  max_quality_regression  RELATIVE ceiling on degradation vs the MEASURED baseline arm. "Do not make my workload materially worse than it is today." OptiML DOES default this, because a customer who configured nothing still expects not to be handed a regression. THE DEFAULT IS 0.02 at 95% one-sided confidence (v7 documented 0.05; superseded by v8). The margin is the safety requirement, not an evaluation-budget knob: sample size adapts to it, not the reverse. A perfect tie needs 133 paired cases at 0.02 and 52 at 0.05. 0.05 remains a supported per-workload override for a workload whose eval suite is a coarse proxy rather than a decisive correctness test; it is not the default and nothing prefers it.
A candidate must satisfy BOTH. A candidate that ties the absolute floor exactly while sitting 10 percentage points under baseline satisfies the first and fails the second; before max_quality_regression existed it was recommended as verified.

Accepted, stored, reported, but NOT verifiable today (never reported as satisfied): require_zero_data_retention, allow_prompt_storage, data_region, require_certifications. See optimization/policies.py UNENFORCEABLE_CONSTRAINTS.';


-- ============================================================================
-- ── 2. STAGED CANDIDATE EVALUATION ──────────────────────────────────────────
--
-- Candidates are evaluated over the case set in stages and dropped as soon as
-- the evidence already gathered makes the verdict certain. NO NEW TABLE and no
-- new column: the per-stage evidence trail is a key inside the existing
-- benchmark_candidate_results.outcome_metrics JSONB.
--
-- THE INVARIANT. Every arm walks the SAME ordered case list from index 0. The
-- baseline is run to completion FIRST — it is the reference, never a candidate
-- for elimination, and costs the same either way — and each candidate is then
-- staged against it. A candidate stopped after k cases is scored against the
-- baseline restricted to those SAME k cases; sample_size on its row is k, and
-- per_case_results holds exactly k rows. Arms are never compared over different
-- case sets, because pairing is the entire basis of the statistics.
--
-- THE BOUND. A candidate is dropped only when finishing the run COULD NOT
-- change its verdict. With b, c the discordant counts over n usable pairs in
-- the prefix, and, among the r cases not yet run, r_pairable yielding a usable
-- baseline verdict of which r_fail are cases the BASELINE failed:
--
--     best_case_final_paired_delta = (c - b + r_fail) / (n + r_pairable)
--
--   The candidate can gain a discordant pair only where the baseline failed, so
--   c - b + r_fail is the largest final difference reachable; a negative ratio
--   is maximised by the largest denominator, hence n + r_pairable. r_fail is a
--   count of MEASURED baseline outcomes, which is why running the baseline
--   first is what makes the bound tight enough to ever fire.
--
--     best_case_final_quality = (passed + K) / (ran + K)
--
--   where K is the number of checks the baseline was measured running on the
--   remaining cases — an exact ceiling, since a check is skipped only for
--   reasons independent of the arm's own output.
--
-- All three of these must hold before anything stops: the observed regression
-- over the shared prefix exceeds the margin, the best possible final regression
-- still exceeds the margin, and the best possible final paired delta is still
-- below -margin. Anything softer is a guess.
--
-- A stopped candidate is candidates_failed_policy — the answer is KNOWN — and
-- is never insufficient_evidence. A candidate that survives every stage without
-- establishing non-inferiority stays promising_candidate_unverified, with its
-- "N more cases" figure recomputed from the cases it actually ran.
--
-- SPEND AVOIDED. cases_not_run and workflow_executions_avoided are exact counts
-- of executions that demonstrably did not happen. The DOLLARS are not
-- measurable — the avoided cases were never executed, so no provider ever
-- priced them — so spend_avoided_usd is ALWAYS NULL with a reason code, and any
-- projection from the arm's own measured mean cost per case lives in the
-- separately named spend_avoided_projected_usd with its basis and inputs. Same
-- measured/estimated split as mean_cost_usd vs mean_cost_estimated_usd.
-- ============================================================================
COMMENT ON COLUMN public.benchmark_candidate_results.outcome_metrics IS
  'Named outcome aggregates for this arm, kept separate per provenance so incompatible signals are never averaged: {"ticket_resolved":{"provenance":"business_outcome","rate":0.83,"n":120}}. Also carries measured facts about HOW this arm was evaluated, never a verdict about it (verdicts are policy-versioned and live on benchmark_conclusions):

  quality_checks_run, cost_variation, cases_measured, cost_basis, mean_cost_estimated_usd, total_cost_estimated_usd, pricing_provenance
  paired_vs_baseline   the four paired cells against the baseline over EXACTLY the cases this arm ran: n_pairs, discordant_b, discordant_c, concordant_pass, concordant_fail, unusable_pairs, baseline_quality_paired, candidate_quality_paired, case_pass_rule. A measurement, not an interpretation.
  staged_evaluation    the per-stage evidence trail (v8). Keys: enabled, stage_sizes, stage_sizes_source, stages_planned, stages_run, cases_planned, cases_run, stopped_early, stopped_at_stage, stop_reason_code, margin, margin_source, bound, stages[], cases_not_run, workflow_executions_avoided, spend_avoided_usd (ALWAYS NULL - the avoided cases were never executed, so were never priced), spend_avoided_reason, spend_avoided_projected_usd, spend_avoided_projection_basis, projected_from_mean_cost_usd, projected_from_cases_measured.
    stages[] holds, per stage, what was known WHEN: stage_index, cases_this_stage, cases_cumulative, quality, baseline_quality_same_cases, observed_regression, paired, best_case_final_paired_delta, best_case_final_regression, decision, decision_reason_code.
    bound holds the derivation that justified a stop, with every count it used, so a stop is re-checkable from the row alone. bound_method = best_case_completion_paired_and_arm.

When staged_evaluation.stopped_early is true, sample_size and per_case_results on this row cover the cases actually run, and every delta on the row (quality_delta, cost_delta_pct, latency_delta_pct) is against the baseline restricted to those same cases. See optimization/staging.py.';


-- ============================================================================
-- ── 3. Nothing else changes. ────────────────────────────────────────────────
--
-- No table, column, index, constraint, policy, grant or row is created,
-- altered or removed by this file.
--
-- -- Useful read: what early stopping did not spend, per benchmark. Measured
-- -- counts only; the dollar column is deliberately absent because it does not
-- -- exist as a measurement.
-- SELECT r.benchmark_id,
--        count(*) FILTER (
--          WHERE (r.outcome_metrics -> ''staged_evaluation'' ->> ''stopped_early'')::bool
--        ) AS candidates_stopped_early,
--        sum(COALESCE(
--          (r.outcome_metrics -> ''staged_evaluation'' ->> ''cases_not_run'')::int, 0
--        )) AS workflow_executions_avoided
--   FROM public.benchmark_candidate_results r
--  WHERE r.org_id = '00000000-0000-0000-0000-000000000000'
--    AND r.arm = 'candidate'
--  GROUP BY r.benchmark_id
--  ORDER BY workflow_executions_avoided DESC;
-- ============================================================================
