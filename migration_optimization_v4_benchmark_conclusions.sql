-- ============================================================================
-- Migration: OptiML optimization layer v4 — explicit benchmark conclusions.
--
-- RUN AFTER: migration_optimization.sql
--            migration_optimization_work_graph.sql
--            migration_optimization_v3_outcomes.sql
-- Idempotent. ADDITIVE ONLY.
--
-- ── WHY ─────────────────────────────────────────────────────────────────────
-- A benchmark's verdict was previously INFERRED from the absence of a verified
-- recommendation. That is the exact dishonesty this product exists to avoid:
-- "we found nothing" and "we could not tell" are opposite epistemic states, and
-- inferring the first from the second would let the UI present ignorance as a
-- clean bill of health.
--
-- From here a benchmark records an EXPLICIT conclusion, plus the structured
-- reasoning behind it, plus whether more data would change it.
--
-- ── EVIDENCE IS NOT A DECISION ──────────────────────────────────────────────
-- A benchmark may complete WITHOUT producing a recommendation. It may be run
-- exploratorily against a workload with no recommendation in existence, so
-- recommendation_id is NULLABLE and workload_id is the required linkage.
-- The benchmark conclusion is the EVIDENCE; optimization_recommendations.status
-- is the DECISION. They are never the same field and never conflated.
--
-- Conclusion -> lifecycle mapping, applied ONLY when a benchmark is attached to
-- a recommendation (see optimization/benchmark.py CONCLUSION_TO_STATUS):
--
--   safe_improvement_found  -> 'verified'
--   candidates_failed_policy-> 'rejected'      (failing constraints recorded)
--   no_material_improvement -> 'rejected'      (see justification below)
--   insufficient_evidence   -> 'inconclusive'
--   benchmark_failed        -> 'failed'
--
-- Why 'no_material_improvement' maps to 'rejected' and not 'inconclusive':
-- it is a KNOWLEDGE state, not an ignorance state. Reaching it requires that
-- the sample cleared the floor, the candidates satisfied policy, and the
-- measured improvement was genuinely below the policy's materiality threshold.
-- We know the answer: not worth changing. 'inconclusive' is reserved strictly
-- for cases where we could not tell. Collapsing the two would reintroduce the
-- ambiguity this migration exists to remove.
-- ============================================================================


-- ============================================================================
-- 1. optimization_benchmarks — explicit conclusion + structured reasoning.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    ADD COLUMN IF NOT EXISTS conclusion                  TEXT,
    ADD COLUMN IF NOT EXISTS conclusion_detail           JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS more_data_changes_conclusion TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS more_data_reason            TEXT,
    ADD COLUMN IF NOT EXISTS materiality_threshold       JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS policy_id                   UUID REFERENCES public.optimization_policies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS confidence                  NUMERIC(4, 3);

ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_conclusion_check;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_conclusion_check
    CHECK (conclusion IS NULL OR conclusion IN (
        'safe_improvement_found',
        'no_material_improvement',
        'candidates_failed_policy',
        'insufficient_evidence',
        'benchmark_failed'));

ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_more_data_check;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_more_data_check
    CHECK (more_data_changes_conclusion IN ('yes', 'no', 'unknown'));

ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_confidence_check;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_confidence_check
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- A completed benchmark MUST state a conclusion. Silence is not a verdict.
ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_completed_needs_conclusion;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_completed_needs_conclusion
    CHECK (status <> 'completed' OR conclusion IS NOT NULL) NOT VALID;

COMMENT ON COLUMN public.optimization_benchmarks.conclusion IS
  'The EXPLICIT verdict of this run. Never inferred from absence.
   safe_improvement_found  = a candidate satisfies policy AND materially improves the objective.
   no_material_improvement = candidates satisfied policy but none improved enough to justify a change. A KNOWLEDGE state.
   candidates_failed_policy= alternatives were tested with sufficient evidence but violated one or more required constraints.
   insufficient_evidence   = we cannot conclude either way (sample size, outcome-signal strength, comparability or coverage inadequate). An IGNORANCE state.
   benchmark_failed        = a technical failure prevented a valid comparison.
   *** insufficient_evidence IS NOT EVIDENCE THAT THE CURRENT CONFIGURATION IS OPTIMAL. *** It must never be counted in any "workloads with no opportunity" aggregate; it belongs in a separate "not yet assessable" bucket.';

COMMENT ON COLUMN public.optimization_benchmarks.conclusion_detail IS
  'Structured reasoning, not prose: {"candidates":[{"fingerprint":...,"label":...,"metrics":{...},"eligible":bool,"violations":[{"constraint":"min_quality","required":0.95,"measured":0.91,"shortfall":0.04}]}], "evidence_source":"replay", "sample_size":42, "objective":"cost", "improvement":{"relative":0.31,"absolute_usd_per_task":0.0004,"projected_monthly_usd":12.40}, "materiality":{"applied":{...},"met":false}, "confidence":0.42, "unenforced_constraints":["data_region"]}. Every number here is measured or explicitly null.';

COMMENT ON COLUMN public.optimization_benchmarks.more_data_changes_conclusion IS
  'Would more data materially change this conclusion? ''yes'' = the verdict is provisional and gathering more evidence is the right next action. ''no'' = the verdict is stable; more of the same data will not move it. ''unknown'' = we cannot even tell that. This is a first-class column, not a note, because it is exactly what separates "we looked and it is fine" from "we could not tell yet".';

COMMENT ON COLUMN public.optimization_benchmarks.more_data_reason IS
  'Why more_data_changes_conclusion has that value, e.g. "sample of 24 gives a cost delta CI that straddles the 5% materiality threshold" or "quality signal is llm_judge only; a deterministic signal would change the verdict".';

COMMENT ON COLUMN public.optimization_benchmarks.materiality_threshold IS
  'The materiality threshold ACTUALLY APPLIED to this run, snapshotted: {"source":"policy|default","policy_id":...,"min_relative_improvement":0.05,"min_absolute_monthly_savings_usd":5.0,"require":"any|all"}. Snapshotted so the same evidence cannot silently yield a different conclusion after a threshold change in a later release.';

COMMENT ON COLUMN public.optimization_benchmarks.confidence IS
  'Confidence in this run''s conclusion, 0..1, from optimization.domain.compute_confidence. NULL when it cannot be computed — never a placeholder.';

CREATE INDEX IF NOT EXISTS idx_optbench_org_conclusion
    ON public.optimization_benchmarks (org_id, conclusion, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optbench_workload_conclusion
    ON public.optimization_benchmarks (workload_id, conclusion, created_at DESC);


-- ============================================================================
-- 2. Linkage correction: a benchmark belongs to a WORKLOAD; a recommendation
--    is optional. An exploratory benchmark has no recommendation.
-- ============================================================================
COMMENT ON COLUMN public.optimization_benchmarks.recommendation_id IS
  'NULLABLE BY DESIGN. A benchmark may be run exploratorily against a workload with no recommendation in existence, and completing a benchmark NEVER auto-creates one — only conclusion=''safe_improvement_found'' may create or advance a recommendation. The required linkage is workload_id.';

ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_workload_id_fkey;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_workload_id_fkey
    FOREIGN KEY (workload_id) REFERENCES public.workloads(id) ON DELETE CASCADE;

-- NOT VALID: enforced for every new row without failing on any pre-existing
-- exploratory row written before this migration.
ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_workload_required;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_workload_required
    CHECK (workload_id IS NOT NULL) NOT VALID;

COMMENT ON COLUMN public.optimization_benchmarks.workload_id IS
  'The workload this evidence is about. REQUIRED for all new rows (enforced by optimization_benchmarks_workload_required). A benchmark is always about a piece of work; it is not always about a recommendation.';


-- ============================================================================
-- 3. optimization_policies — materiality is policy-owned, never hardcoded.
-- ============================================================================
ALTER TABLE public.optimization_policies
    ADD COLUMN IF NOT EXISTS materiality JSONB NOT NULL DEFAULT
    '{"min_relative_improvement": 0.05, "min_absolute_monthly_savings_usd": 5.0, "require": "any"}'::jsonb;

COMMENT ON COLUMN public.optimization_policies.materiality IS
  'What counts as "enough to justify a change" for this workload. Default: a 5% relative improvement in the objective OR at least $5/month of projected absolute savings ("require":"any"). Rationale for the default: below ~5% the measured delta is commonly inside replay noise at realistic sample sizes, and a change that saves less than a few dollars a month does not repay the risk and review cost of touching production. Set "require":"all" to demand both. The threshold actually applied is snapshotted onto optimization_benchmarks.materiality_threshold so a later default change cannot silently rewrite an old verdict.';


-- ============================================================================
-- Verify
-- ============================================================================
-- SELECT conclusion, more_data_changes_conclusion, count(*)
--   FROM public.optimization_benchmarks
--  WHERE org_id = '00000000-0000-0000-0000-000000000000'
--  GROUP BY 1, 2;
--
-- "Not yet assessable" must never be reported as "no opportunity":
-- SELECT count(*) FILTER (WHERE conclusion = 'no_material_improvement') AS no_opportunity,
--        count(*) FILTER (WHERE conclusion = 'insufficient_evidence')   AS not_yet_assessable
--   FROM public.optimization_benchmarks
--  WHERE org_id = '00000000-0000-0000-0000-000000000000';
