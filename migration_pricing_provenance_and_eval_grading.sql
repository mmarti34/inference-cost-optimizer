-- Migration: pricing provenance + LLM-judge eval results
--
-- Two independent additions, both idempotent and safe to re-run:
--
-- 1. usage_logs.pricing_source / pricing_estimated
--    Costs are computed from shared/providers.json. When a model is not in that
--    file, utils/pricing.get_pricing() falls back to a generic default
--    ($0.001/$0.002 per 1K) — a GUESS. Until now nothing recorded which rows
--    were guessed, so savings math silently averaged real prices with invented
--    ones. pricing_estimated = TRUE marks a row whose cost_usd is not grounded
--    in a known list price; exclude or flag those rows in any savings claim.
--
-- 2. eval_run_results LLM-judge columns
--    The model_graded eval check now actually runs a judge (auto_grading.
--    grade_output_sync). These columns record the score, the judge's reasoning,
--    WHICH provider/model graded it and what the grading cost, plus the
--    provenance of the quality signal.
--
--    quality_provenance uses the same vocabulary as optimization.outcomes so the
--    two never get averaged together. For an LLM-judged eval check it is always
--    'llm_judge'. "Passed an LLM judge" is a far weaker claim than a measured
--    business outcome and must never be recorded as one.
--
-- Both blocks no-op if the target table does not exist in this database.

-- ---------------------------------------------------------------------------
-- 1. usage_logs: pricing provenance
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF to_regclass('public.usage_logs') IS NULL THEN
    RAISE NOTICE 'usage_logs does not exist; skipping pricing provenance columns.';
    RETURN;
  END IF;

  ALTER TABLE public.usage_logs
    ADD COLUMN IF NOT EXISTS pricing_source TEXT;
  ALTER TABLE public.usage_logs
    ADD COLUMN IF NOT EXISTS pricing_estimated BOOLEAN NOT NULL DEFAULT FALSE;

  ALTER TABLE public.usage_logs DROP CONSTRAINT IF EXISTS usage_logs_pricing_source_check;
  ALTER TABLE public.usage_logs ADD CONSTRAINT usage_logs_pricing_source_check
    CHECK (pricing_source IS NULL OR pricing_source IN ('exact', 'alias', 'prefix', 'default'));
END $$;

-- Rows whose cost is a guess are the ones analysts need to find fastest.
CREATE INDEX IF NOT EXISTS idx_usage_logs_pricing_estimated
  ON public.usage_logs (org_id, created_at DESC)
  WHERE pricing_estimated;

DO $$
BEGIN
  IF to_regclass('public.usage_logs') IS NULL THEN RETURN; END IF;
  EXECUTE $c$
    COMMENT ON COLUMN public.usage_logs.pricing_source IS
      'How the per-token price behind cost_usd was resolved in shared/providers.json: '
      '''exact'' = model id matched verbatim; ''alias'' = matched a declared short name; '
      '''prefix'' = matched a dated/variant form of a listed id; ''default'' = NO price was '
      'found and a generic fallback was used.'
  $c$;
  EXECUTE $c$
    COMMENT ON COLUMN public.usage_logs.pricing_estimated IS
      'TRUE when cost_usd was computed from the generic fallback price rather than a known '
      'list price. Such rows are guesses: exclude or flag them in savings math, never present '
      'them as measured spend. Non-zero counts mean a model is missing from shared/providers.json '
      '(see GET /api/observability/pricing-misses).'
  $c$;
END $$;

-- ---------------------------------------------------------------------------
-- 2. eval_run_results: LLM-judge grading columns
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF to_regclass('public.eval_run_results') IS NULL THEN
    RAISE NOTICE 'eval_run_results does not exist; skipping LLM-judge columns.';
    RETURN;
  END IF;

  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS score              DOUBLE PRECISION;
  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS grade_reasoning    TEXT;
  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS grader_provider    TEXT;
  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS grader_model       TEXT;
  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS grading_cost_usd   NUMERIC(12, 8);
  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS grading_latency_ms INTEGER;
  ALTER TABLE public.eval_run_results
    ADD COLUMN IF NOT EXISTS quality_provenance TEXT;

  ALTER TABLE public.eval_run_results
    DROP CONSTRAINT IF EXISTS eval_run_results_quality_provenance_check;
  ALTER TABLE public.eval_run_results
    ADD CONSTRAINT eval_run_results_quality_provenance_check
    CHECK (quality_provenance IS NULL OR quality_provenance IN (
      'business_outcome', 'deterministic', 'human', 'user_feedback',
      'automated_test', 'schema', 'llm_judge', 'implicit', 'heuristic', 'unknown'
    ));
END $$;

CREATE INDEX IF NOT EXISTS idx_eval_run_results_provenance
  ON public.eval_run_results (eval_run_id, quality_provenance);

DO $$
BEGIN
  IF to_regclass('public.eval_run_results') IS NULL THEN RETURN; END IF;
  EXECUTE $c$
    COMMENT ON COLUMN public.eval_run_results.quality_provenance IS
      'Where this row''s quality signal came from, using the same vocabulary as '
      'public.outcomes.provenance. model_graded checks are always ''llm_judge'': an LLM''s '
      'opinion, strictly weaker than a measured business outcome and weaker than a '
      'deterministic check. NEVER average across provenance tiers, and never report an '
      'llm_judge score as an outcome.'
  $c$;
  EXECUTE $c$
    COMMENT ON COLUMN public.eval_run_results.grader_provider IS
      'Provider that ran the LLM judge for a model_graded check (openai | anthropic | gemini).'
  $c$;
  EXECUTE $c$
    COMMENT ON COLUMN public.eval_run_results.grader_model IS
      'Exact model id that produced the grade. Grades from different judges are not comparable.'
  $c$;
  EXECUTE $c$
    COMMENT ON COLUMN public.eval_run_results.grading_cost_usd IS
      'USD spent calling the judge for this single check. Grading spend is an OptiML overhead '
      'cost, not part of the workflow''s own cost — do not fold it into candidate_cost.'
  $c$;
END $$;
