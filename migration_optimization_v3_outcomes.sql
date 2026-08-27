-- ============================================================================
-- Migration: OptiML optimization layer v3 — outcomes, workload identity levels,
--            and the Direct Inference surface.
--
-- RUN AFTER: migration_enable_rls.sql
--            migration_optimization.sql
--            migration_optimization_work_graph.sql
-- Idempotent. ADDITIVE ONLY — no drops of columns or data.
-- Paste into the Supabase SQL Editor. (Do NOT use run_migration.py.)
--
-- ── WHAT CHANGED AND WHY ────────────────────────────────────────────────────
--
-- 1. SIGNAL PRECEDENCE IS A DEFAULT, NOT A LAW.  v2 shipped
--    public.outcome_provenance_rank() as a single global hierarchy. That is
--    wrong as a rule: for one workload JSON-schema validity genuinely IS the
--    hard requirement; for another it is conversion rate. The ranking is
--    retained as the DEFAULT ordering used when no policy says otherwise. The
--    authority now lives in optimization_policies.success_signal, per workload.
--
-- 2. OUTCOMES ARE NAMED AND PLURAL.  outcome_type is now an OPEN vocabulary,
--    not an enum. One support response accumulates 'thumbs_up',
--    'ticket_resolved', 'escalation', 'reopened_7d' over days. Collapsing that
--    into one quality_score destroys the information the optimizer needs.
--
-- 3. OUTCOMES ARE CORRECTABLE.  Business data gets revised. A correction is a
--    NEW row that supersedes the old one; the old row is retained and marked,
--    never overwritten — savings math may already have consumed the old value
--    and must be able to see what it consumed.
--
-- 4. WORKLOAD IDENTITY HAS THREE LEVELS: explicit (the customer names it),
--    structural (endpoint / workflow / template / model identifies it), and
--    learned (OptiML discovers repeated clusters). Customers must NOT be
--    required to name a workload, but must be able to. explicit + structural
--    are implemented; learned is a documented extension point.
--
-- 5. DIRECT INFERENCE is a first-class surface. A customer changing one
--    base_url and sending production traffic through OptiML has no Studio
--    workflow and no deployment, and must still feed the same
--    Workload -> Strategy -> Attempt -> Cost -> Outcome architecture.
-- ============================================================================


-- ============================================================================
-- 1. Provenance ranking: re-document as a DEFAULT, not law.
--    The function body is unchanged so the generated column stays valid.
-- ============================================================================
COMMENT ON FUNCTION public.outcome_provenance_rank(TEXT) IS
  'DEFAULT signal-quality ordering (80 strongest .. 10 weakest), used ONLY when an optimization_policies.success_signal does not name the deciding signal for the workload. It is a sensible default, NOT a law: for one workload schema validity is the hard requirement, for another it is conversion rate. Never use this ranking to override an explicit policy, and never average values across different provenances regardless of rank.';

COMMENT ON COLUMN public.outcomes.provenance_rank IS
  'Generated DEFAULT rank from public.outcome_provenance_rank(provenance). Use for ordering and for grouping incompatible signals apart. It does NOT decide success — optimization_policies.success_signal does. See also signal_strength for a caller-supplied override.';


-- ============================================================================
-- 2. outcomes — open vocabulary, explicit signal strength, correctable.
-- ============================================================================

-- 2a. outcome_type becomes an OPEN vocabulary.
ALTER TABLE public.outcomes DROP CONSTRAINT IF EXISTS outcomes_outcome_type_check;
ALTER TABLE public.outcomes ADD CONSTRAINT outcomes_outcome_type_check
    CHECK (outcome_type IS NOT NULL AND length(btrim(outcome_type)) BETWEEN 1 AND 120);

ALTER TABLE public.outcomes
    ADD COLUMN IF NOT EXISTS outcome_category   TEXT,
    ADD COLUMN IF NOT EXISTS signal_strength    NUMERIC(4, 3),
    -- Correction / revision chain
    ADD COLUMN IF NOT EXISTS revision           INT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS is_current         BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS supersedes_outcome_id    UUID REFERENCES public.outcomes(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS superseded_by_outcome_id UUID REFERENCES public.outcomes(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS superseded_at      TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS correction_reason  TEXT;

ALTER TABLE public.outcomes DROP CONSTRAINT IF EXISTS outcomes_outcome_category_check;
ALTER TABLE public.outcomes ADD CONSTRAINT outcomes_outcome_category_check
    CHECK (outcome_category IS NULL OR outcome_category IN (
        'task_success', 'quality_score', 'user_feedback', 'business_value',
        'error', 'escalation', 'human_intervention', 'custom'));

ALTER TABLE public.outcomes DROP CONSTRAINT IF EXISTS outcomes_signal_strength_check;
ALTER TABLE public.outcomes ADD CONSTRAINT outcomes_signal_strength_check
    CHECK (signal_strength IS NULL OR (signal_strength >= 0 AND signal_strength <= 1));

CREATE INDEX IF NOT EXISTS idx_outcomes_type_current
    ON public.outcomes (org_id, outcome_type, occurred_at DESC) WHERE is_current;
CREATE INDEX IF NOT EXISTS idx_outcomes_attempt_current
    ON public.outcomes (attempt_ref, outcome_type) WHERE is_current AND attempt_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_outcomes_supersedes
    ON public.outcomes (supersedes_outcome_id) WHERE supersedes_outcome_id IS NOT NULL;

COMMENT ON COLUMN public.outcomes.outcome_type IS
  'OPEN vocabulary, named per workload: ''thumbs_up'', ''ticket_resolved'', ''escalation'', ''reopened_7d'', ''pr_merged'', ''pr_reverted''. MANY outcome_types may attach to one attempt and they arrive at different times. This is deliberately NOT an enum and deliberately NOT collapsed into a single quality score: the distinct names are what let a policy say which one decides success for THIS workload.';
COMMENT ON COLUMN public.outcomes.outcome_category IS
  'Optional coarse bucket for a named outcome_type, for generic roll-ups and UI grouping only. It never overrides outcome_type and never decides success.';
COMMENT ON COLUMN public.outcomes.signal_strength IS
  'Optional caller-supplied 0..1 strength for THIS specific signal, when the reporter knows better than the default provenance rank (e.g. a verified refund event vs a heuristically-inferred one, both provenance=''business_outcome''). NULL means fall back to provenance_rank. Never fabricate a value to make a signal look stronger.';
COMMENT ON COLUMN public.outcomes.is_current IS
  'FALSE once a correction supersedes this row. Analyses read is_current rows; audits read the whole chain. Corrections NEVER overwrite in place, because savings math may already have consumed the superseded value.';
COMMENT ON COLUMN public.outcomes.revision IS
  'Revision number within a correction chain. The original insert is 1; each correction increments.';
COMMENT ON COLUMN public.outcomes.supersedes_outcome_id IS
  'The earlier outcome row this correction replaces. Together with superseded_by_outcome_id / superseded_at / correction_reason this is the revision audit trail.';


-- ============================================================================
-- 3. workloads — three identity levels + Direct Inference surface.
-- ============================================================================
ALTER TABLE public.workloads
    ADD COLUMN IF NOT EXISTS identity_level TEXT NOT NULL DEFAULT 'structural',
    ADD COLUMN IF NOT EXISTS external_key   TEXT;

ALTER TABLE public.workloads DROP CONSTRAINT IF EXISTS workloads_identity_level_check;
ALTER TABLE public.workloads ADD CONSTRAINT workloads_identity_level_check
    CHECK (identity_level IN ('explicit', 'structural', 'learned'));

-- Widen surface for Direct Inference (POST /v1/chat/completions).
ALTER TABLE public.workloads DROP CONSTRAINT IF EXISTS workloads_surface_check;
ALTER TABLE public.workloads ADD CONSTRAINT workloads_surface_check
    CHECK (surface IN ('runtime', 'direct_inference', 'workforce'));

-- Widen identity_kind: 'explicit' (customer-named) and 'model_endpoint'
-- (direct inference target). 'inferred' remains reserved for learned identity.
ALTER TABLE public.workloads DROP CONSTRAINT IF EXISTS workloads_identity_kind_check;
ALTER TABLE public.workloads ADD CONSTRAINT workloads_identity_kind_check
    CHECK (identity_kind IS NULL OR identity_kind IN (
        'explicit', 'endpoint', 'workflow', 'prompt_template',
        'model_endpoint', 'manual', 'inferred'));

-- A customer-supplied workload name is unique within an org, across surfaces:
-- 'support-refund' means one thing to the customer regardless of how it runs.
CREATE UNIQUE INDEX IF NOT EXISTS idx_workloads_external_key_unique
    ON public.workloads (org_id, external_key) WHERE external_key IS NOT NULL;

COMMENT ON COLUMN public.workloads.identity_level IS
  'How this workload came to be identified. ''explicit'' = the customer named it on the request (external_key). ''structural'' = derived from endpoint / workflow / template / model target. ''learned'' = discovered by clustering repeated semantically-similar work; RESERVED, not produced today. Customers are never REQUIRED to name a workload — structural discovery works without one — but naming one always wins.';
COMMENT ON COLUMN public.workloads.external_key IS
  'The customer''s own name for this workload, e.g. ''support-refund'', sent on the request (X-OptiML-Workload header or a workload field). Stable across surfaces: the same key may cover Studio workflow traffic and Direct Inference traffic.';
COMMENT ON COLUMN public.workloads.surface IS
  'runtime = OptiML-executed Studio workflows/endpoints. direct_inference = traffic sent through OptiML by changing base_url (POST /v1/chat/completions); there is NO workflow and NO deployment behind it. workforce = agentic/workforce surface.';


-- ============================================================================
-- 4. optimization_policies — the policy, not a global constant, decides what
--    "success" means for a workload.
-- ============================================================================
ALTER TABLE public.optimization_policies
    ADD COLUMN IF NOT EXISTS success_signal JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.optimization_policies.success_signal IS
  'Which measured signal decides whether a strategy is acceptable FOR THIS WORKLOAD. Shape: {"outcome_type":"ticket_resolved","provenance":"business_outcome","aggregate":"rate|mean","direction":"higher_is_better|lower_is_better","min_value":0.82,"min_sample":50,"fallback_outcome_types":["thumbs_up"]}. When empty, the DEFAULT provenance ordering (public.outcome_provenance_rank) picks the strongest available signal. A policy naming ''schema_valid'' as the deciding signal OVERRIDES that default — for some workloads schema validity really is the hard requirement, and for others it is conversion rate. Signals of different provenance are never averaged together.';


-- ============================================================================
-- 5. optimization_recommendations — snapshot which signal judged this candidate.
-- ============================================================================
ALTER TABLE public.optimization_recommendations
    ADD COLUMN IF NOT EXISTS success_signal JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.optimization_recommendations.success_signal IS
  'Immutable snapshot of the success_signal that was in force when this recommendation was judged, resolved from optimization_policies (or the default ordering when no policy applied): {"resolved_from":"policy|default","outcome_type":...,"provenance":...,"n":...}. Snapshotted so a later policy change cannot silently rewrite what an old verdict meant.';


-- ============================================================================
-- 6. optimization_benchmarks — same snapshot on the evidence itself.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    ADD COLUMN IF NOT EXISTS success_signal JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.optimization_benchmarks.success_signal IS
  'The success signal this run was judged against, snapshotted at run time. A benchmark judged on schema validity must never be reinterpreted later as evidence about conversion rate.';


-- ============================================================================
-- 7. attempts view — carry project_id (derived through workflow -> project) and
--    keep Direct Inference able to join in later without a rewrite.
-- ============================================================================
-- NOTE: CREATE OR REPLACE VIEW cannot change a view's column names/types or
-- insert a column mid-list. This definition does both relative to the previous
-- migration, so the view MUST be dropped first. Verified against production.
DROP VIEW IF EXISTS public.attempts;
CREATE VIEW public.attempts AS
SELECT
    wr.id                                   AS attempt_id,
    'workflow_run'::TEXT                    AS attempt_source,
    wr.org_id                               AS org_id,
    COALESCE(wl_explicit.id, wl_endpoint.id) AS workload_id,
    'runtime'::TEXT                         AS surface,
    wf.project_id                           AS project_id,
    wr.workflow_id                          AS workflow_id,
    wr.endpoint_slug                        AS endpoint_slug,
    wr.served_version                       AS served_version,
    wr.execution_mode                       AS execution_mode,
    wr.experiment_id                        AS experiment_id,
    wr.variant_name                         AS variant_name,
    wr.total_cost                           AS inference_cost_usd,
    wr.total_latency_ms                     AS duration_ms,
    wr.node_results                         AS step_results,
    wr.created_at                           AS occurred_at
FROM public.workflow_runs wr
LEFT JOIN public.workflows wf
       ON wf.id = wr.workflow_id
LEFT JOIN public.workloads wl_endpoint
       ON wl_endpoint.org_id        = wr.org_id
      AND wl_endpoint.surface       = 'runtime'
      AND wl_endpoint.identity_kind = 'endpoint'
      AND wl_endpoint.identity_ref  = wr.endpoint_slug
LEFT JOIN public.workloads wl_explicit
       ON wl_explicit.org_id        = wr.org_id
      AND wl_explicit.identity_kind = 'workflow'
      AND wl_explicit.identity_ref  = wr.workflow_id::TEXT;

ALTER VIEW public.attempts SET (security_invoker = on);
REVOKE ALL ON public.attempts FROM PUBLIC;
REVOKE ALL ON public.attempts FROM anon;
GRANT SELECT ON public.attempts TO authenticated;
GRANT SELECT ON public.attempts TO service_role;

COMMENT ON VIEW public.attempts IS
  'Domain view of one execution of work, derived from workflow_runs — the existing tracing record, which is NOT duplicated or renamed. project_id is derived through workflows. An explicitly-named workload wins over structural endpoint matching. When Direct Inference lands, UNION ALL a second branch here; no consumer changes. IMPORTANT: node_results (step_results) must be parsed ONLY by optimization/attempts.py — see that module for why.';


-- ============================================================================
-- Verify
-- ============================================================================
-- SELECT column_name, data_type FROM information_schema.columns
--  WHERE table_schema='public' AND table_name='outcomes' ORDER BY ordinal_position;
--
-- Correction chain audit — what did we believe, and what do we believe now?
-- SELECT id, outcome_type, outcome_value, revision, is_current,
--        supersedes_outcome_id, correction_reason, recorded_at
--   FROM public.outcomes
--  WHERE org_id = '00000000-0000-0000-0000-000000000000'
--    AND attempt_ref = '00000000-0000-0000-0000-000000000000'
--  ORDER BY outcome_type, revision;
