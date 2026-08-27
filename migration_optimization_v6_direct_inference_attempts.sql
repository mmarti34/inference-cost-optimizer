-- ============================================================================
-- Migration: OptiML optimization layer v6 — Direct Inference attempts.
--
-- RUN AFTER: migration_optimization.sql
--            migration_optimization_work_graph.sql
--            migration_optimization_v3_outcomes.sql
--            migration_optimization_v4_benchmark_conclusions.sql
--            migration_optimization_v5_evidence_model.sql
-- Idempotent. ADDITIVE ONLY — no columns or data are dropped.
-- Paste into the Supabase SQL Editor. (Do NOT use run_migration.py.)
--
-- ── THE GAP ─────────────────────────────────────────────────────────────────
-- `public.attempts` was a view over `workflow_runs` only. A Direct Inference
-- request (POST /v1/chat/completions, one changed base_url) has NO workflow run
-- and NO deployment, so it produced no attempt row at all: it was identified,
-- attributed to a workload and a strategy, and then dropped. The product's
-- primary onboarding path did not feed the Work Graph.
--
-- ── THE FIX, AND WHAT IT DELIBERATELY DOES NOT DO ───────────────────────────
-- `api_request_log` ALREADY IS the execution record for direct inference. It
-- carries org, status, latency, measured cost, and — in custom_metrics —
-- provider, model, token counts, cost provenance and the workload identity ref.
-- So this migration adds a UNION branch over it. It does NOT copy any of those
-- columns into a second table: there is no second execution table, and no
-- duplication of api_request_log.
--
-- ── WHY A NARROW COMPANION TABLE IS STILL NEEDED ────────────────────────────
-- Exactly two facts cannot be recovered from the log row:
--
--   1. The resolved `workloads.id`. custom_metrics carries the identity REF
--      (a string), so the view can join on it — but that join is a string match
--      and silently breaks if a workload is renamed or its identity kind
--      changes. The link row pins the actual UUID.
--
--   2. The baseline STRATEGY fingerprint. Direct-inference identity is derived
--      partly from the SYSTEM PROMPT, which api_request_log deliberately does
--      not store (it is customer content). The fingerprint is therefore
--      genuinely, permanently unrecoverable from the log row.
--
-- `direct_inference_attempt_links` carries those two facts and nothing else.
-- It has no cost, latency, token or status column, on purpose: any such column
-- would be a second source of truth for a number the product rests on.
--
-- ── ORDERING: THE TWO WRITERS RACE, AND THE VIEW TOLERATES IT ───────────────
-- The router writes api_request_log; the bridge writes the link row. On the
-- streaming path the bridge runs BEFORE the log write; on the non-streaming
-- path both are scheduled concurrently. Neither can wait for the other without
-- adding latency to a customer's production request.
--
-- So the view is driven FROM api_request_log and LEFT JOINs the link:
--   * link missing  -> the attempt still appears, with workload_id resolved by
--                      the identity-ref fallback join and strategy_id NULL.
--   * log missing   -> nothing appears; an attempt with no execution record is
--                      not an attempt.
-- An execution is never invisible merely because attribution failed.
--
-- ── IDENTIFIERS ─────────────────────────────────────────────────────────────
-- The customer-visible id is `chatcmpl-<24 hex>` (X-OptiML-Request-Id), which is
-- NOT a UUID and therefore cannot be api_request_log.id. The view exposes both:
--   attempt_id           = the execution-record row id, cast to TEXT
--   external_attempt_id  = the customer-visible id, used for outcome attachment
-- optimization/attempts.py::get_attempt resolves either.
-- ============================================================================


-- ============================================================================
-- 1. Vocabulary: 'direct_inference' is a first-class attempt source.
-- ============================================================================
ALTER TABLE public.outcomes DROP CONSTRAINT IF EXISTS outcomes_attempt_source_check;
ALTER TABLE public.outcomes ADD CONSTRAINT outcomes_attempt_source_check
    CHECK (attempt_source IN ('workflow_run', 'api_request', 'direct_inference', 'external', 'none'));

ALTER TABLE public.cost_events DROP CONSTRAINT IF EXISTS cost_events_attempt_source_check;
ALTER TABLE public.cost_events ADD CONSTRAINT cost_events_attempt_source_check
    CHECK (attempt_source IN ('workflow_run', 'api_request', 'direct_inference', 'external', 'none'));

-- attempt_ref on outcomes/cost_events is UUID, but a direct-inference attempt is
-- addressed by its `chatcmpl-...` id. Carry it in a TEXT column rather than
-- coercing a non-UUID into a UUID column.
ALTER TABLE public.outcomes
    ADD COLUMN IF NOT EXISTS external_attempt_ref TEXT;
ALTER TABLE public.cost_events
    ADD COLUMN IF NOT EXISTS external_attempt_ref TEXT;

CREATE INDEX IF NOT EXISTS idx_outcomes_external_attempt
    ON public.outcomes (org_id, external_attempt_ref)
    WHERE external_attempt_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cost_events_external_attempt
    ON public.cost_events (org_id, external_attempt_ref)
    WHERE external_attempt_ref IS NOT NULL;

COMMENT ON COLUMN public.outcomes.external_attempt_ref IS
  'The customer-visible attempt id (``chatcmpl-...``, returned as X-OptiML-Request-Id) when attempt_source=''direct_inference''. Used instead of attempt_ref because that id is not a UUID. Exactly one of attempt_ref / external_attempt_ref is set for an attempt-scoped outcome.';
COMMENT ON COLUMN public.cost_events.external_attempt_ref IS
  'The customer-visible attempt id for a direct-inference cost event. See outcomes.external_attempt_ref.';


-- ============================================================================
-- 2. direct_inference_attempt_links — ONLY what the view cannot express.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.direct_inference_attempt_links (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id               UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,

    -- The customer-visible request id. This is the join key to
    -- api_request_log.custom_metrics->>'request_id', and the key a customer
    -- uses to attach an outcome later.
    attempt_id           TEXT NOT NULL,

    workload_id          UUID REFERENCES public.workloads(id) ON DELETE SET NULL,
    strategy_id          UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,
    strategy_fingerprint TEXT,
    executor_id          UUID REFERENCES public.executors(id) ON DELETE SET NULL,

    occurred_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dial_attempt_unique
    ON public.direct_inference_attempt_links (org_id, attempt_id);
CREATE INDEX IF NOT EXISTS idx_dial_workload_occurred
    ON public.direct_inference_attempt_links (workload_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_dial_strategy
    ON public.direct_inference_attempt_links (strategy_id);
CREATE INDEX IF NOT EXISTS idx_dial_org_created
    ON public.direct_inference_attempt_links (org_id, created_at DESC);

COMMENT ON TABLE public.direct_inference_attempt_links IS
  'Attribution for one direct-inference attempt: which workload it belongs to and which baseline strategy it represents. Deliberately NARROW — it has no cost, latency, token or status column, because api_request_log already holds those and a second copy would be a second source of truth for the numbers the product rests on. It exists only for the two facts the attempts view cannot recover: the resolved workload UUID (custom_metrics holds only a string ref) and the strategy fingerprint (derived partly from the system prompt, which api_request_log deliberately does not store).';
COMMENT ON COLUMN public.direct_inference_attempt_links.attempt_id IS
  'The customer-visible ``chatcmpl-...`` id, echoed as X-OptiML-Request-Id and mirrored in api_request_log.custom_metrics->>''request_id''. Not a UUID, which is why it cannot simply be api_request_log.id.';
COMMENT ON COLUMN public.direct_inference_attempt_links.strategy_id IS
  'The baseline execution_strategies row this request represents — the customer''s own current configuration, which is what a candidate must beat. Deduped on fingerprint, so this is one row per distinct configuration, not one per request.';

ALTER TABLE public.direct_inference_attempt_links ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Org members can view direct inference attempt links"
    ON public.direct_inference_attempt_links;
CREATE POLICY "Org members can view direct inference attempt links"
    ON public.direct_inference_attempt_links FOR SELECT
    USING (public.is_org_member(org_id));


-- ============================================================================
-- 3. Join support on the existing log table (index only; no columns added).
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_api_request_log_request_id
    ON public.api_request_log ((custom_metrics->>'request_id'));
CREATE INDEX IF NOT EXISTS idx_api_request_log_direct
    ON public.api_request_log (org_id, created_at DESC)
    WHERE endpoint_slug LIKE 'direct:%';


-- ============================================================================
-- 4. attempts — UNION the two execution surfaces.
--
-- NOTE ON step_results: the direct branch emits NULL, NOT a synthesized
-- node_results array. Building that shape in SQL would put knowledge of the
-- node_results contract in a second place; instead the branch exposes the raw
-- measured columns (provider, model, tokens, cost, success) and
-- optimization/attempts.py::facts_from_direct_row constructs the AttemptFacts.
-- All shape knowledge stays in the one module that owns it.
-- ============================================================================
-- NOTE: CREATE OR REPLACE VIEW cannot change a view's column names/types or
-- insert a column mid-list. This definition does both relative to the previous
-- migration, so the view MUST be dropped first. Verified against production.
DROP VIEW IF EXISTS public.attempts;
CREATE VIEW public.attempts AS

-- ── Runtime: Studio workflows and deployed endpoints ────────────────────────
SELECT
    wr.id::TEXT                             AS attempt_id,
    NULL::TEXT                              AS external_attempt_id,
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
    NULL::NUMERIC                           AS estimated_cost_usd,
    FALSE                                   AS cost_is_estimated,
    wr.total_latency_ms                     AS duration_ms,
    wr.node_results                         AS step_results,
    -- Executor and token facts live INSIDE node_results on this surface, so
    -- these are NULL here and the parser fills them in.
    NULL::TEXT                              AS provider,
    NULL::TEXT                              AS model,
    NULL::INT                               AS prompt_tokens,
    NULL::INT                               AS completion_tokens,
    NULL::BOOLEAN                           AS success,
    NULL::UUID                              AS strategy_id,
    NULL::TEXT                              AS strategy_fingerprint,
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
      AND wl_explicit.identity_ref  = wr.workflow_id::TEXT

UNION ALL

-- ── Direct Inference: POST /v1/chat/completions, no workflow, no deployment ──
SELECT
    arl.id::TEXT                            AS attempt_id,
    COALESCE(lnk.attempt_id, arl.custom_metrics->>'request_id') AS external_attempt_id,
    'direct_inference'::TEXT                AS attempt_source,
    arl.org_id                              AS org_id,
    -- The pinned UUID wins; the identity-ref join is the fallback for when the
    -- attribution write lost the race or failed.
    COALESCE(lnk.workload_id, wl_direct.id) AS workload_id,
    'direct_inference'::TEXT                AS surface,
    NULL::UUID                              AS project_id,
    NULL::UUID                              AS workflow_id,
    arl.endpoint_slug                       AS endpoint_slug,
    NULL::INT                               AS served_version,
    'production'::TEXT                      AS execution_mode,
    arl.experiment_id                       AS experiment_id,
    arl.variant_name                        AS variant_name,
    -- total_cost carries MEASURED spend only; it is NULL when pricing was
    -- estimated, and the estimate is kept separately and clearly labelled.
    arl.total_cost                          AS inference_cost_usd,
    NULLIF(arl.custom_metrics->>'estimated_cost_usd', '')::NUMERIC AS estimated_cost_usd,
    COALESCE((arl.custom_metrics->>'cost_estimated')::BOOLEAN, FALSE) AS cost_is_estimated,
    arl.total_latency_ms                    AS duration_ms,
    -- Deliberately NULL: see the note above section 4.
    NULL::JSONB                             AS step_results,
    arl.custom_metrics->>'provider'         AS provider,
    arl.custom_metrics->>'model'            AS model,
    NULLIF(arl.custom_metrics->>'prompt_tokens', '')::INT     AS prompt_tokens,
    NULLIF(arl.custom_metrics->>'completion_tokens', '')::INT AS completion_tokens,
    arl.success                             AS success,
    lnk.strategy_id                         AS strategy_id,
    lnk.strategy_fingerprint                AS strategy_fingerprint,
    arl.created_at                          AS occurred_at
FROM public.api_request_log arl
LEFT JOIN public.direct_inference_attempt_links lnk
       ON lnk.org_id     = arl.org_id
      AND lnk.attempt_id = arl.custom_metrics->>'request_id'
LEFT JOIN public.workloads wl_direct
       ON wl_direct.org_id       = arl.org_id
      AND wl_direct.surface      = 'direct_inference'
      AND wl_direct.identity_ref = arl.custom_metrics->>'workload_ref'
WHERE arl.endpoint_slug LIKE 'direct:%';

ALTER VIEW public.attempts SET (security_invoker = on);
REVOKE ALL ON public.attempts FROM PUBLIC;
REVOKE ALL ON public.attempts FROM anon;
GRANT SELECT ON public.attempts TO authenticated;
GRANT SELECT ON public.attempts TO service_role;

COMMENT ON VIEW public.attempts IS
  'Domain view of one execution of work, across BOTH execution surfaces. Derived from the existing records — workflow_runs for runtime, api_request_log for direct inference — neither of which is duplicated or renamed. Direct-inference rows are attributed via direct_inference_attempt_links (pinned workload + strategy), falling back to an identity-ref join when that attribution write lost its race, so an execution is never invisible merely because attribution failed. IMPORTANT: step_results (node_results) is parsed ONLY by optimization/attempts.py, and the direct branch deliberately emits NULL there rather than synthesizing that shape in SQL.';


-- ============================================================================
-- Verify
-- ============================================================================
-- Both surfaces, one abstraction:
-- SELECT surface, attempt_source, count(*), sum(inference_cost_usd) AS measured_usd,
--        count(*) FILTER (WHERE cost_is_estimated) AS estimated_rows,
--        count(*) FILTER (WHERE workload_id IS NULL) AS unattributed
--   FROM public.attempts
--  WHERE org_id = '00000000-0000-0000-0000-000000000000'
--  GROUP BY 1, 2;
--
-- Attribution health for direct inference (link row present vs fallback join):
-- SELECT count(*) FILTER (WHERE strategy_id IS NOT NULL) AS pinned,
--        count(*) FILTER (WHERE strategy_id IS NULL)     AS fallback_or_missing
--   FROM public.attempts
--  WHERE org_id = '00000000-0000-0000-0000-000000000000'
--    AND attempt_source = 'direct_inference';
