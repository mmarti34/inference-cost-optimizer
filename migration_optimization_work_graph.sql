-- ============================================================================
-- Migration: OptiML Work Graph — additive evolution of the optimization layer
--
-- RUN AFTER: migration_enable_rls.sql, migration_optimization.sql
-- Idempotent. ADDITIVE ONLY — nothing is dropped, renamed or data-losing.
-- Paste into the Supabase SQL Editor. (Do NOT use run_migration.py.)
--
-- ── WHY ─────────────────────────────────────────────────────────────────────
-- migration_optimization.sql modelled a recommendation as "model/prompt config
-- A vs config B". That is too narrow. The foundational object is WORK, not
-- models. This migration generalises the domain so that the same core tables
-- survive when the thing performing the work is an agent, an external tool,
-- deterministic software, a SaaS API — or, eventually, a person.
--
-- We are still only SHIPPING the Runtime wedge. Nothing below requires a new
-- executor type to exist today; it requires that adding one later is an INSERT,
-- not a schema rewrite.
--
-- ── OLD -> NEW CONCEPT MAPPING ──────────────────────────────────────────────
--   old concept                          new home
--   -----------------------------------  ------------------------------------
--   "candidate model + prompt config"    execution_strategies.steps
--                                        (ordered steps, each -> an executor)
--   "the model"                          executors (executor_type='model');
--                                        agents/software/human are peers, not
--                                        special cases
--   optimization_recommendations
--     .baseline_strategy   (JSONB)       KEPT as the immutable snapshot;
--     .candidate_strategy  (JSONB)       now ALSO linked via
--                                        baseline_strategy_id /
--                                        candidate_strategy_id
--   "did it work?" (implicit, boolean)   outcomes (typed, numeric-capable,
--                                        multiple per attempt, delayed arrival,
--                                        ranked provenance)
--   "cost" == token cost                 cost_events (typed cost with unit);
--                                        runtime inference cost still lives in
--                                        workflow_runs and is NOT duplicated
--   "an execution"                       attempts VIEW over workflow_runs.
--                                        The existing tracing tables are the
--                                        record of truth and are NOT renamed,
--                                        rewritten or duplicated.
--   status 'applied'                     status 'promoted' (old value kept in
--                                        the CHECK for compatibility)
--   evidence_source 'historical_analysis' 'observational'  (old value kept)
--   evidence_source 'online_experiment'   'ab_test'        (old value kept)
--
-- ── THE WORK GRAPH ──────────────────────────────────────────────────────────
--   Work -> Strategy -> Executor -> Config -> Context -> Cost -> Outcome ->
--   Quality -> Human Intervention -> Business Value
--
--   That chain IS workloads + execution_strategies + executors + attempts +
--   cost_events + outcomes. It is a DATA architecture: plain relational tables
--   designed so that "for workloads like this, which strategy produced the
--   better economic outcome" is a natural join. It is not a graph database,
--   not a visualisation, and not the retired knowledge-graph/GraphRAG concept.
--
-- ── HARD BOUNDARY: VENDOR METADATA vs EMPIRICAL EVIDENCE ────────────────────
--   executors.cost_model / .capabilities / .policy_metadata are VENDOR CLAIMS
--   (list price, published context window, advertised region). They are NEVER
--   performance evidence.
--   attempts / cost_events / outcomes / optimization_benchmarks are MEASURED on
--   the customer's own workload. Only these may justify status='verified'.
--   Never write a measured number into an executors column, and never read a
--   vendor column as if it were measured.
-- ============================================================================


-- ============================================================================
-- 0. Shared helper: provenance precedence.
--    Strongest -> weakest signal. Used to rank outcomes so that incompatible
--    signals are never blindly averaged: "passed an LLM judge" must remain
--    distinguishable from "produced a 7% higher actual resolution rate".
-- ============================================================================
CREATE OR REPLACE FUNCTION public.outcome_provenance_rank(p TEXT)
RETURNS SMALLINT
LANGUAGE sql
IMMUTABLE
AS $fn$
  SELECT CASE p
    WHEN 'business_outcome' THEN 80::SMALLINT  -- explicit business result
    WHEN 'deterministic'    THEN 70::SMALLINT  -- exact/verifiable check
    WHEN 'human'            THEN 60::SMALLINT  -- human evaluation
    WHEN 'user_feedback'    THEN 50::SMALLINT  -- end-user signal
    WHEN 'automated_test'   THEN 40::SMALLINT  -- test suite
    WHEN 'schema'           THEN 40::SMALLINT  -- structural/format check
    WHEN 'llm_judge'        THEN 30::SMALLINT  -- model-graded
    WHEN 'implicit'         THEN 20::SMALLINT  -- inferred from behaviour
    WHEN 'heuristic'        THEN 20::SMALLINT
    ELSE 10::SMALLINT                          -- 'unknown'
  END;
$fn$;

COMMENT ON FUNCTION public.outcome_provenance_rank(TEXT) IS
  'Signal-quality precedence for outcome provenance, strongest (80) to weakest (10). Stored as a generated column on outcomes so application code cannot inflate the rank of a weak signal.';


-- ============================================================================
-- 1. workloads — generalise. A Workload is a piece or CLASS of work with an
--    intended outcome. It does NOT assume an LLM performs it.
-- ============================================================================
ALTER TABLE public.workloads
    ADD COLUMN IF NOT EXISTS intended_outcome   TEXT,
    ADD COLUMN IF NOT EXISTS grain              TEXT NOT NULL DEFAULT 'task_class',
    ADD COLUMN IF NOT EXISTS default_objective  TEXT NOT NULL DEFAULT 'cost',
    ADD COLUMN IF NOT EXISTS metadata           JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.workloads DROP CONSTRAINT IF EXISTS workloads_grain_check;
ALTER TABLE public.workloads ADD CONSTRAINT workloads_grain_check
    CHECK (grain IN ('invocation', 'task_class', 'workflow', 'endpoint', 'cluster'));

ALTER TABLE public.workloads DROP CONSTRAINT IF EXISTS workloads_default_objective_check;
ALTER TABLE public.workloads ADD CONSTRAINT workloads_default_objective_check
    CHECK (default_objective IN ('cost', 'quality', 'latency', 'balanced', 'custom'));

COMMENT ON TABLE public.workloads IS
  'A piece or class of WORK with an intended outcome. Deliberately executor-agnostic: nothing here assumes an LLM performs the work. The grain is not locked — a workload may be a single invocation, a repeated task class, a workflow, an endpoint, or (later) a semantic cluster. Engineering work is the first Workforce vertical, not a special table.';
COMMENT ON COLUMN public.workloads.intended_outcome IS
  'What "done" means for this work, in the customer''s terms (e.g. "ticket resolved without escalation"). Free text today; the machine-readable form lives in outcomes.outcome_type/outcome_key.';
COMMENT ON COLUMN public.workloads.grain IS
  'How wide this workload is. Discovery currently emits ''endpoint''. ''cluster'' is reserved for future semantic grouping and is not produced today.';
COMMENT ON COLUMN public.workloads.default_objective IS
  'Default optimization objective for recommendations on this workload. Cost reduction is NOT assumed to be the goal.';


-- ============================================================================
-- 2. executors — the thing that PERFORMS work.
--    VENDOR METADATA ONLY. Never measured performance.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.executors (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,

    executor_type       TEXT NOT NULL
                        CHECK (executor_type IN ('model', 'agent', 'software', 'human')),
    vendor              TEXT,
    external_id         TEXT NOT NULL,
    display_name        TEXT,
    version             TEXT,

    capabilities        JSONB NOT NULL DEFAULT '{}'::jsonb,
    configuration       JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_model          JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,

    integration_source  TEXT NOT NULL DEFAULT 'manual'
                        CHECK (integration_source IN ('providers_json', 'model_registry',
                                                      'custom_endpoint', 'connector', 'manual')),
    enabled             BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_executors_identity_unique
    ON public.executors (org_id, executor_type, COALESCE(vendor, ''), external_id, COALESCE(version, ''));
CREATE INDEX IF NOT EXISTS idx_executors_org_created
    ON public.executors (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_executors_org_type
    ON public.executors (org_id, executor_type) WHERE enabled;

COMMENT ON TABLE public.executors IS
  'Something that can perform work: a model, an agent, deterministic software, a SaaS API — or eventually a human. A model is ONE executor_type, not the centre of the schema. Adding Claude Code, Devin or an internal function later is an INSERT here, not a migration.';
COMMENT ON COLUMN public.executors.executor_type IS
  'model = an LLM/API model. agent = an autonomous agent or external tool (Claude Code, Devin, Codex). software = deterministic code, a rule, a classifier, a SaaS endpoint. human = a person; PERMITTED BY THE SCHEMA BUT NOTHING IS BUILT FOR IT — no human routing, queueing or payout exists.';
COMMENT ON COLUMN public.executors.capabilities IS
  'VENDOR-CLAIMED capabilities: {context_window, modalities, tool_use, streaming, ...}. A published number, not a measurement. NEVER treat as evidence.';
COMMENT ON COLUMN public.executors.cost_model IS
  'VENDOR LIST PRICE, e.g. {"unit":"usd_per_1k_tokens","input":0.00015,"output":0.0006}. For non-model executors the unit may be "usd_per_acu", "usd_per_credit", "usd_per_seat_month", "usd_per_minute". This is a price sheet. Actual spend lives in cost_events / workflow_runs.';
COMMENT ON COLUMN public.executors.policy_metadata IS
  'Vendor/deployment facts that policies filter on: {region, zero_data_retention, stores_prompts, approved, certifications[]}. Sourced from the vendor or the customer''s own attestation — not measured by OptiML.';
COMMENT ON COLUMN public.executors.configuration IS
  'Default per-executor configuration knobs. Strategy steps override these per step.';


-- ============================================================================
-- 3. execution_strategies — HOW a workload should be attempted.
--    The unit of comparison. May chain MULTIPLE executors.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.execution_strategies (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    workload_id         UUID REFERENCES public.workloads(id) ON DELETE CASCADE,

    name                TEXT,
    description         TEXT,
    kind                TEXT NOT NULL DEFAULT 'candidate'
                        CHECK (kind IN ('baseline', 'candidate', 'template')),

    steps               JSONB NOT NULL DEFAULT '[]'::jsonb,
    surface_binding     JSONB NOT NULL DEFAULT '{}'::jsonb,
    dimensions          TEXT[],
    fingerprint         TEXT NOT NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_execstrat_org_fingerprint
    ON public.execution_strategies (org_id, fingerprint);
CREATE INDEX IF NOT EXISTS idx_execstrat_org_created
    ON public.execution_strategies (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execstrat_workload
    ON public.execution_strategies (workload_id);
CREATE INDEX IF NOT EXISTS idx_execstrat_dimensions
    ON public.execution_strategies USING GIN (dimensions);

COMMENT ON TABLE public.execution_strategies IS
  'How a workload should be attempted. THE unit of comparison for a recommendation: baseline STRATEGY vs candidate STRATEGY, never merely model A vs model B. A strategy is an ordered list of steps, each bound to an executor, so a deterministic classifier -> model -> human-approval chain is expressible today even though only model steps execute today.';
COMMENT ON COLUMN public.execution_strategies.steps IS
  'Ordered steps: [{"step_id":..., "order":0, "executor_id":uuid|null, "executor_ref":{"executor_type":"model","vendor":"openai","external_id":"gpt-4o-mini"}, "role":"primary|fallback|verifier|approver|preprocessor", "config":{...}, "on_failure":"fail|next_step|fallback"}]. executor_ref lets a step name an executor that has not been registered yet; executor_id links it once it has.';
COMMENT ON COLUMN public.execution_strategies.surface_binding IS
  'How this strategy is realised on its execution surface. Runtime: {"kind":"workflow_graph","workflow_id":...,"graph_json":{...}} — the concrete graph handed to workflow_runtime.execute_workflow. Other surfaces bind differently without changing this table.';
COMMENT ON COLUMN public.execution_strategies.fingerprint IS
  'Stable hash of the semantically-meaningful strategy content, used to dedupe identical candidates across generators and runs.';
COMMENT ON COLUMN public.execution_strategies.dimensions IS
  'Optimization dimensions this strategy differs on. Runtime v1 implements: model, provider, prompt, context_length, caching, fallback_chain.';


-- ============================================================================
-- 4. attempts — one execution of work.
--
--    *** THE EXISTING TRACING INFRASTRUCTURE IS THE RECORD OF TRUTH. ***
--    workflow_runs already stores runtime executions. This is a thin domain
--    VIEW over it, plus optimization/attempts.py as the Python adapter.
--    Nothing is copied, renamed or rewritten. When a second surface appears,
--    UNION ALL a second branch into this view.
-- ============================================================================
CREATE OR REPLACE VIEW public.attempts AS
SELECT
    wr.id                                   AS attempt_id,
    'workflow_run'::TEXT                    AS attempt_source,
    wr.org_id                               AS org_id,
    w.id                                    AS workload_id,
    'runtime'::TEXT                         AS surface,
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
LEFT JOIN public.workloads w
       ON w.org_id        = wr.org_id
      AND w.surface       = 'runtime'
      AND w.identity_kind = 'endpoint'
      AND w.identity_ref  = wr.endpoint_slug;

-- security_invoker keeps the view honest against the underlying tables' RLS
-- instead of running with the view owner's rights. Backend is service_role.
ALTER VIEW public.attempts SET (security_invoker = on);
REVOKE ALL ON public.attempts FROM PUBLIC;
REVOKE ALL ON public.attempts FROM anon;
GRANT SELECT ON public.attempts TO authenticated;
GRANT SELECT ON public.attempts TO service_role;

COMMENT ON VIEW public.attempts IS
  'Domain view of one execution of work. Derived from workflow_runs — the existing tracing record, which is NOT duplicated or renamed. workload_id is resolved by joining workloads on endpoint identity. Costs and outcomes that have no home in workflow_runs live in cost_events and outcomes, keyed by attempt_id.';


-- ============================================================================
-- 5. outcomes — FIRST CLASS. What actually happened, however late we find out.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.outcomes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    workload_id         UUID REFERENCES public.workloads(id) ON DELETE SET NULL,

    -- attempt_ref is intentionally NOT a foreign key: attempts is a view over
    -- workflow_runs, and an outcome may also describe a workload as a whole
    -- (attempt_ref NULL) or an attempt on a surface OptiML does not execute.
    attempt_ref         UUID,
    attempt_source      TEXT NOT NULL DEFAULT 'workflow_run'
                        CHECK (attempt_source IN ('workflow_run', 'api_request', 'external', 'none')),

    outcome_type        TEXT NOT NULL
                        CHECK (outcome_type IN ('task_success', 'quality_score', 'user_feedback',
                                                'business_value', 'error', 'escalation',
                                                'human_intervention', 'custom')),
    outcome_key         TEXT,

    -- Deliberately NOT forced to boolean. A resolution rate, an NPS score, a
    -- dollar figure and a pass/fail are all representable.
    outcome_value       NUMERIC(20, 6),
    outcome_value_text  TEXT,
    unit                TEXT,
    success             BOOLEAN,

    source              TEXT NOT NULL DEFAULT 'api'
                        CHECK (source IN ('api', 'human', 'system', 'judge', 'connector', 'import')),
    provenance          TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (provenance IN ('business_outcome', 'deterministic', 'human',
                                              'user_feedback', 'automated_test', 'schema',
                                              'llm_judge', 'implicit', 'heuristic', 'unknown')),
    provenance_rank     SMALLINT GENERATED ALWAYS AS (public.outcome_provenance_rank(provenance)) STORED,
    confidence          NUMERIC(4, 3) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),

    occurred_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    idempotency_key     TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotency is per-org and REQUIRED: outcomes arrive from retrying webhooks.
CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_idempotency
    ON public.outcomes (org_id, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_outcomes_org_created
    ON public.outcomes (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_attempt
    ON public.outcomes (attempt_ref) WHERE attempt_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_outcomes_workload_occurred
    ON public.outcomes (workload_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_org_type_occurred
    ON public.outcomes (org_id, outcome_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_org_rank
    ON public.outcomes (org_id, provenance_rank DESC, occurred_at DESC);

COMMENT ON TABLE public.outcomes IS
  'What actually happened as a result of an attempt — the single most important table for honest optimization. MANY outcomes may belong to one attempt (a PR merges tomorrow and is reverted next week: two rows). Outcomes ARRIVE LATE by design: occurred_at is when it happened in the world, recorded_at is when OptiML learned. Attach via POST /api/optimization/{org_id}/outcomes with an idempotency_key.';
COMMENT ON COLUMN public.outcomes.outcome_value IS
  'Numeric outcome value. NOT forced to boolean — a 0.93 resolution rate, a 7.0 CSAT, or 1240.00 USD protected are all valid. Use `success` only when the outcome genuinely is binary; leave it NULL otherwise rather than coercing.';
COMMENT ON COLUMN public.outcomes.provenance IS
  'How strong this signal is. Ranked by public.outcome_provenance_rank(): business_outcome > deterministic > human > user_feedback > automated_test/schema > llm_judge > implicit/heuristic > unknown. NEVER average across provenance tiers — aggregate within a tier and report the tier. "Passed an LLM judge" and "raised actual resolution rate 7%" are not the same number.';
COMMENT ON COLUMN public.outcomes.occurred_at IS
  'When the outcome happened in the real world. May be hours or days BEFORE recorded_at. Always window analyses on occurred_at, never on created_at.';
COMMENT ON COLUMN public.outcomes.idempotency_key IS
  'Caller-supplied dedupe key, unique per org. Re-POSTing the same key returns the existing row unchanged instead of creating a duplicate.';
COMMENT ON COLUMN public.outcomes.attempt_ref IS
  'The attempt this outcome describes: workflow_runs.id when attempt_source=''workflow_run'', api_request_log.id when ''api_request''. NULL means the outcome describes the workload as a whole over a period.';


-- ============================================================================
-- 6. cost_events — cost that is not (only) token cost.
--    Runtime inference cost ALREADY lives in workflow_runs.total_cost and
--    node_results. It is NOT copied here. This table is for cost with no home.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.cost_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    workload_id         UUID REFERENCES public.workloads(id) ON DELETE SET NULL,
    executor_id         UUID REFERENCES public.executors(id) ON DELETE SET NULL,

    attempt_ref         UUID,
    attempt_source      TEXT NOT NULL DEFAULT 'workflow_run'
                        CHECK (attempt_source IN ('workflow_run', 'api_request', 'external', 'none')),

    cost_type           TEXT NOT NULL
                        CHECK (cost_type IN ('inference', 'api_call', 'cache', 'agent_credit',
                                             'saas_usage', 'seat', 'human_time', 'infrastructure', 'other')),
    amount              NUMERIC(20, 8) NOT NULL,
    unit                TEXT NOT NULL DEFAULT 'usd',
    amount_usd          NUMERIC(14, 6),

    quantity            NUMERIC(20, 6),
    quantity_unit       TEXT,

    occurred_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    idempotency_key     TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cost_events_idempotency
    ON public.cost_events (org_id, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_cost_events_org_created
    ON public.cost_events (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_events_workload_occurred
    ON public.cost_events (workload_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_events_attempt
    ON public.cost_events (attempt_ref) WHERE attempt_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cost_events_org_type_occurred
    ON public.cost_events (org_id, cost_type, occurred_at DESC);

COMMENT ON TABLE public.cost_events IS
  'Economic cost as typed events so that "total cost per successful outcome" stays computable when cost stops meaning tokens: Devin ACUs, agent credits, SaaS usage, seat cost, human review minutes, infrastructure. Runtime INFERENCE cost is already recorded in workflow_runs.total_cost/node_results and is deliberately NOT duplicated here.';
COMMENT ON COLUMN public.cost_events.amount IS
  'Amount in `unit`. May be 4200 ACU or 35 minutes — not necessarily dollars.';
COMMENT ON COLUMN public.cost_events.amount_usd IS
  'USD equivalent, ONLY when a real conversion rate is known. Leave NULL when the USD value is unknown — never fabricate a conversion. Aggregations must report how much of the total was unconvertible.';
COMMENT ON COLUMN public.cost_events.idempotency_key IS
  'Caller-supplied dedupe key, unique per org. Billing feeds retry.';


-- ============================================================================
-- 7. optimization_policies — constraints that make a strategy INVALID,
--    not merely worse. Optimization = best strategy WITHIN constraints.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.optimization_policies (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    workload_id         UUID REFERENCES public.workloads(id) ON DELETE CASCADE,

    name                TEXT NOT NULL DEFAULT 'Default policy',
    description         TEXT,
    enabled             BOOLEAN NOT NULL DEFAULT TRUE,
    priority            INT NOT NULL DEFAULT 100,

    constraints         JSONB NOT NULL DEFAULT '{}'::jsonb,
    automation          JSONB NOT NULL DEFAULT
                        '{"auto_benchmark": false, "auto_shadow": false, "auto_canary": false, "auto_promote": false, "require_human_approval": true, "auto_rollback": true}'::jsonb,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_optpolicy_org_created
    ON public.optimization_policies (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optpolicy_workload
    ON public.optimization_policies (workload_id);
CREATE INDEX IF NOT EXISTS idx_optpolicy_org_enabled_priority
    ON public.optimization_policies (org_id, enabled, priority);

COMMENT ON TABLE public.optimization_policies IS
  'Hard constraints on strategy selection, scoped org-wide (workload_id NULL) or per workload. A strategy violating a policy is INVALID, not merely lower-ranked — the optimizer must never trade a policy away for savings.';
COMMENT ON COLUMN public.optimization_policies.constraints IS
  'Validity constraints: {"min_quality":0.95,"max_error_rate":0.02,"max_latency_p95_ms":4000,"max_cost_per_task_usd":0.01,"allowed_vendors":["openai","anthropic"],"blocked_vendors":[],"require_zero_data_retention":true,"allow_prompt_storage":false,"data_region":"eu","require_human_approval":true}. Unknown keys are ignored by the evaluator and reported as unenforced rather than silently treated as satisfied.';
COMMENT ON COLUMN public.optimization_policies.automation IS
  'How much OptiML may do without a human. DEFAULTS ARE CONSERVATIVE: nothing is automatic and human approval is required. An org may later opt into auto_benchmark / auto_shadow / auto_canary / auto_promote. auto_rollback defaults TRUE because rolling back is the safe direction.';


-- ============================================================================
-- 8. allocation_decisions — "OptiML chose strategy X for workload Y because of
--    objective Z under policy P." Internal in v1; the clean path to future
--    autonomous allocation without a schema rewrite.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.allocation_decisions (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    workload_id             UUID REFERENCES public.workloads(id) ON DELETE CASCADE,
    recommendation_id       UUID REFERENCES public.optimization_recommendations(id) ON DELETE SET NULL,
    policy_id               UUID REFERENCES public.optimization_policies(id) ON DELETE SET NULL,

    decision_kind           TEXT NOT NULL DEFAULT 'recommendation'
                            CHECK (decision_kind IN ('recommendation', 'benchmark', 'experiment', 'route')),
    objective               TEXT NOT NULL DEFAULT 'cost'
                            CHECK (objective IN ('cost', 'quality', 'latency', 'balanced', 'custom')),
    objective_config        JSONB NOT NULL DEFAULT '{}'::jsonb,

    considered_strategies   JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_strategy_id    UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,

    expected_cost_usd       NUMERIC(14, 6),
    expected_quality        NUMERIC(6, 4),
    expected_latency_p95_ms INT,
    confidence              NUMERIC(4, 3) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),

    reason                  TEXT,
    decided_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    actual_result           JSONB,
    resolved_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_allocdec_org_created
    ON public.allocation_decisions (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_allocdec_workload_decided
    ON public.allocation_decisions (workload_id, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_allocdec_recommendation
    ON public.allocation_decisions (recommendation_id);
CREATE INDEX IF NOT EXISTS idx_allocdec_selected_strategy
    ON public.allocation_decisions (selected_strategy_id);

COMMENT ON TABLE public.allocation_decisions IS
  'A record of OptiML choosing a strategy for a workload under an objective and a policy, with what it expected and (later) what actually happened. Written today for recommendation/benchmark decisions and NOT surfaced prominently. It is the substrate a future allocation engine reads and writes without needing new tables.';
COMMENT ON COLUMN public.allocation_decisions.considered_strategies IS
  'Everything that was on the table, including rejects: [{"strategy_id":..., "fingerprint":..., "expected_cost_usd":..., "expected_quality":..., "eligible":false, "rejected_reason":"violates policy: data_region"}]. Recording the rejects is what makes the decision auditable.';
COMMENT ON COLUMN public.allocation_decisions.objective_config IS
  'Parameters for objective=''custom'' (e.g. {"formula":"expected_business_value_usd - expected_cost_usd"}). Lets new objectives ship without a schema change.';
COMMENT ON COLUMN public.allocation_decisions.actual_result IS
  'Backfilled once the consequence is known, so expected-vs-actual calibration is measurable. NULL until then — never pre-filled with the expectation.';


-- ============================================================================
-- 9. optimization_recommendations — evolve. ADDITIVE ONLY.
-- ============================================================================
ALTER TABLE public.optimization_recommendations
    ADD COLUMN IF NOT EXISTS objective                 TEXT NOT NULL DEFAULT 'cost',
    ADD COLUMN IF NOT EXISTS objective_config          JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS baseline_strategy_id      UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS candidate_strategy_id     UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS policy_id                 UUID REFERENCES public.optimization_policies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS parent_recommendation_id  UUID REFERENCES public.optimization_recommendations(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS supersedes_id             UUID REFERENCES public.optimization_recommendations(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS bundle_id                 UUID,
    ADD COLUMN IF NOT EXISTS baseline_reference        JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS approval_required         BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS approved_by               UUID,
    ADD COLUMN IF NOT EXISTS approved_at               TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS promoted_at               TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS monitoring_status         TEXT NOT NULL DEFAULT 'not_started',
    ADD COLUMN IF NOT EXISTS realized_window_start     TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS realized_window_end       TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS realized_metrics          JSONB,
    ADD COLUMN IF NOT EXISTS evidence_strength         SMALLINT;

ALTER TABLE public.optimization_recommendations DROP CONSTRAINT IF EXISTS optimization_recommendations_objective_check;
ALTER TABLE public.optimization_recommendations ADD CONSTRAINT optimization_recommendations_objective_check
    CHECK (objective IN ('cost', 'quality', 'latency', 'balanced', 'custom'));

ALTER TABLE public.optimization_recommendations DROP CONSTRAINT IF EXISTS optimization_recommendations_monitoring_status_check;
ALTER TABLE public.optimization_recommendations ADD CONSTRAINT optimization_recommendations_monitoring_status_check
    CHECK (monitoring_status IN ('not_started', 'monitoring', 'healthy', 'degraded', 'stopped'));

-- A recommendation must never claim savings against a baseline that another
-- recommendation already claimed. Guard the shape of that reference.
COMMENT ON COLUMN public.optimization_recommendations.baseline_reference IS
  'WHAT the savings are measured against, so they are never double-counted: {"kind":"promoted_deployment","deployment_id":...,"version":5,"strategy_fingerprint":...,"derived_from_recommendation_id":uuid|null}. If recommendation B optimizes a configuration that recommendation A already produced, B sets derived_from_recommendation_id=A and its savings are attributable ONLY to the delta from A''s result — not from the original baseline. Roll-ups must walk this chain instead of summing naively.';
COMMENT ON COLUMN public.optimization_recommendations.parent_recommendation_id IS
  'The recommendation this one builds on. Used with baseline_reference to keep savings attribution non-overlapping.';
COMMENT ON COLUMN public.optimization_recommendations.supersedes_id IS
  'This recommendation replaces an earlier one (which should move to status=''superseded''). Prevents the same opportunity being surfaced twice.';
COMMENT ON COLUMN public.optimization_recommendations.bundle_id IS
  'Groups candidates that are meant to be evaluated together (e.g. smaller model + compressed prompt + shorter context as one combined change) so the user sees one bundled opportunity plus its parts, instead of a flood of overlapping individual ones. Members of a bundle share bundle_id; the combined candidate is the one whose dimensions is the union.';
COMMENT ON COLUMN public.optimization_recommendations.objective IS
  'What this recommendation is optimizing FOR. The optimizer does not hardcode "minimize dollars": quality, latency and balanced are first-class, and ''custom'' + objective_config allows e.g. expected business value minus cost without a schema change.';
COMMENT ON COLUMN public.optimization_recommendations.approval_required IS
  'TRUE by default. Human approval is the DEFAULT path: discover -> benchmark -> verify -> awaiting_approval -> (human) -> canary -> promoted. Production is never changed autonomously unless an optimization_policies.automation flag says otherwise.';
COMMENT ON COLUMN public.optimization_recommendations.monitoring_status IS
  'Post-promotion monitoring. Realized impact keeps being measured AFTER promotion so an optimization that was initially fine and later deteriorates is detected: ''degraded'' is the rollback trigger.';
COMMENT ON COLUMN public.optimization_recommendations.realized_metrics IS
  'Measured production metrics for the realized window: {cost, quality, latency_p95_ms, error_rate, downstream_outcomes, coverage}. NULL until post-promotion monitoring has actually run.';

-- ── Lifecycle state machine ────────────────────────────────────────────────
-- discovered -> benchmarking -> verified -> awaiting_approval -> shadowing
--            -> canary -> promoted
-- branches/terminals: rejected, inconclusive, failed, superseded, rolled_back
-- ('applied' retained from migration_optimization.sql for compatibility;
--  deprecated in favour of 'promoted' and never emitted by new code.)
ALTER TABLE public.optimization_recommendations DROP CONSTRAINT IF EXISTS optimization_recommendations_status_check;
ALTER TABLE public.optimization_recommendations ADD CONSTRAINT optimization_recommendations_status_check
    CHECK (status IN ('discovered', 'benchmarking', 'verified', 'awaiting_approval', 'shadowing',
                      'canary', 'promoted', 'rejected', 'inconclusive', 'failed', 'superseded',
                      'rolled_back', 'applied'));

COMMENT ON COLUMN public.optimization_recommendations.status IS
  'Lifecycle state. Transitions are validated in optimization/service.py — the DB CHECK only bounds the vocabulary. Happy path: discovered -> benchmarking -> verified -> awaiting_approval -> (shadowing) -> canary -> promoted. Terminals: rejected, superseded, rolled_back. inconclusive = evidence was insufficient to decide (NOT a failure and NOT a rejection). failed = the benchmark itself broke. ''applied'' is a deprecated alias for ''promoted''.';

-- ── evidence_source now encodes COUNTERFACTUAL STRENGTH ────────────────────
-- "we observed A" is much weaker than "we compared A and B on the same inputs".
ALTER TABLE public.optimization_recommendations DROP CONSTRAINT IF EXISTS optimization_recommendations_evidence_source_check;
ALTER TABLE public.optimization_recommendations ADD CONSTRAINT optimization_recommendations_evidence_source_check
    CHECK (evidence_source IN ('none', 'observational', 'replay', 'shadow', 'ab_test', 'canary',
                               'production', 'historical_analysis', 'online_experiment'));

COMMENT ON COLUMN public.optimization_recommendations.evidence_source IS
  'COUNTERFACTUAL STRENGTH of the evidence, not merely its presence. Weakest to strongest: none < observational (we watched A happen; no counterfactual) < replay (A and B on the SAME inputs, offline) < shadow (B ran on live inputs without serving) < ab_test (concurrent split) < canary (B served real traffic at low share) < production (B fully serving). Feeds confidence. Deprecated aliases retained: historical_analysis = observational, online_experiment = ab_test.';

COMMENT ON COLUMN public.optimization_recommendations.evidence_strength IS
  'Numeric mirror of evidence_source (0..80) written by optimization/domain.py so evidence can be ordered and compared in SQL. Never set independently of evidence_source.';

COMMENT ON COLUMN public.optimization_recommendations.confidence IS
  'First-class 0..1 confidence, derived in optimization/domain.py from: sample size, counterfactual strength of evidence_source, strength of the quality signal (outcome provenance rank), observed variance, historical consistency, and whether production confirmed it. 14 replay examples must not look like 180,000 production outcomes. NULL when it cannot be computed — never a placeholder.';

-- Widen quality_provenance to the shared outcome-provenance vocabulary.
ALTER TABLE public.optimization_recommendations DROP CONSTRAINT IF EXISTS optimization_recommendations_quality_provenance_check;
ALTER TABLE public.optimization_recommendations ADD CONSTRAINT optimization_recommendations_quality_provenance_check
    CHECK (quality_provenance IN ('business_outcome', 'deterministic', 'human', 'user_feedback',
                                  'automated_test', 'schema', 'llm_judge', 'implicit',
                                  'heuristic', 'unknown'));

CREATE INDEX IF NOT EXISTS idx_optrec_bundle
    ON public.optimization_recommendations (bundle_id) WHERE bundle_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_optrec_parent
    ON public.optimization_recommendations (parent_recommendation_id);
CREATE INDEX IF NOT EXISTS idx_optrec_org_monitoring
    ON public.optimization_recommendations (org_id, monitoring_status)
    WHERE monitoring_status IN ('monitoring', 'degraded');
CREATE INDEX IF NOT EXISTS idx_optrec_candidate_strategy
    ON public.optimization_recommendations (candidate_strategy_id);


-- ============================================================================
-- 10. optimization_benchmarks — align vocabulary with the new evidence model.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    ADD COLUMN IF NOT EXISTS baseline_strategy_id  UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS candidate_strategy_id UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS objective             TEXT NOT NULL DEFAULT 'cost',
    ADD COLUMN IF NOT EXISTS policy_evaluation     JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_objective_check;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_objective_check
    CHECK (objective IN ('cost', 'quality', 'latency', 'balanced', 'custom'));

ALTER TABLE public.optimization_benchmarks DROP CONSTRAINT IF EXISTS optimization_benchmarks_quality_provenance_check;
ALTER TABLE public.optimization_benchmarks ADD CONSTRAINT optimization_benchmarks_quality_provenance_check
    CHECK (quality_provenance IN ('business_outcome', 'deterministic', 'human', 'user_feedback',
                                  'automated_test', 'schema', 'llm_judge', 'implicit',
                                  'heuristic', 'unknown'));

COMMENT ON COLUMN public.optimization_benchmarks.policy_evaluation IS
  'Which optimization_policies constraints were checked, which passed, and which could NOT be evaluated: {"policy_id":..., "satisfied":[...], "violated":[...], "unenforced":["data_region"]}. An unenforceable constraint is reported, never assumed satisfied.';
COMMENT ON COLUMN public.optimization_benchmarks.objective IS
  'The objective this benchmark was judged against. A run judged on ''cost'' must not later be reinterpreted as evidence for ''quality''.';


-- ============================================================================
-- 11. RLS on the new tables. Backend is service_role and bypasses this.
--     NEVER `USING (true)` without `TO service_role` — that grants PUBLIC.
-- ============================================================================
ALTER TABLE public.executors             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.execution_strategies  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.outcomes              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cost_events           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.optimization_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.allocation_decisions  ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Org members can view executors" ON public.executors;
CREATE POLICY "Org members can view executors"
    ON public.executors FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view execution strategies" ON public.execution_strategies;
CREATE POLICY "Org members can view execution strategies"
    ON public.execution_strategies FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view outcomes" ON public.outcomes;
CREATE POLICY "Org members can view outcomes"
    ON public.outcomes FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view cost events" ON public.cost_events;
CREATE POLICY "Org members can view cost events"
    ON public.cost_events FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view optimization policies" ON public.optimization_policies;
CREATE POLICY "Org members can view optimization policies"
    ON public.optimization_policies FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view allocation decisions" ON public.allocation_decisions;
CREATE POLICY "Org members can view allocation decisions"
    ON public.allocation_decisions FOR SELECT
    USING (public.is_org_member(org_id));


-- ============================================================================
-- Verify
-- ============================================================================
-- SELECT tablename, rowsecurity FROM pg_tables
--  WHERE schemaname='public'
--    AND tablename IN ('workloads','executors','execution_strategies','outcomes',
--                      'cost_events','optimization_policies','allocation_decisions',
--                      'optimization_recommendations','optimization_benchmarks')
--  ORDER BY tablename;
--
-- Work Graph smoke query — "for this workload, which strategy produced the
-- better economic outcome, and how strong is the signal?"
-- SELECT a.workload_id,
--        a.variant_name,
--        count(*)                              AS attempts,
--        avg(a.inference_cost_usd)             AS avg_cost_usd,
--        avg(o.outcome_value) FILTER (WHERE o.provenance_rank >= 70) AS strong_outcome,
--        avg(o.outcome_value) FILTER (WHERE o.provenance_rank <  70) AS weak_outcome
--   FROM public.attempts a
--   LEFT JOIN public.outcomes o ON o.attempt_ref = a.attempt_id
--  WHERE a.org_id = '00000000-0000-0000-0000-000000000000'
--  GROUP BY 1, 2;
