-- ============================================================================
-- Migration: OptiML optimization layer (workloads, recommendations, benchmarks)
-- Run this in the Supabase SQL Editor. Idempotent — safe to re-run.
--
-- Product thesis: the core deliverable is an EVIDENCE-BACKED Optimization
-- Recommendation. A candidate is never promoted because an LLM said it looked
-- good; it is promoted because a replay/benchmark against the customer's own
-- data measured it. The schema below encodes that distinction structurally:
--
--   * optimization_recommendations.rationale       -> LLM prose. NOT evidence.
--   * optimization_recommendations.evidence_source -> where the numbers came from.
--   * optimization_benchmarks                      -> the measurement itself.
--
-- Backend is 100% service_role (bypasses RLS). RLS + an is_org_member SELECT
-- policy is defence-in-depth against direct anon-key access from the browser.
-- `public.is_org_member` is defined in migration_enable_rls.sql — run that first.
-- ============================================================================


-- ============================================================================
-- 1. workloads — a repeated category of AI work
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.workloads (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    project_id      UUID REFERENCES public.projects(id) ON DELETE SET NULL,
    name            TEXT NOT NULL DEFAULT 'Untitled workload',
    description     TEXT,
    surface         TEXT NOT NULL DEFAULT 'runtime'
                    CHECK (surface IN ('runtime', 'workforce')),
    identity_kind   TEXT
                    CHECK (identity_kind IN ('endpoint', 'workflow', 'prompt_template', 'manual', 'inferred')),
    identity_ref    TEXT,
    tags            TEXT[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One workload per concrete identity within an org+surface. Partial so that
-- manually-created workloads with no identity_ref are not forced unique.
CREATE UNIQUE INDEX IF NOT EXISTS idx_workloads_identity_unique
    ON public.workloads (org_id, surface, identity_kind, identity_ref)
    WHERE identity_ref IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_workloads_org_created
    ON public.workloads (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workloads_project
    ON public.workloads (project_id);
CREATE INDEX IF NOT EXISTS idx_workloads_org_surface
    ON public.workloads (org_id, surface);
CREATE INDEX IF NOT EXISTS idx_workloads_tags
    ON public.workloads USING GIN (tags);

COMMENT ON TABLE public.workloads IS
  'A repeated category of AI work (an endpoint, a workflow, a prompt template). The unit that gets observed, benchmarked and optimized. Discovery currently groups production workflow_runs by endpoint_slug, independent of served_version.';
COMMENT ON COLUMN public.workloads.surface IS
  'runtime = OptiML-executed workflows/endpoints; workforce = agentic/workforce surface.';
COMMENT ON COLUMN public.workloads.identity_kind IS
  'How this workload is identified. ''inferred'' is reserved for future semantic clustering and is NOT produced by the current discovery pass.';
COMMENT ON COLUMN public.workloads.identity_ref IS
  'The concrete identifier for identity_kind: endpoint_slug for ''endpoint'', workflows.id for ''workflow'', prompt_templates.id for ''prompt_template''.';


-- ============================================================================
-- 2. optimization_recommendations — the central object
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.optimization_recommendations (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    project_id              UUID REFERENCES public.projects(id) ON DELETE SET NULL,
    workload_id             UUID REFERENCES public.workloads(id) ON DELETE CASCADE,

    status                  TEXT NOT NULL DEFAULT 'discovered'
                            CHECK (status IN ('discovered', 'benchmarking', 'verified', 'rejected',
                                              'canary', 'applied', 'rolled_back')),
    title                   TEXT,

    -- Which optimization dimensions this candidate changes.
    dimensions              TEXT[],

    baseline_strategy       JSONB,
    candidate_strategy      JSONB,
    baseline_version        TEXT,
    candidate_version       TEXT,

    generator               TEXT,
    rationale               TEXT,

    evidence_source         TEXT NOT NULL DEFAULT 'none'
                            CHECK (evidence_source IN ('replay', 'online_experiment', 'historical_analysis', 'none')),
    benchmark_id            UUID,
    sample_size             INT,

    -- ── Cost ────────────────────────────────────────────────────────────────
    baseline_cost           NUMERIC(14, 6),
    candidate_cost          NUMERIC(14, 6),
    projected_savings_usd   NUMERIC(14, 6),
    verified_savings_usd    NUMERIC(14, 6),
    realized_savings_usd    NUMERIC(14, 6),

    -- ── Quality ─────────────────────────────────────────────────────────────
    baseline_quality        NUMERIC(6, 4),
    candidate_quality       NUMERIC(6, 4),
    quality_provenance      TEXT NOT NULL DEFAULT 'unknown'
                            CHECK (quality_provenance IN ('human', 'deterministic', 'schema', 'llm_judge',
                                                          'business_outcome', 'implicit', 'unknown')),

    -- ── Latency / reliability ───────────────────────────────────────────────
    baseline_latency_p95_ms  INT,
    candidate_latency_p95_ms INT,
    baseline_error_rate      NUMERIC(6, 4),
    candidate_error_rate     NUMERIC(6, 4),

    confidence              NUMERIC(4, 3) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    constraints             JSONB NOT NULL DEFAULT '{}'::jsonb,

    decided_by              UUID,
    decided_at              TIMESTAMPTZ,
    deployment_id           UUID,
    experiment_id           UUID,
    rolled_back_at          TIMESTAMPTZ,

    audit                   JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_optrec_org_created
    ON public.optimization_recommendations (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optrec_org_status_created
    ON public.optimization_recommendations (org_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optrec_workload
    ON public.optimization_recommendations (workload_id);
CREATE INDEX IF NOT EXISTS idx_optrec_project
    ON public.optimization_recommendations (project_id);
CREATE INDEX IF NOT EXISTS idx_optrec_benchmark
    ON public.optimization_recommendations (benchmark_id);
CREATE INDEX IF NOT EXISTS idx_optrec_deployment
    ON public.optimization_recommendations (deployment_id);
CREATE INDEX IF NOT EXISTS idx_optrec_dimensions
    ON public.optimization_recommendations USING GIN (dimensions);

COMMENT ON TABLE public.optimization_recommendations IS
  'The central object of the optimization layer: one proposed change to a workload, plus the evidence for or against it. Lifecycle: discovered -> benchmarking -> verified|rejected -> canary -> applied -> rolled_back.';

COMMENT ON COLUMN public.optimization_recommendations.dimensions IS
  'Optimization dimensions changed by this candidate. Vocabulary: model, provider, prompt, context_length, reasoning_effort, temperature, retrieval, reranking, caching, fallback_chain, llm_call_count, workflow_structure, tool_selection, deterministic_code. NOTE: only model, provider and prompt are currently APPLICABLE to graph_json by optimization/strategy.py — the rest are vocabulary reserved for future generators.';

COMMENT ON COLUMN public.optimization_recommendations.rationale IS
  'Human/LLM-written explanation of WHY this candidate was proposed. This is PROSE, NOT EVIDENCE. It must never be used to justify a status transition to ''verified''. Only optimization_benchmarks rows may do that.';

COMMENT ON COLUMN public.optimization_recommendations.evidence_source IS
  'Where the measured numbers on this row came from. ''none'' = nothing has been measured yet (the candidate is a hypothesis). ''replay'' = golden/production replay benchmark. ''online_experiment'' = live traffic split. ''historical_analysis'' = derived from already-observed workflow_runs, not from a fresh controlled run.';

COMMENT ON COLUMN public.optimization_recommendations.quality_provenance IS
  'How baseline_quality/candidate_quality were obtained. ''unknown'' means NOTHING measurable ran and the quality columns MUST be NULL — never substitute a guessed number. ''deterministic'' = exact-match checks against expected_output. ''schema'' = structural/format checks only. ''llm_judge'', ''human'', ''business_outcome'', ''implicit'' are reserved and not yet produced by the backend.';

-- ── The three savings columns are NOT interchangeable ───────────────────────
COMMENT ON COLUMN public.optimization_recommendations.projected_savings_usd IS
  'PROJECTED: extrapolated = (measured or priced per-call delta) x observed traffic volume. A forecast. Never write a measured value here, and never read this as if it were measured.';
COMMENT ON COLUMN public.optimization_recommendations.verified_savings_usd IS
  'VERIFIED: directly MEASURED inside a benchmark or canary, over the benchmark sample only. Written exclusively by optimization/benchmark.py from an optimization_benchmarks row. Never an extrapolation.';
COMMENT ON COLUMN public.optimization_recommendations.realized_savings_usd IS
  'REALIZED: OBSERVED in production after promotion, by comparing post-promotion production spend against the pre-promotion baseline. Written exclusively by post-promotion monitoring. Never a projection and never a benchmark result.';

COMMENT ON COLUMN public.optimization_recommendations.baseline_cost IS
  'Mean cost per case for the baseline arm of the benchmark, in USD. NULL until measured.';
COMMENT ON COLUMN public.optimization_recommendations.candidate_cost IS
  'Mean cost per case for the candidate arm of the benchmark, in USD. NULL until measured.';
COMMENT ON COLUMN public.optimization_recommendations.constraints IS
  'Acceptance constraints checked by the benchmark, e.g. {"min_savings_pct": 5, "min_quality": 0.95, "max_latency_delta_pct": 25, "max_error_rate": 0.05, "min_sample_size": 20}. If min_quality is requested but quality is unmeasurable, the recommendation MUST NOT be verified on cost alone.';
COMMENT ON COLUMN public.optimization_recommendations.audit IS
  'Append-only JSONB array of lifecycle events: [{at, actor, action, from_status, to_status, reason, ...}].';
COMMENT ON COLUMN public.optimization_recommendations.sample_size IS
  'Number of replay cases behind the measured numbers on this row. NULL when nothing has been measured.';


-- ============================================================================
-- 3. optimization_benchmarks — one replay/benchmark run producing evidence
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.optimization_benchmarks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    recommendation_id   UUID REFERENCES public.optimization_recommendations(id) ON DELETE CASCADE,
    workload_id         UUID REFERENCES public.workloads(id) ON DELETE SET NULL,

    method              TEXT NOT NULL DEFAULT 'golden_replay'
                        CHECK (method IN ('golden_replay', 'production_replay', 'shadow', 'online')),
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    sample_size         INT,
    dataset_ref         JSONB,
    baseline_metrics    JSONB,
    candidate_metrics   JSONB,
    per_case_results    JSONB,
    quality_provenance  TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (quality_provenance IN ('human', 'deterministic', 'schema', 'llm_judge',
                                                      'business_outcome', 'implicit', 'unknown')),
    error               TEXT,

    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_optbench_org_created
    ON public.optimization_benchmarks (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optbench_recommendation
    ON public.optimization_benchmarks (recommendation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optbench_workload
    ON public.optimization_benchmarks (workload_id);
CREATE INDEX IF NOT EXISTS idx_optbench_org_status
    ON public.optimization_benchmarks (org_id, status);

COMMENT ON TABLE public.optimization_benchmarks IS
  'One controlled replay/benchmark run. THIS is the evidence: baseline and candidate executed over the SAME inputs with execution_mode=''eval''. optimization_recommendations may only reach status=''verified'' on the back of a completed row here.';
COMMENT ON COLUMN public.optimization_benchmarks.method IS
  'golden_replay = replay of golden_inputs (implemented). production_replay = replay of sampled production workflow_runs inputs. shadow = mirror live traffic. online = live traffic split. Only golden_replay is implemented today.';
COMMENT ON COLUMN public.optimization_benchmarks.dataset_ref IS
  'What was replayed: {"kind": "golden_inputs", "workflow_id": ..., "golden_input_ids": [...], "requested": n, "used": n}. Makes the evidence reproducible.';
COMMENT ON COLUMN public.optimization_benchmarks.baseline_metrics IS
  'Measured aggregate for the baseline arm: {mean_cost, total_cost, latency_p50_ms, latency_p95_ms, error_rate, n, quality (nullable), quality_checks_run}. Every field is measured or NULL — never inferred.';
COMMENT ON COLUMN public.optimization_benchmarks.candidate_metrics IS
  'Measured aggregate for the candidate arm; same shape as baseline_metrics.';
COMMENT ON COLUMN public.optimization_benchmarks.per_case_results IS
  'Per-golden-input measurements for both arms, so a human can audit the aggregate.';
COMMENT ON COLUMN public.optimization_benchmarks.quality_provenance IS
  'How quality was measured for this run, if at all. ''unknown'' means the quality fields inside baseline_metrics/candidate_metrics are NULL and no quality claim may be made.';
COMMENT ON COLUMN public.optimization_benchmarks.error IS
  'Why the run failed or refused to produce evidence, e.g. "sample_size 4 below floor 20".';


-- ============================================================================
-- 4. RLS — defence-in-depth. Backend uses service_role and bypasses this.
--    NOTE: never USING (true) without TO service_role — that grants PUBLIC.
-- ============================================================================
ALTER TABLE public.workloads                    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.optimization_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.optimization_benchmarks      ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Org members can view workloads" ON public.workloads;
CREATE POLICY "Org members can view workloads"
    ON public.workloads FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view optimization recommendations" ON public.optimization_recommendations;
CREATE POLICY "Org members can view optimization recommendations"
    ON public.optimization_recommendations FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view optimization benchmarks" ON public.optimization_benchmarks;
CREATE POLICY "Org members can view optimization benchmarks"
    ON public.optimization_benchmarks FOR SELECT
    USING (public.is_org_member(org_id));


-- ============================================================================
-- Verify
-- ============================================================================
-- SELECT tablename, rowsecurity FROM pg_tables
--  WHERE schemaname = 'public'
--    AND tablename IN ('workloads', 'optimization_recommendations', 'optimization_benchmarks');
