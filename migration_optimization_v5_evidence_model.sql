-- ============================================================================
-- Migration: OptiML optimization layer v5 — evidence precedes recommendation.
--
-- RUN AFTER: migration_optimization.sql
--            migration_optimization_work_graph.sql
--            migration_optimization_v3_outcomes.sql
--            migration_optimization_v4_benchmark_conclusions.sql
-- Idempotent. ADDITIVE ONLY — no columns or data are dropped.
--
-- ── FIVE CORRECTIONS ────────────────────────────────────────────────────────
--
-- 1. NO CUSTOMER-FACING COPY IN THE BACKEND. v4 carried presentation strings as
--    data. That couples the API contract to wording: rephrasing a sentence
--    would become an API behaviour change. Conclusions now carry a stable
--    machine CODE plus structured REASONS with the underlying facts (observed,
--    required, constraint, unit). The frontend derives all wording.
--
-- 2. MATERIALITY IS OBJECTIVE-AWARE. "5% or $5/month" only makes sense for a
--    cost objective. A latency objective needs "150ms p95", a resolution-rate
--    objective needs "1.5 percentage points". Materiality is now
--    metric + comparator + value + unit, OR/AND-combined. Savings-shaped fields
--    remain as sugar, not as the domain model.
--
-- 3. CONCLUSIONS ARE IMMUTABLE AND POLICY-VERSIONED. A benchmark result must be
--    reproducible as (evidence + policy version + objective) -> conclusion. If
--    a customer relaxes quality >= 0.95 to >= 0.94 tomorrow, yesterday's
--    'candidates_failed_policy' may become today's 'safe_improvement_found' —
--    and yesterday's row MUST NOT change. Re-evaluation writes a NEW
--    benchmark_conclusions row against the new policy version. Policies are
--    versioned and edited by INSERT, never in place.
--
-- 4. CANDIDATE RESULTS SURVIVE THEIR CONCLUSION. A candidate that saved 51% but
--    landed 0.7pp under the quality floor is commercially valuable evidence. It
--    is stored in its own queryable table and is NOT discarded because the
--    overall verdict was 'candidates_failed_policy'. When the customer relaxes
--    the threshold, OptiML re-reads it instead of re-measuring everything.
--
-- 5. EVIDENCE PRECEDES RECOMMENDATION, MANY-TO-MANY.
--        Workload
--          |- Benchmark  (baseline, candidates, evidence, conclusion)
--          `- Optimization Recommendation  <- produced FROM qualifying evidence
--    Benchmarks discover FACTS. Recommendations propose ACTIONS. A benchmark may
--    live forever with no recommendation; a recommendation must point back to
--    the evidence that justified it. SEVERAL benchmarks may support ONE
--    recommendation, and ONE benchmark may lead to MULTIPLE recommendations, so
--    the linkage is a join table. optimization_benchmarks.recommendation_id is
--    DEPRECATED (retained, never written) in favour of recommendation_evidence.
-- ============================================================================


-- ============================================================================
-- 1. optimization_policies — versioned, immutable history.
-- ============================================================================
ALTER TABLE public.optimization_policies
    ADD COLUMN IF NOT EXISTS policy_key    UUID,
    ADD COLUMN IF NOT EXISTS version       INT  NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS is_current    BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS supersedes_policy_id UUID REFERENCES public.optimization_policies(id) ON DELETE SET NULL;

-- Backfill policy_key for rows created before versioning existed: each becomes
-- the first version of its own lineage.
UPDATE public.optimization_policies SET policy_key = id WHERE policy_key IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_optpolicy_key_version
    ON public.optimization_policies (policy_key, version);
CREATE UNIQUE INDEX IF NOT EXISTS idx_optpolicy_key_current
    ON public.optimization_policies (policy_key) WHERE is_current;
CREATE INDEX IF NOT EXISTS idx_optpolicy_org_current
    ON public.optimization_policies (org_id, is_current, priority);

COMMENT ON COLUMN public.optimization_policies.policy_key IS
  'Stable identity of a policy across all of its versions. `id` identifies ONE VERSION; `policy_key` identifies the policy. Benchmarks reference the version they were judged under, never the lineage.';
COMMENT ON COLUMN public.optimization_policies.version IS
  'Monotonic version within a policy_key lineage. Editing a policy INSERTS a new version and marks the previous one is_current=false; rows are NEVER updated in place, because historical conclusions must remain reproducible.';
COMMENT ON COLUMN public.optimization_policies.is_current IS
  'Exactly one version per policy_key is current. Evaluation uses the current version; audit reads the whole lineage.';

COMMENT ON COLUMN public.optimization_policies.materiality IS
  'What counts as "enough to justify a change", expressed against the OBJECTIVE''s own metric and units — not as savings.
   Shape: {"thresholds":[{"metric":"cost","comparator":"relative_decrease_at_least","value":0.05,"unit":"ratio"},
                         {"metric":"cost","comparator":"absolute_decrease_at_least","value":1000,"unit":"usd_per_month"}],
           "combine":"any"}
   metric:     cost | latency_p95_ms | quality | outcome_rate | error_rate | custom
   comparator: relative_decrease_at_least | relative_increase_at_least |
               absolute_decrease_at_least | absolute_increase_at_least
   unit:       ratio | usd_per_month | usd_per_task | ms | percentage_points | score
   combine:    any (OR) | all (AND)
   Examples: minimize cost -> >=5% reduction OR >=$1,000/month. minimize latency
   -> >=150ms p95 improvement. maximize resolution rate -> >=1.5 percentage points.
   The legacy savings-shaped keys min_relative_improvement /
   min_absolute_monthly_savings_usd are accepted as SUGAR and normalised into
   cost thresholds by optimization/domain.py. They are not the domain model.';


-- ============================================================================
-- 2. benchmark_candidate_results — evidence that outlives its interpretation.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.benchmark_candidate_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    benchmark_id        UUID NOT NULL REFERENCES public.optimization_benchmarks(id) ON DELETE CASCADE,
    workload_id         UUID NOT NULL REFERENCES public.workloads(id) ON DELETE CASCADE,

    strategy_id         UUID REFERENCES public.execution_strategies(id) ON DELETE SET NULL,
    strategy_fingerprint TEXT,
    arm                 TEXT NOT NULL DEFAULT 'candidate'
                        CHECK (arm IN ('baseline', 'candidate')),
    label               TEXT,
    generator           TEXT,
    dimensions          TEXT[],
    executor_refs       JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- MEASURED. Every one of these is nullable and NULL means "not measured".
    sample_size         INT,
    mean_cost_usd       NUMERIC(18, 8),
    total_cost_usd      NUMERIC(18, 8),
    latency_p50_ms      INT,
    latency_p95_ms      INT,
    error_rate          NUMERIC(6, 4),
    quality             NUMERIC(6, 4),
    quality_provenance  TEXT NOT NULL DEFAULT 'unknown',
    outcome_metrics     JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Deltas vs the baseline arm of the same benchmark.
    cost_delta_pct      NUMERIC(10, 4),
    latency_delta_pct   NUMERIC(10, 4),
    quality_delta       NUMERIC(10, 6),

    evidence_source     TEXT NOT NULL DEFAULT 'replay',
    per_case_results    JSONB,
    error               TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bcr_org_created
    ON public.benchmark_candidate_results (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bcr_benchmark
    ON public.benchmark_candidate_results (benchmark_id);
CREATE INDEX IF NOT EXISTS idx_bcr_workload_created
    ON public.benchmark_candidate_results (workload_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bcr_fingerprint
    ON public.benchmark_candidate_results (org_id, strategy_fingerprint);
-- "Show me every candidate we ever measured for this workload that beat
-- baseline on cost, regardless of what we concluded at the time."
CREATE INDEX IF NOT EXISTS idx_bcr_workload_cost_delta
    ON public.benchmark_candidate_results (workload_id, cost_delta_pct);

COMMENT ON TABLE public.benchmark_candidate_results IS
  'One measured arm of a benchmark, stored INDEPENDENTLY of any conclusion. A conclusion is an interpretation of evidence under a particular policy version; the evidence itself must survive that interpretation. A candidate that saved 51%% but missed the quality floor by 0.7pp is retained here as a first-class result, so that relaxing the threshold later is a re-read rather than a re-measurement.';
COMMENT ON COLUMN public.benchmark_candidate_results.arm IS
  'baseline = the configuration currently in force. candidate = a proposed alternative. Both are measured over the SAME inputs in the same run — that sameness is what makes this a counterfactual rather than an observation.';
COMMENT ON COLUMN public.benchmark_candidate_results.quality IS
  'Measured quality 0..1, or NULL. NULL means quality was not measurable in this run — never a substituted guess. Read together with quality_provenance.';
COMMENT ON COLUMN public.benchmark_candidate_results.outcome_metrics IS
  'Named outcome aggregates for this arm, kept separate per provenance so incompatible signals are never averaged: {"ticket_resolved":{"provenance":"business_outcome","rate":0.83,"n":120}}.';


-- ============================================================================
-- 3. benchmark_conclusions — immutable, policy-versioned interpretations.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.benchmark_conclusions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    benchmark_id        UUID NOT NULL REFERENCES public.optimization_benchmarks(id) ON DELETE CASCADE,
    workload_id         UUID NOT NULL REFERENCES public.workloads(id) ON DELETE CASCADE,

    policy_id           UUID REFERENCES public.optimization_policies(id) ON DELETE SET NULL,
    policy_key          UUID,
    policy_version      INT,
    objective           TEXT NOT NULL DEFAULT 'cost'
                        CHECK (objective IN ('cost', 'quality', 'latency', 'balanced', 'custom')),

    conclusion          TEXT NOT NULL
                        CHECK (conclusion IN ('safe_improvement_found', 'no_material_improvement',
                                              'candidates_failed_policy', 'insufficient_evidence',
                                              'benchmark_failed')),
    reasons             JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence          NUMERIC(4, 3) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    confidence_band     TEXT CHECK (confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')),

    materiality_applied JSONB NOT NULL DEFAULT '{}'::jsonb,
    success_signal      JSONB NOT NULL DEFAULT '{}'::jsonb,
    more_data_changes_conclusion TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (more_data_changes_conclusion IN ('yes', 'no', 'unknown')),
    more_data_reasons   JSONB NOT NULL DEFAULT '[]'::jsonb,

    selected_candidate_result_id UUID REFERENCES public.benchmark_candidate_results(id) ON DELETE SET NULL,

    is_current          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bconc_org_created
    ON public.benchmark_conclusions (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bconc_benchmark
    ON public.benchmark_conclusions (benchmark_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bconc_workload_current
    ON public.benchmark_conclusions (workload_id, conclusion) WHERE is_current;
CREATE UNIQUE INDEX IF NOT EXISTS idx_bconc_benchmark_policy_version
    ON public.benchmark_conclusions (benchmark_id, COALESCE(policy_id, '00000000-0000-0000-0000-000000000000'::uuid), objective);

COMMENT ON TABLE public.benchmark_conclusions IS
  'An IMMUTABLE interpretation of one benchmark''s evidence under ONE policy version and ONE objective. Rows are never updated. Re-evaluating the same evidence under a relaxed policy INSERTS a new row and flips is_current on the old one; the original verdict and the policy version that produced it remain readable forever. This auditability is what makes realized-savings claims and production decisions defensible.';
COMMENT ON COLUMN public.benchmark_conclusions.reasons IS
  'Structured facts, NOT prose: [{"code":"sample_size_below_threshold","observed":82,"required":500},{"code":"quality_below_threshold","constraint":"min_quality","observed":0.943,"required":0.95,"unit":"score","shortfall":0.007}]. Codes are a stable documented vocabulary (see optimization/domain.py REASON_CODES). The backend returns codes and facts; ALL customer-facing wording is derived by the frontend. No user-facing sentence is ever returned from the API.';
COMMENT ON COLUMN public.benchmark_conclusions.confidence_band IS
  'Coarse band derived from confidence, for callers that need a stable categorical. NULL when confidence could not be computed.';
COMMENT ON COLUMN public.benchmark_conclusions.is_current IS
  'TRUE for the latest evaluation of this benchmark under the current policy. Older evaluations stay in the table with is_current=false and are never modified.';


-- ============================================================================
-- 4. recommendation_evidence — many-to-many: evidence precedes recommendation.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.recommendation_evidence (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    recommendation_id   UUID NOT NULL REFERENCES public.optimization_recommendations(id) ON DELETE CASCADE,
    benchmark_id        UUID NOT NULL REFERENCES public.optimization_benchmarks(id) ON DELETE CASCADE,
    evidence_role       TEXT NOT NULL DEFAULT 'primary'
                        CHECK (evidence_role IN ('primary', 'supporting')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_recev_unique
    ON public.recommendation_evidence (recommendation_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_recev_recommendation
    ON public.recommendation_evidence (recommendation_id);
CREATE INDEX IF NOT EXISTS idx_recev_benchmark
    ON public.recommendation_evidence (benchmark_id);
CREATE INDEX IF NOT EXISTS idx_recev_org
    ON public.recommendation_evidence (org_id);

COMMENT ON TABLE public.recommendation_evidence IS
  'Many-to-many linkage from a recommendation to the benchmark evidence it CITES. This is not an abstraction over "evidence objects" in general: today the only concrete evidence object is a benchmark, and generalising before canaries, production observations and human reviews are real citable objects would be speculation. The join exists to correct a CAUSAL DIRECTION, not to be extensible: a recommendation_id column on optimization_benchmarks would encode "recommendation owns benchmark", when the real semantics are "benchmark produces evidence; recommendation cites evidence". Several benchmarks may support one recommendation and one benchmark may yield several recommendations, which no single column on either side can express.';
COMMENT ON COLUMN public.recommendation_evidence.evidence_role IS
  'Which cited benchmark is the decisive one when a recommendation cites several (a replay plus a later shadow run). Kept minimal on purpose: exactly primary and supporting.';


COMMENT ON COLUMN public.optimization_benchmarks.recommendation_id IS
  'DEPRECATED. Retained so no historical row loses data; never written by current code. The linkage is public.recommendation_evidence, which is many-to-many. A benchmark''s required owner is workload_id: benchmarks discover facts about WORK, and may exist forever without a recommendation.';


-- ============================================================================
-- 5. The conclusion columns on optimization_benchmarks are now a MIRROR of the
--    current benchmark_conclusions row, kept for cheap single-row reads.
-- ============================================================================
COMMENT ON COLUMN public.optimization_benchmarks.conclusion IS
  'Denormalised mirror of the CURRENT public.benchmark_conclusions row for this benchmark. The record of truth — and the entire evaluation history — is benchmark_conclusions. Written only by optimization/benchmark.py at the same time as the conclusion row. Never edit this column directly: a historical conclusion must remain reproducible as (evidence + policy version + objective).';
COMMENT ON COLUMN public.optimization_benchmarks.conclusion_detail IS
  'Denormalised mirror of the current conclusion''s structured `reasons` array. Codes and facts only — never customer-facing prose.';


-- ============================================================================
-- 6. RLS on the new tables.
-- ============================================================================
ALTER TABLE public.benchmark_candidate_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.benchmark_conclusions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.recommendation_evidence     ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Org members can view benchmark candidate results" ON public.benchmark_candidate_results;
CREATE POLICY "Org members can view benchmark candidate results"
    ON public.benchmark_candidate_results FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view benchmark conclusions" ON public.benchmark_conclusions;
CREATE POLICY "Org members can view benchmark conclusions"
    ON public.benchmark_conclusions FOR SELECT
    USING (public.is_org_member(org_id));

DROP POLICY IF EXISTS "Org members can view recommendation evidence" ON public.recommendation_evidence;
CREATE POLICY "Org members can view recommendation evidence"
    ON public.recommendation_evidence FOR SELECT
    USING (public.is_org_member(org_id));


-- ============================================================================
-- Verify — Optimization Coverage, in SPEND terms, not workload counts.
--
-- Coverage = the share of eligible workload spend for which OptiML has
-- sufficient evidence to make an optimization determination.
--   COVERED     : safe_improvement_found, no_material_improvement,
--                 candidates_failed_policy
--   NOT COVERED : insufficient_evidence, benchmark_failed
-- ============================================================================
-- WITH spend AS (
--   SELECT a.workload_id, SUM(a.inference_cost_usd) AS monthly_usd
--     FROM public.attempts a
--    WHERE a.org_id = '00000000-0000-0000-0000-000000000000'
--      AND a.execution_mode = 'production'
--      AND a.occurred_at >= now() - interval '30 days'
--    GROUP BY 1
-- ), verdict AS (
--   SELECT DISTINCT ON (c.workload_id) c.workload_id, c.conclusion
--     FROM public.benchmark_conclusions c
--    WHERE c.org_id = '00000000-0000-0000-0000-000000000000' AND c.is_current
--    ORDER BY c.workload_id, c.created_at DESC
-- )
-- SELECT
--   SUM(s.monthly_usd) FILTER (
--     WHERE v.conclusion IN ('safe_improvement_found','no_material_improvement','candidates_failed_policy')
--   ) AS assessable_usd,
--   SUM(s.monthly_usd) FILTER (
--     WHERE v.conclusion IS NULL OR v.conclusion IN ('insufficient_evidence','benchmark_failed')
--   ) AS awaiting_evidence_usd
-- FROM spend s LEFT JOIN verdict v USING (workload_id);
