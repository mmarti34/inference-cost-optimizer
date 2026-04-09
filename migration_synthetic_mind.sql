-- Synthetic Mind Phase 1: Observations + Memory Units
-- Run against Supabase SQL editor.

-- ─── Observations ────────────────────────────────────────────────
-- Raw structured observations extracted from every public API call.
-- Processed by the consolidation engine into memory units.

CREATE TABLE IF NOT EXISTS sm_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID REFERENCES api_request_log(id) ON DELETE SET NULL,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workflow_id UUID REFERENCES workflows(id) ON DELETE SET NULL,
    endpoint_slug TEXT,
    user_id UUID,
    session_id TEXT,
    observation_type TEXT NOT NULL DEFAULT 'response'
        CHECK (observation_type IN ('request', 'response', 'feedback', 'error')),
    input_summary TEXT,
    output_summary TEXT,
    entities_mentioned JSONB DEFAULT '[]'::jsonb,
    model TEXT,
    provider TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL,
    total_latency_ms INTEGER,
    error_message TEXT,
    consolidated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sm_obs_org_created
    ON sm_observations(org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sm_obs_workflow
    ON sm_observations(workflow_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sm_obs_unconsolidated
    ON sm_observations(org_id, consolidated) WHERE consolidated = FALSE;
CREATE INDEX IF NOT EXISTS idx_sm_obs_endpoint
    ON sm_observations(org_id, endpoint_slug, created_at DESC);

COMMENT ON TABLE sm_observations IS 'Synthetic Mind: raw observations from public API calls.';


-- ─── Memory Units ────────────────────────────────────────────────
-- Consolidated understanding derived from observations.
-- Each unit is scoped to org/team/workflow/user and has decaying confidence.

CREATE TABLE IF NOT EXISTS sm_memory_units (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type TEXT NOT NULL
        CHECK (memory_type IN ('entity', 'relation', 'abstraction', 'procedure', 'preference', 'fact')),
    -- Hierarchical scope
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    team_id UUID,
    workflow_id UUID REFERENCES workflows(id) ON DELETE SET NULL,
    user_id UUID,
    session_id TEXT,
    -- Content
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object JSONB NOT NULL,
    evidence JSONB DEFAULT '[]'::jsonb,       -- array of observation IDs
    -- Confidence + lifecycle
    confidence DECIMAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMPTZ DEFAULT now(),
    last_confirmed_at TIMESTAMPTZ DEFAULT now(),
    confirmed_count INTEGER DEFAULT 1,
    supersedes UUID REFERENCES sm_memory_units(id) ON DELETE SET NULL,
    protected BOOLEAN DEFAULT FALSE,
    -- Causal graph edges stored as JSONB array
    causal_edges JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    -- Soft delete
    forgotten_at TIMESTAMPTZ,
    forgotten_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_sm_mem_org_type
    ON sm_memory_units(org_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_sm_mem_workflow
    ON sm_memory_units(workflow_id) WHERE workflow_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sm_mem_confidence
    ON sm_memory_units(org_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_sm_mem_active
    ON sm_memory_units(org_id, forgotten_at) WHERE forgotten_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sm_mem_subject
    ON sm_memory_units(org_id, subject);

COMMENT ON TABLE sm_memory_units IS 'Synthetic Mind: consolidated understanding with hierarchical scopes and confidence decay.';


-- ─── Memory Scopes ───────────────────────────────────────────────
-- Registry of active memory scopes for inheritance resolution.

CREATE TABLE IF NOT EXISTS sm_memory_scopes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope_type TEXT NOT NULL
        CHECK (scope_type IN ('platform', 'organization', 'team', 'workflow', 'user', 'session')),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    parent_scope_id UUID REFERENCES sm_memory_scopes(id) ON DELETE SET NULL,
    -- Polymorphic reference to the scoped entity
    entity_id UUID,         -- workflow_id, user_id, team_id, etc.
    entity_slug TEXT,       -- human-readable identifier
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sm_scopes_org
    ON sm_memory_scopes(org_id, scope_type);
CREATE INDEX IF NOT EXISTS idx_sm_scopes_parent
    ON sm_memory_scopes(parent_scope_id);

COMMENT ON TABLE sm_memory_scopes IS 'Synthetic Mind: hierarchical scope registry for memory inheritance.';


-- ─── Consolidation Runs ──────────────────────────────────────────
-- Audit trail of consolidation engine executions.

CREATE TABLE IF NOT EXISTS sm_consolidation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    observations_processed INTEGER DEFAULT 0,
    memory_units_created INTEGER DEFAULT 0,
    memory_units_updated INTEGER DEFAULT 0,
    abstractions_formed INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_sm_consol_org
    ON sm_consolidation_runs(org_id, started_at DESC);

COMMENT ON TABLE sm_consolidation_runs IS 'Synthetic Mind: audit trail for consolidation engine runs.';


-- ─── RLS Policies ────────────────────────────────────────────────
-- Enable RLS on all Synthetic Mind tables.

ALTER TABLE sm_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE sm_memory_units ENABLE ROW LEVEL SECURITY;
ALTER TABLE sm_memory_scopes ENABLE ROW LEVEL SECURITY;
ALTER TABLE sm_consolidation_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view observations"
    ON sm_observations FOR SELECT
    USING (public.is_org_member(org_id));

CREATE POLICY "Org members can view memory units"
    ON sm_memory_units FOR SELECT
    USING (public.is_org_member(org_id));

CREATE POLICY "Org members can view memory scopes"
    ON sm_memory_scopes FOR SELECT
    USING (public.is_org_member(org_id));

CREATE POLICY "Org members can view consolidation runs"
    ON sm_consolidation_runs FOR SELECT
    USING (public.is_org_member(org_id));
