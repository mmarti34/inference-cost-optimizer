-- Synthetic Mind v2 Phase 4: Variable Consolidations
-- Stores consolidated summaries of repeated variable data per scope.
-- When the same scope_value (e.g. concert_id) sends the same variable content
-- 3+ times, the SM consolidates it into a compact summary.
-- Run against Supabase SQL editor.

CREATE TABLE IF NOT EXISTS sm_variable_consolidations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    endpoint_slug TEXT NOT NULL,
    scope_value TEXT NOT NULL,
    consolidated_text TEXT NOT NULL,
    raw_token_count INTEGER,
    consolidated_token_count INTEGER,
    confidence DECIMAL DEFAULT 0.8
        CHECK (confidence >= 0 AND confidence <= 1),
    hit_count INTEGER DEFAULT 0,
    variable_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sm_var_consol_scope
    ON sm_variable_consolidations(org_id, endpoint_slug, scope_value);
CREATE INDEX IF NOT EXISTS idx_sm_var_consol_confidence
    ON sm_variable_consolidations(org_id, confidence DESC);

COMMENT ON TABLE sm_variable_consolidations IS 'Synthetic Mind v2: consolidated variable data per scope value, replacing raw variable content with compact summaries.';

-- RLS
ALTER TABLE sm_variable_consolidations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view variable consolidations"
    ON sm_variable_consolidations FOR SELECT
    USING (public.is_org_member(org_id));
