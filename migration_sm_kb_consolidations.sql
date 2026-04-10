-- Synthetic Mind v2 Phase 2: KB Context Consolidations
-- Stores consolidated summaries of frequently-retrieved KB asset combinations.
-- Run against Supabase SQL editor.

CREATE TABLE IF NOT EXISTS sm_kb_consolidations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    asset_ids TEXT[] NOT NULL,
    consolidated_text TEXT NOT NULL,
    raw_token_count INTEGER,
    consolidated_token_count INTEGER,
    confidence DECIMAL DEFAULT 0.5
        CHECK (confidence >= 0 AND confidence <= 1),
    hit_count INTEGER DEFAULT 0,
    asset_versions_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sm_kb_consol_org
    ON sm_kb_consolidations(org_id);
CREATE INDEX IF NOT EXISTS idx_sm_kb_consol_asset_ids
    ON sm_kb_consolidations USING GIN (asset_ids);
CREATE INDEX IF NOT EXISTS idx_sm_kb_consol_confidence
    ON sm_kb_consolidations(org_id, confidence DESC);

COMMENT ON TABLE sm_kb_consolidations IS 'Synthetic Mind v2: consolidated KB context summaries replacing raw asset content.';

-- RLS
ALTER TABLE sm_kb_consolidations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view KB consolidations"
    ON sm_kb_consolidations FOR SELECT
    USING (public.is_org_member(org_id));
