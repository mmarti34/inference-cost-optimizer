-- migration_context_assets.sql
-- Knowledge base: org-level context assets for workflow context injection

CREATE TABLE IF NOT EXISTS context_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id),
    name TEXT NOT NULL,
    description TEXT,
    asset_type TEXT NOT NULL DEFAULT 'text',
    content TEXT,
    source_ref JSONB,
    metadata JSONB DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_context_assets_org ON context_assets(org_id);
CREATE INDEX IF NOT EXISTS idx_context_assets_org_type ON context_assets(org_id, asset_type);
CREATE INDEX IF NOT EXISTS idx_context_assets_org_status ON context_assets(org_id, status);

-- No RLS: backend uses service role for all access (consistent with other OptiML tables)
