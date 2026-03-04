-- Org-level secrets store for tool definitions and webhook auth.
-- Secrets are encrypted with AES-256-GCM (same pattern as api_keys table).
-- Referenced in graph_json via {{secrets.NAME}} syntax, resolved at runtime.

CREATE TABLE IF NOT EXISTS org_secrets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    encrypted_value TEXT NOT NULL,       -- AES-256-GCM encrypted
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_org_secrets_org_name UNIQUE (org_id, name)
);

-- RLS: service role only (backend manages access via auth middleware)
ALTER TABLE org_secrets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on org_secrets"
    ON org_secrets
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Index for fast lookups by org_id
CREATE INDEX IF NOT EXISTS idx_org_secrets_org_id ON org_secrets(org_id);
