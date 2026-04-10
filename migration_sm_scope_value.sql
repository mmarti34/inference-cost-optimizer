-- Synthetic Mind v2: Add scope_value for per-entity context consolidation.
-- When a workflow has a scope_key configured (e.g. "concert_id", "customer_id"),
-- the resolved value is stored here so consolidations are scoped per-entity
-- instead of shared across all calls.
-- Run against Supabase SQL editor.

-- Add scope_value column (nullable — NULL means "shared across all calls")
ALTER TABLE sm_kb_consolidations
    ADD COLUMN IF NOT EXISTS scope_value TEXT;

-- Index for scoped lookups
CREATE INDEX IF NOT EXISTS idx_sm_kb_consol_scope
    ON sm_kb_consolidations(org_id, scope_value);

COMMENT ON COLUMN sm_kb_consolidations.scope_value IS 'Entity scope value (e.g. concert_id, customer_id). NULL = shared across all calls.';
