-- Add metadata JSONB column to sm_observations for SM v2 Phase 2-4 data.
-- Stores scope_value, variable_hash, variable_text, kb_asset_ids, agent_tool_calls.
-- Run against Supabase SQL editor.

ALTER TABLE sm_observations ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

-- Index for finding observations by scope_value (Phase 4 consolidation)
CREATE INDEX IF NOT EXISTS idx_sm_observations_metadata_scope
    ON sm_observations USING gin (metadata jsonb_path_ops);

COMMENT ON COLUMN sm_observations.metadata IS 'Structured metadata for SM v2: scope_value, variable_hash, variable_text, kb_asset_ids, agent_tool_calls';
