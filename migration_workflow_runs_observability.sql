-- Add endpoint_slug, version, execution_mode to workflow_runs for observability
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS endpoint_slug TEXT;
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS version INTEGER;
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS execution_mode TEXT DEFAULT 'draft' CHECK (execution_mode IN ('draft', 'production'));

CREATE INDEX IF NOT EXISTS idx_workflow_runs_endpoint_slug ON workflow_runs(endpoint_slug);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_execution_mode ON workflow_runs(execution_mode);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_org_mode_created ON workflow_runs(org_id, execution_mode, created_at DESC);

COMMENT ON COLUMN workflow_runs.endpoint_slug IS 'Deployment endpoint slug for production runs; null for draft.';
COMMENT ON COLUMN workflow_runs.version IS 'Deployment version for production runs; null for draft.';
COMMENT ON COLUMN workflow_runs.execution_mode IS 'draft = Studio simulation; production = public endpoint.';
