-- Projects-first: workflows belong to a project; workflow_deployments denormalize project_id.
-- Run after workflows and workflow_deployments exist.

-- 1. Ensure projects table exists (id, org_id, name, ...)
-- (Already in create_missing_tables.sql)

-- 2. Add project_id and slug to workflows (nullable for migration; new workflows require project_id)
ALTER TABLE workflows ADD COLUMN IF NOT EXISTS project_id UUID REFERENCES projects(id) ON DELETE CASCADE;
ALTER TABLE workflows ADD COLUMN IF NOT EXISTS slug TEXT;
ALTER TABLE workflows ADD COLUMN IF NOT EXISTS variables JSONB DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_workflows_project_id ON workflows(project_id);
COMMENT ON COLUMN workflows.project_id IS 'Project this workflow belongs to.';
COMMENT ON COLUMN workflows.slug IS 'URL-safe identifier for this workflow endpoint (per workflow).';
COMMENT ON COLUMN workflows.variables IS 'Workflow variables: [{ name, type, required, description }].';

-- 3. Add project_id to workflow_deployments (denormalized for fast lookup)
ALTER TABLE workflow_deployments ADD COLUMN IF NOT EXISTS project_id UUID REFERENCES projects(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_workflow_deployments_project_id ON workflow_deployments(project_id);
