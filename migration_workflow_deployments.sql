-- Deployed versions per workflow (persisted deploy history)
CREATE TABLE IF NOT EXISTS workflow_deployments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  version INTEGER NOT NULL,
  endpoint_slug TEXT NOT NULL,
  graph_json JSONB NOT NULL DEFAULT '{"nodes":[],"edges":[]}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_deployments_workflow_version
  ON workflow_deployments(workflow_id, version);
CREATE INDEX IF NOT EXISTS idx_workflow_deployments_workflow_id ON workflow_deployments(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_deployments_org_id ON workflow_deployments(org_id);
CREATE INDEX IF NOT EXISTS idx_workflow_deployments_created_at ON workflow_deployments(workflow_id, created_at DESC);

COMMENT ON TABLE workflow_deployments IS 'Deployment history per workflow; version increments per workflow.';
