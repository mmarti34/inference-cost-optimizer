-- Record which version was replaced when a rollback creates a new deployment.
-- When a new deployment is created with graph from an older version, set rolled_back_from_version = the previous current version.
ALTER TABLE workflow_deployments ADD COLUMN IF NOT EXISTS rolled_back_from_version INTEGER DEFAULT NULL;
