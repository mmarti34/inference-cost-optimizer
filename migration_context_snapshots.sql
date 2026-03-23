-- migration_context_snapshots.sql
-- Snapshot asset content at deployment time for deterministic production execution

CREATE TABLE IF NOT EXISTS context_asset_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES context_assets(id),
    deployment_id UUID NOT NULL REFERENCES workflow_deployments(id),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_deployment ON context_asset_snapshots(deployment_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_asset_deployment ON context_asset_snapshots(asset_id, deployment_id);
