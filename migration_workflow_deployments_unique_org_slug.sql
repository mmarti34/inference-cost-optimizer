-- Enforce UNIQUE(org_id, endpoint_slug) for workflow_deployments (enterprise hardening).
-- Run after workflow_deployments and organizations exist.

-- 1. Backfill endpoint_slug if any rows have null (defensive; column is NOT NULL in schema)
UPDATE workflow_deployments
SET endpoint_slug = 'workflow-' || workflow_id::text || '-v' || version
WHERE endpoint_slug IS NULL OR trim(endpoint_slug) = '';

-- 2. Deduplicate: for (org_id, endpoint_slug) with multiple rows, keep newest by created_at (then version), suffix others
WITH numbered AS (
  SELECT id, org_id, endpoint_slug, version, created_at,
    row_number() OVER (
      PARTITION BY org_id, endpoint_slug
      ORDER BY created_at DESC NULLS LAST, version DESC
    ) AS rn
  FROM workflow_deployments
)
UPDATE workflow_deployments w
SET endpoint_slug = w.endpoint_slug || '--dup-' || w.version
FROM numbered n
WHERE w.id = n.id AND n.rn > 1;

-- 3. Add unique constraint (skip if already exists)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'workflow_deployments_org_slug_unique'
  ) THEN
    ALTER TABLE workflow_deployments
    ADD CONSTRAINT workflow_deployments_org_slug_unique UNIQUE (org_id, endpoint_slug);
  END IF;
END $$;

COMMENT ON CONSTRAINT workflow_deployments_org_slug_unique ON workflow_deployments IS
  'Endpoint slug is unique per organization; deploy returns 409 on conflict.';
