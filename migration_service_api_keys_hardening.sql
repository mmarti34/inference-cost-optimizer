-- Server API Keys hardening: named keys, status, last_used_at
-- Run after migration_service_api_keys_hashed.sql and migration_service_api_keys_rate_limit.sql

-- name: required for display; existing rows get default
ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS name TEXT;
UPDATE service_api_keys SET name = 'Server Key' WHERE name IS NULL;
ALTER TABLE service_api_keys ALTER COLUMN name SET DEFAULT 'Server Key';
-- Make NOT NULL after backfill (if any row still null, set again)
UPDATE service_api_keys SET name = 'Server Key' WHERE name IS NULL;
ALTER TABLE service_api_keys ALTER COLUMN name SET NOT NULL;

-- status: active | revoked; existing keys default active
ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
UPDATE service_api_keys SET status = 'active' WHERE status IS NULL;
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'service_api_keys_status_check'
  ) THEN
    ALTER TABLE service_api_keys ADD CONSTRAINT service_api_keys_status_check
      CHECK (status IN ('active', 'revoked'));
  END IF;
END $$;
ALTER TABLE service_api_keys ALTER COLUMN status SET NOT NULL;

-- last_used_at: set on successful validation
ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP WITH TIME ZONE;

COMMENT ON COLUMN service_api_keys.name IS 'Display name for the key (e.g. Production, Staging).';
COMMENT ON COLUMN service_api_keys.status IS 'active = valid for auth; revoked = rejected at verification.';
COMMENT ON COLUMN service_api_keys.last_used_at IS 'Updated on successful Bearer validation; not updated on failed attempts.';

CREATE INDEX IF NOT EXISTS idx_service_api_keys_org_status ON service_api_keys(org_id, status);
