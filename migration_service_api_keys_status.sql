-- Server API key status: active-only enforcement (enterprise hardening).
-- Idempotent; safe to run after migration_service_api_keys_hardening.sql.

ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
UPDATE service_api_keys SET status = 'active' WHERE status IS NULL;
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'service_api_keys_status_check') THEN
    ALTER TABLE service_api_keys ADD CONSTRAINT service_api_keys_status_check
      CHECK (status IN ('active', 'revoked'));
  END IF;
END $$;
ALTER TABLE service_api_keys ALTER COLUMN status SET NOT NULL;

COMMENT ON COLUMN service_api_keys.status IS 'active = valid for auth; revoked = rejected at verification.';
