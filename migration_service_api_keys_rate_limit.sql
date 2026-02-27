-- Add key_type and rate_limit_per_minute to service_api_keys (universal API keys)
-- Do not add updated_at; do not reference updated_at in app code.
ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS key_type TEXT DEFAULT 'live' CHECK (key_type IN ('live', 'test'));
ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS rate_limit_per_minute INTEGER DEFAULT 60;

COMMENT ON COLUMN service_api_keys.key_type IS 'live or test key.';
COMMENT ON COLUMN service_api_keys.rate_limit_per_minute IS 'Max requests per minute per org for public endpoint (default 60).';
