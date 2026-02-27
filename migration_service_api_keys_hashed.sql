-- Universal API keys: store only hashed_key, never plaintext.
-- Run in Supabase SQL Editor. New keys use hashed_key; existing api_key column kept for legacy.

-- Add hashed_key column if missing (nullable for backward compat)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'service_api_keys' AND column_name = 'hashed_key'
  ) THEN
    ALTER TABLE service_api_keys ADD COLUMN hashed_key TEXT;
  END IF;
END $$;

-- Optional: add project_id later for "one key per project"
-- ALTER TABLE service_api_keys ADD COLUMN IF NOT EXISTS project_id UUID REFERENCES projects(id);

-- Index for fast lookup by hashed_key (for Authorization: Bearer verification)
CREATE INDEX IF NOT EXISTS idx_service_api_keys_hashed_key ON service_api_keys(hashed_key);

-- Allow api_key to be NULL for new rows that use hashed_key only
ALTER TABLE service_api_keys ALTER COLUMN api_key DROP NOT NULL;

-- Keep api_key column for legacy rows; new rows should set hashed_key and leave api_key NULL.
