-- Add optional name (and user_id) to api_keys if missing.
-- Run in Supabase SQL Editor. Safe to run multiple times.
-- After running, reload PostgREST schema: Project Settings -> API -> Reload schema cache (or run NOTIFY pgrst, 'reload schema').

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'api_keys' AND column_name = 'name'
  ) THEN
    ALTER TABLE api_keys ADD COLUMN name TEXT;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'api_keys' AND column_name = 'user_id'
  ) THEN
    ALTER TABLE api_keys ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;
  END IF;
END $$;
