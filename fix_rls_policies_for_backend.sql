-- Fix RLS policies to allow backend service role to access data
-- The backend uses SERVICE_ROLE_KEY which should bypass RLS, but we need to ensure
-- the policies don't block service role operations

-- For projects: Allow service role to bypass RLS
CREATE POLICY IF NOT EXISTS "Service role can manage projects" ON projects
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- For prompt_templates: Allow service role to bypass RLS  
CREATE POLICY IF NOT EXISTS "Service role can manage prompt templates" ON prompt_templates
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- For service_api_keys: Allow service role to bypass RLS
CREATE POLICY IF NOT EXISTS "Service role can manage service API keys" ON service_api_keys
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- For api_keys: Allow service role to bypass RLS
CREATE POLICY IF NOT EXISTS "Service role can manage API keys" ON api_keys
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Note: If the above doesn't work, we may need to disable RLS for these tables
-- or use a different approach. The service role key should bypass RLS by default,
-- but if it's not working, check:
-- 1. Is SUPABASE_KEY set to the service role key (not anon key)?
-- 2. Are the RLS policies correctly configured?

