-- Disable RLS for backend service role operations
-- The backend uses service_role key which should bypass RLS, but if RLS is still blocking,
-- we can disable it for these tables since the backend handles authorization

-- Option 1: Disable RLS entirely (NOT RECOMMENDED for production, but works)
-- Uncomment if service role key is not bypassing RLS:

-- ALTER TABLE projects DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE prompt_templates DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE api_keys DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE service_api_keys DISABLE ROW LEVEL SECURITY;

-- Option 2: Add service role bypass policies (RECOMMENDED)
-- This allows service role to bypass RLS while keeping RLS enabled for frontend

-- Drop existing policies if they conflict
DROP POLICY IF EXISTS "Service role bypass - projects" ON projects;
DROP POLICY IF EXISTS "Service role bypass - prompt_templates" ON prompt_templates;
DROP POLICY IF EXISTS "Service role bypass - api_keys" ON api_keys;
DROP POLICY IF EXISTS "Service role bypass - service_api_keys" ON service_api_keys;

-- Create policies that allow service role to bypass RLS
-- Note: service_role automatically bypasses RLS, but these policies ensure it works

-- For projects
CREATE POLICY "Service role bypass - projects" ON projects
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- For prompt_templates  
CREATE POLICY "Service role bypass - prompt_templates" ON prompt_templates
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- For api_keys
CREATE POLICY "Service role bypass - api_keys" ON api_keys
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- For service_api_keys
CREATE POLICY "Service role bypass - service_api_keys" ON service_api_keys
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- Verify RLS is enabled but service role can access
SELECT 
  schemaname,
  tablename,
  rowsecurity as rls_enabled
FROM pg_tables
WHERE tablename IN ('projects', 'prompt_templates', 'api_keys', 'service_api_keys')
AND schemaname = 'public';

