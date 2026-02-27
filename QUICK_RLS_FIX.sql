-- QUICK FIX: Disable RLS for backend-managed tables
-- Run this in Supabase SQL Editor to fix projects, prompts, and API keys issues
-- The backend handles authorization, so RLS is not needed for these tables

-- Disable RLS for projects
ALTER TABLE projects DISABLE ROW LEVEL SECURITY;

-- Disable RLS for prompt_templates
ALTER TABLE prompt_templates DISABLE ROW LEVEL SECURITY;

-- Disable RLS for api_keys
ALTER TABLE api_keys DISABLE ROW LEVEL SECURITY;

-- Disable RLS for service_api_keys
ALTER TABLE service_api_keys DISABLE ROW LEVEL SECURITY;

-- Verify RLS is disabled
SELECT 
  schemaname,
  tablename,
  rowsecurity as rls_enabled
FROM pg_tables
WHERE tablename IN ('projects', 'prompt_templates', 'api_keys', 'service_api_keys')
AND schemaname = 'public';

-- Expected result: rls_enabled should be false for all tables

