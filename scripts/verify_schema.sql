-- verify_schema.sql
-- Run in Supabase SQL Editor to confirm which schema each table lives in.
-- PGRST104/PGRST106 mean PostgREST cannot see the schema; tables must be in an exposed schema (e.g. public).

-- 1) List schema and table for all app tables
SELECT
  table_schema,
  table_name
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
  AND table_name IN (
    'api_keys',
    'service_api_keys',
    'projects',
    'prompt_templates',
    'organizations',
    'organization_members',
    'user_profiles',
    'join_requests',
    'usage_logs',
    'optimizer_recommendations',
    'request_logs',
    'request_spans',
    'model_stats_daily',
    'replay_logs'
  )
ORDER BY table_schema, table_name;

-- 2) Tables that are NOT in public (investigate if any appear)
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
  AND table_name IN (
    'api_keys', 'service_api_keys', 'projects', 'prompt_templates',
    'organizations', 'organization_members', 'user_profiles', 'join_requests',
    'usage_logs', 'optimizer_recommendations', 'request_logs', 'request_spans',
    'model_stats_daily', 'replay_logs'
  )
  AND table_schema != 'public';

-- 3) PostgREST exposed schemas (run as superuser or check Dashboard)
-- Supabase exposes schemas in: Project Settings -> API -> Exposed schemas (e.g. public, storage, graphql_public)
-- This query only shows where tables are; it cannot show API "exposed schemas" setting.
