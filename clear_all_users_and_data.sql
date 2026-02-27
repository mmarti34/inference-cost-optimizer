-- Clear all users and their data (organizations, projects, workflows, etc.)
-- Run this in Supabase SQL Editor. Order matters: delete children before parents.
-- WARNING: This permanently deletes all application data. Optionally clears auth.users.
-- If you get "relation does not exist" for request_spans/request_logs/model_stats_daily,
-- comment out those three DELETE lines (observability tables may not exist in all envs).

BEGIN;

-- 1. Observability / logging (comment out if these tables don't exist)
DELETE FROM request_spans;
DELETE FROM request_logs;
DELETE FROM model_stats_daily;

-- 2. Optimizer recommendations (refs prompt_templates, projects, orgs)
DELETE FROM optimizer_recommendations;

-- 3. Usage logs (refs org, project, user)
DELETE FROM usage_logs;

-- 4. Workflow execution data (refs workflows, orgs)
DELETE FROM workflow_runs;
DELETE FROM workflow_deployments;

-- 5. Workflows (refs orgs, projects)
DELETE FROM workflows;

-- 6. Prompt templates (refs org, project, user)
DELETE FROM prompt_templates;

-- 7. API keys (refs org, user)
DELETE FROM api_keys;
DELETE FROM service_api_keys;

-- 8. Join requests (refs org, user)
DELETE FROM join_requests;

-- 9. Projects (refs org, user)
DELETE FROM projects;

-- 10. Organization membership (refs org, user)
DELETE FROM organization_members;

-- 11. User profiles (refs auth.users)
DELETE FROM user_profiles;

-- 12. Organizations (refs auth.users)
DELETE FROM organizations;

-- 13. Auth users (Supabase sign-in accounts). Removes all users; sessions/identities
--     are typically removed by CASCADE. Use for full reset (e.g. dev/staging).
DELETE FROM auth.users;

COMMIT;
