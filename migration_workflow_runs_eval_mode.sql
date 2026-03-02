-- Allow 'eval' execution_mode for workflow_runs (experiment A/B eval runs).
-- The original migration only allowed ('draft', 'production'), but eval runs
-- need execution_mode = 'eval'.
-- Also disable RLS on workflow_runs as belt-and-suspenders fix (same as
-- routing_latency_facts) — the backend uses the service role key.

-- 1. Drop the old CHECK constraint and re-add with 'eval'
ALTER TABLE workflow_runs DROP CONSTRAINT IF EXISTS workflow_runs_execution_mode_check;
ALTER TABLE workflow_runs ADD CONSTRAINT workflow_runs_execution_mode_check
  CHECK (execution_mode IN ('draft', 'production', 'eval'));

-- 2. Disable RLS (service role key bypasses anyway, but avoids any edge cases)
ALTER TABLE workflow_runs DISABLE ROW LEVEL SECURITY;

-- 3. Add experiment columns if missing (used by experiment traffic splitting)
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS experiment_id UUID;
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS variant_name TEXT;
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS served_version INTEGER;

CREATE INDEX IF NOT EXISTS idx_workflow_runs_experiment_id ON workflow_runs(experiment_id)
  WHERE experiment_id IS NOT NULL;
