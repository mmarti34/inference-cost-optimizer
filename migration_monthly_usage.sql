-- Migration: monthly_usage table for plan enforcement
-- Tracks API request counts per organization per month.
-- Used by plan_enforcement.py to enforce monthly request limits.

-- 1. Create the monthly_usage table
CREATE TABLE IF NOT EXISTS monthly_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  month TEXT NOT NULL,  -- 'YYYY-MM' format (e.g. '2026-03')
  request_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(org_id, month)
);

-- 2. Index for fast lookups by org + month
CREATE INDEX IF NOT EXISTS idx_monthly_usage_org_month
  ON monthly_usage(org_id, month);

-- 3. Atomic increment function (avoids read-then-write race)
--    Returns the new request_count after incrementing.
CREATE OR REPLACE FUNCTION increment_monthly_usage(p_org_id UUID, p_month TEXT)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
  new_count INTEGER;
BEGIN
  INSERT INTO monthly_usage (org_id, month, request_count)
  VALUES (p_org_id, p_month, 1)
  ON CONFLICT (org_id, month)
  DO UPDATE SET request_count = monthly_usage.request_count + 1
  RETURNING request_count INTO new_count;

  RETURN new_count;
END;
$$;

-- 4. RLS: disable for this table (backend uses service role only)
ALTER TABLE monthly_usage ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
CREATE POLICY "Service role full access on monthly_usage"
  ON monthly_usage
  FOR ALL
  USING (true)
  WITH CHECK (true);
