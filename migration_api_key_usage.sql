-- Rate limiting: per-org, per-endpoint, per-minute bucket
CREATE TABLE IF NOT EXISTS api_key_usage (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  endpoint_slug TEXT NOT NULL,
  minute_bucket TIMESTAMP WITH TIME ZONE NOT NULL,
  request_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_api_key_usage_bucket
  ON api_key_usage(org_id, endpoint_slug, minute_bucket);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_org_id ON api_key_usage(org_id);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_minute_bucket ON api_key_usage(minute_bucket);

COMMENT ON TABLE api_key_usage IS 'Per-minute request counts for rate limiting public workflow endpoints.';
