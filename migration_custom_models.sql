-- Custom model configs: named LLM presets per org (display name, provider, base_model, defaults).
-- Run in Supabase SQL Editor. No select("*"); no updated_at in API responses.

CREATE TABLE IF NOT EXISTS custom_models (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  provider TEXT NOT NULL,
  base_model TEXT NOT NULL,
  temperature FLOAT DEFAULT 0.7,
  max_tokens INTEGER DEFAULT 1024,
  system_prefix TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_custom_models_org_id ON custom_models(org_id);

COMMENT ON TABLE custom_models IS 'User-defined LLM configs for Studio model nodes (display name, provider, base_model, defaults).';
