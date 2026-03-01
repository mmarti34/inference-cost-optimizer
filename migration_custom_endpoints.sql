-- Migration: custom_endpoints + model_registry tables
-- Supports two target types:
--   1. Custom model ID: model identifier string on a known provider (includes fine-tuned IDs).
--      Same API contract as the base provider; only the model string changes.
--   2. Custom endpoint: OpenAI-compatible API hosted elsewhere (vLLM, Anyscale, etc.).
--      Uses OpenAI router formatting with base_url + auth override.

-- 1. Custom endpoints: OpenAI-compatible servers with their own auth
CREATE TABLE IF NOT EXISTS custom_endpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,                          -- e.g. "https://my-vllm.example.com/v1"
    auth_type TEXT NOT NULL DEFAULT 'bearer'
        CHECK (auth_type IN ('bearer', 'header')),   -- bearer = "Authorization: Bearer <secret>"
    auth_header_name TEXT,                            -- only for auth_type='header'; e.g. "X-Api-Key"
    auth_secret_encrypted TEXT NOT NULL,              -- AES-256-GCM encrypted; same scheme as api_keys
    default_headers_json JSONB DEFAULT '{}'::jsonb,   -- extra headers merged into every request
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,

    -- Prevent duplicate endpoint names within an org
    CONSTRAINT uq_custom_endpoints_org_name UNIQUE (org_id, name)
);

CREATE INDEX IF NOT EXISTS idx_custom_endpoints_org_id ON custom_endpoints(org_id);

-- 2. Model registry: stable identifiers for model targets (both provider and custom-endpoint)
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,                               -- display name, e.g. "GPT-4o Fine-tuned (support)"
    target_type TEXT NOT NULL
        CHECK (target_type IN ('provider_model', 'openai_compatible_endpoint')),

    -- For provider_model targets
    provider TEXT,                                    -- "openai", "anthropic", etc. (lowercase)
    model_name TEXT,                                  -- model identifier sent to provider API

    -- For openai_compatible_endpoint targets
    endpoint_id UUID REFERENCES custom_endpoints(id) ON DELETE SET NULL,
    endpoint_model_name TEXT,                         -- model name to send to the custom endpoint

    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,

    -- At least one side must be populated
    CONSTRAINT model_registry_target_check CHECK (
        (target_type = 'provider_model' AND provider IS NOT NULL AND model_name IS NOT NULL)
        OR
        (target_type = 'openai_compatible_endpoint' AND endpoint_id IS NOT NULL AND endpoint_model_name IS NOT NULL)
    ),

    -- Prevent duplicate registry names within an org
    CONSTRAINT uq_model_registry_org_name UNIQUE (org_id, name)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_org_id ON model_registry(org_id);
CREATE INDEX IF NOT EXISTS idx_model_registry_endpoint_id ON model_registry(endpoint_id);

-- 3. RLS policies (disabled for backend service role, matching existing pattern)
-- The backend connects via the service role key which bypasses RLS.
-- If RLS is enabled on these tables later, add policies matching the api_keys pattern.

COMMENT ON TABLE custom_endpoints IS 'OpenAI-compatible custom endpoints with encrypted auth. Secrets use same AES-256-GCM as api_keys.';
COMMENT ON TABLE model_registry IS 'Stable model target identifiers. Nodes reference model_registry.id instead of raw provider+model strings.';
