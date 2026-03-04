-- Webhook triggers: allow external services to trigger workflow executions.
-- Each webhook gets a unique URL path, optional secret for signature verification,
-- and can transform the incoming payload into workflow input.

CREATE TABLE IF NOT EXISTS webhook_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    name TEXT NOT NULL,
    endpoint_path TEXT NOT NULL,           -- Unique path segment (e.g., 'abc123')
    secret TEXT,                           -- Optional HMAC secret for signature verification
    payload_template TEXT DEFAULT '{{body}}', -- Template to extract input from webhook payload
    is_active BOOLEAN DEFAULT true,
    last_triggered_at TIMESTAMPTZ,
    trigger_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_webhook_endpoint_path UNIQUE (endpoint_path)
);

-- RLS: service role only
ALTER TABLE webhook_triggers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on webhook_triggers"
    ON webhook_triggers
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_webhook_triggers_org_id ON webhook_triggers(org_id);
CREATE INDEX IF NOT EXISTS idx_webhook_triggers_endpoint_path ON webhook_triggers(endpoint_path);
