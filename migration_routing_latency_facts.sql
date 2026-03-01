-- Migration: routing_latency_facts — internal latency instrumentation
--
-- Purpose: quantify "OptiML middleman overhead" vs provider latency.
-- NOT user-facing. Written by workflow_runtime.py on every model/ai-step
-- execution.  Query with service-role for internal analysis.
--
-- Three latencies per call:
--   total_latency_ms    = wall-clock time from node execution start to end
--   provider_latency_ms = time spent on outbound provider request only
--   gateway_overhead_ms = total - provider (resolution, auth, serialization, retries backoff)
--
-- provider_latency_ms is nullable: only instrumented routers populate it
-- (openai_router first; others will follow).

CREATE TABLE IF NOT EXISTS routing_latency_facts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Dimensional keys for slicing
    org_id          UUID,
    workflow_run_id UUID,
    workflow_id     UUID,
    node_id         TEXT NOT NULL,
    node_type       TEXT NOT NULL,               -- 'model', 'ai-step', 'optimizer'

    -- Model target metadata
    target_type     TEXT,                         -- 'provider_model' | 'openai_compatible_endpoint' | null
    provider_label  TEXT NOT NULL,                -- 'openai', 'anthropic', 'custom:<host>'
    model_name      TEXT,
    endpoint_id     UUID,
    resolved_base_url TEXT,

    -- Routing context
    endpoint_slug   TEXT,                         -- production endpoint slug (null for draft)
    version         INT,                          -- deployment version (null for draft)
    execution_mode  TEXT,                         -- 'draft' | 'production' | 'eval'

    -- Latency measurements (ms, monotonic clock)
    total_latency_ms    INT NOT NULL,
    provider_latency_ms INT,                      -- null when router not yet instrumented
    gateway_overhead_ms INT,                      -- max(total - provider, 0); null when provider is null

    -- Token counts for correlation analysis
    input_tokens    INT,
    output_tokens   INT,

    -- Outcome
    success         BOOLEAN NOT NULL,
    error_type      TEXT,                         -- e.g. 'timeout', 'rate_limit', 'auth', 'provider_error'
    http_status     INT
);

-- Query patterns: time-series by provider, by org, by model, by target_type, by workflow
CREATE INDEX IF NOT EXISTS idx_rlf_ts                ON routing_latency_facts (ts);
CREATE INDEX IF NOT EXISTS idx_rlf_org_ts            ON routing_latency_facts (org_id, ts);
CREATE INDEX IF NOT EXISTS idx_rlf_provider_ts       ON routing_latency_facts (provider_label, ts);
CREATE INDEX IF NOT EXISTS idx_rlf_target_type_ts    ON routing_latency_facts (target_type, ts);
CREATE INDEX IF NOT EXISTS idx_rlf_workflow_ts       ON routing_latency_facts (workflow_id, ts);
CREATE INDEX IF NOT EXISTS idx_rlf_model_ts          ON routing_latency_facts (model_name, ts);

COMMENT ON TABLE routing_latency_facts IS 'Internal-only latency instrumentation. One row per model/ai-step node execution. Not exposed via API.';
