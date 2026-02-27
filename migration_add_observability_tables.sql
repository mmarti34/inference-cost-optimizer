-- Migration to add observability tables for request logging, tracing, and metrics

-- 1. Request logs table (one row per request)
CREATE TABLE IF NOT EXISTS request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL UNIQUE,  -- External request ID (from client or generated)
    trace_id UUID NOT NULL,  -- Groups related spans
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    user_identifier_hash TEXT,  -- Hashed user identifier for privacy
    prompt_id UUID REFERENCES prompt_templates(id) ON DELETE SET NULL,
    
    -- Routing information
    route_policy_name TEXT NOT NULL,
    route_policy_version TEXT NOT NULL,
    chosen_provider TEXT NOT NULL,
    chosen_model TEXT NOT NULL,
    reason_code TEXT,
    
    -- Token usage
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Cost and latency
    cost_usd DECIMAL(10, 6),
    latency_ms INTEGER,  -- Overall request latency
    
    -- Outcome
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'fallback_success', 'failure', 'refused')),
    failure_reason TEXT CHECK (failure_reason IN ('timeout', 'rate_limit', 'provider_5xx', 'invalid_json', 'tool_error', 'safety', 'unknown')),
    
    -- Validation results
    json_valid BOOLEAN,
    schema_valid BOOLEAN,
    
    -- Prompt storage (optional, controlled by env flag)
    prompt_text TEXT,  -- Only stored if OPTIML_STORE_PROMPTS=true
    response_text TEXT,  -- Only stored if OPTIML_STORE_PROMPTS=true
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Request spans table (one row per span/attempt)
CREATE TABLE IF NOT EXISTS request_spans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL REFERENCES request_logs(request_id) ON DELETE CASCADE,
    trace_id UUID NOT NULL,
    span_type TEXT NOT NULL CHECK (span_type IN ('gateway_receive', 'route_decision', 'provider_call', 'fallback_call', 'post_processing', 'validation')),
    provider TEXT,
    model TEXT,
    attempt_number INTEGER DEFAULT 1,  -- 1 for primary, 2+ for fallbacks
    
    -- Timing
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    -- Outcome
    status TEXT NOT NULL CHECK (status IN ('success', 'failure', 'timeout', 'rate_limited')),
    error_message TEXT,
    
    -- Token usage for this span
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd DECIMAL(10, 6),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Model stats daily table (rollups for dashboards)
CREATE TABLE IF NOT EXISTS model_stats_daily (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    
    -- Aggregated metrics
    request_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    fallback_count INTEGER DEFAULT 0,
    
    -- Token aggregates
    total_prompt_tokens BIGINT DEFAULT 0,
    total_completion_tokens BIGINT DEFAULT 0,
    total_tokens BIGINT DEFAULT 0,
    
    -- Cost aggregates
    total_cost_usd DECIMAL(10, 6) DEFAULT 0,
    
    -- Latency aggregates (in milliseconds)
    p50_latency_ms INTEGER,
    p95_latency_ms INTEGER,
    p99_latency_ms INTEGER,
    avg_latency_ms INTEGER,
    
    -- Error breakdown
    timeout_count INTEGER DEFAULT 0,
    rate_limit_count INTEGER DEFAULT 0,
    provider_5xx_count INTEGER DEFAULT 0,
    invalid_json_count INTEGER DEFAULT 0,
    tool_error_count INTEGER DEFAULT 0,
    safety_count INTEGER DEFAULT 0,
    
    -- Validation stats
    json_valid_count INTEGER DEFAULT 0,
    schema_valid_count INTEGER DEFAULT 0,
    refusal_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    UNIQUE(date, org_id, project_id, provider, model)
);

-- 4. Replay logs table (for offline replay results)
CREATE TABLE IF NOT EXISTS replay_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_request_id UUID NOT NULL REFERENCES request_logs(request_id) ON DELETE CASCADE,
    replay_request_id UUID NOT NULL UNIQUE,  -- New request_id for the replay
    trace_id UUID NOT NULL,
    
    -- Replay configuration
    replayed_provider TEXT NOT NULL,
    replayed_model TEXT NOT NULL,
    replay_prompt_text TEXT,  -- If original prompt wasn't stored
    
    -- Results
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd DECIMAL(10, 6),
    latency_ms INTEGER,
    
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'fallback_success', 'failure', 'refused')),
    failure_reason TEXT,
    
    response_text TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_request_logs_request_id ON request_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_trace_id ON request_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_org_id ON request_logs(org_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_project_id ON request_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_outcome ON request_logs(outcome);
CREATE INDEX IF NOT EXISTS idx_request_logs_provider_model ON request_logs(chosen_provider, chosen_model);

CREATE INDEX IF NOT EXISTS idx_request_spans_request_id ON request_spans(request_id);
CREATE INDEX IF NOT EXISTS idx_request_spans_trace_id ON request_spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_request_spans_span_type ON request_spans(span_type);

CREATE INDEX IF NOT EXISTS idx_model_stats_daily_date ON model_stats_daily(date);
CREATE INDEX IF NOT EXISTS idx_model_stats_daily_org_project ON model_stats_daily(org_id, project_id);
CREATE INDEX IF NOT EXISTS idx_model_stats_daily_provider_model ON model_stats_daily(provider, model);

CREATE INDEX IF NOT EXISTS idx_replay_logs_original_request_id ON replay_logs(original_request_id);
CREATE INDEX IF NOT EXISTS idx_replay_logs_replay_request_id ON replay_logs(replay_request_id);

-- Add comments
COMMENT ON TABLE request_logs IS 'Main request log table - one row per LLM request';
COMMENT ON TABLE request_spans IS 'Trace spans - one row per span/attempt (gateway, routing, provider calls, fallbacks)';
COMMENT ON TABLE model_stats_daily IS 'Daily aggregated statistics per model/provider for dashboard queries';
COMMENT ON TABLE replay_logs IS 'Offline replay results - stores replay attempts for research';

