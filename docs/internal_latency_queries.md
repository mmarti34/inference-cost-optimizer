# Internal Latency Instrumentation — Query Reference

> **Audience:** OptiML engineers only. This data is NOT exposed via API.
> **Table:** `routing_latency_facts`
> **Written by:** `workflow_runtime.py` on every model/ai-step/optimizer node execution.

## Schema Quick Reference

| Column | Type | Notes |
|--------|------|-------|
| `ts` | timestamptz | Row creation time (indexed) |
| `org_id` | uuid | Organization |
| `workflow_id` | uuid | Workflow |
| `node_id` | text | Node within workflow |
| `node_type` | text | `model`, `ai-step`, `optimizer` |
| `target_type` | text | `provider_model`, `openai_compatible_endpoint`, null |
| `provider_label` | text | `openai`, `anthropic`, `custom:<host>` |
| `model_name` | text | e.g. `gpt-4o`, `claude-sonnet-4-5-20250929` |
| `endpoint_id` | uuid | For custom endpoints |
| `total_latency_ms` | int | Wall-clock: node start → node end |
| `provider_latency_ms` | int | SDK request only (null for uninstrumented routers) |
| `gateway_overhead_ms` | int | `max(total - provider, 0)` (null when provider is null) |
| `success` | boolean | Whether the call succeeded |
| `error_type` | text | `timeout`, `rate_limit`, `auth`, `not_found`, `connection`, `provider_error` |

## Queries

### 1. P50/P95 Gateway Overhead — Last 7 Days by Provider

```sql
SELECT
    provider_label,
    count(*)                                              AS calls,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p50_overhead_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p95_overhead_ms,
    round(avg(gateway_overhead_ms)::numeric, 1)           AS avg_overhead_ms
FROM routing_latency_facts
WHERE ts > now() - interval '7 days'
  AND success = true
  AND gateway_overhead_ms IS NOT NULL
GROUP BY provider_label
ORDER BY calls DESC;
```

### 2. P50/P95 Total Latency by Model

```sql
SELECT
    model_name,
    provider_label,
    count(*)                                              AS calls,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY total_latency_ms)    AS p50_total_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms)    AS p95_total_ms,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY provider_latency_ms) AS p50_provider_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY provider_latency_ms) AS p95_provider_ms
FROM routing_latency_facts
WHERE ts > now() - interval '7 days'
  AND success = true
GROUP BY model_name, provider_label
ORDER BY calls DESC;
```

### 3. Overhead as Percentage of Provider Latency

```sql
SELECT
    provider_label,
    count(*)                                              AS calls,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY provider_latency_ms) AS p95_provider_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p95_overhead_ms,
    round(
        100.0 * percentile_cont(0.95) WITHIN GROUP (ORDER BY gateway_overhead_ms)
        / NULLIF(percentile_cont(0.95) WITHIN GROUP (ORDER BY provider_latency_ms), 0),
        1
    )                                                     AS overhead_pct
FROM routing_latency_facts
WHERE ts > now() - interval '7 days'
  AND success = true
  AND provider_latency_ms IS NOT NULL
GROUP BY provider_label
ORDER BY calls DESC;
```

### 4. Custom Endpoints vs Standard Providers — Overhead Comparison

```sql
SELECT
    target_type,
    count(*)                                              AS calls,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p50_overhead_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p95_overhead_ms
FROM routing_latency_facts
WHERE ts > now() - interval '7 days'
  AND success = true
  AND gateway_overhead_ms IS NOT NULL
GROUP BY target_type
ORDER BY calls DESC;
```

### 5. Error Rate and Latency by Error Type

```sql
SELECT
    provider_label,
    error_type,
    count(*)                                              AS error_count,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY total_latency_ms) AS p50_total_ms
FROM routing_latency_facts
WHERE ts > now() - interval '7 days'
  AND success = false
GROUP BY provider_label, error_type
ORDER BY error_count DESC;
```

### 6. Per-Org Overhead (for answering customer questions)

```sql
SELECT
    org_id,
    count(*)                                              AS calls,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p50_overhead_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p95_overhead_ms,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY total_latency_ms)    AS p50_total_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms)    AS p95_total_ms
FROM routing_latency_facts
WHERE ts > now() - interval '30 days'
  AND success = true
  AND gateway_overhead_ms IS NOT NULL
  AND org_id = '<ORG_UUID>'    -- replace with customer org_id
GROUP BY org_id;
```

### 7. Hourly Trend — Spot Regressions

```sql
SELECT
    date_trunc('hour', ts)                                AS hour,
    count(*)                                              AS calls,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p50_overhead_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY gateway_overhead_ms) AS p95_overhead_ms
FROM routing_latency_facts
WHERE ts > now() - interval '48 hours'
  AND success = true
  AND gateway_overhead_ms IS NOT NULL
GROUP BY hour
ORDER BY hour;
```

## Expected Result Shapes

**Query 1** (overhead by provider):
```
 provider_label | calls | p50_overhead_ms | p95_overhead_ms | avg_overhead_ms
----------------+-------+-----------------+-----------------+----------------
 openai         |  1234 |              12 |              45 |           18.3
 anthropic      |   567 |            NULL |            NULL |           NULL
 custom:my-vllm |    89 |              15 |              52 |           22.1
```
> `anthropic` shows NULL because its router is not yet instrumented (provider_latency_ms is NULL → gateway_overhead_ms is NULL).

**Query 3** (overhead percentage):
```
 provider_label | calls | p95_provider_ms | p95_overhead_ms | overhead_pct
----------------+-------+-----------------+-----------------+-------------
 openai         |  1234 |            2100 |              45 |         2.1
 custom:my-vllm |    89 |             850 |              52 |         6.1
```
> Target: overhead < 5% of provider latency.

## Retention

No automatic retention policy yet. If table grows large, consider:
```sql
DELETE FROM routing_latency_facts WHERE ts < now() - interval '90 days';
```
Or add a pg_cron job.
