"""
Synthetic Mind — Consolidation Engine ("Sleep Phase").

Processes raw observations into reusable understanding:
  a) Abstraction Formation — cluster observations, extract patterns
  b) Causal Binding — mine sequential patterns for temporal relationships
  c) Belief Revision — update contradicted understanding with audit trail

Runs on a schedule (not on the hot path).
"""
import logging
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from synthetic_mind.memory_store import (
    get_unconsolidated_observations,
    mark_observations_consolidated,
    upsert_memory_unit,
    start_consolidation_run,
    complete_consolidation_run,
    decay_confidence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstraction Formation
# ---------------------------------------------------------------------------

def _extract_patterns(observations: list[dict]) -> list[dict]:
    """
    Extract patterns from a batch of observations.

    Phase 1: Simple frequency-based pattern extraction.
    - Most common models used per workflow/endpoint
    - Average token counts and costs
    - Common entities across requests
    - Input/output length distributions
    """
    patterns: list[dict] = []

    # Group by endpoint_slug
    by_endpoint: dict[str, list[dict]] = defaultdict(list)
    for obs in observations:
        slug = obs.get("endpoint_slug") or "unknown"
        by_endpoint[slug].append(obs)

    for slug, obs_group in by_endpoint.items():
        if len(obs_group) < 3:
            continue

        # Model usage pattern
        model_counts = Counter(o.get("model") for o in obs_group if o.get("model"))
        if model_counts:
            top_model, count = model_counts.most_common(1)[0]
            patterns.append({
                "memory_type": "fact",
                "subject": f"endpoint:{slug}",
                "predicate": "most_used_model",
                "object": {"model": top_model, "count": count, "total": len(obs_group)},
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Average cost pattern
        costs = [o.get("total_cost") or 0 for o in obs_group if o.get("total_cost")]
        if costs:
            avg_cost = sum(costs) / len(costs)
            patterns.append({
                "memory_type": "fact",
                "subject": f"endpoint:{slug}",
                "predicate": "average_cost_per_call",
                "object": {"avg_cost": round(avg_cost, 6), "sample_size": len(costs)},
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Token usage pattern
        input_tokens = [o.get("input_tokens") or 0 for o in obs_group if o.get("input_tokens")]
        output_tokens = [o.get("output_tokens") or 0 for o in obs_group if o.get("output_tokens")]
        if input_tokens:
            patterns.append({
                "memory_type": "fact",
                "subject": f"endpoint:{slug}",
                "predicate": "typical_token_usage",
                "object": {
                    "avg_input_tokens": round(sum(input_tokens) / len(input_tokens)),
                    "avg_output_tokens": round(sum(output_tokens) / len(output_tokens)) if output_tokens else 0,
                    "sample_size": len(input_tokens),
                },
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Entity frequency across observations
        entity_counts: Counter = Counter()
        for o in obs_group:
            for e in (o.get("entities_mentioned") or []):
                entity_counts[e] += 1
        common_entities = [e for e, c in entity_counts.most_common(10) if c >= 2]
        if common_entities:
            patterns.append({
                "memory_type": "entity",
                "subject": f"endpoint:{slug}",
                "predicate": "common_entities",
                "object": {"entities": common_entities},
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Error rate
        errors = [o for o in obs_group if o.get("observation_type") == "error"]
        if errors:
            patterns.append({
                "memory_type": "fact",
                "subject": f"endpoint:{slug}",
                "predicate": "error_rate",
                "object": {
                    "error_count": len(errors),
                    "total_count": len(obs_group),
                    "rate": round(len(errors) / len(obs_group), 4),
                },
                "evidence": [o["id"] for o in errors[:10]],
            })

    return patterns


# ---------------------------------------------------------------------------
# Main consolidation entry point
# ---------------------------------------------------------------------------

def consolidate_org(org_id: str) -> dict[str, int]:
    """
    Run one consolidation cycle for an org.

    1. Fetch unconsolidated observations
    2. Extract patterns → memory units
    3. Apply confidence decay
    4. Mark observations as consolidated
    5. Log the run

    Returns summary stats.
    """
    run_id = start_consolidation_run(org_id)
    stats = {
        "observations_processed": 0,
        "memory_units_created": 0,
        "memory_units_updated": 0,
        "abstractions_formed": 0,
    }

    try:
        observations = get_unconsolidated_observations(org_id, limit=500)
        if not observations:
            if run_id:
                complete_consolidation_run(run_id, **stats, status="completed")
            return stats

        stats["observations_processed"] = len(observations)

        # Extract patterns
        patterns = _extract_patterns(observations)
        stats["abstractions_formed"] = len(patterns)

        # Upsert memory units from patterns
        for pattern in patterns:
            obs = observations[0]  # Use first observation for scope
            unit = {
                "id": str(uuid.uuid4()),
                "memory_type": pattern["memory_type"],
                "org_id": org_id,
                "workflow_id": obs.get("workflow_id"),
                "subject": pattern["subject"],
                "predicate": pattern["predicate"],
                "object": pattern["object"],
                "evidence": pattern.get("evidence", []),
                "confidence": min(1.0, 0.5 + len(observations) * 0.01),
            }
            result = upsert_memory_unit(unit)
            if result:
                stats["memory_units_created"] += 1

        # Mark all processed observations as consolidated
        obs_ids = [o["id"] for o in observations]
        mark_observations_consolidated(obs_ids)

        # Apply confidence decay
        decay_confidence(org_id)

        if run_id:
            complete_consolidation_run(run_id, **stats)

    except Exception as e:
        logger.error("Consolidation failed for org %s: %s", org_id, e)
        if run_id:
            complete_consolidation_run(
                run_id, **stats, status="failed", error_message=str(e)
            )

    return stats
