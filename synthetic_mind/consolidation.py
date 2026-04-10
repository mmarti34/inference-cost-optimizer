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
    upsert_kb_consolidation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstraction Formation
# ---------------------------------------------------------------------------

def _extract_patterns(observations: list[dict]) -> list[dict]:
    """
    Extract patterns from a batch of observations.

    Extracts both infrastructure stats AND content understanding:
    - Model/cost/token patterns (infrastructure)
    - Common topics and keywords (what users ask about)
    - Intent distribution (why users call this endpoint)
    - Endpoint purpose (inferred from content patterns)
    - Frequent question patterns (what specific things users ask)
    - Common entities mentioned in content
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

        # --- Infrastructure patterns (kept from phase 1) ---

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

        # --- Content understanding patterns (new) ---

        # Topic extraction from entities_mentioned (which now includes keywords)
        topic_counts: Counter = Counter()
        for o in obs_group:
            for entity in (o.get("entities_mentioned") or []):
                topic_counts[entity] += 1

        # Split into model/provider entities vs content topics
        _model_providers = {"openai", "anthropic", "google", "gemini", "groq", "together",
                           "mistral", "cohere", "deepseek", "fireworks"}
        content_topics = [(t, c) for t, c in topic_counts.most_common(30)
                         if t not in _model_providers
                         and not any(t.startswith(p) for p in ("gpt-", "claude-", "gemini-", "llama", "mistral-"))
                         and c >= 2]
        top_topics = [t for t, _ in content_topics[:15]]

        if top_topics:
            patterns.append({
                "memory_type": "abstraction",
                "subject": f"endpoint:{slug}",
                "predicate": "common_topics",
                "object": {
                    "topics": top_topics,
                    "topic_counts": {t: c for t, c in content_topics[:15]},
                    "sample_size": len(obs_group),
                },
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Intent distribution from input summaries
        intent_counts: Counter = Counter()
        for o in obs_group:
            summary = o.get("input_summary") or ""
            intent = _classify_intent_from_summary(summary)
            intent_counts[intent] += 1

        if intent_counts:
            total_intents = sum(intent_counts.values())
            intent_dist = {
                intent: round(count / total_intents, 3)
                for intent, count in intent_counts.most_common()
                if count / total_intents >= 0.05  # only include intents ≥5%
            }
            top_intent = intent_counts.most_common(1)[0][0]
            patterns.append({
                "memory_type": "abstraction",
                "subject": f"endpoint:{slug}",
                "predicate": "intent_distribution",
                "object": {
                    "distribution": intent_dist,
                    "primary_intent": top_intent,
                    "sample_size": total_intents,
                },
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Endpoint purpose — inferred from top topics + intent distribution
        purpose = _infer_endpoint_purpose(top_topics, intent_counts, slug)
        if purpose:
            patterns.append({
                "memory_type": "abstraction",
                "subject": f"endpoint:{slug}",
                "predicate": "endpoint_purpose",
                "object": {"description": purpose, "sample_size": len(obs_group)},
                "evidence": [o["id"] for o in obs_group[:10]],
            })

        # Frequent question patterns — extract representative questions
        question_patterns = _extract_question_patterns(obs_group)
        if question_patterns:
            patterns.append({
                "memory_type": "procedure",
                "subject": f"endpoint:{slug}",
                "predicate": "common_question_patterns",
                "object": {
                    "patterns": question_patterns,
                    "sample_size": len(obs_group),
                },
                "evidence": [o["id"] for o in obs_group[:10]],
            })

    return patterns


def _classify_intent_from_summary(text: str) -> str:
    """Classify user intent from an observation's input summary."""
    if not text:
        return "unknown"
    t = text.lower().strip()
    if any(w in t for w in ["how do i", "how to", "walk me through", "can you show", "steps to", "how does"]):
        return "how_to"
    if any(w in t for w in ["price", "pricing", "cost", "plan", "tier", "upgrade", "billing", "subscribe", "pay", "refund"]):
        return "pricing"
    if any(w in t for w in ["error", "bug", "broken", "not working", "can't", "won't", "issue",
                            "problem", "help me", "fix", "stuck", "failing", "disappeared", "stopped",
                            "not syncing", "slow", "loading", "crash", "wrong"]):
        return "troubleshooting"
    if any(w in t for w in ["what is", "what's", "what are", "does it", "do you", "does ",
                            "is there", "can i", "can you", "support", "have a", "have an"]):
        return "feature_question"
    if any(w in t for w in ["hello", "hi ", "hi!", "hey", "new user", "just signed", "getting started", "where do i start"]):
        return "onboarding"
    return "general"


def _infer_endpoint_purpose(topics: list[str], intents: Counter, slug: str) -> Optional[str]:
    """Infer what this endpoint is for based on topics and intents."""
    if not topics and not intents:
        return None

    top_intent = intents.most_common(1)[0][0] if intents else "general"
    topic_str = ", ".join(topics[:8]) if topics else "various topics"

    # Build a human-readable purpose description
    intent_labels = {
        "how_to": "instructional/how-to",
        "pricing": "pricing and billing",
        "troubleshooting": "troubleshooting and support",
        "feature_question": "feature discovery",
        "onboarding": "user onboarding",
        "general": "general assistance",
    }
    intent_desc = intent_labels.get(top_intent, "general assistance")

    return f"Handles {intent_desc} queries. Common topics: {topic_str}."


def _extract_question_patterns(obs_group: list[dict]) -> list[str]:
    """
    Extract representative question patterns from observations.
    Groups similar questions and returns the most common patterns.
    """
    # Collect first sentences from input summaries as question patterns
    raw_questions: list[str] = []
    for o in obs_group:
        summary = o.get("input_summary") or ""
        if not summary:
            continue
        # Take first sentence/question
        for sep in ["?", ".", "!"]:
            idx = summary.find(sep)
            if idx > 0:
                raw_questions.append(summary[:idx + 1].strip())
                break
        else:
            raw_questions.append(summary[:100].strip())

    if not raw_questions:
        return []

    # Group by leading words to find patterns
    pattern_groups: dict[str, int] = Counter()
    for q in raw_questions:
        words = q.lower().split()
        if len(words) >= 3:
            # Use first 3-4 words as the pattern key
            key_len = min(4, len(words))
            key = " ".join(words[:key_len])
            pattern_groups[key] += 1

    # Return patterns that appeared at least twice, with example
    result = []
    for pattern, count in pattern_groups.most_common(10):
        if count >= 2:
            # Find a representative full question for this pattern
            for q in raw_questions:
                if q.lower().startswith(pattern):
                    result.append(f"{q} ({count}x)")
                    break
    return result[:8]


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

    # Phase 2: consolidate KB patterns from observations that tracked asset usage
    try:
        _consolidate_kb_patterns(org_id, observations)
    except Exception as e:
        logger.warning("KB pattern consolidation failed for org %s: %s", org_id, e)

    # Phase 3: consolidate agent tool patterns
    try:
        _consolidate_agent_tool_patterns(org_id, observations)
    except Exception as e:
        logger.warning("Agent tool pattern consolidation failed for org %s: %s", org_id, e)

    return stats


# ---------------------------------------------------------------------------
# Phase 2: KB Context Consolidation
# ---------------------------------------------------------------------------

def _consolidate_kb_patterns(org_id: str, observations: list[dict]) -> None:
    """
    Find recurring KB asset combinations across observations and generate
    consolidated summaries. Only creates consolidations when the same set
    of assets has been retrieved 3+ times.
    """
    import hashlib
    import os
    from supabase_client import supabase

    # Count (org, scope_value, asset_set) combos from observations that recorded kb_asset_ids
    # Key: (scope_value, asset_ids_key) → list of occurrences
    asset_set_counts: dict[tuple, list[list[str]]] = defaultdict(list)
    scope_values: dict[tuple, str | None] = {}
    for obs in observations:
        metadata = obs.get("metadata") or {}
        if isinstance(metadata, str):
            continue
        kb_ids = metadata.get("kb_asset_ids") or []
        if not kb_ids or len(kb_ids) < 1:
            continue
        obs_scope = metadata.get("scope_value")
        asset_key = "|".join(sorted(kb_ids))
        combo_key = (obs_scope, asset_key)
        asset_set_counts[combo_key].append(kb_ids)
        scope_values[combo_key] = obs_scope

    for combo_key, occurrences in asset_set_counts.items():
        if len(occurrences) < 3:
            continue  # Need 3+ hits before consolidating

        asset_ids = sorted(occurrences[0])
        scope_value = scope_values.get(combo_key)

        # Check if consolidation already exists (scoped)
        from synthetic_mind.memory_store import get_kb_consolidation
        existing = get_kb_consolidation(org_id, asset_ids, min_confidence=0.0, scope_value=scope_value)
        if existing:
            # Already consolidated — confidence gets bumped via hits at serve time
            continue

        # Fetch raw asset content
        try:
            result = (
                supabase.table("context_assets")
                .select("id, name, content, updated_at")
                .in_("id", asset_ids)
                .eq("org_id", org_id)
                .execute()
            )
            if not result.data or len(result.data) != len(asset_ids):
                continue
        except Exception:
            continue

        # Compute versions hash for staleness detection
        timestamps = sorted(f"{r['id']}:{r['updated_at']}" for r in result.data)
        versions_hash = hashlib.sha256("|".join(timestamps).encode()).hexdigest()[:16]

        # Build combined text
        raw_parts = []
        raw_chars = 0
        for asset in result.data:
            content = asset.get("content") or ""
            raw_parts.append(f"## {asset.get('name', 'Asset')}\n{content}")
            raw_chars += len(content)

        if raw_chars < 200:
            continue  # Too short to bother consolidating

        combined = "\n\n---\n\n".join(raw_parts)
        raw_token_count = max(1, raw_chars // 4)

        # Call gpt-4o-mini to consolidate
        summary = _call_kb_consolidator(combined, raw_chars)
        if not summary:
            continue

        consolidated_token_count = max(1, len(summary) // 4)

        # Only store if we actually save tokens (at least 2x compression)
        if consolidated_token_count > raw_token_count // 2:
            continue

        upsert_kb_consolidation(
            org_id=org_id,
            asset_ids=asset_ids,
            consolidated_text=summary,
            raw_token_count=raw_token_count,
            consolidated_token_count=consolidated_token_count,
            asset_versions_hash=versions_hash,
            scope_value=scope_value,
        )

        logger.info(
            "KB consolidation created: %d assets, %d→%d tokens, org=%s",
            len(asset_ids), raw_token_count, consolidated_token_count, org_id[:8],
        )


def _call_kb_consolidator(combined_text: str, raw_chars: int) -> Optional[str]:
    """Call gpt-4o-mini to compress multiple KB assets into a consolidated summary."""
    import os
    api_key = os.environ.get("SYSTEM_OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        target_chars = max(200, raw_chars // 4)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Compress the following knowledge base content into a concise summary of about {target_chars} characters. "
                        "Preserve ALL key facts, data points, rules, procedures, and important details. "
                        "This summary will replace the raw content in future LLM calls, "
                        "so it must contain everything needed to answer questions accurately. "
                        "Be factual and precise — do not generalize or lose specifics."
                    ),
                },
                {"role": "user", "content": combined_text},
            ],
            max_tokens=target_chars // 3,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("KB consolidation summarizer failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Phase 3: Agent Cross-Run Memory
# ---------------------------------------------------------------------------

def _consolidate_agent_tool_patterns(org_id: str, observations: list[dict]) -> None:
    """
    Extract tool usage patterns from agent observations and store as memory units.
    When an agent reads the same file or searches the same KB multiple times across
    runs, we consolidate what it learned into reusable prior knowledge.
    """
    # Collect agent tool call data from observations
    tool_usage: dict[str, list[dict]] = defaultdict(list)  # tool_name → [calls]

    for obs in observations:
        metadata = obs.get("metadata") or {}
        if isinstance(metadata, str):
            continue
        agent_tools = metadata.get("agent_tool_calls") or []
        for tool_call in agent_tools:
            tool_name = tool_call.get("tool_name", "")
            if not tool_name:
                continue
            tool_usage[tool_name].append(tool_call)

    if not tool_usage:
        return

    # For file-reading tools, consolidate repeated reads of same files
    for tool_name in ("get_knowledge_asset", "search_knowledge_base"):
        calls = tool_usage.get(tool_name, [])
        if len(calls) < 2:
            continue

        # Group by the input (asset_id or query)
        by_input: dict[str, list[dict]] = defaultdict(list)
        for call in calls:
            input_key = call.get("input_key", "")
            if input_key:
                by_input[input_key].append(call)

        for input_key, repeated_calls in by_input.items():
            if len(repeated_calls) < 2:
                continue

            # Extract common output patterns
            outputs = [c.get("output_summary", "") for c in repeated_calls if c.get("output_summary")]
            if not outputs:
                continue

            # Store as a memory unit for agent prior knowledge
            unit = {
                "id": str(uuid.uuid4()),
                "memory_type": "procedure",
                "org_id": org_id,
                "subject": f"agent_tool:{tool_name}",
                "predicate": "prior_knowledge",
                "object": {
                    "tool_name": tool_name,
                    "input_key": input_key,
                    "call_count": len(repeated_calls),
                    "output_summary": outputs[0][:500],  # Use first output as representative
                },
                "evidence": [c.get("observation_id", "") for c in repeated_calls[:10]],
                "confidence": min(1.0, 0.4 + len(repeated_calls) * 0.1),
            }
            upsert_memory_unit(unit)
