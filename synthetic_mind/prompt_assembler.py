"""
Synthetic Mind — Prompt Assembler.

Assembles the thinnest possible prompt by referencing accumulated
understanding instead of resending raw context.

Phase 1: Memory summary generation (injected as system context).
Prompt thinning (delta computation) comes in Phase 3.
"""
import logging
from typing import Any, Optional

from synthetic_mind.scopes import load_scoped_memories

logger = logging.getLogger(__name__)

# Max tokens for the memory summary injection
MAX_SUMMARY_TOKENS = 300  # ~300 tokens ≈ 1200 chars


def generate_memory_summary(
    org_id: str,
    *,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a compact memory summary for injection into the system prompt.

    Returns a string like:
      [CONTEXT: endpoint=recipe-gen, typical_model=gpt-4o, avg_cost=$0.003,
       common_entities=[recipe, nutrition, vegan], avg_input=450tok]

    Returns None if no relevant memories exist yet.
    """
    memories = load_scoped_memories(
        org_id,
        workflow_id=workflow_id,
        user_id=user_id,
        endpoint_slug=endpoint_slug,
        min_confidence=0.3,
        limit_per_scope=15,
    )

    if not memories:
        return None

    # Build summary parts
    parts: list[str] = []

    if endpoint_slug:
        parts.append(f"endpoint={endpoint_slug}")

    for mem in memories:
        pred = mem.get("predicate", "")
        obj = mem.get("object", {})

        if pred == "most_used_model" and isinstance(obj, dict):
            parts.append(f"typical_model={obj.get('model', '?')}")
        elif pred == "average_cost_per_call" and isinstance(obj, dict):
            avg = obj.get("avg_cost", 0)
            parts.append(f"avg_cost=${avg:.4f}")
        elif pred == "typical_token_usage" and isinstance(obj, dict):
            parts.append(f"avg_input={obj.get('avg_input_tokens', 0)}tok")
            parts.append(f"avg_output={obj.get('avg_output_tokens', 0)}tok")
        elif pred == "common_entities" and isinstance(obj, dict):
            ents = obj.get("entities", [])[:5]
            if ents:
                parts.append(f"common_entities=[{', '.join(ents)}]")
        elif pred == "error_rate" and isinstance(obj, dict):
            rate = obj.get("rate", 0)
            if rate > 0.05:
                parts.append(f"error_rate={rate:.1%}")

    if not parts:
        return None

    summary = "[CONTEXT: " + ", ".join(parts) + "]"

    # Truncate if too long
    if len(summary) > MAX_SUMMARY_TOKENS * 4:
        summary = summary[: MAX_SUMMARY_TOKENS * 4 - 3] + "...]"

    return summary
