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

    Produces a human-readable context block that helps the LLM understand:
    - What this endpoint does (purpose, domain)
    - What users typically ask about (topics, intents)
    - Common question patterns
    - Relevant infrastructure context (model, cost only if noteworthy)

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

    # Organize memories by predicate for structured assembly
    mem_by_pred: dict[str, dict] = {}
    for mem in memories:
        pred = mem.get("predicate", "")
        obj = mem.get("object", {})
        if isinstance(obj, dict):
            mem_by_pred[pred] = obj

    sections: list[str] = []

    # 1. Endpoint purpose (most valuable — tells the LLM what it's doing)
    purpose = mem_by_pred.get("endpoint_purpose", {})
    if purpose.get("description"):
        sections.append(purpose["description"])

    # 2. Common topics (what users ask about)
    topics = mem_by_pred.get("common_topics", {})
    topic_list = topics.get("topics", [])
    if topic_list:
        sections.append(f"Frequently discussed: {', '.join(topic_list[:10])}.")

    # 3. Intent distribution (why users call)
    intents = mem_by_pred.get("intent_distribution", {})
    dist = intents.get("distribution", {})
    if dist:
        intent_parts = []
        for intent, pct in sorted(dist.items(), key=lambda x: -x[1]):
            label = intent.replace("_", " ")
            intent_parts.append(f"{label} ({pct:.0%})")
        sections.append(f"User intent breakdown: {', '.join(intent_parts)}.")

    # 4. Common question patterns (specific examples of what users ask)
    q_patterns = mem_by_pred.get("common_question_patterns", {})
    patterns = q_patterns.get("patterns", [])
    if patterns:
        examples = "; ".join(p for p in patterns[:5])
        sections.append(f"Frequent questions: {examples}.")

    # 5. Light infrastructure context (only if noteworthy)
    model_info = mem_by_pred.get("most_used_model", {})
    error_info = mem_by_pred.get("error_rate", {})
    infra_parts = []
    if model_info.get("model"):
        infra_parts.append(f"model={model_info['model']}")
    if error_info.get("rate", 0) > 0.05:
        infra_parts.append(f"error_rate={error_info['rate']:.1%}")
    if infra_parts:
        sections.append(f"Infrastructure: {', '.join(infra_parts)}.")

    if not sections:
        return None

    summary = "[LEARNED CONTEXT]\n" + "\n".join(sections)

    # Truncate if too long
    if len(summary) > MAX_SUMMARY_TOKENS * 4:
        summary = summary[: MAX_SUMMARY_TOKENS * 4 - 3] + "..."

    return summary
