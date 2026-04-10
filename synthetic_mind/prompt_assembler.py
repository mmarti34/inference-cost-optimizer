"""
Synthetic Mind — Prompt Assembler.

Assembles the thinnest possible prompt by referencing accumulated
understanding instead of resending raw context.

Phase 1: Memory summary generation (injected as system context).
Phase 3: Agent prior knowledge generation from cross-run tool patterns.
"""
import logging
from typing import Any, Optional

from synthetic_mind.scopes import load_scoped_memories
from synthetic_mind.memory_store import get_active_memories

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


# Max chars for agent prior knowledge block
MAX_PRIOR_KNOWLEDGE_CHARS = 1500


def generate_agent_prior_knowledge(
    org_id: str,
    *,
    workflow_id: Optional[str] = None,
    scope_value: Optional[str] = None,
) -> Optional[str]:
    """
    SM v2 Phase 3: Generate a prior knowledge block for agent nodes.

    Fetches consolidated tool knowledge from past agent runs and formats
    it as a compact reference the agent can use to avoid redundant tool calls.

    Returns None if no relevant prior knowledge exists.
    The agent decides whether to use this or re-call tools — not forced.
    """
    try:
        memories = get_active_memories(
            org_id,
            workflow_id=workflow_id,
            memory_type="procedure",
            min_confidence=0.5,
            limit=20,
        )

        if not memories:
            return None

        # Filter to agent tool prior knowledge
        tool_memories = [
            m for m in memories
            if m.get("predicate") == "prior_knowledge"
            and (m.get("subject") or "").startswith("agent_tool:")
        ]

        if not tool_memories:
            return None

        lines: list[str] = []
        for mem in tool_memories:
            obj = mem.get("object", {})
            if not isinstance(obj, dict):
                continue
            tool_name = obj.get("tool_name", "")
            input_key = obj.get("input_key", "")
            output_summary = obj.get("output_summary", "")
            call_count = obj.get("call_count", 0)

            if not output_summary:
                continue

            if tool_name == "get_knowledge_asset" and input_key:
                lines.append(f"- Asset {input_key}: {output_summary}")
            elif tool_name == "search_knowledge_base" and input_key:
                lines.append(f"- KB search \"{input_key}\": {output_summary}")
            else:
                lines.append(f"- {tool_name}: {output_summary}")

        if not lines:
            return None

        block = "[PRIOR TOOL KNOWLEDGE]\n" + "\n".join(lines)

        # Truncate if too long
        if len(block) > MAX_PRIOR_KNOWLEDGE_CHARS:
            block = block[:MAX_PRIOR_KNOWLEDGE_CHARS - 3] + "..."

        return block

    except Exception as e:
        logger.warning("Failed to generate agent prior knowledge: %s", e)
        return None
