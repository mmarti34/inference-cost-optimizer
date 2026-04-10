"""
Conversation state for multi-turn public API calls.
When conversation_id is provided, we load history, trim to fit, and prepend to AI step messages;
after completion we save the user turn and assistant turn.

Synthetic Mind v2: conversation history compression.
When a conversation exceeds COMPRESS_THRESHOLD turns, older turns are replaced
with a compact LLM-generated summary. Only the most recent turns are sent raw.
This keeps context cost roughly flat instead of growing linearly.
"""
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from supabase_client import supabase

_logger = logging.getLogger(__name__)

# Rough token estimate: ~4 chars per token for English
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 8000
DEFAULT_MAX_TURNS = 50

# Agent conversations are verbose (tool calls, reasoning) — higher limits
AGENT_MAX_TOKENS = 16000
AGENT_MAX_TURNS = 20

# SM v2: compress history when turn count exceeds this threshold
COMPRESS_THRESHOLD = 6
# Number of recent turns to keep in full (not summarized)
RECENT_TURNS_TO_KEEP = 3
# Role used to store the SM summary as a synthetic turn
SM_SUMMARY_ROLE = "sm_summary"


def get_or_create_conversation(
    org_id: str,
    endpoint_slug: str,
    conversation_id: str | None,
) -> dict | None:
    """
    If conversation_id is provided and valid UUID, return existing conversation or create one.
    If conversation_id is None or invalid, return None (no conversation).
    """
    if not conversation_id or not (conversation_id or "").strip():
        return None
    cid = (conversation_id or "").strip()
    try:
        uuid.UUID(cid)
    except ValueError:
        return None
    try:
        existing = (
            supabase.table("conversations")
            .select("id, org_id, endpoint_slug, created_at, updated_at, metadata")
            .eq("id", cid)
            .eq("org_id", org_id)
            .eq("endpoint_slug", endpoint_slug)
            .limit(1)
            .execute()
        )
        if existing.data and len(existing.data) > 0:
            return existing.data[0]
    except Exception:
        pass
    try:
        row = {
            "id": cid,
            "org_id": org_id,
            "endpoint_slug": endpoint_slug,
        }
        supabase.table("conversations").insert(row).execute()
        return {"id": cid, "org_id": org_id, "endpoint_slug": endpoint_slug}
    except Exception:
        return None


def load_conversation_turns(conversation_id: str, limit: int = 100) -> list[dict]:
    """Load turns for a conversation, oldest first."""
    try:
        result = (
            supabase.table("conversation_turns")
            .select("id, turn_number, role, content, variables, request_id, served_version, created_at")
            .eq("conversation_id", conversation_id)
            .order("turn_number", desc=False)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        return []


def trim_history_to_fit(
    turns: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[dict]:
    """Keep the most recent turns that fit within max_tokens and max_turns."""
    if not turns:
        return []
    by_tokens = 0
    keep: list[dict] = []
    for t in reversed(turns):
        if len(keep) >= max_turns:
            break
        content = (t.get("content") or "") if isinstance(t, dict) else ""
        by_tokens += max(1, len(content) // CHARS_PER_TOKEN)
        if by_tokens > max_tokens:
            break
        keep.append(t)
    keep.reverse()
    return keep


def format_history_as_prefix(turns: list[dict]) -> str:
    """Format conversation turns as a single string to prepend to the next user message."""
    if not turns:
        return ""
    parts = []
    for t in turns:
        role = (t.get("role") or "user").strip().lower()
        content = (t.get("content") or "").strip()
        if role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts) + "\n\n"


def format_agent_history_as_prefix(turns: list[dict]) -> str:
    """Format conversation turns with structured delimiters and tool usage summaries.

    Optimised for agent nodes — preserves tool call metadata stored in the
    ``variables`` JSON column so the agent has richer context on follow-up turns.
    """
    if not turns:
        return ""
    parts: list[str] = []
    for t in turns:
        role = (t.get("role") or "user").strip().lower()
        content = (t.get("content") or "").strip()
        metadata = t.get("variables") or {}

        if role == "assistant":
            tool_summary = metadata.get("agent_tool_summary", "") if isinstance(metadata, dict) else ""
            if tool_summary:
                parts.append(f"[Previous assistant response]\n{tool_summary}\nFinal answer: {content}")
            else:
                parts.append(f"[Previous assistant response]\n{content}")
        else:
            parts.append(f"[Previous user message]\n{content}")
    return "\n---\n".join(parts) + "\n---\n"


def save_conversation_turn(
    conversation_id: str,
    turn_number: int,
    role: str,
    content: str,
    variables: dict[str, Any] | None = None,
    request_id: str | None = None,
    served_version: int | None = None,
) -> None:
    """Append one turn to the conversation."""
    try:
        row = {
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "role": role,
            "content": content or "",
        }
        if variables is not None:
            row["variables"] = variables
        if request_id is not None:
            row["request_id"] = request_id
        if served_version is not None:
            row["served_version"] = served_version
        supabase.table("conversation_turns").insert(row).execute()
    except Exception:
        pass


def get_next_turn_number(conversation_id: str) -> int:
    """
    Atomically allocate the next turn_number for a conversation.

    Uses a Postgres RPC function `next_conversation_turn_number` that locks
    the conversation row (FOR UPDATE) and returns MAX(turn_number) + 1.
    This prevents two concurrent requests from both getting the same number.

    Falls back to application-level MAX query if the RPC is not yet deployed.
    """
    try:
        result = supabase.rpc("next_conversation_turn_number", {
            "p_conversation_id": conversation_id,
        }).execute()

        next_num = result.data
        if isinstance(next_num, list) and len(next_num) > 0:
            next_num = next_num[0]
        if isinstance(next_num, dict):
            next_num = next_num.get("next_conversation_turn_number", next_num.get("max_turn", 0))
        return int(next_num) if next_num is not None else 0
    except Exception:
        # RPC may not exist yet; fall back to app-level query (has TOCTOU race under concurrency)
        pass

    try:
        result = (
            supabase.table("conversation_turns")
            .select("turn_number")
            .eq("conversation_id", conversation_id)
            .order("turn_number", desc=True)
            .limit(1)
            .execute()
        )
        if result.data and len(result.data) > 0:
            return int(result.data[0].get("turn_number", 0)) + 1
    except Exception:
        pass
    return 0


def update_conversation_updated_at(conversation_id: str) -> None:
    """Touch conversation.updated_at."""
    try:
        supabase.table("conversations").update({"updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", conversation_id).execute()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Synthetic Mind v2 — Conversation History Compression
# ---------------------------------------------------------------------------

def compress_history_prefix(
    turns: list[dict],
    conversation_id: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_turns: int = DEFAULT_MAX_TURNS,
    is_agent: bool = False,
) -> str:
    """
    Format conversation history with SM compression when beneficial.

    If the conversation has more than COMPRESS_THRESHOLD turns AND a summary
    exists for the older turns, returns:
        [Summary of earlier conversation]
        {summary}
        ---
        {recent raw turns}

    Otherwise falls back to the standard format (raw turns, trimmed to fit).
    Never adds latency — only uses summaries that were pre-generated async.
    """
    if not turns:
        return ""

    # Filter out any existing SM summary turns from the raw list
    real_turns = [t for t in turns if (t.get("role") or "") != SM_SUMMARY_ROLE]

    # If not enough turns to bother compressing, use standard format
    if len(real_turns) <= COMPRESS_THRESHOLD:
        trimmed = trim_history_to_fit(real_turns, max_tokens, max_turns)
        if is_agent:
            return format_agent_history_as_prefix(trimmed)
        return format_history_as_prefix(trimmed)

    # Check for an existing summary
    summary = _get_existing_summary(conversation_id)

    if not summary:
        # No summary yet — fall back to raw turns
        trimmed = trim_history_to_fit(real_turns, max_tokens, max_turns)
        if is_agent:
            return format_agent_history_as_prefix(trimmed)
        return format_history_as_prefix(trimmed)

    summary_text = summary.get("content", "")
    covers_up_to = summary.get("variables", {}).get("covers_up_to_turn", 0) if isinstance(summary.get("variables"), dict) else 0

    # Split: recent turns are those AFTER what the summary covers
    recent_turns = [t for t in real_turns if (t.get("turn_number") or 0) > covers_up_to]

    # If the summary doesn't cover enough old turns, or too many recent,
    # just use raw turns
    if len(recent_turns) >= len(real_turns) - 1:
        trimmed = trim_history_to_fit(real_turns, max_tokens, max_turns)
        if is_agent:
            return format_agent_history_as_prefix(trimmed)
        return format_history_as_prefix(trimmed)

    # Check compression ratio — only use summary if it actually saves tokens
    summary_tokens = max(1, len(summary_text) // CHARS_PER_TOKEN)
    old_turns_raw = [t for t in real_turns if (t.get("turn_number") or 0) <= covers_up_to]
    old_turns_tokens = sum(max(1, len((t.get("content") or "")) // CHARS_PER_TOKEN) for t in old_turns_raw)

    if summary_tokens >= old_turns_tokens * 0.8:
        # Summary isn't saving enough — use raw turns
        trimmed = trim_history_to_fit(real_turns, max_tokens, max_turns)
        if is_agent:
            return format_agent_history_as_prefix(trimmed)
        return format_history_as_prefix(trimmed)

    # Build compressed prefix: summary + recent raw turns
    if is_agent:
        recent_formatted = format_agent_history_as_prefix(recent_turns)
    else:
        recent_formatted = format_history_as_prefix(recent_turns)

    compressed = f"[Summary of earlier conversation]\n{summary_text}\n---\n{recent_formatted}"

    # Log savings for observability
    raw_tokens = old_turns_tokens + sum(max(1, len((t.get("content") or "")) // CHARS_PER_TOKEN) for t in recent_turns)
    compressed_tokens = summary_tokens + sum(max(1, len((t.get("content") or "")) // CHARS_PER_TOKEN) for t in recent_turns)
    _logger.info(
        "SM history compression: %d raw tokens → %d compressed (%d%% saved), conversation=%s",
        raw_tokens, compressed_tokens, int((1 - compressed_tokens / max(raw_tokens, 1)) * 100),
        conversation_id[:8],
    )

    return compressed


def _get_existing_summary(conversation_id: str) -> dict | None:
    """Fetch the most recent SM summary turn for a conversation."""
    try:
        result = (
            supabase.table("conversation_turns")
            .select("id, turn_number, role, content, variables, created_at")
            .eq("conversation_id", conversation_id)
            .eq("role", SM_SUMMARY_ROLE)
            .order("turn_number", desc=True)
            .limit(1)
            .execute()
        )
        if result.data and len(result.data) > 0:
            return result.data[0]
    except Exception:
        pass
    return None


def generate_history_summary(conversation_id: str) -> None:
    """
    Generate (or update) a compressed summary of older turns.

    Called ASYNC after a response is saved — never on the hot path.
    Stores the summary as a synthetic turn with role='sm_summary'.
    The summary covers all turns except the most recent RECENT_TURNS_TO_KEEP.
    """
    try:
        turns = load_conversation_turns(conversation_id)
        real_turns = [t for t in turns if (t.get("role") or "") != SM_SUMMARY_ROLE]

        if len(real_turns) <= COMPRESS_THRESHOLD:
            return  # Not enough turns to compress

        # Turns to summarize: all except the most recent N
        old_turns = real_turns[:-RECENT_TURNS_TO_KEEP]
        if len(old_turns) < 2:
            return  # Need at least 2 turns to make a summary worthwhile

        # Check if existing summary already covers these turns
        existing = _get_existing_summary(conversation_id)
        if existing:
            covers_up_to = (existing.get("variables") or {}).get("covers_up_to_turn", 0) if isinstance(existing.get("variables"), dict) else 0
            max_old_turn = max(t.get("turn_number", 0) for t in old_turns)
            if covers_up_to >= max_old_turn:
                return  # Summary is already up to date

        # Build the text to summarize
        parts = []
        for t in old_turns:
            role = (t.get("role") or "user").strip().lower()
            content = (t.get("content") or "").strip()
            if role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        history_text = "\n\n".join(parts)

        # Call gpt-4o-mini to generate summary
        summary_text = _call_summarizer(history_text)
        if not summary_text:
            return

        covers_up_to_turn = max(t.get("turn_number", 0) for t in old_turns)

        # Upsert: update existing summary or create new one
        if existing:
            supabase.table("conversation_turns").update({
                "content": summary_text,
                "variables": {
                    "covers_up_to_turn": covers_up_to_turn,
                    "turns_summarized": len(old_turns),
                    "raw_chars": len(history_text),
                    "summary_chars": len(summary_text),
                },
            }).eq("id", existing["id"]).execute()
        else:
            supabase.table("conversation_turns").insert({
                "conversation_id": conversation_id,
                "turn_number": -1,  # Synthetic turn, sorts before real turns
                "role": SM_SUMMARY_ROLE,
                "content": summary_text,
                "variables": {
                    "covers_up_to_turn": covers_up_to_turn,
                    "turns_summarized": len(old_turns),
                    "raw_chars": len(history_text),
                    "summary_chars": len(summary_text),
                },
            }).execute()

        _logger.info(
            "SM summary generated: %d turns → %d chars (was %d chars), conversation=%s",
            len(old_turns), len(summary_text), len(history_text), conversation_id[:8],
        )

    except Exception as e:
        _logger.warning("SM summary generation failed: %s", e)


def _call_summarizer(history_text: str) -> str | None:
    """Call gpt-4o-mini to compress conversation history into a summary."""
    api_key = os.environ.get("SYSTEM_OPENAI_API_KEY")
    if not api_key:
        _logger.warning("SYSTEM_OPENAI_API_KEY not set, cannot generate summary")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        target_chars = max(200, len(history_text) // 4)  # Target ~25% of original

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Compress the following conversation into a concise summary of about {target_chars} characters. "
                        "Preserve all key facts, decisions, user preferences, questions asked, problems discussed, "
                        "and the current state of the discussion. Write in third person past tense. "
                        "This summary will replace the raw conversation history in future LLM calls, "
                        "so it must contain everything the assistant needs to continue the conversation coherently."
                    ),
                },
                {"role": "user", "content": history_text},
            ],
            max_tokens=target_chars // 3,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        _logger.warning("Summarizer call failed: %s", e)
        return None
