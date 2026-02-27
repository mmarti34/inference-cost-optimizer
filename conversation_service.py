"""
Conversation state for multi-turn public API calls.
When conversation_id is provided, we load history, trim to fit, and prepend to AI step messages;
after completion we save the user turn and assistant turn.
"""
import uuid
from datetime import datetime, timezone
from typing import Any

from supabase_client import supabase


# Rough token estimate: ~4 chars per token for English
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 8000
DEFAULT_MAX_TURNS = 50


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
