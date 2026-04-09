"""
Synthetic Mind — Forgetting Engine.

Manages what to forget: prevents memory bloat while preserving critical info.

Forgetting criteria:
  - Confidence decay below threshold
  - Contradiction (superseded by newer observations)
  - Irrelevance (never retrieved)
  - Scope death (deleted users, archived workflows)

Safety constraints:
  - Protected memories are never auto-forgotten
  - Every forgetting event is logged (what, when, why)
  - Soft delete with configurable cold-storage retention
  - Budget: max N memories forgotten per cycle
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)

# Max memories to forget per cycle (prevents catastrophic loss)
DEFAULT_FORGETTING_BUDGET = 50

# Confidence threshold below which memories become candidates
DEFAULT_CONFIDENCE_THRESHOLD = 0.05


def forget_low_confidence(
    org_id: str,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    budget: int = DEFAULT_FORGETTING_BUDGET,
) -> int:
    """
    Soft-delete memory units whose confidence has decayed below threshold.
    Protected memories are never forgotten.
    Returns count of forgotten memories.
    """
    try:
        # Fetch candidates
        r = (
            supabase.table("sm_memory_units")
            .select("id, subject, predicate, confidence")
            .eq("org_id", org_id)
            .is_("forgotten_at", "null")
            .eq("protected", False)
            .lte("confidence", threshold)
            .order("confidence", desc=False)
            .limit(budget)
            .execute()
        )
        candidates = r.data or []
        if not candidates:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        forgotten = 0
        for mem in candidates:
            try:
                supabase.table("sm_memory_units").update({
                    "forgotten_at": now,
                    "forgotten_reason": f"confidence_decay:below_{threshold}",
                }).eq("id", mem["id"]).execute()
                forgotten += 1
                logger.info(
                    "Forgot memory %s (%s/%s) confidence=%.4f",
                    mem["id"], mem["subject"], mem["predicate"], mem["confidence"],
                )
            except Exception as e:
                logger.warning("Failed to forget memory %s: %s", mem["id"], e)

        return forgotten

    except Exception as e:
        logger.warning("Forgetting engine failed for org %s: %s", org_id, e)
        return 0


def forget_superseded(org_id: str, budget: int = DEFAULT_FORGETTING_BUDGET) -> int:
    """
    Forget memories that have been superseded by newer ones.
    A memory is superseded when another memory with the same
    (subject, predicate) has higher confidence and a more recent
    last_confirmed_at.
    """
    # Phase 1: this is handled implicitly by upsert_memory_unit
    # which updates existing units rather than creating duplicates.
    # Full supersession tracking comes in Phase 2.
    return 0


def run_forgetting_cycle(org_id: str) -> dict[str, int]:
    """Run a complete forgetting cycle for an org."""
    stats = {
        "low_confidence_forgotten": forget_low_confidence(org_id),
        "superseded_forgotten": forget_superseded(org_id),
    }
    logger.info("Forgetting cycle for org %s: %s", org_id, stats)
    return stats
