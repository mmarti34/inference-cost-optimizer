"""
Synthetic Mind — Memory Store.

CRUD operations for memory units, scopes, and observations.
All DB access goes through the Supabase service-role client.
"""
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)

# Default confidence half-life: memory loses half its confidence
# after this many days without re-confirmation.
DEFAULT_HALF_LIFE_DAYS = 30


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def get_unconsolidated_observations(
    org_id: str,
    limit: int = 200,
) -> list[dict]:
    """Fetch oldest unconsolidated observations for an org."""
    try:
        r = (
            supabase.table("sm_observations")
            .select("*")
            .eq("org_id", org_id)
            .eq("consolidated", False)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return r.data or []
    except Exception as e:
        logger.warning("Failed to fetch unconsolidated observations: %s", e)
        return []


def mark_observations_consolidated(observation_ids: list[str]) -> None:
    """Mark a batch of observations as consolidated."""
    if not observation_ids:
        return
    try:
        for obs_id in observation_ids:
            supabase.table("sm_observations").update(
                {"consolidated": True}
            ).eq("id", obs_id).execute()
    except Exception as e:
        logger.warning("Failed to mark observations consolidated: %s", e)


def get_observations_for_workflow(
    org_id: str,
    workflow_id: str,
    limit: int = 100,
) -> list[dict]:
    """Fetch recent observations for a specific workflow."""
    try:
        r = (
            supabase.table("sm_observations")
            .select("*")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return r.data or []
    except Exception as e:
        logger.warning("Failed to fetch workflow observations: %s", e)
        return []


# ---------------------------------------------------------------------------
# Memory Units
# ---------------------------------------------------------------------------

def upsert_memory_unit(unit: dict[str, Any]) -> Optional[str]:
    """
    Insert or update a memory unit.  If a unit with the same
    (org_id, subject, predicate, scope) exists and is active,
    boost its confidence instead of creating a duplicate.
    Returns the memory unit ID.
    """
    try:
        # Check for existing active unit with same subject+predicate in scope
        query = (
            supabase.table("sm_memory_units")
            .select("id, confidence, confirmed_count, evidence")
            .eq("org_id", unit["org_id"])
            .eq("subject", unit["subject"])
            .eq("predicate", unit["predicate"])
            .is_("forgotten_at", "null")
        )
        if unit.get("workflow_id"):
            query = query.eq("workflow_id", unit["workflow_id"])
        if unit.get("user_id"):
            query = query.eq("user_id", unit["user_id"])

        r = query.limit(1).execute()

        if r.data and len(r.data) > 0:
            existing = r.data[0]
            # Boost confidence and merge evidence
            new_confidence = min(1.0, existing["confidence"] + 0.1)
            new_count = (existing.get("confirmed_count") or 1) + 1
            old_evidence = existing.get("evidence") or []
            new_evidence = list(set(old_evidence + (unit.get("evidence") or [])))[:50]

            supabase.table("sm_memory_units").update({
                "confidence": new_confidence,
                "confirmed_count": new_count,
                "last_confirmed_at": datetime.now(timezone.utc).isoformat(),
                "evidence": new_evidence,
                "object": unit["object"],
            }).eq("id", existing["id"]).execute()
            return existing["id"]
        else:
            r = supabase.table("sm_memory_units").insert(unit).execute()
            if r.data and len(r.data) > 0:
                return r.data[0]["id"]
            return unit.get("id")
    except Exception as e:
        logger.warning("Failed to upsert memory unit: %s", e)
        return None


def get_active_memories(
    org_id: str,
    *,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    memory_type: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
) -> list[dict]:
    """Fetch active (non-forgotten) memory units for a scope."""
    try:
        query = (
            supabase.table("sm_memory_units")
            .select("*")
            .eq("org_id", org_id)
            .is_("forgotten_at", "null")
            .gte("confidence", min_confidence)
            .order("confidence", desc=True)
            .limit(limit)
        )
        if workflow_id:
            query = query.eq("workflow_id", workflow_id)
        if user_id:
            query = query.eq("user_id", user_id)
        if memory_type:
            query = query.eq("memory_type", memory_type)
        r = query.execute()
        return r.data or []
    except Exception as e:
        logger.warning("Failed to fetch active memories: %s", e)
        return []


def decay_confidence(org_id: str, half_life_days: int = DEFAULT_HALF_LIFE_DAYS) -> int:
    """
    Apply exponential confidence decay to all active memories for an org.
    confidence *= 2^(-days_since_last_confirmed / half_life)
    Returns number of units updated.
    """
    try:
        memories = (
            supabase.table("sm_memory_units")
            .select("id, confidence, last_confirmed_at, protected")
            .eq("org_id", org_id)
            .is_("forgotten_at", "null")
            .eq("protected", False)
            .execute()
        )
        updated = 0
        now = datetime.now(timezone.utc)
        for mem in (memories.data or []):
            last_confirmed = datetime.fromisoformat(
                mem["last_confirmed_at"].replace("Z", "+00:00")
            )
            days_elapsed = (now - last_confirmed).total_seconds() / 86400
            if days_elapsed < 1:
                continue
            decay_factor = math.pow(2, -days_elapsed / half_life_days)
            new_confidence = round(mem["confidence"] * decay_factor, 4)
            if new_confidence < 0.01:
                new_confidence = 0.0
            if abs(new_confidence - mem["confidence"]) > 0.001:
                supabase.table("sm_memory_units").update(
                    {"confidence": new_confidence}
                ).eq("id", mem["id"]).execute()
                updated += 1
        return updated
    except Exception as e:
        logger.warning("Failed to decay confidence: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

def ensure_scope(
    org_id: str,
    scope_type: str,
    entity_id: Optional[str] = None,
    entity_slug: Optional[str] = None,
    parent_scope_id: Optional[str] = None,
) -> Optional[str]:
    """Get or create a memory scope. Returns scope ID."""
    try:
        query = (
            supabase.table("sm_memory_scopes")
            .select("id")
            .eq("org_id", org_id)
            .eq("scope_type", scope_type)
        )
        if entity_id:
            query = query.eq("entity_id", entity_id)
        r = query.limit(1).execute()

        if r.data and len(r.data) > 0:
            return r.data[0]["id"]

        row = {
            "org_id": org_id,
            "scope_type": scope_type,
            "entity_id": entity_id,
            "entity_slug": entity_slug,
            "parent_scope_id": parent_scope_id,
        }
        r = supabase.table("sm_memory_scopes").insert(row).execute()
        if r.data and len(r.data) > 0:
            return r.data[0]["id"]
        return None
    except Exception as e:
        logger.warning("Failed to ensure scope: %s", e)
        return None


# ---------------------------------------------------------------------------
# Consolidation Runs
# ---------------------------------------------------------------------------

def start_consolidation_run(org_id: str) -> Optional[str]:
    """Create a new consolidation run record. Returns run ID."""
    try:
        r = supabase.table("sm_consolidation_runs").insert({
            "org_id": org_id,
            "status": "running",
        }).execute()
        if r.data and len(r.data) > 0:
            return r.data[0]["id"]
        return None
    except Exception as e:
        logger.warning("Failed to start consolidation run: %s", e)
        return None


def complete_consolidation_run(
    run_id: str,
    *,
    observations_processed: int = 0,
    memory_units_created: int = 0,
    memory_units_updated: int = 0,
    abstractions_formed: int = 0,
    status: str = "completed",
    error_message: Optional[str] = None,
) -> None:
    """Update a consolidation run with results."""
    try:
        supabase.table("sm_consolidation_runs").update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "observations_processed": observations_processed,
            "memory_units_created": memory_units_created,
            "memory_units_updated": memory_units_updated,
            "abstractions_formed": abstractions_formed,
            "error_message": (error_message or "")[:500] if error_message else None,
        }).eq("id", run_id).execute()
    except Exception as e:
        logger.warning("Failed to complete consolidation run: %s", e)


# ---------------------------------------------------------------------------
# KB Consolidations (SM v2 Phase 2)
# ---------------------------------------------------------------------------

def get_kb_consolidation(
    org_id: str,
    asset_ids: list[str],
    min_confidence: float = 0.7,
    scope_value: Optional[str] = None,
) -> dict | None:
    """
    Look up an existing KB consolidation for a set of asset IDs.
    When scope_value is provided, only returns consolidations for that scope.
    Returns the consolidation record if found and valid, None otherwise.
    """
    if not asset_ids:
        return None
    sorted_ids = sorted(asset_ids)
    try:
        query = (
            supabase.table("sm_kb_consolidations")
            .select("*")
            .eq("org_id", org_id)
            .contains("asset_ids", sorted_ids)
            .gte("confidence", min_confidence)
            .order("confidence", desc=True)
            .limit(1)
        )
        if scope_value:
            query = query.eq("scope_value", scope_value)
        else:
            query = query.is_("scope_value", "null")
        r = query.execute()
        if r.data and len(r.data) > 0:
            rec = r.data[0]
            # Check that asset_ids match exactly (contains is superset check)
            if sorted(rec.get("asset_ids", [])) == sorted_ids:
                return rec
    except Exception as e:
        logger.warning("Failed to lookup KB consolidation: %s", e)
    return None


def upsert_kb_consolidation(
    org_id: str,
    asset_ids: list[str],
    consolidated_text: str,
    raw_token_count: int,
    consolidated_token_count: int,
    asset_versions_hash: str,
    scope_value: Optional[str] = None,
) -> str | None:
    """Create or update a KB consolidation record. Returns record ID."""
    sorted_ids = sorted(asset_ids)
    try:
        # Check for existing (scoped)
        existing = get_kb_consolidation(org_id, sorted_ids, min_confidence=0.0, scope_value=scope_value)
        if existing:
            supabase.table("sm_kb_consolidations").update({
                "consolidated_text": consolidated_text,
                "raw_token_count": raw_token_count,
                "consolidated_token_count": consolidated_token_count,
                "asset_versions_hash": asset_versions_hash,
                "confidence": min(1.0, existing.get("confidence", 0.5) + 0.1),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", existing["id"]).execute()
            return existing["id"]
        else:
            row = {
                "org_id": org_id,
                "asset_ids": sorted_ids,
                "consolidated_text": consolidated_text,
                "raw_token_count": raw_token_count,
                "consolidated_token_count": consolidated_token_count,
                "confidence": 0.5,
                "hit_count": 0,
                "asset_versions_hash": asset_versions_hash,
            }
            if scope_value:
                row["scope_value"] = scope_value
            r = supabase.table("sm_kb_consolidations").insert(row).execute()
            if r.data and len(r.data) > 0:
                return r.data[0]["id"]
    except Exception as e:
        logger.warning("Failed to upsert KB consolidation: %s", e)
    return None


def bump_kb_consolidation_hit(consolidation_id: str) -> None:
    """Increment hit count and boost confidence when a consolidation is served."""
    try:
        rec = supabase.table("sm_kb_consolidations").select("hit_count, confidence").eq("id", consolidation_id).limit(1).execute()
        if rec.data:
            supabase.table("sm_kb_consolidations").update({
                "hit_count": (rec.data[0].get("hit_count") or 0) + 1,
                "confidence": min(1.0, (rec.data[0].get("confidence") or 0.5) + 0.05),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", consolidation_id).execute()
    except Exception as e:
        logger.warning("Failed to bump KB consolidation hit: %s", e)
