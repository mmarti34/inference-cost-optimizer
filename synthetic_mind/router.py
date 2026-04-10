"""
Synthetic Mind — API endpoints.

POST /api/synthetic-mind/consolidate       — trigger consolidation for an org
GET  /api/synthetic-mind/stats             — observation + memory stats
GET  /api/synthetic-mind/memories          — list active memories for an org
POST /api/synthetic-mind/forget            — run forgetting cycle
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional

from auth_dependency import require_auth
from supabase_client import supabase
from synthetic_mind.consolidation import consolidate_org
from synthetic_mind.forgetting import run_forgetting_cycle
from synthetic_mind.memory_store import get_active_memories
from synthetic_mind.prompt_assembler import generate_memory_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/synthetic-mind", tags=["synthetic-mind"])

# Secret for cron-triggered consolidation (no user auth needed)
import os
CRON_SECRET = os.getenv("SM_CRON_SECRET", "sm-cron-default-secret")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ConsolidateRequest(BaseModel):
    org_id: str


class ConsolidateResponse(BaseModel):
    observations_processed: int
    memory_units_created: int
    memory_units_updated: int
    abstractions_formed: int


class StatsResponse(BaseModel):
    total_observations: int
    unconsolidated_observations: int
    total_memory_units: int
    active_memory_units: int
    forgotten_memory_units: int
    consolidation_runs: int
    memory_summary_preview: Optional[str] = None


class MemoryListResponse(BaseModel):
    memories: list[dict]
    count: int


class ForgetResponse(BaseModel):
    low_confidence_forgotten: int
    superseded_forgotten: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/consolidate-cron")
async def cron_consolidation(request: Request):
    """
    Cron-triggered consolidation for ALL orgs with unconsolidated observations.
    Secured by SM_CRON_SECRET header, no user auth needed.
    Called by Railway cron or external scheduler.
    """
    secret = request.headers.get("x-sm-cron-secret", "")
    if secret != CRON_SECRET:
        raise HTTPException(status_code=403, detail="Invalid cron secret")

    try:
        # Find all orgs with unconsolidated observations
        r = supabase.table("sm_observations").select(
            "org_id", count="exact"
        ).eq("consolidated", False).execute()

        org_ids = list({row["org_id"] for row in (r.data or [])})
        if not org_ids:
            return {"message": "No unconsolidated observations", "orgs_processed": 0}

        results = []
        for org_id in org_ids:
            stats = consolidate_org(org_id)
            run_forgetting_cycle(org_id)
            results.append({"org_id": org_id, **stats})

        return {
            "message": f"Consolidated {len(org_ids)} orgs",
            "orgs_processed": len(org_ids),
            "results": results,
        }
    except Exception as e:
        logger.error("Cron consolidation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/observations")
async def list_observations(
    org_id: str,
    limit: int = 50,
    endpoint_slug: Optional[str] = None,
    user=Depends(require_auth),
):
    """List recent observations for an org."""
    _verify_org_access(user, org_id)
    try:
        query = (
            supabase.table("sm_observations")
            .select("*")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if endpoint_slug:
            query = query.eq("endpoint_slug", endpoint_slug)
        r = query.execute()
        return {"observations": r.data or [], "count": len(r.data or [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate", response_model=ConsolidateResponse)
async def trigger_consolidation(body: ConsolidateRequest, user=Depends(require_auth)):
    """
    Trigger a consolidation cycle for the specified org.
    Processes unconsolidated observations into memory units.
    """
    # Verify user has access to this org
    _verify_org_access(user, body.org_id)

    try:
        stats = consolidate_org(body.org_id)
        return ConsolidateResponse(**stats)
    except Exception as e:
        logger.error("Consolidation trigger failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats(org_id: str, user=Depends(require_auth)):
    """Get Synthetic Mind statistics for an org."""
    _verify_org_access(user, org_id)

    try:
        # Total observations
        total_obs = supabase.table("sm_observations").select("id", count="exact").eq("org_id", org_id).execute()
        total_observations = total_obs.count or 0

        # Unconsolidated
        uncons = supabase.table("sm_observations").select("id", count="exact").eq("org_id", org_id).eq("consolidated", False).execute()
        unconsolidated = uncons.count or 0

        # Memory units
        all_mem = supabase.table("sm_memory_units").select("id", count="exact").eq("org_id", org_id).execute()
        total_mem = all_mem.count or 0

        active_mem = supabase.table("sm_memory_units").select("id", count="exact").eq("org_id", org_id).is_("forgotten_at", "null").execute()
        active = active_mem.count or 0

        forgotten = total_mem - active

        # Consolidation runs
        runs = supabase.table("sm_consolidation_runs").select("id", count="exact").eq("org_id", org_id).execute()
        run_count = runs.count or 0

        # Preview memory summary
        summary = generate_memory_summary(org_id)

        return StatsResponse(
            total_observations=total_observations,
            unconsolidated_observations=unconsolidated,
            total_memory_units=total_mem,
            active_memory_units=active,
            forgotten_memory_units=forgotten,
            consolidation_runs=run_count,
            memory_summary_preview=summary,
        )
    except Exception as e:
        logger.error("Stats fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    org_id: str,
    workflow_id: Optional[str] = None,
    memory_type: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
    user=Depends(require_auth),
):
    """List active memory units for an org, optionally filtered by workflow."""
    _verify_org_access(user, org_id)

    memories = get_active_memories(
        org_id,
        workflow_id=workflow_id,
        memory_type=memory_type,
        min_confidence=min_confidence,
        limit=limit,
    )
    return MemoryListResponse(memories=memories, count=len(memories))


@router.post("/forget", response_model=ForgetResponse)
async def trigger_forgetting(body: ConsolidateRequest, user=Depends(require_auth)):
    """Run a forgetting cycle — prune low-confidence memories."""
    _verify_org_access(user, body.org_id)

    try:
        stats = run_forgetting_cycle(body.org_id)
        return ForgetResponse(**stats)
    except Exception as e:
        logger.error("Forgetting cycle failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Forgetting failed: {str(e)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_org_access(user, org_id: str) -> None:
    """Verify the authenticated user belongs to the requested org."""
    user_id = getattr(user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        r = (
            supabase.table("organization_members")
            .select("id")
            .eq("user_id", user_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        if not r.data or len(r.data) == 0:
            raise HTTPException(status_code=403, detail="Not a member of this organization")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=403, detail="Access check failed")
