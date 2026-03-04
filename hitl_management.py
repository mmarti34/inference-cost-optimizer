"""
Human-in-the-Loop (HITL) review management.
When a human_review node is reached during workflow execution, a pending review is created.
Reviewers can list, approve, or reject reviews via these endpoints.
"""
import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from supabase_client import supabase
from auth_dependency import require_org_member, AuthenticatedUser

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ReviewResolve(BaseModel):
    status: str  # 'approved' or 'rejected'
    reviewer_response: Optional[str] = None
    reviewer_email: Optional[str] = None


# Endpoints

@router.get("/reviews/{org_id}")
async def list_reviews(
    org_id: str,
    status: Optional[str] = None,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """List pending reviews for an org, optionally filtered by status."""
    try:
        query = (
            supabase.table("pending_reviews")
            .select("*")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
        )
        if status:
            query = query.eq("status", status)
        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.error("Failed to list reviews for org %s: %s", org_id, e)
        raise HTTPException(status_code=500, detail="Failed to list reviews")


@router.get("/reviews/{org_id}/{review_id}")
async def get_review(
    org_id: str,
    review_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Get a single review by ID."""
    try:
        result = (
            supabase.table("pending_reviews")
            .select("*")
            .eq("id", review_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Review not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get review %s: %s", review_id, e)
        raise HTTPException(status_code=500, detail="Failed to get review")


@router.post("/reviews/{org_id}/{review_id}/resolve")
async def resolve_review(
    org_id: str,
    review_id: str,
    body: ReviewResolve,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Approve or reject a pending review."""
    if body.status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Status must be 'approved' or 'rejected'")

    try:
        # Check current status
        existing = (
            supabase.table("pending_reviews")
            .select("id, status")
            .eq("id", review_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Review not found")
        if existing.data[0]["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"Review already {existing.data[0]['status']}")

        now = datetime.now(timezone.utc).isoformat()
        result = (
            supabase.table("pending_reviews")
            .update({
                "status": body.status,
                "reviewer_response": body.reviewer_response or "",
                "reviewer_email": body.reviewer_email or auth_user.email or "",
                "resolved_at": now,
                "updated_at": now,
            })
            .eq("id", review_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Review not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve review %s: %s", review_id, e)
        raise HTTPException(status_code=500, detail="Failed to resolve review")


@router.get("/reviews/{org_id}/count/pending")
async def count_pending_reviews(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Get count of pending reviews for badge display."""
    try:
        result = (
            supabase.table("pending_reviews")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("status", "pending")
            .execute()
        )
        return {"count": result.count or 0}
    except Exception as e:
        logger.error("Failed to count pending reviews: %s", e)
        raise HTTPException(status_code=500, detail="Failed to count reviews")
