import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from supabase_client import supabase
from auth_dependency import require_auth, require_org_member, AuthenticatedUser

logger = logging.getLogger(__name__)


class OrganizationPlanUpdate(BaseModel):
    plan: str


class OrganizationUpdate(BaseModel):
    """Allowed fields for org update. Slug is never accepted — it is permanent after creation."""
    name: Optional[str] = None
    plan: Optional[str] = None
    logo: Optional[str] = None

    class Config:
        extra = "forbid"  # Reject any extra fields (e.g. slug) at parse time


def _org_update_payload_safe(data: dict) -> dict:
    """Strip slug from any org update payload. Never allow slug to be updated."""
    out = {k: v for k, v in data.items() if k != "slug"}
    return out


router = APIRouter()

# Pydantic models
class MemberResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    role: str
    status: str
    created_at: str
    updated_at: str

class UserProfileResponse(BaseModel):
    user_id: str
    subscription_tier: Optional[str]
    subscription_status: Optional[str]


def _resolve_member_emails(members: list) -> list:
    """
    For active members missing invited_email (e.g. org creator),
    look up their email from Supabase auth admin API.
    Returns enriched member list with 'email' field.
    """
    needs_lookup = [
        m for m in members
        if m.get("status") == "active" and m.get("user_id") and not m.get("invited_email")
    ]
    email_map: dict[str, str] = {}

    for m in needs_lookup:
        uid = m["user_id"]
        if uid in email_map:
            continue
        try:
            user_resp = supabase.auth.admin.get_user_by_id(uid)
            if user_resp and user_resp.user and user_resp.user.email:
                email_map[uid] = user_resp.user.email
        except Exception:
            pass  # Gracefully skip — will show invited_email or "Member"

    enriched = []
    for m in members:
        row = {**m}
        if row.get("invited_email"):
            row["email"] = row["invited_email"]
        elif row.get("user_id") and row["user_id"] in email_map:
            row["email"] = email_map[row["user_id"]]
        else:
            row["email"] = None
        # Keep user_id for member actions (remove) but don't display it
        enriched.append(row)
    return enriched


# Organization Member Endpoints
@router.get("/organizations/{org_id}/members/{user_id}/role")
async def get_member_role(
    org_id: str,
    user_id: str,
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    """Get member role for a specific user in an organization"""
    try:
        result = supabase.table("organization_members").select("role, status").eq("org_id", org_id).eq("user_id", user_id).eq("status", "active").single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Member not found or not active")

        return {
            "role": result.data["role"],
            "status": result.data["status"],
            "is_admin": result.data["role"] == "admin"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching member role: {str(e)}")

@router.get("/organizations/{org_id}/members")
async def get_organization_members(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Get all members of an organization. Requires authenticated org member."""
    try:
        result = supabase.table("organization_members").select(
            "id, org_id, user_id, role, status, invited_email, created_at"
        ).eq("org_id", org_id).order("created_at", desc=True).execute()

        if not result.data:
            return []

        return _resolve_member_emails(result.data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching organization members: {str(e)}")

@router.get("/organizations/{org_id}/join-requests")
async def get_join_requests(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Get pending join requests for an organization"""
    try:
        result = supabase.table("join_requests").select("id").eq("org_id", org_id).eq("status", "pending").execute()

        if not result.data:
            return []

        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching join requests: {str(e)}")

@router.get("/user-profiles/{user_id}/subscription")
async def get_user_subscription(
    user_id: str,
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    """Get user's subscription information. User can only query their own profile."""
    # Only allow users to query their own subscription
    if auth_user.user_id != user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's subscription.")
    try:
        result = supabase.table("user_profiles").select("subscription_tier, subscription_status").eq("user_id", user_id).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        return {
            "subscription_tier": result.data.get("subscription_tier"),
            "subscription_status": result.data.get("subscription_status")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user subscription: {str(e)}")

@router.put("/organizations/{org_id}/plan")
async def update_organization_plan(
    org_id: str,
    plan_data: OrganizationPlanUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update organization plan. Slug is never updated."""
    try:
        update_data = _org_update_payload_safe({"plan": plan_data.plan})
        result = supabase.table("organizations").update(update_data).eq("id", org_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Organization not found")

        return {"message": "Organization plan updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating organization plan: {str(e)}")


@router.patch("/organizations/{org_id}")
async def update_organization(
    org_id: str,
    payload: OrganizationUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update organization (name, plan, logo). Slug is immutable and must never be sent or updated."""
    try:
        update_data = {}
        if payload.name is not None:
            update_data["name"] = payload.name
        if payload.plan is not None:
            update_data["plan"] = payload.plan
        if payload.logo is not None:
            update_data["logo"] = payload.logo
        if not update_data:
            raise HTTPException(status_code=400, detail="No allowed fields provided (name, plan, logo). Slug cannot be updated.")
        update_data = _org_update_payload_safe(update_data)
        result = supabase.table("organizations").update(update_data).eq("id", org_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Organization not found")
        return {"message": "Organization updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating organization: {str(e)}")
