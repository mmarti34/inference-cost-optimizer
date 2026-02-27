from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from supabase_client import supabase

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
    if "slug" in data:
        # Log or ignore; slug must remain immutable
        pass
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

# Organization Member Endpoints
@router.get("/organizations/{org_id}/members/{user_id}/role")
async def get_member_role(org_id: str, user_id: str):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching member role: {str(e)}")

@router.get("/organizations/{org_id}/members")
async def get_organization_members(org_id: str):
    """Get all members of an organization"""
    try:
        result = supabase.table("organization_members").select("id, org_id, user_id, role, status, invited_email, created_at").eq("org_id", org_id).order("created_at", desc=True).execute()
        
        if not result.data:
            return []
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching organization members: {str(e)}")

@router.get("/organizations/{org_id}/join-requests")
async def get_join_requests(org_id: str):
    """Get pending join requests for an organization"""
    try:
        result = supabase.table("join_requests").select("id").eq("org_id", org_id).eq("status", "pending").execute()
        
        if not result.data:
            return []
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching join requests: {str(e)}")

@router.get("/user-profiles/{user_id}/subscription")
async def get_user_subscription(user_id: str):
    """Get user's subscription information"""
    try:
        result = supabase.table("user_profiles").select("subscription_tier, subscription_status").eq("user_id", user_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return {
            "subscription_tier": result.data.get("subscription_tier"),
            "subscription_status": result.data.get("subscription_status")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user subscription: {str(e)}")

@router.put("/organizations/{org_id}/plan")
async def update_organization_plan(org_id: str, plan_data: OrganizationPlanUpdate):
    """Update organization plan. Slug is never updated."""
    try:
        update_data = _org_update_payload_safe({"plan": plan_data.plan})
        result = supabase.table("organizations").update(update_data).eq("id", org_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        return {"message": "Organization plan updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating organization plan: {str(e)}")


@router.patch("/organizations/{org_id}")
async def update_organization(org_id: str, payload: OrganizationUpdate):
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
