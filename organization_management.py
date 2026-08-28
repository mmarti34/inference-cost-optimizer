import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import audit
from supabase_client import supabase
from auth_dependency import require_auth, require_org_member, require_org_admin, AuthenticatedUser
from plan_enforcement import get_monthly_usage, get_org_plan_tier, _get_plan_limit

logger = logging.getLogger(__name__)


class OrganizationUpdate(BaseModel):
    """
    Allowed fields for a client org update.

    `slug` is permanent after creation. `plan` is BILLING STATE and is not
    client-writable at all: every user is admin of their own Personal
    workspace, so accepting a plan string here let anyone set
    plan='enterprise' and turn every limit in PLAN_LIMITS into -1 (unlimited).
    The only writer of organizations.plan is the Stripe webhook
    (stripe_webhook._propagate_tier_to_orgs).
    """
    name: Optional[str] = None
    logo: Optional[str] = None

    class Config:
        extra = "forbid"  # Reject any extra fields (e.g. slug, plan) at parse time


#: Columns a client may never write on `organizations`, whatever the payload says.
_ORG_CLIENT_WRITABLE_FIELDS = frozenset({"name", "logo"})


def _org_update_payload_safe(data: dict) -> dict:
    """
    Reduce an org update payload to the client-writable allow-list.

    Deny-by-default: anything not explicitly listed (slug, plan, type,
    created_by, ...) is dropped rather than passed through to Supabase.
    """
    return {k: v for k, v in data.items() if k in _ORG_CLIENT_WRITABLE_FIELDS}


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
        result = supabase.table("organization_members").select("role, status").eq("org_id", org_id).eq("user_id", user_id).eq("status", "active").limit(1).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="Member not found or not active")

        row = result.data[0]
        return {
            "role": row["role"],
            "status": row["status"],
            "is_admin": row["role"] == "admin"
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
        result = supabase.table("user_profiles").select("subscription_tier, subscription_status").eq("user_id", user_id).maybe_single().execute()

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

@router.get("/organizations/{org_id}/usage/monthly")
async def get_org_monthly_usage(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Return the current month's API request count, plan tier, and monthly limit.
    Used by the frontend to render the usage bar in the sidebar.
    """
    try:
        plan = get_org_plan_tier(org_id)
        limit = _get_plan_limit(plan, "requests_per_month")
        count = get_monthly_usage(org_id)
        return {
            "plan": plan,
            "request_count": count,
            "limit": limit,  # -1 = unlimited
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching monthly usage: {str(e)}")


# NOTE: `PUT /organizations/{org_id}/plan` was removed deliberately.
# It accepted an arbitrary plan string with no Stripe validation, and every user
# is admin of their own Personal workspace — so it was a one-request upgrade to
# `enterprise` (unlimited everything). Plan changes now come exclusively from
# the Stripe webhook. It had no frontend caller.


@router.patch("/organizations/{org_id}")
async def update_organization(
    request: Request,
    org_id: str,
    payload: OrganizationUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_admin),
):
    """
    Update organization (name, logo). Admin only.

    Slug is immutable. Plan is billing state and is rejected outright — it is
    settable only by the Stripe webhook.
    """
    try:
        update_data = {}
        if payload.name is not None:
            update_data["name"] = payload.name
        if payload.logo is not None:
            update_data["logo"] = payload.logo
        update_data = _org_update_payload_safe(update_data)
        if not update_data:
            raise HTTPException(
                status_code=400,
                detail="No allowed fields provided (name, logo). Slug and plan cannot be updated.",
            )
        result = supabase.table("organizations").update(update_data).eq("id", org_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Organization not found")
        # The only org-settings mutation a client can reach. `name` and `logo`
        # are the only writable columns (see _ORG_CLIENT_WRITABLE_FIELDS) and
        # both are customer free text, so the VALUES are not recorded — the row
        # id and the actor are what an incident timeline needs.
        audit.record(
            audit.ORGANIZATION_UPDATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORGANIZATION,
            resource_id=org_id,
            request=request,
        )
        return {"message": "Organization updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating organization: {str(e)}")
