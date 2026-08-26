"""Usage logs API. Frontend fetches via backend only (no direct Supabase)."""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Any

from supabase_client import supabase
from auth_dependency import require_org_member, AuthenticatedUser
from plan_enforcement import get_org_plan_tier, get_monthly_usage
from org_access_control import PLAN_LIMITS

router = APIRouter()


@router.get("/usage-logs/{org_id}")
async def get_usage_logs(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
) -> List[Any]:
    """List usage logs for an organization. Scoped by org_id."""
    try:
        result = (
            supabase.table("usage_logs")
            .select("id, org_id, user_id, project_id, provider, model, input_tokens, output_tokens, cost_usd, created_at")
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            return []
        sorted_data = sorted(result.data, key=lambda x: x.get("created_at", ""), reverse=True)
        return sorted_data
    except Exception as e:
        msg = str(e)
        if "permission denied" in msg.lower() or "policy" in msg.lower():
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Use service_role key for backend.",
            )
        raise HTTPException(status_code=500, detail=f"Error fetching usage logs: {msg}")


@router.get("/plan-usage/{org_id}")
async def get_plan_usage(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
) -> dict:
    """Return current resource usage vs plan limits for the billing page."""
    try:
        plan = get_org_plan_tier(org_id)
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

        # Count projects
        proj_result = supabase.table("projects").select("id", count="exact").eq("org_id", org_id).execute()
        project_count = proj_result.count if proj_result.count is not None else len(proj_result.data or [])

        # Count workflows
        wf_result = supabase.table("workflows").select("id", count="exact").eq("org_id", org_id).execute()
        workflow_count = wf_result.count if wf_result.count is not None else len(wf_result.data or [])

        # Count active members
        mem_result = (
            supabase.table("organization_members")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("status", "active")
            .execute()
        )
        member_count = mem_result.count if mem_result.count is not None else len(mem_result.data or [])

        # Count active server keys
        key_result = (
            supabase.table("service_api_keys")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("status", "active")
            .execute()
        )
        key_count = key_result.count if key_result.count is not None else len(key_result.data or [])

        # Monthly API requests
        monthly_requests = get_monthly_usage(org_id)

        return {
            "plan": plan,
            "usage": {
                "projects": {"current": project_count, "limit": limits.get("projects", 0)},
                "workflows": {"current": workflow_count, "limit": limits.get("workflows", 0)},
                "members": {"current": member_count, "limit": limits.get("members", 0)},
                "server_keys": {"current": key_count, "limit": limits.get("server_keys", 0)},
                "requests_per_month": {"current": monthly_requests, "limit": limits.get("requests_per_month", 0)},
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching plan usage: {str(e)}")
