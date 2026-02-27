"""Usage logs API. Frontend fetches via backend only (no direct Supabase)."""
from fastapi import APIRouter, HTTPException
from typing import List, Any

from supabase_client import supabase

router = APIRouter()


@router.get("/usage-logs/{org_id}")
async def get_usage_logs(org_id: str) -> List[Any]:
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
