from fastapi import APIRouter, HTTPException, Body
from supabase_client import supabase

router = APIRouter()

# Plan limits
PLAN_LIMITS = {
    "free": {"orgs": 1, "members": 1},
    "starter": {"orgs": 1, "members": 3},
    "team": {"orgs": 1, "members": 20},
    "pro": {"orgs": 1, "members": 99999},
    "enterprise": {"orgs": 99999, "members": 99999},
}

def get_org_plan(org):
    return org.get("plan", "free")

@router.get("/api/organizations/test")
def test_connection():
    """Test endpoint to check if the router and database connection work"""
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase client not initialized"}
        
        # Test basic query
        result = supabase.table("organizations").select("count").limit(1).execute()
        return {"status": "success", "message": "Database connection working", "data": result.data}
    except Exception as e:
        return {"status": "error", "message": f"Database connection failed: {str(e)}"}

@router.post("/api/organizations/create")
def create_organization(user_id: str = Body(...), org_name: str = Body(...), plan: str = Body("free")):
    try:
        print(f"Creating organization: user_id={user_id}, org_name={org_name}, plan={plan}")
        
        # Check if Supabase client is initialized
        if not supabase:
            print("Error: Supabase client not initialized")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Test basic Supabase connection first
        try:
            test_result = supabase.table("organizations").select("count").limit(1).execute()
            print("Supabase connection test successful")
        except Exception as db_error:
            print(f"Database connection test failed: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(db_error)}")
        
        # 1. Fetch user's orgs
        print("Fetching user's organizations...")
        orgs_result = supabase.table("organizations").select("*").eq("created_by", user_id).execute()
        orgs = orgs_result.data if orgs_result.data else []
        print(f"Found {len(orgs)} existing organizations")
        
        # Use the highest plan among user's orgs, or the provided plan
        user_plan = plan
        if orgs:
            user_plan = max([get_org_plan(o) for o in orgs] + [plan])
        
        org_limit = PLAN_LIMITS.get(user_plan, PLAN_LIMITS["free"])["orgs"]
        if len(orgs) >= org_limit:
            raise HTTPException(status_code=403, detail=f"Your plan ({user_plan}) only allows {org_limit} organization(s).")
        
        # 2. Create org
        print("Creating new organization...")
        new_org_result = supabase.table("organizations").insert({
            "name": org_name, 
            "created_by": user_id, 
            "plan": user_plan,
            "type": "Organization"
        }).execute()
        
        if not new_org_result.data:
            print("Error: No data returned from organization creation")
            raise HTTPException(status_code=500, detail="Failed to create organization")
        
        print(f"Organization created successfully: {new_org_result.data}")
        return new_org_result.data
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in create_organization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/api/organizations/invite")
def invite_member(org_id: str = Body(...), email: str = Body(...)):
    # 1. Fetch org and plan
    org = supabase.table("organizations").select("*").eq("id", org_id).single().execute().data
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found.")
    plan = get_org_plan(org)
    members = supabase.table("organization_members").select("*").eq("org_id", org_id).eq("status", "active").execute().data
    member_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["members"]
    if len(members) >= member_limit:
        raise HTTPException(status_code=403, detail=f"Member limit reached for your plan ({plan}).")
    # 2. Add member
    new_member = supabase.table("organization_members").insert({"org_id": org_id, "invited_email": email, "status": "pending"}).execute()
    return new_member.data
