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

def get_upgrade_suggestion(plan):
    """Get upgrade suggestion based on current plan"""
    if plan == "free":
        return " Consider upgrading to Starter plan ($29/mo) for up to 3 team members."
    elif plan == "starter":
        return " Consider upgrading to Team plan ($99/mo) for up to 20 team members."
    elif plan == "team":
        return " Consider upgrading to Pro plan ($299/mo) for unlimited team members."
    return ""

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
        
        # 1. Get user's actual plan from user_profiles table
        print("Fetching user's actual plan...")
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).single().execute()
        user_actual_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_actual_plan = user_profile_result.data["subscription_tier"]
        print(f"User's actual plan: {user_actual_plan}")
        
        # Use the provided plan or user's actual plan, whichever is higher
        plan_priority = {"free": 0, "starter": 1, "team": 2, "pro": 3, "enterprise": 4}
        effective_plan = max([plan, user_actual_plan], key=lambda p: plan_priority.get(p, 0))
        print(f"Effective plan for organization: {effective_plan}")
        
        # 2. Fetch user's orgs (only count Organization type, not Personal)
        print("Fetching user's organizations...")
        orgs_result = supabase.table("organizations").select("*").eq("created_by", user_id).eq("type", "Organization").execute()
        orgs = orgs_result.data if orgs_result.data else []
        print(f"Found {len(orgs)} existing Organization type organizations")
        
        org_limit = PLAN_LIMITS.get(effective_plan, PLAN_LIMITS["free"])["orgs"]
        if len(orgs) >= org_limit:
            upgrade_msg = get_upgrade_suggestion(effective_plan)
            raise HTTPException(status_code=403, detail=f"Your plan ({effective_plan}) only allows {org_limit} organization(s).{upgrade_msg}")
        
        # 3. Create org
        print("Creating new organization...")
        new_org_result = supabase.table("organizations").insert({
            "name": org_name, 
            "created_by": user_id, 
            "plan": effective_plan,
            "type": "Organization"
        }).execute()
        
        if not new_org_result.data:
            print("Error: No data returned from organization creation")
            raise HTTPException(status_code=500, detail="Failed to create organization")
        
        new_org = new_org_result.data[0] if isinstance(new_org_result.data, list) else new_org_result.data
        org_id = new_org["id"]
        
        # 4. Add user as admin member
        print("Adding user as admin member...")
        member_result = supabase.table("organization_members").insert({
            "org_id": org_id,
            "user_id": user_id,
            "role": "admin",
            "status": "active"
        }).execute()
        
        if not member_result.data:
            print("Error: Failed to add user as admin member")
            # Clean up the created org
            supabase.table("organizations").delete().eq("id", org_id).execute()
            raise HTTPException(status_code=500, detail="Failed to add user as admin member")
        
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
    try:
        print(f"Inviting member to org {org_id}: {email}")
        
        # 1. Fetch org and plan
        org = supabase.table("organizations").select("*").eq("id", org_id).single().execute().data
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found.")
        
        plan = get_org_plan(org)
        print(f"Organization plan: {plan}")
        
        # 2. Count current active members in this org
        members = supabase.table("organization_members").select("*").eq("org_id", org_id).eq("status", "active").execute().data
        current_member_count = len(members) if members else 0
        print(f"Current member count: {current_member_count}")
        
        # 3. Check member limit for this org's plan
        member_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["members"]
        print(f"Member limit for {plan} plan: {member_limit}")
        
        if current_member_count >= member_limit:
            upgrade_msg = get_upgrade_suggestion(plan)
            raise HTTPException(
                status_code=403, 
                detail=f"Member limit reached for your plan ({plan}). You can have up to {member_limit} members.{upgrade_msg}"
            )
        
        # 4. Check if email is already invited or a member
        existing_invite = supabase.table("organization_members").select("*").eq("org_id", org_id).eq("invited_email", email).execute().data
        if existing_invite:
            raise HTTPException(status_code=400, detail="This email has already been invited or is already a member.")
        
        # 5. Add member invitation
        new_member = supabase.table("organization_members").insert({
            "org_id": org_id, 
            "invited_email": email, 
            "status": "invited"
        }).execute()
        
        print(f"Invitation sent successfully")
        return new_member.data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in invite_member: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/api/organizations/join")
def join_organization(user_id: str = Body(...), org_id: str = Body(...)):
    """Join an organization - check plan limits before allowing"""
    try:
        print(f"User {user_id} attempting to join organization {org_id}")
        
        # 1. Get user's actual plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        print(f"User's plan: {user_plan}")
        
        # 2. Check if user is already a member
        existing_member = supabase.table("organization_members").select("*").eq("user_id", user_id).eq("org_id", org_id).execute()
        if existing_member.data:
            raise HTTPException(status_code=400, detail="User is already a member of this organization")
        
        # Note: We don't check user's plan limits for JOINING organizations
        # Plan limits only apply to CREATING organizations
        print(f"User {user_plan} plan - can join any organization")
