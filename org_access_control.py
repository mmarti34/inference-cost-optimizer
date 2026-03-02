import re
from fastapi import APIRouter, HTTPException, Body
from supabase_client import supabase

router = APIRouter()


def _slugify_org_name(name: str) -> str:
    """URL-safe slug from org name (aligned with frontend and migration)."""
    if not name or not str(name).strip():
        return "org"
    s = re.sub(r"\s+", "-", str(name).strip().lower())
    s = re.sub(r"[^a-z0-9-]", "", s).strip("-")
    return s or "org"

# Plan limits — must stay in sync with frontend lib/planLimits.ts
# Tiers: free → startup ($49/mo) → team ($149/mo) → enterprise (custom)
PLAN_LIMITS = {
    "free":       {"orgs": 0,  "members": 1,     "admins": 0,     "projects": 2,  "workflows": 5,   "server_keys": 1,  "requests_per_month": 1_000},
    "startup":    {"orgs": 1,  "members": 5,     "admins": 1,     "projects": 10, "workflows": 25,  "server_keys": 5,  "requests_per_month": 50_000},
    "team":       {"orgs": 5,  "members": 20,    "admins": 5,     "projects": -1, "workflows": -1,  "server_keys": 20, "requests_per_month": 500_000},
    "enterprise": {"orgs": -1, "members": -1,    "admins": -1,    "projects": -1, "workflows": -1,  "server_keys": -1, "requests_per_month": -1},
}
# -1 = unlimited

def get_org_plan(org):
    return org.get("plan", "free")

def get_upgrade_suggestion(plan):
    """Get upgrade suggestion based on current plan."""
    if plan == "free":
        return " Upgrade to Startup ($49/mo) for 5 team members, 10 projects, and 50K API requests/mo."
    elif plan == "startup":
        return " Upgrade to Team ($149/mo) for unlimited projects, 20 team members, and 500K API requests/mo."
    elif plan == "team":
        return " Contact us about Enterprise for unlimited everything, SSO, and dedicated support."
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
        plan_priority = {"free": 0, "startup": 1, "team": 2, "enterprise": 3}
        effective_plan = max([plan, user_actual_plan], key=lambda p: plan_priority.get(p, 0))
        print(f"Effective plan for organization: {effective_plan}")
        
        # 2. Check if user can create organizations (Free users cannot create orgs)
        if effective_plan == "free":
            upgrade_msg = get_upgrade_suggestion(effective_plan)
            raise HTTPException(status_code=403, detail=f"Free users cannot create organizations. Please upgrade to create organizations.{upgrade_msg}")
        
        # 3. Check admin limits (how many orgs user can be admin of)
        print("Checking admin limits...")
        admin_orgs_result = supabase.table("organization_members").select("org_id").eq("user_id", user_id).eq("role", "admin").eq("status", "active").execute()
        admin_orgs = admin_orgs_result.data if admin_orgs_result.data else []
        print(f"User is currently admin of {len(admin_orgs)} organizations")
        
        admin_limit = PLAN_LIMITS.get(effective_plan, PLAN_LIMITS["free"])["admins"]
        if len(admin_orgs) >= admin_limit:
            upgrade_msg = get_upgrade_suggestion(effective_plan)
            raise HTTPException(status_code=403, detail=f"Your plan ({effective_plan}) only allows {admin_limit} admin position(s).{upgrade_msg}")
        
        # 4. Create org (organizations don't have plans - billing is per user). Set slug deterministically.
        print("Creating new organization...")
        slug_base = _slugify_org_name(org_name)
        slug = slug_base
        suffix = 2
        while True:
            try:
                new_org_result = supabase.table("organizations").insert({
                    "name": org_name,
                    "created_by": user_id,
                    "type": "Organization",
                    "slug": slug,
                }).execute()
                break
            except Exception as e:
                err_str = str(e).lower()
                if "23505" in err_str or "unique" in err_str or "duplicate" in err_str:
                    slug = f"{slug_base}-{suffix}"
                    suffix += 1
                    if suffix > 1000:
                        raise HTTPException(status_code=409, detail="Organization slug conflict; try a different name.")
                else:
                    raise
        
        if not new_org_result.data:
            print("Error: No data returned from organization creation")
            raise HTTPException(status_code=500, detail="Failed to create organization")
        
        new_org = new_org_result.data[0] if isinstance(new_org_result.data, list) else new_org_result.data
        org_id = new_org["id"]
        
        # 5. Add user as admin member
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
        org = supabase.table("organizations").select("id, name, type, plan, created_by, logo, created_at").eq("id", org_id).single().execute().data
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found.")
        
        plan = get_org_plan(org)
        print(f"Organization plan: {plan}")
        
        # 2. Count current active members in this org
        members = supabase.table("organization_members").select("id, org_id, user_id, role, status, invited_email, created_at").eq("org_id", org_id).eq("status", "active").execute().data
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
        existing_invite = supabase.table("organization_members").select("id").eq("org_id", org_id).eq("invited_email", email).execute().data
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
        
        # 2. Check if user already has too many org memberships
        user_orgs_result = supabase.table("organization_members").select("org_id").eq("user_id", user_id).eq("status", "active").execute()
        user_orgs = user_orgs_result.data if user_orgs_result.data else []
        
        # Get the org types for user's current memberships
        org_types = []
        for membership in user_orgs:
            org_result = supabase.table("organizations").select("type").eq("id", membership["org_id"]).single().execute()
            if org_result.data:
                org_types.append(org_result.data["type"])
        
        # Count only Organization type memberships
        org_count = org_types.count("Organization")
        print(f"User currently has {org_count} Organization memberships")
        
        org_limit = PLAN_LIMITS.get(user_plan, PLAN_LIMITS["free"])["orgs"]
        if org_count >= org_limit:
            upgrade_msg = get_upgrade_suggestion(user_plan)
            raise HTTPException(status_code=403, detail=f"Your plan ({user_plan}) only allows {org_limit} organization(s).{upgrade_msg}")
        
        # 3. Check if user is already a member
        existing_member = supabase.table("organization_members").select("id").eq("user_id", user_id).eq("org_id", org_id).execute()
        if existing_member.data:
            raise HTTPException(status_code=400, detail="User is already a member of this organization")
        
        # 4. Check if org has reached member limit (based on org's plan, not user's plan)
        org = supabase.table("organizations").select("id, name, type, plan, created_by, logo, created_at").eq("id", org_id).single().execute().data
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found.")
        
        plan = get_org_plan(org)
        current_members = supabase.table("organization_members").select("id, org_id, user_id, role, status, invited_email, created_at").eq("org_id", org_id).eq("status", "active").execute().data
        current_member_count = len(current_members) if current_members else 0
        
        member_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["members"]
        if current_member_count >= member_limit:
            upgrade_msg = get_upgrade_suggestion(plan)
            raise HTTPException(
                status_code=403, 
                detail=f"Organization has reached its member limit ({member_limit}) for the {plan} plan.{upgrade_msg}"
            )
        
        # 5. Add user as member
        member_result = supabase.table("organization_members").insert({
            "org_id": org_id,
            "user_id": user_id,
            "role": "member",
            "status": "active"
        }).execute()
        
        return member_result.data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in join_organization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/api/organizations/members/{org_id}/{user_id}")
def remove_member(org_id: str, user_id: str):
    """Remove a member from an organization and clean up related data"""
    try:
        print(f"Removing user {user_id} from organization {org_id}")
        
        # 1. Check if the requesting user is an admin
        # (This would typically come from authentication middleware)
        # For now, we'll assume the request is authorized
        
        # 2. Remove the user from organization_members
        member_result = supabase.table("organization_members").delete().eq("org_id", org_id).eq("user_id", user_id).execute()
        print(f"Member removal result: {member_result}")
        
        # 3. Clean up any pending join requests for this user and org
        join_request_result = supabase.table("join_requests").delete().eq("org_id", org_id).eq("user_id", user_id).execute()
        print(f"Join request cleanup result: {join_request_result}")
        
        # 4. Clean up any pending invitations for this user and org
        invite_result = supabase.table("organization_members").delete().eq("org_id", org_id).eq("invited_email", user_id).eq("status", "invited").execute()
        print(f"Invitation cleanup result: {invite_result}")
        
        return {"message": "Member removed successfully", "data": member_result.data}
        
    except Exception as e:
        print(f"Error in remove_member: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def check_org_access_permission(user_plan: str, org_plan: str, org_type: str = "Organization") -> bool:
    """Check if user can access an organization based on their current plan"""
    # Personal organizations are always accessible
    if org_type == "Personal":
        return True

    # For Organization type, check if user's plan can access this org's plan
    # Plan priority (higher number = higher tier)
    plan_priority = {"free": 0, "startup": 1, "team": 2, "enterprise": 3}
    
    user_priority = plan_priority.get(user_plan, 0)
    org_priority = plan_priority.get(org_plan, 0)
    
    # User can access orgs of their plan level or lower
    # Users can access organizations of their plan level or lower
    return user_priority >= org_priority

@router.get("/api/organizations/check-access/{org_id}")
def check_organization_access(org_id: str, user_id: str):
    """Check if a user can access an organization based on their current plan"""
    try:
        print(f"Checking access for user {user_id} to organization {org_id}")
        
        # 1. Get user's current plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        print(f"User's current plan: {user_plan}")
        
        # 2. Get organization details
        org_result = supabase.table("organizations").select("id, name, type, plan, created_by, logo, created_at").eq("id", org_id).single().execute()
        if not org_result.data:
            raise HTTPException(status_code=404, detail="Organization not found.")
        
        org = org_result.data
        org_plan = get_org_plan(org)
        print(f"Organization plan: {org_plan}")
        
        # 3. Check if user is a member of this org
        membership_result = supabase.table("organization_members").select("id, org_id, user_id, role, status, invited_email, created_at").eq("org_id", org_id).eq("user_id", user_id).eq("status", "active").single().execute()
        
        # If user is already a member, they can access it regardless of plan
        if membership_result.data:
            print(f"User is existing member - granting access")
            return {
                "can_access": True,
                "user_plan": user_plan,
                "org_plan": org_plan,
                "message": "Access granted (existing member)"
            }
        
        # 4. Check access permission based on plan
        can_access = check_org_access_permission(user_plan, org_plan, org.get("type", "Organization"))
        
        if not can_access:
            upgrade_msg = get_upgrade_suggestion(user_plan)
            raise HTTPException(
                status_code=403, 
                detail=f"Access denied. This organization requires a {org_plan} plan or higher. Your current plan is {user_plan}.{upgrade_msg}"
            )
        
        return {
            "can_access": True,
            "user_plan": user_plan,
            "org_plan": org_plan,
            "message": "Access granted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in check_organization_access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/api/organizations/user-accessible")
def get_user_accessible_organizations(user_id: str):
    """Get all organizations that a user can access based on their current plan"""
    try:
        print(f"Getting accessible organizations for user {user_id}")
        
        # 1. Get user's current plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        print(f"User's current plan: {user_plan}")
        
        # 2. Get all organizations where user is a member
        memberships_result = supabase.table("organization_members").select("""
            org_id,
            role,
            status,
            organizations (*)
        """).eq("user_id", user_id).eq("status", "active").execute()
        
        if not memberships_result.data:
            return {"accessible_orgs": [], "inaccessible_orgs": []}
        
        accessible_orgs = []
        inaccessible_orgs = []
        
        for membership in memberships_result.data:
            if (hasattr(membership, "organizations") and membership.organizations) or (isinstance(membership, dict) and "organizations" in membership and membership["organizations"]):
                org = membership.organizations if hasattr(membership, "organizations") else membership["organizations"]
                org_plan = get_org_plan(org)
                user_role = membership.get("role", "member") if isinstance(membership, dict) else getattr(membership, "role", "member")                
                # Check if user can access this org based on their plan
                access_message = "Access granted"
                
                # Free users cannot be admins of any organization (except their personal workspace)
                if user_role == "admin" and user_plan == "free" and org.get("type", "Organization") != "Personal":
                    can_access = False
                    access_message = f"Access denied. Your {user_plan} plan doesn't allow admin roles in organizations. You can only access your personal workspace."
                else:
                    can_access = check_org_access_permission(user_plan, org_plan, org.get("type", "Organization"))
                
                org_data = {
                    "id": org["id"],
                    "name": org["name"],
                    "type": org.get("type", "Organization"),
                    "plan": org_plan,
                    "role": user_role,
                    "can_access": can_access,
                    "access_message": access_message
                }
                
                if can_access:
                    accessible_orgs.append(org_data)
                else:
                    inaccessible_orgs.append(org_data)
        
        return {
            "accessible_orgs": accessible_orgs,
            "inaccessible_orgs": inaccessible_orgs,
            "user_plan": user_plan
        }
        
    except Exception as e:
        print(f"Error in get_user_accessible_organizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/api/organizations/validate-api-key-access")
def validate_api_key_access(user_id: str, org_id: str):
    """Validate if a user can use API keys for a specific organization based on their subscription plan"""
    try:
        print(f"Validating API key access for user {user_id} to organization {org_id}")
        
        # 1. Get user's current plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        print(f"User's current plan: {user_plan}")
        
        # 2. Get organization details
        org_result = supabase.table("organizations").select("id, name, type, plan, created_by, logo, created_at").eq("id", org_id).single().execute()
        if not org_result.data:
            raise HTTPException(status_code=404, detail="Organization not found.")
        
        org = org_result.data
        org_type = org.get("type", "Organization")
        
        # 3. Check if user is a member of this org
        membership_result = supabase.table("organization_members").select("id, org_id, user_id, role, status, invited_email, created_at").eq("org_id", org_id).eq("user_id", user_id).eq("status", "active").single().execute()
        
        if not membership_result.data:
            raise HTTPException(status_code=403, detail="You are not a member of this organization.")
        
        membership = membership_result.data
        user_role = membership.get("role", "member")
        
        # 4. Check access based on user's plan and role
        can_access = True
        access_message = "API key access granted"
        
        # Free users cannot use API keys for organizations (only personal workspace)
        if user_plan == "free" and org_type != "Personal":
            can_access = False
            access_message = "API key access denied. Free users can only use API keys for their personal workspace. Upgrade to use organization API keys."
        
        # Check if the organization's admin is on a Free plan
        elif user_plan != "free":
            admin_result = supabase.table("organization_members").select("user_id").eq("org_id", org_id).eq("role", "admin").eq("status", "active").single().execute()
            if admin_result.data:
                admin_user_id = admin_result.data["user_id"]
                admin_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", admin_user_id).single().execute()
                admin_plan = "free"
                if admin_profile_result.data and admin_profile_result.data.get("subscription_tier"):
                    admin_plan = admin_profile_result.data["subscription_tier"]
                
                # If admin is on Free plan, block API key access for everyone
                if admin_plan == "free":
                    can_access = False
                    access_message = "API key access denied. This organization's admin is on a Free plan. The organization is temporarily unavailable until the admin upgrades their subscription."
        
        return {
            "can_access": can_access,
            "user_plan": user_plan,
            "org_type": org_type,
            "user_role": user_role,
            "message": access_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in validate_api_key_access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
