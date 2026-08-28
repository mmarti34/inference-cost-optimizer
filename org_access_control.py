import re
import uuid
import logging
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Body, Query, Request
import audit
from supabase_client import supabase
from auth_dependency import require_auth, require_org_member, require_org_admin, AuthenticatedUser
from email_service import send_invite_email

logger = logging.getLogger(__name__)

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
    """
    Liveness probe for this router. Intentionally unauthenticated, but it no
    longer runs a query or echoes database state / driver errors back to an
    anonymous caller — it used to return an organizations row count and the raw
    exception text on failure.
    """
    return {
        "status": "success" if supabase else "error",
        "message": "Router reachable" if supabase else "Supabase client not initialized",
    }

@router.post("/api/organizations/create")
def create_organization(
    request: Request,
    org_name: str = Body(...),
    user_id: Optional[str] = Body(None),
    plan: str = Body("free"),
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    """
    Create an organization owned by the authenticated caller.

    ``user_id`` and ``plan`` in the body are ignored: the org is always created
    for the caller, and its plan always comes from the caller's Stripe-backed
    ``user_profiles.subscription_tier``. A client-supplied plan used to be able
    to mint an ``enterprise`` org with every limit set to unlimited.
    """
    user_id = auth_user.user_id
    try:
        logger.info(f"Creating organization: user_id={user_id}, org_name={org_name}, plan={plan}")
        
        # Check if Supabase client is initialized
        if not supabase:
            logger.info("Error: Supabase client not initialized")
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Test basic Supabase connection first
        try:
            test_result = supabase.table("organizations").select("count").limit(1).execute()
            logger.info("Supabase connection test successful")
        except Exception as db_error:
            logger.info(f"Database connection test failed: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(db_error)}")
        
        # 1. Get user's actual plan from user_profiles table
        print("Fetching user's actual plan...")
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).maybe_single().execute()
        user_actual_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_actual_plan = user_profile_result.data["subscription_tier"]
        logger.info(f"User's actual plan: {user_actual_plan}")
        
        # The org plan is the caller's own subscription tier. The client-supplied
        # `plan` value is deliberately NOT considered — plan is billing state and
        # only Stripe (via stripe_webhook.py) may raise it.
        effective_plan = user_actual_plan if user_actual_plan in PLAN_LIMITS else "free"
        logger.info(f"Effective plan for organization: {effective_plan}")
        
        # 2. Check if user can create organizations (Free users cannot create orgs)
        if effective_plan == "free":
            upgrade_msg = get_upgrade_suggestion(effective_plan)
            raise HTTPException(status_code=403, detail=f"Free users cannot create organizations. Please upgrade to create organizations.{upgrade_msg}")
        
        # 3. Check admin limits (how many orgs user can be admin of)
        print("Checking admin limits...")
        admin_orgs_result = supabase.table("organization_members").select("org_id").eq("user_id", user_id).eq("role", "admin").eq("status", "active").execute()
        admin_orgs = admin_orgs_result.data if admin_orgs_result.data else []
        logger.info(f"User is currently admin of {len(admin_orgs)} organizations")
        
        admin_limit = PLAN_LIMITS.get(effective_plan, PLAN_LIMITS["free"])["admins"]
        if len(admin_orgs) >= admin_limit:
            upgrade_msg = get_upgrade_suggestion(effective_plan)
            raise HTTPException(status_code=403, detail=f"Your plan ({effective_plan}) only allows {admin_limit} admin position(s).{upgrade_msg}")
        
        # 4. Create org — inherit the admin's subscription tier as the org plan.
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
                    "plan": effective_plan,
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
            logger.info("Error: No data returned from organization creation")
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
            logger.info("Error: Failed to add user as admin member")
            # Clean up the created org
            supabase.table("organizations").delete().eq("id", org_id).execute()
            raise HTTPException(status_code=500, detail="Failed to add user as admin member")
        
        logger.info(f"Organization created successfully: {new_org_result.data}")
        # The org did not exist until the insert above, so there is no verified
        # org to derive from — the id comes from the row the SERVER just wrote.
        # `org_name` is customer free text and is deliberately not recorded.
        audit.record_server_derived(
            audit.ORGANIZATION_CREATED,
            org_id=org_id,
            derived_from="organizations.id (row inserted by this handler)",
            actor_id=user_id,
            resource_type=audit.RESOURCE_ORGANIZATION,
            resource_id=org_id,
            metadata={"role": "admin"},
            request=request,
        )
        return new_org_result.data
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.info(f"Unexpected error in create_organization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/api/organizations/invite")
def invite_member(
    request: Request,
    org_id: str = Body(...),
    email: str = Body(...),
    user_id: Optional[str] = Body(None),
    inviter_email: str = Body("a team admin"),
    auth_user: AuthenticatedUser = Depends(require_org_admin),
):
    """
    Invite a member to an organization. Admin of org_id only.

    Was previously unauthenticated: anyone could invite their own address into
    any organization and then accept the emailed token, a full takeover.
    """
    user_id = auth_user.user_id
    inviter_email = auth_user.email or inviter_email
    try:
        logger.info("Inviting %s to org %s by user %s", email, org_id, user_id)

        # 1. Fetch org
        org = supabase.table("organizations").select("id, name, type, plan").eq("id", org_id).single().execute().data
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found.")

        # Block invites to Personal workspaces
        if org.get("type") == "Personal":
            raise HTTPException(status_code=400, detail="Cannot invite members to a personal workspace. Create an organization first.")

        plan = get_org_plan(org)

        # 2. Count active + invited members (both count toward limit)
        members_result = supabase.table("organization_members") \
            .select("id, status") \
            .eq("org_id", org_id) \
            .in_("status", ["active", "invited"]) \
            .execute()
        current_count = len(members_result.data) if members_result.data else 0

        # 3. Check member limit
        member_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["members"]
        if member_limit != -1 and current_count >= member_limit:
            upgrade_msg = get_upgrade_suggestion(plan)
            raise HTTPException(
                status_code=403,
                detail=f"Member limit reached ({current_count}/{member_limit}) for the {plan} plan.{upgrade_msg}"
            )

        # 4. Check for duplicate invite by email
        existing = supabase.table("organization_members") \
            .select("id, status") \
            .eq("org_id", org_id) \
            .eq("invited_email", email.lower().strip()) \
            .execute().data
        if existing:
            status = existing[0].get("status")
            if status == "active":
                raise HTTPException(status_code=400, detail="This email is already a member.")
            if status == "invited":
                raise HTTPException(status_code=400, detail="This email has already been invited.")

        # 5. Create organization_members row
        new_member = supabase.table("organization_members").insert({
            "org_id": org_id,
            "invited_email": email.lower().strip(),
            "status": "invited",
            "role": "member",
        }).execute()

        if not new_member.data:
            raise HTTPException(status_code=500, detail="Failed to create invitation.")

        member_row = new_member.data[0] if isinstance(new_member.data, list) else new_member.data

        # 6. Generate invite token
        invite_token = str(uuid.uuid4())
        token_result = supabase.table("invite_tokens").insert({
            "token": invite_token,
            "org_id": org_id,
            "invited_email": email.lower().strip(),
            "invited_by": user_id,
            "member_id": member_row["id"],
            "status": "pending",
        }).execute()

        if not token_result.data:
            supabase.table("organization_members").delete().eq("id", member_row["id"]).execute()
            raise HTTPException(status_code=500, detail="Failed to generate invite token.")

        # 7. Send invite email (inviter_email passed from frontend)
        email_sent = send_invite_email(
            to_email=email.lower().strip(),
            org_name=org.get("name", "an organization"),
            inviter_email=inviter_email,
            invite_token=invite_token,
        )

        logger.info("Invite created for %s (email_sent=%s)", email, email_sent)
        # The invitee's EMAIL is not recorded. It is PII, and the
        # organization_members row named by resource_id resolves it — an audit
        # table is the wrong place to keep a second copy of a contact address.
        audit.record(
            audit.MEMBER_INVITED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORG_MEMBER,
            resource_id=member_row.get("id"),
            metadata={"role": "member"},
            request=request,
        )
        return {
            "member": member_row,
            "email_sent": email_sent,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in invite_member: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/api/organizations/accept-invite")
def accept_invite(
    request: Request,
    token: str = Body(...),
    user_id: Optional[str] = Body(None),
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    """
    Accept an organization invite by token.

    The membership is always created for the *authenticated* caller — the
    ``user_id`` body field is ignored (it used to let anyone holding a token
    grant membership to an arbitrary account). The invite must also have been
    addressed to the caller's own verified email address.
    """
    # Never trust a body-supplied user id.
    user_id = auth_user.user_id
    caller_email = (auth_user.email or "").lower().strip()
    try:
        # 1. Look up the token
        token_result = supabase.table("invite_tokens") \
            .select("*") \
            .eq("token", token) \
            .single().execute()

        if not token_result.data:
            raise HTTPException(status_code=404, detail="Invitation not found or has already been used.")

        invite = token_result.data

        # 1b. The invite must be addressed to this caller's own verified email.
        invited_email = (invite.get("invited_email") or "").lower().strip()
        if not caller_email or not auth_user.email_verified or invited_email != caller_email:
            logger.warning(
                "accept_invite rejected: invite email does not match authenticated caller (user=%s)",
                user_id,
            )
            raise HTTPException(
                status_code=403,
                detail="This invitation was sent to a different email address. Sign in with the invited address to accept it.",
            )

        # 2. Check token status
        if invite["status"] == "accepted":
            raise HTTPException(status_code=400, detail="This invitation has already been accepted.")
        if invite["status"] == "revoked":
            raise HTTPException(status_code=400, detail="This invitation has been revoked.")
        if invite["status"] == "expired":
            raise HTTPException(status_code=400, detail="This invitation has expired.")

        # 3. Check expiry
        expires_at_str = invite["expires_at"]
        if isinstance(expires_at_str, str):
            expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        else:
            expires_at = expires_at_str
        if datetime.now(timezone.utc) > expires_at:
            supabase.table("invite_tokens").update({"status": "expired"}).eq("id", invite["id"]).execute()
            raise HTTPException(status_code=400, detail="This invitation has expired.")

        # 4. Check if user is already an active member
        existing_member = supabase.table("organization_members") \
            .select("id, status") \
            .eq("org_id", invite["org_id"]) \
            .eq("user_id", user_id) \
            .execute().data
        if existing_member:
            for m in existing_member:
                if m["status"] == "active":
                    supabase.table("invite_tokens").update({
                        "status": "accepted",
                        "accepted_at": datetime.now(timezone.utc).isoformat(),
                        "accepted_by": user_id,
                    }).eq("id", invite["id"]).execute()
                    return {"message": "You are already a member of this organization.", "org_id": invite["org_id"], "org_name": ""}

        # 6. Activate the membership
        member_id = invite.get("member_id")
        if member_id:
            supabase.table("organization_members").update({
                "user_id": user_id,
                "status": "active",
                "role": "member",
            }).eq("id", member_id).execute()
        else:
            supabase.table("organization_members").insert({
                "org_id": invite["org_id"],
                "user_id": user_id,
                "invited_email": invite["invited_email"],
                "status": "active",
                "role": "member",
            }).execute()

        # 7. Mark token as accepted
        supabase.table("invite_tokens").update({
            "status": "accepted",
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "accepted_by": user_id,
        }).eq("id", invite["id"]).execute()

        # 8. Get org name for response
        org_name = ""
        try:
            org_result = supabase.table("organizations").select("name").eq("id", invite["org_id"]).single().execute()
            if org_result.data:
                org_name = org_result.data.get("name", "")
        except Exception:
            pass

        logger.info("Invite accepted: user=%s org=%s", user_id, invite["org_id"])
        # Guarded by require_auth only: the caller is not yet a member, so there
        # is no verified org. The org is the one named on the invite_tokens row
        # the presented token matched — read by the server, not sent by the
        # client. The token itself is never recorded.
        audit.record_server_derived(
            audit.MEMBER_INVITE_ACCEPTED,
            org_id=invite["org_id"],
            derived_from="invite_tokens.org_id",
            actor_id=user_id,
            resource_type=audit.RESOURCE_ORG_MEMBER,
            resource_id=invite.get("member_id"),
            metadata={"role": "member", "target_user_id": user_id},
            request=request,
        )
        return {
            "message": "Invitation accepted successfully.",
            "org_id": invite["org_id"],
            "org_name": org_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in accept_invite: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/api/organizations/revoke-invite")
def revoke_invite(
    request: Request,
    org_id: str = Body(...),
    member_id: str = Body(...),
    user_id: Optional[str] = Body(None),
    auth_user: AuthenticatedUser = Depends(require_org_admin),
):
    """Revoke a pending invite. Admin of org_id only (verified by the guard)."""
    # The guard proved the caller is an admin of org_id; the body-supplied
    # user_id is ignored (it used to be the only "authorization" here).
    user_id = auth_user.user_id
    try:
        # 1. Revoke invite token(s) for this member
        supabase.table("invite_tokens") \
            .update({"status": "revoked"}) \
            .eq("member_id", member_id) \
            .eq("org_id", org_id) \
            .eq("status", "pending") \
            .execute()

        # 2. Delete the organization_members row (only if still invited),
        #    re-filtered by the org the guard verified.
        supabase.table("organization_members") \
            .delete() \
            .eq("id", member_id) \
            .eq("org_id", org_id) \
            .eq("status", "invited") \
            .execute()

        logger.info("Invite revoked: member_id=%s org=%s by user=%s", member_id, org_id, user_id)
        audit.record(
            audit.MEMBER_INVITE_REVOKED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORG_MEMBER,
            resource_id=member_id,
            request=request,
        )
        return {"message": "Invitation revoked successfully."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in revoke_invite: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/api/organizations/pending-invites")
def get_pending_invites_for_user(
    email: str = Query(None),
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    """
    Get pending invite tokens addressed to the *authenticated caller's own*
    verified email address.

    The ``email`` query parameter is accepted for backwards compatibility but is
    never used to select rows: previously any anonymous caller could enumerate
    invite tokens for an arbitrary address and take over the organization.
    """
    caller_email = (auth_user.email or "").lower().strip()
    if not caller_email or not auth_user.email_verified:
        # Cursor tokens and unverified accounts have no verified address to
        # scope by — return nothing rather than leaking someone else's invites.
        return []
    if email and email.lower().strip() != caller_email:
        raise HTTPException(
            status_code=403,
            detail="You can only view invitations sent to your own email address.",
        )
    try:
        results = supabase.table("invite_tokens") \
            .select("id, token, org_id, invited_email, expires_at, created_at, organizations(id, name, logo)") \
            .eq("invited_email", caller_email) \
            .eq("status", "pending") \
            .execute()

        if not results.data:
            return []

        # Filter out expired ones
        now = datetime.now(timezone.utc)
        pending = []
        for invite in results.data:
            expires_str = invite.get("expires_at", "")
            if isinstance(expires_str, str) and expires_str:
                try:
                    expires_at = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                    if now > expires_at:
                        supabase.table("invite_tokens").update({"status": "expired"}).eq("id", invite["id"]).execute()
                        continue
                except (ValueError, TypeError):
                    pass
            pending.append(invite)

        return pending

    except Exception as e:
        logger.exception("Error in get_pending_invites_for_user: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/api/organizations/join")
def join_organization(
    org_id: str = Body(...),
    user_id: Optional[str] = Body(None),
    auth_user: AuthenticatedUser = Depends(require_auth),
):
    """
    Request to join an organization.

    This creates a *pending* row in ``join_requests`` — it never grants
    membership. An admin of the target org approves or rejects the request from
    the Join Requests screen, which is what actually inserts the
    organization_members row.

    Previously this endpoint was unauthenticated and inserted an **active**
    member directly, which let anyone join any organization they knew the id of.
    """
    # Never trust a body-supplied user id — always the authenticated caller.
    user_id = auth_user.user_id
    try:
        logger.info(f"User {user_id} requesting to join organization {org_id}")
        
        # 1. Get user's actual plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).maybe_single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        logger.info(f"User's plan: {user_plan}")
        
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
        logger.info(f"User currently has {org_count} Organization memberships")
        
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
        
        # 5. Record a PENDING join request. An org admin must approve it before
        #    any organization_members row is created.
        existing_request = (
            supabase.table("join_requests")
            .select("id, status")
            .eq("org_id", org_id)
            .eq("user_id", user_id)
            .execute()
        )
        for row in (existing_request.data or []):
            if row.get("status") == "pending":
                return {
                    "status": "pending",
                    "message": "A join request for this organization is already awaiting approval.",
                    "join_request_id": row.get("id"),
                }

        request_row = {
            "org_id": org_id,
            "user_id": user_id,
            "user_email": auth_user.email or "",
            "status": "pending",
        }
        if existing_request.data:
            # Re-open a previously rejected/approved request (table is UNIQUE on
            # (org_id, user_id), so update rather than insert).
            jr_result = (
                supabase.table("join_requests")
                .update({"status": "pending"})
                .eq("org_id", org_id)
                .eq("user_id", user_id)
                .execute()
            )
        else:
            jr_result = supabase.table("join_requests").insert(request_row).execute()

        jr = (jr_result.data or [{}])[0]
        return {
            "status": "pending",
            "message": "Join request submitted. An organization admin must approve it.",
            "join_request_id": jr.get("id"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"Error in join_organization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/api/organizations/members/{org_id}/{user_id}")
async def remove_member(
    request: Request,
    org_id: str,
    user_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Remove a member from an organization. Requires admin role."""
    try:
        # 1. Verify the requester is an admin of this org
        admin_check = supabase.table("organization_members") \
            .select("role") \
            .eq("org_id", org_id) \
            .eq("user_id", auth_user.user_id) \
            .eq("status", "active") \
            .limit(1) \
            .execute()
        if not admin_check.data or admin_check.data[0].get("role") != "admin":
            # A non-admin member trying to evict someone is a privilege-boundary
            # probe worth keeping, not just a 403 in an access log.
            audit.record(
                audit.MEMBER_REMOVE_REFUSED,
                principal=auth_user,
                resource_type=audit.RESOURCE_ORG_MEMBER,
                metadata={
                    "target_user_id": user_id,
                    "reason_code": audit.REASON_NOT_ADMIN,
                },
                request=request,
            )
            raise HTTPException(status_code=403, detail="Only admins can remove members.")

        # 2. Prevent removing yourself (admins should use leave instead)
        if auth_user.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot remove yourself. Use leave instead.")

        # 3. Remove the user from organization_members
        member_result = supabase.table("organization_members").delete().eq("org_id", org_id).eq("user_id", user_id).execute()

        # 4. Clean up any pending join requests for this user and org
        supabase.table("join_requests").delete().eq("org_id", org_id).eq("user_id", user_id).execute()

        logger.info("Member removed: user=%s org=%s by=%s", user_id, org_id, auth_user.user_id)
        audit.record(
            audit.MEMBER_REMOVED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORG_MEMBER,
            resource_id=((member_result.data or [{}])[0].get("id") if member_result.data else None),
            metadata={"target_user_id": user_id},
            request=request,
        )
        return {"message": "Member removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in remove_member: %s", e)
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
def check_organization_access(org_id: str, auth_user: AuthenticatedUser = Depends(require_auth)):
    """Check if a user can access an organization based on their current plan"""
    user_id = auth_user.user_id
    try:
        logger.info("Checking access for user %s to organization %s", user_id, org_id)
        
        # 1. Get user's current plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).maybe_single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        logger.info(f"User's current plan: {user_plan}")
        
        # 2. Get organization details
        org_result = supabase.table("organizations").select("id, name, type, plan, created_by, logo, created_at").eq("id", org_id).single().execute()
        if not org_result.data:
            raise HTTPException(status_code=404, detail="Organization not found.")
        
        org = org_result.data
        org_plan = get_org_plan(org)
        logger.info(f"Organization plan: {org_plan}")
        
        # 3. Check if user is a member of this org
        membership_result = supabase.table("organization_members").select("id, org_id, user_id, role, status, invited_email, created_at").eq("org_id", org_id).eq("user_id", user_id).eq("status", "active").single().execute()
        
        # If user is already a member, they can access it regardless of plan
        if membership_result.data:
            logger.info(f"User is existing member - granting access")
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
        logger.info(f"Error in check_organization_access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/api/organizations/user-accessible")
def get_user_accessible_organizations(auth_user: AuthenticatedUser = Depends(require_auth)):
    """Get all organizations that a user can access based on their current plan"""
    user_id = auth_user.user_id
    try:
        logger.info("Getting accessible organizations for user %s", user_id)
        
        # 1. Get user's current plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).maybe_single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        logger.info(f"User's current plan: {user_plan}")
        
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
                # Existing members always have access to orgs they belong to.
                # Plan limits only gate *creating/joining* new orgs, not accessing existing ones.
                # This matches check_organization_access() which also grants access to existing members.
                can_access = True
                access_message = "Access granted"
                
                org_data = {
                    "id": org["id"],
                    "name": org["name"],
                    "slug": org.get("slug") or "",
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
        logger.info(f"Error in get_user_accessible_organizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/api/organizations/validate-api-key-access")
def validate_api_key_access(org_id: str, auth_user: AuthenticatedUser = Depends(require_auth)):
    """Validate if a user can use API keys for a specific organization based on their subscription plan"""
    user_id = auth_user.user_id
    try:
        logger.info("Validating API key access for user %s to organization %s", user_id, org_id)
        
        # 1. Get user's current plan
        user_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", user_id).maybe_single().execute()
        user_plan = "free"
        if user_profile_result.data and user_profile_result.data.get("subscription_tier"):
            user_plan = user_profile_result.data["subscription_tier"]
        logger.info(f"User's current plan: {user_plan}")
        
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
                admin_profile_result = supabase.table("user_profiles").select("subscription_tier").eq("user_id", admin_user_id).maybe_single().execute()
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
        logger.info(f"Error in validate_api_key_access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
