"""Cursor tokens: create long-lived tokens for Cursor plugin. GET /api/cursor/me for token holder."""
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth_dependency import require_auth, require_org_member, AuthenticatedUser
from supabase_client import supabase
from utils.encryption import hash_service_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

CURSOR_TOKEN_PREFIX = "optml_"
MAX_CURSOR_TOKENS_PER_USER = 10


class CreateCursorTokenBody(BaseModel):
    org_id: str
    name: str = "Cursor"


@router.post("/api/cursor-tokens")
async def create_cursor_token(
    body: CreateCursorTokenBody,
    _user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Create a Cursor token for the given org. Requires Supabase auth (user must be org member).
    Returns plaintext token once; store it as OPTIML_CURSOR_TOKEN. Token is scoped to this org.
    """
    org_id = body.org_id.strip()
    name = (body.name or "Cursor").strip() or "Cursor"

    # Count existing ACTIVE tokens for this user — revoked ones don't count.
    existing = (
        supabase.table("cursor_tokens")
        .select("id, status")
        .eq("user_id", _user.user_id)
        .execute()
    )
    total = len([
        r for r in (existing.data or [])
        if (r.get("status") or "active") == "active"
    ])
    if total >= MAX_CURSOR_TOKENS_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {MAX_CURSOR_TOKENS_PER_USER} active Cursor tokens per user. "
                f"Revoke one first (DELETE /api/cursor-tokens/{{org_id}}/{{token_id}})."
            ),
        )

    plaintext = f"{CURSOR_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    token_hash = hash_service_api_key(plaintext)

    supabase.table("cursor_tokens").insert({
        "user_id": _user.user_id,
        "org_id": org_id,
        "token_hash": token_hash,
        "name": name,
        "status": "active",
    }).execute()

    # Get org slug for response
    org_row = supabase.table("organizations").select("id, slug").eq("id", org_id).single().execute()
    org_slug = (org_row.data or {}).get("slug") or org_id

    return {
        "cursor_token": plaintext,
        "org_id": org_id,
        "org_slug": org_slug,
        "name": name,
    }


@router.get("/api/cursor-tokens/{org_id}")
async def list_cursor_tokens(
    org_id: str,
    _user: AuthenticatedUser = Depends(require_org_member),
):
    """
    List the caller's own Cursor tokens for this org. Never returns token_hash
    or any part of the plaintext token — those are shown once, at creation.
    """
    result = (
        supabase.table("cursor_tokens")
        .select("id, org_id, name, status, created_at, revoked_at")
        .eq("org_id", org_id)
        .eq("user_id", _user.user_id)
        .order("created_at", desc=True)
        .execute()
    )
    rows = []
    for row in (result.data or []):
        rows.append({
            "id": row.get("id"),
            "org_id": row.get("org_id"),
            "name": row.get("name"),
            "status": row.get("status") or "active",
            "created_at": row.get("created_at"),
            "revoked_at": row.get("revoked_at"),
        })
    return rows


@router.delete("/api/cursor-tokens/{org_id}/{token_id}")
async def revoke_cursor_token(
    org_id: str,
    token_id: str,
    _user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Revoke one of the caller's own Cursor tokens.

    Marks it `revoked`; auth_dependency._verify_cursor_token refuses any token
    whose status is not 'active', so the bearer stops working immediately.
    Re-filtered by both org_id (the org the guard verified) and user_id, so a
    member cannot revoke another member's token.
    """
    result = (
        supabase.table("cursor_tokens")
        .update({"status": "revoked", "revoked_at": "now()"})
        .eq("id", token_id)
        .eq("org_id", org_id)
        .eq("user_id", _user.user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Cursor token not found")
    logger.info("Cursor token revoked: token_id=%s org=%s", token_id, org_id)
    return {"status": "revoked", "id": token_id}


@router.get("/api/cursor/me")
async def cursor_me(_user: AuthenticatedUser = Depends(require_auth)):
    """
    When called with a Cursor token, returns org_id and org_slug for that token.
    When called with Supabase JWT, returns first accessible org (for compatibility).
    """
    if getattr(_user, "_cursor_org_id", None):
        org_id = _user._cursor_org_id
        org_row = supabase.table("organizations").select("id, slug").eq("id", org_id).single().execute()
        org_slug = (org_row.data or {}).get("slug") or org_id
        return {"org_id": org_id, "org_slug": org_slug}

    # Supabase: return first org the user is a member of
    members = (
        supabase.table("organization_members")
        .select("org_id, organizations(slug)")
        .eq("user_id", _user.user_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    if not members.data or len(members.data) == 0:
        raise HTTPException(status_code=404, detail="No organization found. Create or join an org first.")
    row = members.data[0]
    org_id = row.get("org_id")
    org_slug = ""
    if isinstance(row.get("organizations"), dict):
        org_slug = (row["organizations"] or {}).get("slug") or org_id
    return {"org_id": org_id, "org_slug": org_slug or org_id}


@router.get("/api/cursor/endpoints")
async def cursor_endpoints(_user: AuthenticatedUser = Depends(require_auth)):
    """
    List deployed endpoints for the authenticated user's org.
    Use with Cursor token (or Supabase JWT) so apps can populate OPTIML_ORG_SLUG and OPTIML_ENDPOINT_SLUG.
    Returns promoted deployments only (live endpoints).
    """
    org_id = None
    if getattr(_user, "_cursor_org_id", None):
        org_id = _user._cursor_org_id
    else:
        members = (
            supabase.table("organization_members")
            .select("org_id")
            .eq("user_id", _user.user_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
        if members.data and len(members.data) > 0:
            org_id = members.data[0].get("org_id")
    if not org_id:
        raise HTTPException(status_code=404, detail="No organization found.")
    org_row = supabase.table("organizations").select("slug").eq("id", org_id).single().execute()
    org_slug = (org_row.data or {}).get("slug") or org_id
    dep_result = (
        supabase.table("workflow_deployments")
        .select("endpoint_slug, workflow_id")
        .eq("org_id", org_id)
        .eq("status", "promoted")
        .execute()
    )
    seen = set()
    endpoints = []
    for row in (dep_result.data or []):
        slug = (row.get("endpoint_slug") or "").strip()
        if slug and slug not in seen:
            seen.add(slug)
            endpoints.append({"endpoint_slug": slug, "workflow_id": row.get("workflow_id")})
    return {"org_id": org_id, "org_slug": org_slug, "endpoints": endpoints}
