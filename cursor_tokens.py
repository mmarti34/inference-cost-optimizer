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

    # Count existing tokens for this user
    existing = (
        supabase.table("cursor_tokens")
        .select("id")
        .eq("user_id", _user.user_id)
        .execute()
    )
    total = len(existing.data or [])
    if total >= MAX_CURSOR_TOKENS_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_CURSOR_TOKENS_PER_USER} Cursor tokens per user. Revoke one in Settings first.",
        )

    plaintext = f"{CURSOR_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    token_hash = hash_service_api_key(plaintext)

    supabase.table("cursor_tokens").insert({
        "user_id": _user.user_id,
        "org_id": org_id,
        "token_hash": token_hash,
        "name": name,
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
