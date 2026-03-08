"""
Supabase JWT authentication dependency for internal endpoints.

Verifies the Authorization: Bearer <supabase_jwt> header by calling
Supabase auth.getUser(). Returns the authenticated user_id.

Usage:
    from auth_dependency import require_auth, require_org_access

    @app.get("/some-endpoint")
    def endpoint(user: AuthenticatedUser = Depends(require_auth)):
        ...

    @app.get("/org-endpoint/{org_id}")
    def endpoint(org_id: str, user: AuthenticatedUser = Depends(require_org_access)):
        ...
"""
import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request
from supabase_client import supabase

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedUser:
    user_id: str
    email: Optional[str] = None
    _org_role: Optional[str] = None


async def require_auth(
    authorization: Optional[str] = Header(None),
) -> AuthenticatedUser:
    """
    Verify Supabase JWT from Authorization header.
    Returns AuthenticatedUser with user_id extracted from the token.
    Raises 401 if missing/invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    try:
        # Use Supabase's auth.get_user() to validate the JWT and extract user info.
        # This calls Supabase's GoTrue server, which verifies signature + expiry.
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        user = user_response.user
        return AuthenticatedUser(
            user_id=str(user.id),
            email=user.email,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Auth verification failed: %s", type(e).__name__)
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from e


async def require_org_member(
    request: Request,
    auth_user: AuthenticatedUser = Depends(require_auth),
    x_org_id: Optional[str] = Header(None),
) -> AuthenticatedUser:
    """
    Verify the authenticated user is a member of the org_id in the request path or body.
    Extracts org_id from path params, query params, JSON body, or X-Org-Id header.
    Raises 403 if user is not a member of that org.
    """
    # Try to get org_id from path params
    org_id = request.path_params.get("org_id")

    # Try query params
    if not org_id:
        org_id = request.query_params.get("org_id")

    # Try to get from body (for POST/PUT requests) — already parsed by FastAPI
    if not org_id:
        try:
            body = await request.json()
            org_id = body.get("org_id") if isinstance(body, dict) else None
        except Exception:
            pass

    # Fall back to X-Org-Id header (used by frontend for endpoints where org_id
    # is not in the path, e.g. DELETE /workflows/{id}, GET /experiments/{id})
    if not org_id:
        org_id = x_org_id

    if not org_id:
        raise HTTPException(
            status_code=400,
            detail="org_id is required but could not be found in path, query, body, or X-Org-Id header.",
        )

    # Check membership
    try:
        result = (
            supabase.table("organization_members")
            .select("role, status")
            .eq("org_id", org_id)
            .eq("user_id", auth_user.user_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this organization.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Org membership check failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="Error verifying organization access.") from e

    # Stash the role so require_org_admin can use it without re-querying
    auth_user._org_role = result.data[0].get("role", "member") if result.data else "member"
    return auth_user


async def require_org_admin(
    request: Request,
    auth_user: AuthenticatedUser = Depends(require_org_member),
) -> AuthenticatedUser:
    """
    Verify the authenticated user is an admin of the org.
    Chains on require_org_member (which already verified membership + stashed role).
    Raises 403 if user is not an admin.
    """
    if getattr(auth_user, "_org_role", None) != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required for this operation.",
        )
    return auth_user
