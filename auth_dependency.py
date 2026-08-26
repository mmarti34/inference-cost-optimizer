"""
Supabase JWT authentication dependency for internal endpoints.
Also supports Cursor tokens (Bearer optml_...) for plugin/API access.

Verifies the Authorization: Bearer <supabase_jwt> or Bearer <cursor_token> header.
Returns the authenticated user_id (and for Cursor tokens: _cursor_org_id).

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
from utils.encryption import hash_service_api_key

logger = logging.getLogger(__name__)

CURSOR_TOKEN_PREFIX = "optml_"


@dataclass
class AuthenticatedUser:
    user_id: str
    email: Optional[str] = None
    #: True only when the identity provider confirmed the address (Supabase
    #: ``email_confirmed_at``). Never trust ``email`` for authorization unless
    #: this is True.
    email_verified: bool = False
    _org_role: Optional[str] = None
    _cursor_org_id: Optional[str] = None
    #: The org_id that ``require_org_member``/``require_org_admin`` actually
    #: verified membership against. Handlers whose path has no ``{org_id}``
    #: MUST re-filter their queries by this value — never by an org id taken
    #: straight from the request.
    _verified_org_id: Optional[str] = None


def _verify_cursor_token(token: str) -> Optional[AuthenticatedUser]:
    """Look up cursor_tokens by hash; return AuthenticatedUser with _cursor_org_id or None."""
    if not token.startswith(CURSOR_TOKEN_PREFIX) or len(token) < len(CURSOR_TOKEN_PREFIX) + 10:
        return None
    try:
        token_hash = hash_service_api_key(token)
        result = (
            supabase.table("cursor_tokens")
            .select("user_id, org_id, status")
            .eq("token_hash", token_hash)
            .limit(1)
            .execute()
        )
        if not result.data or len(result.data) == 0:
            return None
        row = result.data[0]
        # Revoked tokens must never authenticate. Rows written before the
        # status column existed come back as None -> treated as active.
        status = row.get("status")
        if status is not None and str(status) != "active":
            return None
        return AuthenticatedUser(
            user_id=str(row["user_id"]),
            _cursor_org_id=str(row["org_id"]),
        )
    except Exception as e:
        logger.debug("Cursor token lookup failed: %s", type(e).__name__)
        return None


async def require_auth(
    authorization: Optional[str] = Header(None),
) -> AuthenticatedUser:
    """
    Verify Authorization: Bearer. Tries Cursor token (optml_...) first, then Supabase JWT.
    Returns AuthenticatedUser. Raises 401 if missing/invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    # Cursor token (long-lived, org-scoped)
    cursor_user = _verify_cursor_token(token)
    if cursor_user is not None:
        return cursor_user

    # Supabase JWT
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        user = user_response.user
        return AuthenticatedUser(
            user_id=str(user.id),
            email=user.email,
            email_verified=bool(
                getattr(user, "email_confirmed_at", None)
                or getattr(user, "confirmed_at", None)
            ),
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

    # Cursor token is scoped to one org: require request org matches token's org
    if getattr(auth_user, "_cursor_org_id", None):
        if org_id != auth_user._cursor_org_id:
            raise HTTPException(
                status_code=403,
                detail="Cursor token is scoped to a different organization.",
            )
        auth_user._org_role = "member"
        auth_user._verified_org_id = org_id
        return auth_user

    # Check membership (Supabase-authenticated user)
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
    auth_user._verified_org_id = org_id
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


def verified_org_id(auth_user: AuthenticatedUser) -> str:
    """
    Return the org_id that require_org_member/require_org_admin actually verified.

    Use this in handlers whose path does not contain {org_id}: the guard resolved
    the org from the path/query/body/X-Org-Id header and proved membership, so
    this is the only org the caller may touch on this request. Re-filter every
    query by it.
    """
    org_id = getattr(auth_user, "_verified_org_id", None)
    if not org_id:
        # Should be unreachable: the guard always sets it. Fail closed.
        raise HTTPException(status_code=403, detail="Organization access could not be verified.")
    return org_id
