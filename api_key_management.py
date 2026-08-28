import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import audit
from supabase_client import supabase
from utils.encryption import encrypt_api_key
from plan_enforcement import check_server_key_limit
from auth_dependency import require_auth, require_org_member, AuthenticatedUser, verified_org_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _owning_org_for_audit(table: str, row_id: str) -> Optional[str]:
    """The org that owns `row_id`, FOR FILING A REFUSAL ROW AND NOTHING ELSE.

    THIS IS NOT AN AUTHORIZATION READ, and nothing it returns is ever compared,
    branched on, or handed to the caller. By the time it runs, the org-scoped
    statement has already matched nothing and the 404 is already decided; the
    response is byte-identical whatever this returns.

    It exists because of one question: "did anyone try to revoke MY production
    key?" These endpoints carry the caller's own org in the path, so a refusal
    filed under `principal` lands in the ATTACKER'S audit trail and the tenant
    who actually owns the key never sees it. `main.delete_service_api_key`
    already files this class of refusal under the owning org for exactly this
    reason; doing the same here keeps one resource class from having two
    different answers depending on which route was used.

    Selects `org_id` alone — no key material, no name, no status — and returns
    None on anything unexpected, in which case the caller falls back to filing
    under the verified org.
    """
    rid = str(row_id or "").strip()
    if not rid:
        return None
    try:
        result = supabase.table(table).select("org_id").eq("id", rid).limit(1).execute()
    except Exception as e:
        logger.debug("audit owner lookup on %s failed: %s", table, type(e).__name__)
        return None
    rows = result.data or []
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return None
    org = rows[0].get("org_id")
    return str(org) if org else None

# Pydantic models
class APIKeyCreate(BaseModel):
    provider: str
    api_key: str
    org_id: str
    name: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        extra = "ignore"

class APIKeyResponse(BaseModel):
    id: str
    provider: str
    org_id: str
    name: Optional[str] = None
    user_id: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        extra = "ignore"

class ServiceAPIKeyCreate(BaseModel):
    api_key: str
    org_id: str

class ServiceAPIKeyResponse(BaseModel):
    id: str
    org_id: str
    created_at: str
    key_type: Optional[str] = None
    rate_limit_per_minute: Optional[int] = None
    name: Optional[str] = None
    status: Optional[str] = None
    last_used_at: Optional[str] = None

class ServiceAPIKeyUpdate(BaseModel):
    rate_limit_per_minute: Optional[int] = None
    status: Optional[str] = None  # 'active' | 'revoked'


# ─── Provider API Keys ───────────────────────────────────────────────


@router.get("/api-keys/{org_id}", response_model=List[APIKeyResponse])
async def get_api_keys(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Get all API keys for an organization. Never returns the raw key."""
    try:
        result = supabase.table("api_keys").select(
            "id, org_id, provider, created_at"
        ).eq("org_id", org_id).execute()

        if not result.data:
            return []

        normalized_data = []
        for item in result.data:
            api_key_response = {
                "id": item.get("id", ""),
                "provider": item.get("provider", ""),
                "org_id": item.get("org_id", ""),
                "name": item.get("name"),
                "user_id": None,  # Never expose user_id in list
                "created_at": item.get("created_at", ""),
                "updated_at": item.get("created_at", ""),
            }
            normalized_data.append(api_key_response)

        normalized_data = sorted(normalized_data, key=lambda x: x.get("created_at", ""), reverse=True)
        return normalized_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_api_keys failed for org_id=%s: %s", org_id, e)
        if "permission denied" in str(e).lower() or "policy" in str(e).lower():
            raise HTTPException(status_code=403, detail="Permission denied.")
        raise HTTPException(status_code=500, detail=f"Error fetching API keys: {str(e)}")


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: Request,
    api_key_data: APIKeyCreate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new provider API key. The raw key is encrypted at rest and never returned."""
    # The credential is stored against the org membership was PROVEN for, not
    # the org named in the body — the same rule `create_service_api_key` below
    # already follows. The guard refuses a request whose org_id sources
    # disagree, so this is the second layer rather than the only one, and it
    # makes the audit row's org and the stored row's org identical by
    # construction rather than by that refusal holding.
    org_id = verified_org_id(auth_user)
    try:
        key_id = str(uuid.uuid4())
        encrypted_key = encrypt_api_key(api_key_data.api_key)

        result = supabase.table("api_keys").insert({
            "id": key_id,
            "provider": api_key_data.provider,
            "api_key": encrypted_key,
            "org_id": org_id,
        }).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create API key")

        row = result.data[0]
        # A provider credential now exists that can spend this tenant's money.
        # The row names the provider and the key id; the key itself is never
        # handed to the writer, and `audit`'s metadata allow-list would drop it
        # if it were.
        audit.record(
            audit.PROVIDER_CREDENTIAL_CREATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_PROVIDER_CREDENTIAL,
            resource_id=row.get("id"),
            metadata={"provider": api_key_data.provider},
            request=request,
        )
        return {
            "id": row["id"],
            "provider": row["provider"],
            "org_id": row["org_id"],
            "name": api_key_data.name,
            "user_id": None,
            "created_at": row["created_at"],
            "updated_at": row.get("created_at", row["created_at"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating API key: {str(e)}")


@router.delete("/api-keys/{org_id}/{key_id}")
async def delete_api_key(
    request: Request,
    org_id: str,
    key_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a provider API key"""
    verified = verified_org_id(auth_user)
    try:
        # `.limit(1)` rather than `.single()`: PostgREST raises on a zero-row
        # `.single()`, so the handler's own 404 branch was unreachable and both
        # "not yours" and "no such key" surfaced as a 500 from the outer
        # handler. Same opacity either way, but a refusal cannot be recorded
        # from inside a generic exception handler without also recording every
        # genuine database fault as an attack. This is the shape
        # `resource_access.fetch_owned_row` already uses.
        existing = supabase.table("api_keys").select("id").eq("id", key_id).eq("org_id", verified).limit(1).execute()
        rows = existing.data or []
        if isinstance(rows, dict):
            rows = [rows]

        if not rows:
            # Refused. Filed under the org that OWNS the key when there is one,
            # so the tenant whose credential was reached for can see the
            # attempt; under the caller's own org when the id matches nothing
            # at all. The response is identical in both cases.
            _victim = _owning_org_for_audit("api_keys", key_id)
            if _victim and _victim != verified:
                audit.record_server_derived(
                    audit.PROVIDER_CREDENTIAL_DELETE_REFUSED,
                    org_id=_victim,
                    derived_from="api_keys.org_id",
                    actor_id=auth_user.user_id,
                    resource_type=audit.RESOURCE_PROVIDER_CREDENTIAL,
                    resource_id=key_id,
                    metadata={"reason_code": audit.REASON_CROSS_TENANT},
                    request=request,
                )
            else:
                audit.record(
                    audit.PROVIDER_CREDENTIAL_DELETE_REFUSED,
                    principal=auth_user,
                    resource_type=audit.RESOURCE_PROVIDER_CREDENTIAL,
                    resource_id=key_id,
                    metadata={"reason_code": audit.REASON_NOT_FOUND},
                    request=request,
                )
            raise HTTPException(status_code=404, detail="API key not found")

        supabase.table("api_keys").delete().eq("id", key_id).eq("org_id", verified).execute()
        audit.record(
            audit.PROVIDER_CREDENTIAL_DELETED,
            principal=auth_user,
            resource_type=audit.RESOURCE_PROVIDER_CREDENTIAL,
            resource_id=key_id,
            request=request,
        )
        return {"message": "API key deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting API key: {str(e)}")


# ─── Service API Keys ────────────────────────────────────────────────


@router.get("/service-api-keys/{org_id}", response_model=List[ServiceAPIKeyResponse])
async def get_service_api_keys(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Get all service API keys for an organization. Never returns the raw key."""
    try:
        result = supabase.table("service_api_keys").select(
            "id, org_id, created_at, key_type, rate_limit_per_minute, name, status, last_used_at"
        ).eq("org_id", org_id).order("created_at", desc=True).execute()
    except Exception:
        try:
            result = supabase.table("service_api_keys").select(
                "id, org_id, created_at, key_type, rate_limit_per_minute"
            ).eq("org_id", org_id).order("created_at", desc=True).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching service API keys: {str(e)}") from e

    if not result.data:
        return []

    out = []
    for row in result.data:
        out.append({
            **row,
            "name": row.get("name") or "Server Key",
            "status": row.get("status") or "active",
            "last_used_at": row.get("last_used_at"),
        })
    return out


@router.post("/service-api-keys", response_model=ServiceAPIKeyResponse)
async def create_service_api_key(
    request: Request,
    api_key_data: ServiceAPIKeyCreate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new service API key"""
    # A service API key authenticates the public execution surface for an org.
    # Minting one into an org must depend on the org membership was proven
    # against, not on the org named in the body — otherwise the body picks
    # which tenant's endpoints the new credential can call.
    org_id = verified_org_id(auth_user)
    try:
        check_server_key_limit(org_id)

        key_id = str(uuid.uuid4())
        encrypted_key = encrypt_api_key(api_key_data.api_key)

        result = supabase.table("service_api_keys").insert({
            "id": key_id,
            "api_key": encrypted_key,
            "org_id": org_id,
        }).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create service API key")

        # A live credential for the PUBLIC EXECUTION SURFACE now exists for this
        # org. The row identifies the key without containing it.
        audit.record(
            audit.SERVER_KEY_CREATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_SERVER_API_KEY,
            resource_id=result.data[0].get("id"),
            request=request,
        )

        return {
            "id": result.data[0]["id"],
            "org_id": result.data[0]["org_id"],
            "created_at": result.data[0]["created_at"],
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "permission denied" in error_msg.lower() or "policy" in error_msg.lower():
            raise HTTPException(status_code=403, detail="Permission denied.")
        raise HTTPException(status_code=500, detail=f"Error creating service API key: {error_msg}")


@router.put("/service-api-keys/{org_id}/{key_id}", response_model=ServiceAPIKeyResponse)
async def update_service_api_key(
    request: Request,
    org_id: str,
    key_id: str,
    payload: ServiceAPIKeyUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update a service API key (e.g. rate_limit_per_minute, status).

    THE QUIETEST PATH TO A REVOKE. `status="revoked"` here takes a production
    credential out of service just as `DELETE` does, but it reads as an
    ordinary settings edit, so it is the route an attacker would prefer and the
    one an operator would least expect to check. It is recorded as a revoke.
    """
    verified = verified_org_id(auth_user)
    # A revoke attempt and a rate-limit edit are refused the same way but are
    # not the same event; decide which this is before anything can fail.
    _is_revoke = payload.status == "revoked"
    try:
        # `.limit(1)`, not `.single()` — see delete_api_key.
        existing_res = supabase.table("service_api_keys").select(
            "id, org_id, created_at"
        ).eq("id", key_id).eq("org_id", verified).limit(1).execute()
        _rows = existing_res.data or []
        if isinstance(_rows, dict):
            _rows = [_rows]

        if not _rows:
            _victim = _owning_org_for_audit("service_api_keys", key_id)
            _action = (
                audit.SERVER_KEY_REVOKE_REFUSED if _is_revoke
                else audit.SERVER_KEY_UPDATE_REFUSED
            )
            if _victim and _victim != verified:
                # Filed under the tenant whose production key was reached for.
                audit.record_server_derived(
                    _action,
                    org_id=_victim,
                    derived_from="service_api_keys.org_id",
                    actor_id=auth_user.user_id,
                    resource_type=audit.RESOURCE_SERVER_API_KEY,
                    resource_id=key_id,
                    metadata={"reason_code": audit.REASON_CROSS_TENANT},
                    request=request,
                )
            else:
                audit.record(
                    _action,
                    principal=auth_user,
                    resource_type=audit.RESOURCE_SERVER_API_KEY,
                    resource_id=key_id,
                    metadata={"reason_code": audit.REASON_NOT_FOUND},
                    request=request,
                )
            raise HTTPException(status_code=404, detail="Service API key not found")

        existing_row = _rows[0]

        update_data = {}
        if payload.rate_limit_per_minute is not None:
            update_data["rate_limit_per_minute"] = payload.rate_limit_per_minute
        if payload.status is not None and payload.status in ("active", "revoked"):
            update_data["status"] = payload.status

        if not update_data:
            # Nothing changed, so nothing to record: an audit row for a no-op
            # would make the log count edits that never happened.
            return {
                **existing_row,
                "key_type": existing_row.get("key_type"),
                "rate_limit_per_minute": existing_row.get("rate_limit_per_minute"),
            }

        result = supabase.table("service_api_keys").update(update_data).eq("id", key_id).eq("org_id", verified).execute()
        audit.record(
            audit.SERVER_KEY_REVOKED if _is_revoke else audit.SERVER_KEY_UPDATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_SERVER_API_KEY,
            resource_id=key_id,
            metadata={"new_status": update_data.get("status")},
            request=request,
        )
        if not result.data:
            return {
                **existing_row,
                "key_type": existing_row.get("key_type"),
                "rate_limit_per_minute": update_data.get(
                    "rate_limit_per_minute",
                    existing_row.get("rate_limit_per_minute"),
                ),
            }
        row = result.data[0]
        return {
            "id": row["id"],
            "org_id": row["org_id"],
            "created_at": row["created_at"],
            "key_type": row.get("key_type"),
            "rate_limit_per_minute": row.get("rate_limit_per_minute"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating service API key: {str(e)}")


@router.delete("/service-api-keys/{org_id}/{key_id}")
async def delete_service_api_key(
    request: Request,
    org_id: str,
    key_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a service API key"""
    verified = verified_org_id(auth_user)
    try:
        # `.limit(1)`, not `.single()` — see delete_api_key.
        existing = supabase.table("service_api_keys").select(
            "id, org_id, created_at"
        ).eq("id", key_id).eq("org_id", verified).limit(1).execute()
        rows = existing.data or []
        if isinstance(rows, dict):
            rows = [rows]

        if not rows:
            _victim = _owning_org_for_audit("service_api_keys", key_id)
            if _victim and _victim != verified:
                audit.record_server_derived(
                    audit.SERVER_KEY_REVOKE_REFUSED,
                    org_id=_victim,
                    derived_from="service_api_keys.org_id",
                    actor_id=auth_user.user_id,
                    resource_type=audit.RESOURCE_SERVER_API_KEY,
                    resource_id=key_id,
                    metadata={"reason_code": audit.REASON_CROSS_TENANT},
                    request=request,
                )
            else:
                audit.record(
                    audit.SERVER_KEY_REVOKE_REFUSED,
                    principal=auth_user,
                    resource_type=audit.RESOURCE_SERVER_API_KEY,
                    resource_id=key_id,
                    metadata={"reason_code": audit.REASON_NOT_FOUND},
                    request=request,
                )
            raise HTTPException(status_code=404, detail="Service API key not found")

        supabase.table("service_api_keys").delete().eq("id", key_id).eq("org_id", verified).execute()
        # Destroying the row and setting status='revoked' have the same effect
        # on the caller holding that key, so they are the same action.
        audit.record(
            audit.SERVER_KEY_REVOKED,
            principal=auth_user,
            resource_type=audit.RESOURCE_SERVER_API_KEY,
            resource_id=key_id,
            request=request,
        )
        return {"message": "Service API key deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting service API key: {str(e)}")
