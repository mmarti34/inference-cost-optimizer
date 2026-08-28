import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from supabase_client import supabase
from utils.encryption import encrypt_api_key
from plan_enforcement import check_server_key_limit
from auth_dependency import require_auth, require_org_member, AuthenticatedUser, verified_org_id

logger = logging.getLogger(__name__)

router = APIRouter()

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
    api_key_data: APIKeyCreate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new provider API key. The raw key is encrypted at rest and never returned."""
    try:
        key_id = str(uuid.uuid4())
        encrypted_key = encrypt_api_key(api_key_data.api_key)

        result = supabase.table("api_keys").insert({
            "id": key_id,
            "provider": api_key_data.provider,
            "api_key": encrypted_key,
            "org_id": api_key_data.org_id,
        }).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create API key")

        row = result.data[0]
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
    org_id: str,
    key_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a provider API key"""
    try:
        existing = supabase.table("api_keys").select("id").eq("id", key_id).eq("org_id", org_id).single().execute()

        if not existing.data:
            raise HTTPException(status_code=404, detail="API key not found")

        supabase.table("api_keys").delete().eq("id", key_id).eq("org_id", org_id).execute()
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
    org_id: str,
    key_id: str,
    payload: ServiceAPIKeyUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update a service API key (e.g. rate_limit_per_minute, status)."""
    try:
        existing = supabase.table("service_api_keys").select(
            "id, org_id, created_at"
        ).eq("id", key_id).eq("org_id", org_id).single().execute()

        if not existing.data:
            raise HTTPException(status_code=404, detail="Service API key not found")

        update_data = {}
        if payload.rate_limit_per_minute is not None:
            update_data["rate_limit_per_minute"] = payload.rate_limit_per_minute
        if payload.status is not None and payload.status in ("active", "revoked"):
            update_data["status"] = payload.status

        if not update_data:
            return {
                **existing.data,
                "key_type": existing.data.get("key_type"),
                "rate_limit_per_minute": existing.data.get("rate_limit_per_minute"),
            }

        result = supabase.table("service_api_keys").update(update_data).eq("id", key_id).eq("org_id", org_id).execute()
        if not result.data:
            return {
                **existing.data,
                "key_type": existing.data.get("key_type"),
                "rate_limit_per_minute": update_data.get(
                    "rate_limit_per_minute",
                    existing.data.get("rate_limit_per_minute"),
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
    org_id: str,
    key_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a service API key"""
    try:
        existing = supabase.table("service_api_keys").select(
            "id, org_id, created_at"
        ).eq("id", key_id).eq("org_id", org_id).single().execute()

        if not existing.data:
            raise HTTPException(status_code=404, detail="Service API key not found")

        supabase.table("service_api_keys").delete().eq("id", key_id).eq("org_id", org_id).execute()
        return {"message": "Service API key deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting service API key: {str(e)}")
