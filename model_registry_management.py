"""
CRUD routes for custom_endpoints and model_registry tables.

Secrets are encrypted at rest using the same AES-256-GCM scheme as provider
api_keys.  Decryption only happens server-side during workflow execution —
the GET endpoints never return raw secrets.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from supabase_client import supabase
from utils.encryption import encrypt_api_key, decrypt_api_key
from model_target import ModelTarget

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Pydantic models ──────────────────────────────────────────────────────────

class CustomEndpointCreate(BaseModel):
    org_id: str
    name: str
    base_url: str
    auth_type: str = "bearer"          # "bearer" | "header"
    auth_header_name: Optional[str] = None  # required when auth_type="header"
    auth_secret: str                   # plaintext; encrypted before storage
    default_headers: Optional[dict] = None

    class Config:
        extra = "ignore"


class CustomEndpointResponse(BaseModel):
    id: str
    org_id: str
    name: str
    base_url: str
    auth_type: str
    auth_header_name: Optional[str] = None
    default_headers_json: Optional[dict] = None
    is_active: bool
    created_at: str

    class Config:
        extra = "ignore"


class ModelRegistryCreate(BaseModel):
    org_id: str
    name: str
    target_type: str                   # "provider_model" | "openai_compatible_endpoint"

    # For provider_model
    provider: Optional[str] = None
    model_name: Optional[str] = None

    # For openai_compatible_endpoint
    endpoint_id: Optional[str] = None
    endpoint_model_name: Optional[str] = None

    class Config:
        extra = "ignore"


class ModelRegistryResponse(BaseModel):
    id: str
    org_id: str
    name: str
    target_type: str
    provider: Optional[str] = None
    model_name: Optional[str] = None
    endpoint_id: Optional[str] = None
    endpoint_model_name: Optional[str] = None
    is_active: bool
    created_at: str
    # joined from custom_endpoints when present
    endpoint_name: Optional[str] = None
    endpoint_base_url: Optional[str] = None

    class Config:
        extra = "ignore"


# ── Custom Endpoints ─────────────────────────────────────────────────────────

ENDPOINT_COLS = "id, org_id, name, base_url, auth_type, auth_header_name, default_headers_json, is_active, created_at"


@router.post("/custom-endpoints", response_model=CustomEndpointResponse)
def create_custom_endpoint(payload: CustomEndpointCreate):
    if payload.auth_type not in ("bearer", "header"):
        raise HTTPException(status_code=400, detail="auth_type must be 'bearer' or 'header'")
    if payload.auth_type == "header" and not payload.auth_header_name:
        raise HTTPException(status_code=400, detail="auth_header_name required when auth_type='header'")
    if not payload.base_url.startswith("http"):
        raise HTTPException(status_code=400, detail="base_url must be a valid HTTP(S) URL")
    if not payload.auth_secret or len(payload.auth_secret.strip()) < 1:
        raise HTTPException(status_code=400, detail="auth_secret is required")

    encrypted = encrypt_api_key(payload.auth_secret.strip())

    row = {
        "org_id": payload.org_id,
        "name": payload.name.strip(),
        "base_url": payload.base_url.strip().rstrip("/"),
        "auth_type": payload.auth_type,
        "auth_header_name": payload.auth_header_name.strip() if payload.auth_header_name else None,
        "auth_secret_encrypted": encrypted,
        "default_headers_json": payload.default_headers or {},
    }

    try:
        result = supabase.table("custom_endpoints").insert(row).execute()
    except Exception as e:
        logger.error("Failed to create custom endpoint: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create endpoint") from e

    data = result.data
    if not data or len(data) == 0:
        raise HTTPException(status_code=500, detail="Insert returned no data")
    return data[0]


@router.get("/custom-endpoints/{org_id}", response_model=List[CustomEndpointResponse])
def list_custom_endpoints(org_id: str):
    try:
        result = (
            supabase.table("custom_endpoints")
            .select(ENDPOINT_COLS)
            .eq("org_id", org_id)
            .eq("is_active", True)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as e:
        logger.error("Failed to list custom endpoints: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list endpoints") from e
    return result.data or []


@router.delete("/custom-endpoints/{endpoint_id}")
def delete_custom_endpoint(endpoint_id: str):
    """Soft-delete: sets is_active=False."""
    try:
        supabase.table("custom_endpoints").update({"is_active": False}).eq("id", endpoint_id).execute()
    except Exception as e:
        logger.error("Failed to delete custom endpoint: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete endpoint") from e
    return {"ok": True}


# ── Model Registry ───────────────────────────────────────────────────────────

REGISTRY_COLS = "id, org_id, name, target_type, provider, model_name, endpoint_id, endpoint_model_name, is_active, created_at"


@router.post("/models", response_model=ModelRegistryResponse)
def create_model(payload: ModelRegistryCreate):
    if payload.target_type not in ("provider_model", "openai_compatible_endpoint"):
        raise HTTPException(status_code=400, detail="target_type must be 'provider_model' or 'openai_compatible_endpoint'")

    if payload.target_type == "provider_model":
        if not payload.provider or not payload.model_name:
            raise HTTPException(status_code=400, detail="provider and model_name are required for provider_model")
    else:
        if not payload.endpoint_id or not payload.endpoint_model_name:
            raise HTTPException(status_code=400, detail="endpoint_id and endpoint_model_name are required for openai_compatible_endpoint")
        # Verify endpoint exists
        ep = supabase.table("custom_endpoints").select("id").eq("id", payload.endpoint_id).eq("is_active", True).execute()
        if not ep.data:
            raise HTTPException(status_code=404, detail="Custom endpoint not found or inactive")

    row = {
        "org_id": payload.org_id,
        "name": payload.name.strip(),
        "target_type": payload.target_type,
        "provider": payload.provider.lower().strip() if payload.provider else None,
        "model_name": payload.model_name.strip() if payload.model_name else None,
        "endpoint_id": payload.endpoint_id,
        "endpoint_model_name": payload.endpoint_model_name.strip() if payload.endpoint_model_name else None,
    }

    try:
        result = supabase.table("model_registry").insert(row).execute()
    except Exception as e:
        logger.error("Failed to create model registry entry: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create model") from e

    data = result.data
    if not data or len(data) == 0:
        raise HTTPException(status_code=500, detail="Insert returned no data")
    return data[0]


@router.get("/models/{org_id}", response_model=List[ModelRegistryResponse])
def list_models(org_id: str):
    try:
        result = (
            supabase.table("model_registry")
            .select(REGISTRY_COLS)
            .eq("org_id", org_id)
            .eq("is_active", True)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as e:
        logger.error("Failed to list model registry: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list models") from e

    models = result.data or []

    # Enrich with endpoint info where applicable
    endpoint_ids = {m["endpoint_id"] for m in models if m.get("endpoint_id")}
    endpoints_map: dict[str, dict] = {}
    if endpoint_ids:
        try:
            ep_result = (
                supabase.table("custom_endpoints")
                .select("id, name, base_url")
                .in_("id", list(endpoint_ids))
                .execute()
            )
            endpoints_map = {e["id"]: e for e in (ep_result.data or [])}
        except Exception:
            pass  # graceful degrade — no endpoint enrichment

    for m in models:
        ep = endpoints_map.get(m.get("endpoint_id") or "")
        m["endpoint_name"] = ep["name"] if ep else None
        m["endpoint_base_url"] = ep["base_url"] if ep else None

    return models


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Soft-delete: sets is_active=False."""
    try:
        supabase.table("model_registry").update({"is_active": False}).eq("id", model_id).execute()
    except Exception as e:
        logger.error("Failed to delete model: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete model") from e
    return {"ok": True}


# ── Resolution helper (used by workflow_runtime) ─────────────────────────────

def resolve_model_target(model_registry_id: str, org_id: str) -> ModelTarget:
    """
    Fetch a model_registry row + its custom_endpoint (if applicable),
    decrypt auth secrets, and return a fully-resolved ModelTarget ready
    for the router layer.

    Raises HTTPException(404) if not found / inactive.
    """
    try:
        result = (
            supabase.table("model_registry")
            .select(REGISTRY_COLS)
            .eq("id", model_registry_id)
            .eq("org_id", org_id)
            .eq("is_active", True)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model registry entry not found: {model_registry_id}")

    mr = result.data
    if not mr:
        raise HTTPException(status_code=404, detail=f"Model registry entry not found: {model_registry_id}")

    if mr["target_type"] == "provider_model":
        return ModelTarget(
            target_type="provider_model",
            provider=mr["provider"],
            model_name=mr["model_name"],
            model_registry_id=mr["id"],
        )

    # openai_compatible_endpoint
    ep_id = mr["endpoint_id"]
    try:
        ep_result = (
            supabase.table("custom_endpoints")
            .select("id, base_url, auth_type, auth_header_name, auth_secret_encrypted, default_headers_json")
            .eq("id", ep_id)
            .eq("is_active", True)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Custom endpoint not found: {ep_id}")

    ep = ep_result.data
    if not ep:
        raise HTTPException(status_code=404, detail=f"Custom endpoint not found: {ep_id}")

    # Decrypt auth secret
    try:
        secret = decrypt_api_key(ep["auth_secret_encrypted"])
    except Exception as e:
        logger.error("Failed to decrypt endpoint auth secret for %s: %s", ep_id, type(e).__name__)
        raise HTTPException(status_code=500, detail="Failed to decrypt endpoint credentials") from e

    # Build auth header
    if ep["auth_type"] == "bearer":
        auth_header_name = "Authorization"
        auth_header_value = f"Bearer {secret}"
    else:
        auth_header_name = ep.get("auth_header_name") or "X-Api-Key"
        auth_header_value = secret

    return ModelTarget(
        target_type="openai_compatible_endpoint",
        provider="openai",  # always reuse openai router
        model_name=mr["endpoint_model_name"],
        base_url=ep["base_url"],
        auth_header_name=auth_header_name,
        auth_header_value=auth_header_value,
        default_headers=ep.get("default_headers_json") or {},
        model_registry_id=mr["id"],
        endpoint_id=ep["id"],
    )
