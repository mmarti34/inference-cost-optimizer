"""
CRUD endpoints for org-level encrypted secrets.
Secrets are referenced in workflow graph_json via {{secrets.NAME}} syntax.
At runtime, the workflow executor resolves these references with decrypted values.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import audit
from supabase_client import supabase
from auth_dependency import require_org_member, AuthenticatedUser
from utils.encryption import encrypt_api_key, decrypt_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SecretCreate(BaseModel):
    org_id: str
    name: str
    value: str  # plaintext — encrypted before storage
    description: str = ""


class SecretUpdate(BaseModel):
    value: Optional[str] = None  # plaintext — encrypted before storage
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/secrets/{org_id}")
async def list_secrets(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """List all secrets for an org. Values are masked."""
    try:
        result = (
            supabase.table("org_secrets")
            .select("id, org_id, name, description, created_at, updated_at")
            .eq("org_id", org_id)
            .order("created_at", desc=False)
            .execute()
        )
        secrets = result.data or []
        # Add masked value indicator
        for s in secrets:
            s["has_value"] = True
            s["masked_value"] = "••••••••"
        return secrets
    except Exception as e:
        logger.error("Failed to list secrets for org %s: %s", org_id, e)
        raise HTTPException(status_code=500, detail="Failed to list secrets")


@router.post("/secrets")
async def create_secret(
    request: Request,
    body: SecretCreate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new encrypted secret."""
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="Secret name is required")
    if not body.value:
        raise HTTPException(status_code=400, detail="Secret value is required")

    # Validate name format: alphanumeric + underscores only
    clean_name = body.name.strip().upper().replace("-", "_").replace(" ", "_")
    if not all(c.isalnum() or c == "_" for c in clean_name):
        raise HTTPException(status_code=400, detail="Secret name must be alphanumeric with underscores only")

    try:
        encrypted = encrypt_api_key(body.value)
    except Exception as e:
        logger.error("Failed to encrypt secret: %s", e)
        raise HTTPException(status_code=500, detail="Failed to encrypt secret value")

    try:
        result = (
            supabase.table("org_secrets")
            .insert({
                "org_id": body.org_id,
                "name": clean_name,
                "encrypted_value": encrypted,
                "description": body.description or "",
            })
            .execute()
        )
        row = result.data[0] if result.data else {}
        # The NAME is not recorded: it is customer-chosen free text on a secret,
        # and the row id resolves it. The VALUE never reaches this module.
        audit.record(
            audit.ORG_SECRET_CREATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORG_SECRET,
            resource_id=row.get("id"),
            request=request,
        )
        return {
            "id": row.get("id"),
            "org_id": row.get("org_id"),
            "name": row.get("name"),
            "description": row.get("description"),
            "created_at": row.get("created_at"),
            "has_value": True,
            "masked_value": "••••••••",
        }
    except Exception as e:
        detail = str(e)
        if "uq_org_secrets_org_name" in detail or "duplicate key" in detail.lower():
            raise HTTPException(status_code=409, detail=f"Secret '{clean_name}' already exists")
        logger.error("Failed to create secret: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create secret")


@router.put("/secrets/{org_id}/{secret_id}")
async def update_secret(
    request: Request,
    org_id: str,
    secret_id: str,
    body: SecretUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update a secret's value and/or description."""
    patch: dict = {"updated_at": "now()"}

    if body.value is not None:
        try:
            patch["encrypted_value"] = encrypt_api_key(body.value)
        except Exception as e:
            logger.error("Failed to encrypt secret: %s", e)
            raise HTTPException(status_code=500, detail="Failed to encrypt secret value")

    if body.description is not None:
        patch["description"] = body.description

    try:
        result = (
            supabase.table("org_secrets")
            .update(patch)
            .eq("id", secret_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            # Scoped by org, so this is "no such secret in YOUR org" — which is
            # also what a cross-tenant attempt looks like from in here.
            audit.record(
                audit.ORG_SECRET_UPDATE_REFUSED,
                principal=auth_user,
                resource_type=audit.RESOURCE_ORG_SECRET,
                resource_id=secret_id,
                metadata={"reason_code": audit.REASON_NOT_FOUND},
                request=request,
            )
            raise HTTPException(status_code=404, detail="Secret not found")
        row = result.data[0]
        audit.record(
            audit.ORG_SECRET_UPDATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORG_SECRET,
            resource_id=secret_id,
            request=request,
        )
        return {
            "id": row.get("id"),
            "org_id": row.get("org_id"),
            "name": row.get("name"),
            "description": row.get("description"),
            "updated_at": row.get("updated_at"),
            "has_value": True,
            "masked_value": "••••••••",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update secret: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update secret")


@router.delete("/secrets/{org_id}/{secret_id}")
async def delete_secret(
    request: Request,
    org_id: str,
    secret_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a secret."""
    try:
        result = (
            supabase.table("org_secrets")
            .delete()
            .eq("id", secret_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            audit.record(
                audit.ORG_SECRET_DELETE_REFUSED,
                principal=auth_user,
                resource_type=audit.RESOURCE_ORG_SECRET,
                resource_id=secret_id,
                metadata={"reason_code": audit.REASON_NOT_FOUND},
                request=request,
            )
            raise HTTPException(status_code=404, detail="Secret not found")
        audit.record(
            audit.ORG_SECRET_DELETED,
            principal=auth_user,
            resource_type=audit.RESOURCE_ORG_SECRET,
            resource_id=secret_id,
            request=request,
        )
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete secret: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete secret")
