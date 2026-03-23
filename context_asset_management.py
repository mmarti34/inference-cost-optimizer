"""CRUD API for org-level context assets (knowledge base)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth_dependency import require_org_member, AuthenticatedUser
from supabase_client import supabase

logger = logging.getLogger(__name__)

router = APIRouter(tags=["context-assets"])

_ALLOWED_TYPES = {"text", "document", "url", "notion_page"}
_MAX_CONTENT_CHARS = 500_000


class CreateContextAssetRequest(BaseModel):
    org_id: str
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    asset_type: str = "text"
    content: str | None = None
    source_ref: dict[str, Any] | None = None


class UpdateContextAssetRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    content: str | None = None
    source_ref: dict[str, Any] | None = None


@router.get("/context-assets/{org_id}")
async def list_context_assets(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
    asset_type: str | None = None,
):
    query = (
        supabase.table("context_assets")
        .select("id, org_id, name, description, asset_type, metadata, status, created_at, updated_at")
        .eq("org_id", org_id)
        .neq("status", "deleted")
        .order("created_at", desc=True)
    )
    if asset_type:
        query = query.eq("asset_type", asset_type)
    result = query.execute()
    return result.data or []


@router.post("/context-assets")
async def create_context_asset(
    body: CreateContextAssetRequest,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    if body.asset_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"asset_type must be one of: {', '.join(sorted(_ALLOWED_TYPES))}")

    status = "active"
    if body.asset_type == "text":
        if not body.content:
            raise HTTPException(400, "content is required for text assets")
        if len(body.content) > _MAX_CONTENT_CHARS:
            raise HTTPException(400, f"content exceeds maximum of {_MAX_CONTENT_CHARS} characters")
    else:
        status = "processing"

    metadata = {}
    if body.content:
        metadata["char_count"] = len(body.content)
        metadata["word_count"] = len(body.content.split())

    org_id = body.org_id

    row = {
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "name": body.name,
        "description": body.description,
        "asset_type": body.asset_type,
        "content": body.content,
        "source_ref": body.source_ref,
        "metadata": metadata,
        "status": status,
    }
    result = supabase.table("context_assets").insert(row).execute()
    return result.data[0] if result.data else row


@router.get("/context-assets/{org_id}/{asset_id}")
async def get_context_asset(
    org_id: str,
    asset_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    result = (
        supabase.table("context_assets")
        .select("*")
        .eq("id", asset_id)
        .eq("org_id", org_id)
        .neq("status", "deleted")
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Context asset not found")
    return result.data[0]


@router.put("/context-assets/{org_id}/{asset_id}")
async def update_context_asset(
    org_id: str,
    asset_id: str,
    body: UpdateContextAssetRequest,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    existing = (
        supabase.table("context_assets")
        .select("id, asset_type")
        .eq("id", asset_id)
        .eq("org_id", org_id)
        .neq("status", "deleted")
        .execute()
    )
    if not existing.data:
        raise HTTPException(404, "Context asset not found")

    updates: dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if body.name is not None:
        updates["name"] = body.name
    if body.description is not None:
        updates["description"] = body.description
    if body.content is not None:
        if len(body.content) > _MAX_CONTENT_CHARS:
            raise HTTPException(400, f"content exceeds maximum of {_MAX_CONTENT_CHARS} characters")
        updates["content"] = body.content
        updates["metadata"] = {
            "char_count": len(body.content),
            "word_count": len(body.content.split()),
        }
    if body.source_ref is not None:
        updates["source_ref"] = body.source_ref

    result = (
        supabase.table("context_assets")
        .update(updates)
        .eq("id", asset_id)
        .eq("org_id", org_id)
        .execute()
    )
    return result.data[0] if result.data else updates


@router.delete("/context-assets/{org_id}/{asset_id}")
async def delete_context_asset(
    org_id: str,
    asset_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    existing = (
        supabase.table("context_assets")
        .select("id")
        .eq("id", asset_id)
        .eq("org_id", org_id)
        .neq("status", "deleted")
        .execute()
    )
    if not existing.data:
        raise HTTPException(404, "Context asset not found")

    supabase.table("context_assets").update({
        "status": "deleted",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", asset_id).eq("org_id", org_id).execute()

    return {"deleted": True}
