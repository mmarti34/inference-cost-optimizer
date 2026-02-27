"""
Custom model configs: named LLM presets per org.
Explicit columns only; no select("*"); no updated_at.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from supabase_client import supabase

router = APIRouter()

COLS = "id, org_id, name, provider, base_model, temperature, max_tokens, system_prefix, created_at"


class CustomModelCreate(BaseModel):
    org_id: str
    name: str
    provider: str
    base_model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    system_prefix: Optional[str] = None

    class Config:
        extra = "ignore"


class CustomModelUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    base_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prefix: Optional[str] = None

    class Config:
        extra = "ignore"


class CustomModelResponse(BaseModel):
    id: str
    org_id: str
    name: str
    provider: str
    base_model: str
    temperature: float
    max_tokens: int
    system_prefix: Optional[str]
    created_at: str

    class Config:
        extra = "ignore"


@router.get("/custom-models", response_model=List[CustomModelResponse])
async def get_custom_models(org_id: str):
    """List custom models for an organization."""
    try:
        result = (
            supabase.table("custom_models")
            .select(COLS)
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result.data or []


@router.post("/custom-models", response_model=CustomModelResponse)
async def create_custom_model(payload: CustomModelCreate):
    """Create a custom model config."""
    try:
        data = {
            "org_id": payload.org_id,
            "name": payload.name,
            "provider": payload.provider,
            "base_model": payload.base_model,
            "temperature": payload.temperature if payload.temperature is not None else 0.7,
            "max_tokens": payload.max_tokens if payload.max_tokens is not None else 1024,
            "system_prefix": payload.system_prefix,
        }
        result = supabase.table("custom_models").insert(data).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=500, detail="Failed to create custom model")
    row = result.data[0]
    return {
        "id": row["id"],
        "org_id": row["org_id"],
        "name": row["name"],
        "provider": row["provider"],
        "base_model": row["base_model"],
        "temperature": float(row.get("temperature") or 0.7),
        "max_tokens": int(row.get("max_tokens") or 1024),
        "system_prefix": row.get("system_prefix"),
        "created_at": row["created_at"],
    }


@router.put("/custom-models/{model_id}", response_model=CustomModelResponse)
async def update_custom_model(model_id: str, payload: CustomModelUpdate):
    """Update a custom model config."""
    try:
        existing = (
            supabase.table("custom_models")
            .select(COLS)
            .eq("id", model_id)
            .single()
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not existing.data:
        raise HTTPException(status_code=404, detail="Custom model not found")
    update_data = {k: v for k, v in payload.dict().items() if v is not None}
    if not update_data:
        return existing.data
    try:
        result = (
            supabase.table("custom_models")
            .update(update_data)
            .eq("id", model_id)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=500, detail="Failed to update custom model")
    row = result.data[0]
    return {
        "id": row["id"],
        "org_id": row["org_id"],
        "name": row["name"],
        "provider": row["provider"],
        "base_model": row["base_model"],
        "temperature": float(row.get("temperature") or 0.7),
        "max_tokens": int(row.get("max_tokens") or 1024),
        "system_prefix": row.get("system_prefix"),
        "created_at": row["created_at"],
    }


@router.delete("/custom-models/{model_id}")
async def delete_custom_model(model_id: str):
    """Delete a custom model config."""
    try:
        result = supabase.table("custom_models").delete().eq("id", model_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Custom model deleted"}


def get_custom_model_by_id(custom_model_id: str, org_id: str) -> Optional[dict]:
    """Resolve custom_model_id to provider, base_model, and optional params. Used by workflow runtime."""
    try:
        result = (
            supabase.table("custom_models")
            .select("id, org_id, name, provider, base_model, temperature, max_tokens, system_prefix")
            .eq("id", custom_model_id)
            .eq("org_id", org_id)
            .single()
            .execute()
        )
    except Exception:
        return None
    return result.data if result.data else None
