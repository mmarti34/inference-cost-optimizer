from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from supabase_client import supabase
import uuid

router = APIRouter()

# Pydantic models
class PromptTemplateCreate(BaseModel):
    name: str
    description: str
    provider: str
    model: str
    prompt: str
    project_id: Optional[str] = None
    org_id: str

class PromptTemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    project_id: Optional[str] = None

class PromptTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    provider: str
    model: str
    prompt: str
    project_id: Optional[str]
    org_id: str
    created_at: str
    updated_at: str

# Prompt Template Endpoints
@router.post("/prompt-templates", response_model=PromptTemplateResponse)
async def create_prompt_template(prompt_data: PromptTemplateCreate):
    """Create a new prompt template"""
    try:
        # Generate UUID for the prompt template
        prompt_id = str(uuid.uuid4())
        
        # Insert the prompt template - user_id is now nullable
        result = supabase.table("prompt_templates").insert({
            "id": prompt_id,
            "name": prompt_data.name,
            "description": prompt_data.description,
            "provider": prompt_data.provider,
            "model": prompt_data.model,
            "prompt": prompt_data.prompt,
            "project_id": prompt_data.project_id,
            "org_id": prompt_data.org_id
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create prompt template")
        
        return result.data[0]
    except Exception as e:
        error_msg = str(e)
        # Check if it's an RLS policy error
        if "permission denied" in error_msg.lower() or "policy" in error_msg.lower():
            raise HTTPException(
                status_code=403, 
                detail=f"Permission denied. Ensure SUPABASE_KEY is set to service_role key (not anon key). Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Error creating prompt template: {error_msg}")

@router.get("/prompt-templates/{org_id}", response_model=List[PromptTemplateResponse])
async def get_prompt_templates(org_id: str):
    """Get all prompt templates for an organization"""
    try:
        result = supabase.table("prompt_templates").select("id, org_id, name, description, provider, model, prompt, project_id, created_at, updated_at").eq("org_id", org_id).execute()
        
        # Sort in Python if needed
        if result.data:
            result.data = sorted(result.data, key=lambda x: x.get("created_at", ""), reverse=True)
        
        if not result.data:
            return []
        
        return result.data
    except Exception as e:
        error_msg = str(e)
        # Check if it's an RLS policy error
        if "permission denied" in error_msg.lower() or "policy" in error_msg.lower() or "row-level security" in error_msg.lower():
            raise HTTPException(
                status_code=403, 
                detail=f"Permission denied. Ensure SUPABASE_KEY is set to service_role key (not anon key). Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Error fetching prompt templates: {error_msg}")

@router.get("/prompt-templates/{org_id}/{prompt_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(org_id: str, prompt_id: str):
    """Get a specific prompt template"""
    try:
        result = supabase.table("prompt_templates").select("id, org_id, name, description, provider, model, prompt, project_id, created_at, updated_at").eq("id", prompt_id).eq("org_id", org_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompt template: {str(e)}")

@router.put("/prompt-templates/{org_id}/{prompt_id}", response_model=PromptTemplateResponse)
async def update_prompt_template(org_id: str, prompt_id: str, prompt_data: PromptTemplateUpdate):
    """Update a prompt template"""
    try:
        # First check if the prompt template exists and belongs to the org
        existing = supabase.table("prompt_templates").select("id, org_id, name, description, provider, model, prompt, project_id, created_at, updated_at").eq("id", prompt_id).eq("org_id", org_id).single().execute()
        
        if not existing.data:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        
        # Update only provided fields
        update_data = {k: v for k, v in prompt_data.dict().items() if v is not None}
        
        if not update_data:
            return existing.data
        
        result = supabase.table("prompt_templates").update(update_data).eq("id", prompt_id).eq("org_id", org_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to update prompt template")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prompt template: {str(e)}")

@router.delete("/prompt-templates/{org_id}/{prompt_id}")
async def delete_prompt_template(org_id: str, prompt_id: str):
    """Delete a prompt template"""
    try:
        # First check if the prompt template exists and belongs to the org
        existing = supabase.table("prompt_templates").select("id").eq("id", prompt_id).eq("org_id", org_id).single().execute()
        
        if not existing.data:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        
        # Delete the prompt template
        result = supabase.table("prompt_templates").delete().eq("id", prompt_id).eq("org_id", org_id).execute()
        
        return {"message": "Prompt template deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting prompt template: {str(e)}")
