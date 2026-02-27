import os
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from supabase_client import supabase

router = APIRouter()

# Pydantic models
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    monthly_budget: Optional[float] = None
    org_id: str

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    monthly_budget: Optional[float] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    monthly_budget: Optional[float]
    org_id: str
    user_id: Optional[str] = None
    created_at: str

# Project Endpoints
@router.post("/projects", response_model=ProjectResponse)
async def create_project(project_data: ProjectCreate):
    """Create a new project"""
    try:
        # Generate UUID for the project
        project_id = str(uuid.uuid4())
        
        # Insert the project - user_id is now nullable
        result = supabase.table("projects").insert({
            "id": project_id,
            "name": project_data.name,
            "description": project_data.description,
            "monthly_budget": project_data.monthly_budget,
            "org_id": project_data.org_id
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create project")
        
        return result.data[0]
    except Exception as e:
        error_msg = str(e)
        # Check if it's an RLS policy error
        if "permission denied" in error_msg.lower() or "policy" in error_msg.lower():
            raise HTTPException(
                status_code=403, 
                detail=f"Permission denied. Ensure SUPABASE_KEY is set to service_role key (not anon key). Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Error creating project: {error_msg}")

@router.get("/projects/{org_id}", response_model=List[ProjectResponse])
async def get_projects(org_id: str):
    """Get all projects for an organization"""
    try:
        result = supabase.table("projects").select("id, name, description, monthly_budget, org_id, user_id, created_at").eq("org_id", org_id).execute()
        
        if not result.data:
            return []
        
        # Sort in Python if needed
        sorted_data = sorted(result.data, key=lambda x: x.get("created_at", ""), reverse=True)
        return sorted_data
    except Exception as e:
        error_msg = str(e)
        # Check if it's an RLS policy error
        if "permission denied" in error_msg.lower() or "policy" in error_msg.lower() or "row-level security" in error_msg.lower():
            raise HTTPException(
                status_code=403, 
                detail=f"Permission denied. Ensure SUPABASE_KEY is set to service_role key (not anon key). Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {error_msg}")

@router.get("/projects/{org_id}/{project_id}", response_model=ProjectResponse)
async def get_project(org_id: str, project_id: str):
    """Get a specific project"""
    try:
        result = supabase.table("projects").select("id, name, description, monthly_budget, org_id, user_id, created_at").eq("id", project_id).eq("org_id", org_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching project: {str(e)}")

@router.put("/projects/{org_id}/{project_id}", response_model=ProjectResponse)
async def update_project(org_id: str, project_id: str, project_data: ProjectUpdate):
    """Update a project"""
    try:
        # First check if the project exists and belongs to the org
        existing = supabase.table("projects").select("id, name, description, monthly_budget, org_id, user_id, created_at").eq("id", project_id).eq("org_id", org_id).single().execute()
        
        if not existing.data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Update only provided fields
        update_data = {k: v for k, v in project_data.dict().items() if v is not None}
        
        if not update_data:
            return existing.data
        
        result = supabase.table("projects").update(update_data).eq("id", project_id).eq("org_id", org_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to update project")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating project: {str(e)}")

@router.delete("/projects/{org_id}/{project_id}")
async def delete_project(org_id: str, project_id: str):
    """Delete a project"""
    try:
        # First check if the project exists and belongs to the org
        existing = supabase.table("projects").select("id").eq("id", project_id).eq("org_id", org_id).single().execute()
        
        if not existing.data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if there are any prompt templates using this project
        prompts_result = supabase.table("prompt_templates").select("id").eq("project_id", project_id).limit(1).execute()
        
        if prompts_result.data:
            raise HTTPException(status_code=400, detail="Cannot delete project that has prompt templates. Please reassign or delete the prompt templates first.")
        
        # Delete the project
        result = supabase.table("projects").delete().eq("id", project_id).eq("org_id", org_id).execute()
        
        return {"message": "Project deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")

# Debug endpoints (at the end to avoid route conflicts). Only when OPTIML_DEBUG=true.
def _debug_enabled() -> bool:
    return os.environ.get("OPTIML_DEBUG", "false").lower() == "true"

@router.get("/debug/projects")
async def debug_projects():
    """Debug endpoint to test projects table access. Disabled unless OPTIML_DEBUG=true."""
    if not _debug_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        # Test basic table access
        result = supabase.table("projects").select("id, name, description, monthly_budget, org_id, user_id, created_at").limit(1).execute()
        return {
            "status": "success",
            "message": "Projects table accessible",
            "count": len(result.data) if result.data else 0,
            "sample_data": result.data[0] if result.data else None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Projects table error: {str(e)}",
            "error_type": type(e).__name__
        }

@router.get("/debug/projects/{org_id}")
async def debug_get_projects(org_id: str):
    """Debug endpoint to test GET projects with org_id. Disabled unless OPTIML_DEBUG=true."""
    if not _debug_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        result = supabase.table("projects").select("id, name, description, monthly_budget, org_id, user_id, created_at").eq("org_id", org_id).execute()
        
        # Sort in Python if needed
        if result.data:
            result.data = sorted(result.data, key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Test serialization with ProjectResponse
        projects = []
        for project in result.data:
            project_response = ProjectResponse(
                id=project["id"],
                name=project["name"],
                description=project.get("description"),
                monthly_budget=project.get("monthly_budget"),
                org_id=project["org_id"],
                user_id=project.get("user_id"),
                created_at=project["created_at"]
            )
            projects.append(project_response.dict())
        
        return {
            "status": "success",
            "message": f"Found {len(projects)} projects for org {org_id}",
            "projects": projects
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"GET projects error: {str(e)}",
            "error_type": type(e).__name__
        }
