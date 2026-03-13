"""
Deploy from parsed: create workflow + deployment from parse-import result.
Used by Cursor plugin so user can replace code and deploy without opening optiml.one.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth_dependency import require_org_member, AuthenticatedUser
from plan_enforcement import check_workflow_limit, check_server_key_limit
from supabase_client import supabase
from utils.encryption import hash_service_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

# api_type -> (node_type, default_user_var, user_var_type)
API_TYPE_CONFIG = {
    "chat": ("ai-step", "user_message", "string"),
    "vision": ("vision_step", "user_message", "string"),
    "tool_call": ("tool_call", "user_message", "string"),
    "agent": ("agent", "user_message", "string"),
    "image_generation": ("image_gen_step", "prompt", "string"),
    "tts": ("tts_step", "text", "string"),
    "stt": ("stt_step", "audio", "audio"),
    "embeddings": ("embedding_step", "text", "string"),
}


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = "".join(c if c.isalnum() or c == "-" else "-" for c in s)
    return "-".join(filter(None, s.split("-"))) or "workflow"


def build_graph_from_parsed(parsed: dict) -> tuple[dict, list[dict]]:
    """Port of frontend buildGraphFromParsed. Returns (graph_json, variables)."""
    api_type = (parsed.get("api_type") or "chat").replace("-", "_")
    if api_type not in API_TYPE_CONFIG:
        api_type = "chat"
    node_type, default_user_var, user_var_type = API_TYPE_CONFIG[api_type]
    user_var = (parsed.get("user_variable") or "").strip() or default_user_var
    user_content = parsed.get("user_content")
    is_hardcoded = not parsed.get("user_variable") and bool(user_content)
    img_var = (parsed.get("image_variable") or "").strip() or "image_url"

    # input variables
    if api_type == "vision":
        input_vars = []
        if not is_hardcoded:
            input_vars.append({"name": user_var, "type": "string"})
        input_vars.append({"name": img_var, "type": "image"})
    elif api_type == "stt":
        input_vars = [{"name": user_var, "type": "audio"}]
    else:
        input_vars = [{"name": user_var, "type": user_var_type}]

    # ai node data
    if api_type in ("chat", "tool_call", "agent"):
        task_desc = (user_content or "") if is_hardcoded else f"{{{{{user_var}}}}}"
        ai_data = {
            "label": "agent" if api_type == "agent" else "respond",
            "provider": parsed.get("provider") or "openai",
            "modelName": parsed.get("model") or "gpt-4o-mini",
            "systemInstructions": (parsed.get("system_prompt") or "") or "",
            "taskDescription": task_desc,
            "temperature": float(parsed.get("temperature") or 0.7),
            "maxTokens": int(parsed.get("max_tokens") or 1024),
        }
        if api_type in ("tool_call", "agent"):
            ai_data["tools"] = []
    elif api_type == "vision":
        prompt = (parsed.get("system_prompt") or "") or ((user_content or "") if is_hardcoded else f"{{{{{user_var}}}}}")
        ai_data = {
            "label": "vision",
            "provider": parsed.get("provider") or "openai",
            "model": parsed.get("model") or "gpt-4o-mini",
            "prompt": prompt,
            "image_source": "input",
            "image_variable": img_var,
        }
    elif api_type == "image_generation":
        ai_data = {
            "label": "image gen",
            "provider": parsed.get("provider") or "openai",
            "model": parsed.get("model") or "dall-e-3",
            "prompt_source": "static" if is_hardcoded else "input",
        }
        if is_hardcoded:
            ai_data["prompt"] = user_content or ""
        else:
            ai_data["prompt_variable"] = user_var
    elif api_type == "tts":
        ai_data = {
            "label": "tts",
            "provider": parsed.get("provider") or "openai",
            "model": parsed.get("model") or "tts-1",
            "text_source": "static" if is_hardcoded else "input",
        }
        if is_hardcoded:
            ai_data["text"] = user_content or ""
        else:
            ai_data["input_variable"] = user_var
        if parsed.get("voice"):
            ai_data["voice"] = parsed["voice"]
        if parsed.get("instructions"):
            ai_data["instructions"] = parsed["instructions"]
    elif api_type == "stt":
        ai_data = {
            "label": "stt",
            "provider": parsed.get("provider") or "openai",
            "model": parsed.get("model") or "whisper-1",
            "audio_variable": user_var,
        }
        if parsed.get("instructions"):
            ai_data["prompt"] = parsed["instructions"]
    else:
        ai_data = {
            "label": "embeddings",
            "provider": parsed.get("provider") or "openai",
            "model": parsed.get("model") or "text-embedding-3-small",
            "text_source": "static" if is_hardcoded else "input",
        }
        if is_hardcoded:
            ai_data["text"] = user_content or ""

    nodes = [
        {"id": "input", "type": "input", "position": {"x": 100, "y": 200}, "data": {"label": "input", "inputVariables": input_vars}},
        {"id": "ai_1", "type": node_type, "position": {"x": 380, "y": 200}, "data": ai_data},
        {"id": "output", "type": "output", "position": {"x": 660, "y": 200}, "data": {"label": "output", "response_format": "auto"}},
    ]
    edges = [
        {"id": "e1", "source": "input", "target": "ai_1"},
        {"id": "e2", "source": "ai_1", "target": "output"},
    ]
    graph = {"nodes": nodes, "edges": edges}

    # variables for workflow (mirror frontend graphVariables)
    if api_type == "vision":
        variables = []
        if not is_hardcoded:
            variables.append({"name": user_var, "type": "string", "required": True, "description": ""})
        variables.append({"name": img_var, "type": "image", "required": True, "description": ""})
    else:
        variables = [{"name": user_var, "type": user_var_type, "required": True, "description": ""}]

    return graph, variables


class DeployFromParsedBody(BaseModel):
    org_id: str
    parsed: dict
    endpoint_slug: str

    class Config:
        extra = "ignore"


@router.post("/api/cursor/deploy-from-parsed")
async def deploy_from_parsed(
    body: DeployFromParsedBody,
    _user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Create workflow + deployment from a parse-import result. Uses Cursor or Supabase auth.
    Promotes the deployment so the endpoint is live immediately. Returns org_slug, endpoint_slug, server_key.
    """
    org_id = body.org_id.strip()
    parsed = body.parsed or {}
    endpoint_slug = (body.endpoint_slug or "").strip() or _slugify(parsed.get("suggestedName") or "workflow")

    if not parsed.get("provider") or not parsed.get("model"):
        raise HTTPException(status_code=400, detail="parsed must include provider and model (from parse-import).")

    try:
        graph_json, variables = build_graph_from_parsed(parsed)
    except Exception as e:
        logger.warning("build_graph_from_parsed failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid parsed shape: {e}") from e

    check_workflow_limit(org_id)

    # Resolve project
    projs = supabase.table("projects").select("id").eq("org_id", org_id).limit(1).execute()
    project_id = projs.data[0]["id"] if projs.data and len(projs.data) > 0 else None
    if not project_id:
        new_proj = supabase.table("projects").insert({"org_id": org_id, "name": "Default"}).execute()
        if new_proj.data:
            project_id = new_proj.data[0]["id"]
    if not project_id:
        raise HTTPException(status_code=400, detail="Could not resolve or create project.")

    workflow_name = (parsed.get("suggestedName") or endpoint_slug or "workflow").strip() or "workflow"
    slug = _slugify(workflow_name)

    wf_insert = {
        "org_id": org_id,
        "project_id": project_id,
        "name": workflow_name,
        "slug": slug,
        "graph_json": graph_json,
        "variables": variables,
    }
    wf_result = supabase.table("workflows").insert(wf_insert).execute()
    if not wf_result.data:
        raise HTTPException(status_code=500, detail="Failed to create workflow.")
    workflow_id = str(wf_result.data[0]["id"])

    dep_insert = {
        "workflow_id": workflow_id,
        "org_id": org_id,
        "version": 1,
        "endpoint_slug": endpoint_slug,
        "graph_json": graph_json,
        "status": "promoted",
        "promoted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_id": project_id,
    }
    dep_result = supabase.table("workflow_deployments").insert(dep_insert).execute()
    if not dep_result.data:
        raise HTTPException(status_code=500, detail="Failed to create deployment.")

    # Create a new server key so we can return plaintext (user adds to .env). Existing keys are hashed only.
    check_server_key_limit(org_id)
    import secrets as sec
    plaintext_key = sec.token_urlsafe(32)
    hashed = hash_service_api_key(plaintext_key)
    insert_payload = {"org_id": org_id, "hashed_key": hashed, "name": "Cursor deploy", "status": "active"}
    try:
        supabase.table("service_api_keys").insert(insert_payload).execute()
        server_key = plaintext_key
    except Exception as e:
        if "null value" in str(e).lower() and "api_key" in str(e).lower():
            insert_payload["api_key"] = ""
            supabase.table("service_api_keys").insert(insert_payload).execute()
            server_key = plaintext_key
        else:
            server_key = None

    org_row = supabase.table("organizations").select("slug").eq("id", org_id).single().execute()
    org_slug = (org_row.data or {}).get("slug") or org_id

    return {
        "org_slug": org_slug,
        "endpoint_slug": endpoint_slug,
        "endpoint_url": f"https://api.optiml.one/api/public/{org_slug}/{endpoint_slug}",
        "workflow_id": workflow_id,
        "server_key": server_key,
    }
