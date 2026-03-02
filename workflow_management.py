"""
Workflow CRUD for Studio persistence + execute-workflow endpoint.
Do NOT select updated_at in responses if the column is removed from schema later.
"""
import asyncio
import json
import math
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime, timezone, timedelta
from supabase_client import supabase

from workflow_runtime import execute_workflow
from routing.resolver import get_promoted_deployment_by_version, get_latest_promoted_deployment

router = APIRouter()


def _nr_has_error(nr: Any) -> bool:
    """Return True if a node_result entry indicates an error or output quality warning.

    Centralised check used by all observability / experiment / rollback queries.
    Covers:
      - explicit errors:  status == "error"  or  error == True
      - output quality:   status == "warning"  (empty output, refusal, provider_error)
      - output_warning:   output_warning field is non-empty
    """
    if not isinstance(nr, dict):
        return False
    _st = nr.get("status")
    if _st == "error" or _st == "warning":
        return True
    if nr.get("error"):
        return True
    if nr.get("output_warning"):
        return True
    return False


def _sse_line(data: Any) -> str:
    """One SSE data line (JSON) plus newline. Send two newlines after for event boundary."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"data: {payload}\n\n"


def _summarize_node(node: dict) -> str:
    """Short summary for node_added display."""
    data = node.get("data") or {}
    node_type = (node.get("type") or "").lower()
    if node_type == "ai-step":
        model = data.get("modelName") or data.get("model") or "—"
        provider = data.get("provider") or ""
        return f"{provider} / {model}".strip(" /") or "AI step"
    if node_type == "prompt":
        return (data.get("label") or data.get("preview") or "prompt")[:60]
    if node_type == "condition":
        return (data.get("condition") or "condition")[:60]
    return data.get("label") or node.get("id") or "node"


def _compute_graph_diff(graph_a: dict, graph_b: dict) -> dict:
    """Compare two workflow graphs; return changes and summary."""
    nodes_a = {n["id"]: n for n in graph_a.get("nodes", []) if n.get("id")}
    nodes_b = {n["id"]: n for n in graph_b.get("nodes", []) if n.get("id")}
    edges_a = set(f"{e.get('source')}->{e.get('target')}" for e in graph_a.get("edges", []) if e.get("source") and e.get("target"))
    edges_b = set(f"{e.get('source')}->{e.get('target')}" for e in graph_b.get("edges", []) if e.get("source") and e.get("target"))

    changes: List[dict] = []

    for nid in nodes_b:
        if nid not in nodes_a:
            nb = nodes_b[nid]
            changes.append({
                "type": "node_added",
                "node_id": nid,
                "node_label": (nb.get("data") or {}).get("label") or nid,
                "details": _summarize_node(nb),
            })

    for nid in nodes_a:
        if nid not in nodes_b:
            na = nodes_a[nid]
            changes.append({
                "type": "node_removed",
                "node_id": nid,
                "node_label": (na.get("data") or {}).get("label") or nid,
            })

    for nid in nodes_a:
        if nid not in nodes_b:
            continue
        na_data = nodes_a[nid].get("data") or {}
        nb_data = nodes_b[nid].get("data") or {}
        node_label = nb_data.get("label") or nid

        model_a = f"{na_data.get('provider', '')} / {na_data.get('modelName', '') or na_data.get('model', '')}".strip(" /")
        model_b = f"{nb_data.get('provider', '')} / {nb_data.get('modelName', '') or nb_data.get('model', '')}".strip(" /")
        if model_a != model_b and (na_data.get("modelName") or na_data.get("model") or nb_data.get("modelName") or nb_data.get("model")):
            changes.append({
                "type": "model_changed",
                "node_id": nid,
                "node_label": node_label,
                "before": model_a or "—",
                "after": model_b or "—",
            })

        prompt_a = (na_data.get("systemInstructions") or na_data.get("system_prefix") or "") + "\n" + (na_data.get("taskDescription") or na_data.get("task") or "")
        prompt_b = (nb_data.get("systemInstructions") or nb_data.get("system_prefix") or "") + "\n" + (nb_data.get("taskDescription") or nb_data.get("task") or "")
        if prompt_a.strip() != prompt_b.strip():
            changes.append({
                "type": "prompt_changed",
                "node_id": nid,
                "node_label": node_label,
                "before_preview": (prompt_a[:120] + ("..." if len(prompt_a) > 120 else "")).replace("\n", " "),
                "after_preview": (prompt_b[:120] + ("..." if len(prompt_b) > 120 else "")).replace("\n", " "),
                "char_diff": abs(len(prompt_b) - len(prompt_a)),
            })

        temp_a = na_data.get("temperature")
        temp_b = nb_data.get("temperature")
        if temp_a != temp_b and (temp_a is not None or temp_b is not None):
            changes.append({
                "type": "temperature_changed",
                "node_id": nid,
                "node_label": node_label,
                "before": temp_a,
                "after": temp_b,
            })

        mt_a = na_data.get("maxTokens") or na_data.get("max_tokens")
        mt_b = nb_data.get("maxTokens") or nb_data.get("max_tokens")
        if mt_a != mt_b and (mt_a is not None or mt_b is not None):
            changes.append({
                "type": "max_tokens_changed",
                "node_id": nid,
                "node_label": node_label,
                "before": mt_a,
                "after": mt_b,
            })

        rule_a = na_data.get("condition") or ""
        rule_b = nb_data.get("condition") or ""
        if rule_a != rule_b and (rule_a or rule_b):
            changes.append({
                "type": "condition_changed",
                "node_id": nid,
                "node_label": node_label,
                "before": rule_a,
                "after": rule_b,
            })

    added_edges = edges_b - edges_a
    removed_edges = edges_a - edges_b
    if added_edges:
        changes.append({"type": "connections_added", "count": len(added_edges), "details": list(added_edges)[:10]})
    if removed_edges:
        changes.append({"type": "connections_removed", "count": len(removed_edges), "details": list(removed_edges)[:10]})

    type_counts: dict = {}
    for c in changes:
        t = c["type"].replace("_", " ")
        type_counts[t] = type_counts.get(t, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in sorted(type_counts.items(), key=lambda x: -x[1]))

    return {"changes": changes, "summary": summary or "no changes detected"}


class WorkflowCreate(BaseModel):
    org_id: str
    project_id: Optional[str] = None  # Required for project-first; if missing, use or create default project
    name: str = "Untitled workflow"
    slug: Optional[str] = None
    graph_json: Optional[dict] = None
    variables: Optional[List[dict]] = None

    class Config:
        extra = "ignore"


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    slug: Optional[str] = None
    graph_json: Optional[dict] = None
    variables: Optional[List[dict]] = None

    class Config:
        extra = "ignore"


class WorkflowResponse(BaseModel):
    id: str
    org_id: str
    project_id: Optional[str] = None
    name: str
    slug: Optional[str] = None
    graph_json: dict
    variables: Optional[List[dict]] = None
    created_at: str

    class Config:
        extra = "ignore"


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = "".join(c if c.isalnum() or c == "-" else "-" for c in s)
    return "-".join(filter(None, s.split("-"))) or "workflow"


def _row_to_response(row: dict) -> dict:
    return {
        "id": row["id"],
        "org_id": row["org_id"],
        "project_id": row.get("project_id"),
        "name": row.get("name") or "Untitled workflow",
        "slug": row.get("slug"),
        "graph_json": row.get("graph_json") or {"nodes": [], "edges": []},
        "variables": row.get("variables") if row.get("variables") is not None else [],
        "created_at": row["created_at"],
    }


_WF_COLS = "id, org_id, project_id, name, slug, graph_json, variables, created_at"


@router.get("/workflows/{org_id}", response_model=List[WorkflowResponse])
async def get_workflows(org_id: str, project_id: Optional[str] = None):
    """List workflows for an organization. Optional project_id to filter by project."""
    try:
        q = (
            supabase.table("workflows")
            .select(_WF_COLS)
            .eq("org_id", org_id)
            .order("created_at", desc=True)
        )
        if project_id:
            q = q.eq("project_id", project_id)
        result = q.execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching workflows: {str(e)}")
    if not result.data:
        return []
    return [_row_to_response(row) for row in result.data]


@router.get("/projects/{org_id}/{project_id}/workflows", response_model=List[WorkflowResponse])
async def get_project_workflows(org_id: str, project_id: str):
    """List workflows for a project. Verifies project belongs to org."""
    try:
        proj = supabase.table("projects").select("id").eq("id", project_id).eq("org_id", org_id).single().execute()
        if not proj.data:
            raise HTTPException(status_code=404, detail="Project not found")
        result = (
            supabase.table("workflows")
            .select(_WF_COLS)
            .eq("project_id", project_id)
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [_row_to_response(row) for row in (result.data or [])]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching project workflows: {str(e)}")


@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(payload: WorkflowCreate):
    """Create a new workflow. Uses project_id if provided; otherwise uses or creates default project for org."""
    try:
        project_id = payload.project_id
        if not project_id:
            projs = supabase.table("projects").select("id").eq("org_id", payload.org_id).limit(1).execute()
            if projs.data and len(projs.data) > 0:
                project_id = projs.data[0]["id"]
            else:
                new_proj = supabase.table("projects").insert({"org_id": payload.org_id, "name": "Default"}).execute()
                if new_proj.data:
                    project_id = new_proj.data[0]["id"]
            if not project_id:
                raise HTTPException(status_code=400, detail="project_id required. Create a project first.")
        else:
            proj = supabase.table("projects").select("id").eq("id", project_id).eq("org_id", payload.org_id).single().execute()
            if not proj.data:
                raise HTTPException(status_code=404, detail="Project not found")
        slug = (payload.slug or "").strip() or _slugify(payload.name or "Untitled workflow")
        data = {
            "org_id": payload.org_id,
            "project_id": project_id,
            "name": payload.name or "Untitled workflow",
            "slug": slug,
            "graph_json": payload.graph_json if payload.graph_json is not None else {"nodes": [], "edges": []},
            "variables": payload.variables if payload.variables is not None else [],
        }
        result = supabase.table("workflows").insert(data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create workflow")
        return _row_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating workflow: {str(e)}")


@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: str, payload: WorkflowUpdate):
    """Update workflow name, slug, graph_json, variables."""
    try:
        update_data = {}
        if payload.name is not None:
            update_data["name"] = payload.name
        if payload.slug is not None:
            update_data["slug"] = payload.slug.strip() or None
        if payload.graph_json is not None:
            update_data["graph_json"] = payload.graph_json
        if payload.variables is not None:
            update_data["variables"] = payload.variables
        if not update_data:
            result = supabase.table("workflows").select(_WF_COLS).eq("id", workflow_id).single().execute()
            if not result.data:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return _row_to_response(result.data[0])
        result = supabase.table("workflows").update(update_data).eq("id", workflow_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return _row_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating workflow: {str(e)}")


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow and all associated data: runs, deployments, golden_inputs, eval_suites, then the workflow."""
    try:
        # Verify workflow exists and get org_id for RLS (optional; delete may still work with service role)
        wf = supabase.table("workflows").select("id, org_id").eq("id", workflow_id).execute()
        if not wf.data or len(wf.data) == 0:
            raise HTTPException(status_code=404, detail="Workflow not found")
        # Delete in order to respect FKs: runs -> deployments (eval_runs CASCADE from deployment) -> golden_inputs, eval_suites -> workflow
        supabase.table("workflow_runs").delete().eq("workflow_id", workflow_id).execute()
        supabase.table("workflow_deployments").delete().eq("workflow_id", workflow_id).execute()
        supabase.table("golden_inputs").delete().eq("workflow_id", workflow_id).execute()
        supabase.table("eval_suites").delete().eq("workflow_id", workflow_id).execute()
        supabase.table("workflows").delete().eq("id", workflow_id).execute()
        return {"message": "Workflow deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting workflow: {str(e)}")


class ExecuteWorkflowPayload(BaseModel):
    workflow_id: str
    input: Optional[str] = None
    variables: Optional[dict] = None
    user_id: str = ""
    org_id: Optional[str] = None
    graph_json: Optional[dict] = None
    version: Optional[str] = None

    class Config:
        extra = "ignore"


def _parse_version(version: Optional[str]) -> Optional[int]:
    if not version:
        return None
    s = (version or "").strip()
    if s.startswith("v") or s.startswith("V"):
        s = s[1:]
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Workflow deployments (versioned deploy history)
# ---------------------------------------------------------------------------

class DeploymentCreate(BaseModel):
    workflow_id: str
    org_id: str
    endpoint_slug: str
    graph_json: dict
    rolled_back_from_version: Optional[int] = None

    class Config:
        extra = "ignore"


_DEP_COLS = "id, workflow_id, project_id, org_id, version, endpoint_slug, graph_json, created_at, rolled_back_from_version, status, promoted_at, eval_run_id, override_by, override_reason"


def _deployment_row_to_response(row: dict) -> dict:
    out = {
        "id": row["id"],
        "workflow_id": row["workflow_id"],
        "org_id": row["org_id"],
        "version": row["version"],
        "endpoint_slug": row.get("endpoint_slug") or "",
        "graph_json": row.get("graph_json") or {"nodes": [], "edges": []},
        "created_at": row["created_at"],
    }
    if row.get("project_id") is not None:
        out["project_id"] = row["project_id"]
    if row.get("rolled_back_from_version") is not None:
        out["rolled_back_from_version"] = row["rolled_back_from_version"]
    if row.get("status") is not None:
        out["status"] = row["status"]
    if row.get("promoted_at") is not None:
        out["promoted_at"] = row["promoted_at"]
    if row.get("eval_run_id") is not None:
        out["eval_run_id"] = str(row["eval_run_id"])
    if row.get("override_by") is not None:
        out["override_by"] = str(row["override_by"])
    if row.get("override_reason") is not None:
        out["override_reason"] = row["override_reason"]
    return out


@router.get("/workflow-deployments/latest")
async def get_latest_deployment(workflow_id: str, org_id: str, promoted_only: bool = True):
    """Get the latest deployment for a workflow. Default: latest promoted only (what's live). Set promoted_only=false for latest by version (e.g. candidate)."""
    try:
        q = (
            supabase.table("workflow_deployments")
            .select(_DEP_COLS)
            .eq("workflow_id", workflow_id)
            .eq("org_id", org_id)
        )
        if promoted_only:
            q = q.eq("status", "promoted")
        result = q.order("version", desc=True).limit(1).execute()
        if not result.data or len(result.data) == 0:
            return None
        return _deployment_row_to_response(result.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching latest deployment: {str(e)}")


@router.get("/workflow-deployments")
async def list_workflow_deployments(workflow_id: str, org_id: str, limit: int = 50, promoted_only: bool = False):
    """List deployments for a workflow (history, newest first). Set promoted_only=true to only return promoted (e.g. for canary/experiments)."""
    try:
        q = (
            supabase.table("workflow_deployments")
            .select(_DEP_COLS)
            .eq("workflow_id", workflow_id)
            .eq("org_id", org_id)
        )
        if promoted_only:
            q = q.eq("status", "promoted")
        result = q.order("version", desc=True).limit(max(1, min(limit, 100))).execute()
        return [_deployment_row_to_response(row) for row in (result.data or [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing deployments: {str(e)}")


@router.get("/workflow-deployments/diff")
async def diff_deployments(org_id: str, endpoint_slug: str, version_a: int, version_b: int):
    """Compare two promoted deployment versions (e.g. for experiment v5 vs v6). Returns changes and summary."""
    try:
        dep_a = await get_promoted_deployment_by_version(org_id, endpoint_slug.strip(), version_a)
        dep_b = await get_promoted_deployment_by_version(org_id, endpoint_slug.strip(), version_b)
        if not dep_a or not dep_b:
            raise HTTPException(status_code=404, detail="Deployment version not found")
        graph_a = dep_a.get("graph_json") or {"nodes": [], "edges": []}
        graph_b = dep_b.get("graph_json") or {"nodes": [], "edges": []}
        return _compute_graph_diff(graph_a, graph_b)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint (deployed workflow) limits per plan tier. Count = distinct workflow_id with ≥1 deployment.
_ENDPOINT_LIMITS = {"free": 1, "startup": 10, "enterprise": float("inf")}
# Map DB plan values (organizations.plan) to limit tier; DB may not have plan_tier column.
_PLAN_TO_TIER = {"starter": "startup", "team": "startup", "pro": "startup"}


def _endpoint_limit_for_plan(plan: Optional[str]) -> float:
    if not plan:
        return _ENDPOINT_LIMITS["free"]
    tier = (plan or "").strip().lower()
    tier = _PLAN_TO_TIER.get(tier, tier)
    return _ENDPOINT_LIMITS.get(tier, _ENDPOINT_LIMITS["free"])


@router.post("/workflow-deployments")
async def create_workflow_deployment(payload: DeploymentCreate):
    """Create a new deployment. Version increments automatically per workflow.
    - If this workflow already has deployments: reuse existing endpoint_slug; reject different slug with 400.
    - First deploy: use payload endpoint_slug; 409 if slug already used by another workflow in this org.
    - Plan limit: count distinct workflow_id with deployments; 402 if at limit when creating first deploy for a new workflow."""
    try:
        client_slug = (payload.endpoint_slug or "").strip() or "workflow"

        # Existing deployments for this workflow in this org (for slug reuse and version)
        existing_for_workflow = (
            supabase.table("workflow_deployments")
            .select("version, endpoint_slug")
            .eq("workflow_id", payload.workflow_id)
            .eq("org_id", payload.org_id)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        has_existing = existing_for_workflow.data and len(existing_for_workflow.data) > 0
        existing_slug = (existing_for_workflow.data[0].get("endpoint_slug") or "").strip() if has_existing else None

        if has_existing and existing_slug:
            # Redeploy: slug is permanent — reuse existing, reject change
            if client_slug != existing_slug:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot change endpoint_slug after first deploy. Existing slug: {existing_slug}",
                )
            endpoint_slug = existing_slug
        else:
            # First deploy for this workflow: use client slug; enforce org uniqueness
            endpoint_slug = client_slug
            slug_collision = (
                supabase.table("workflow_deployments")
                .select("id")
                .eq("org_id", payload.org_id)
                .eq("endpoint_slug", endpoint_slug)
                .limit(1)
                .execute()
            )
            if slug_collision.data and len(slug_collision.data) > 0:
                raise HTTPException(
                    status_code=409,
                    detail="An endpoint with this slug already exists in this organization.",
                )

            # Plan limit: count distinct workflows with ≥1 deployment (this workflow not yet counted)
            all_deployments = (
                supabase.table("workflow_deployments")
                .select("workflow_id")
                .eq("org_id", payload.org_id)
                .execute()
            )
            workflow_ids = {r.get("workflow_id") for r in (all_deployments.data or []) if r.get("workflow_id")}
            deployed_workflow_count = len(workflow_ids)
            org_row = supabase.table("organizations").select("plan").eq("id", payload.org_id).single().execute()
            raw_plan = (org_row.data or {}).get("plan") or "free"
            limit = _endpoint_limit_for_plan(raw_plan)
            if deployed_workflow_count >= limit:
                raise HTTPException(
                    status_code=402,
                    detail=f"Your plan allows {int(limit)} deployed endpoint(s). Upgrade to deploy more workflows.",
                )

        next_version = 1
        if has_existing and existing_for_workflow.data:
            next_version = int(existing_for_workflow.data[0].get("version") or 0) + 1

        wf_row = supabase.table("workflows").select("project_id, org_id").eq("id", payload.workflow_id).single().execute()
        project_id = wf_row.data.get("project_id") if wf_row.data else None

        data = {
            "workflow_id": payload.workflow_id,
            "org_id": payload.org_id,
            "version": next_version,
            "endpoint_slug": endpoint_slug,
            "graph_json": payload.graph_json if payload.graph_json is not None else {"nodes": [], "edges": []},
            "status": "candidate",
        }
        if project_id is not None:
            data["project_id"] = project_id
        if payload.rolled_back_from_version is not None:
            data["rolled_back_from_version"] = payload.rolled_back_from_version
        result = supabase.table("workflow_deployments").insert(data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create deployment")
        deployment = result.data[0]
        deployment_id = str(deployment["id"])
        resp = _deployment_row_to_response(deployment)
        try:
            suite_row = (
                supabase.table("eval_suites")
                .select("id")
                .eq("org_id", payload.org_id)
                .eq("workflow_id", payload.workflow_id)
                .limit(1)
                .execute()
            )
            eval_suite_id = suite_row.data[0]["id"] if suite_row.data and len(suite_row.data) > 0 else None
            eval_insert = {
                "org_id": payload.org_id,
                "deployment_id": deployment_id,
                "eval_suite_id": eval_suite_id,
                "status": "running",
            }
            eval_result = supabase.table("eval_runs").insert(eval_insert).execute()
            if eval_result.data and len(eval_result.data) > 0:
                eval_run_id = str(eval_result.data[0]["id"])
                resp["eval_run_id"] = eval_run_id
                asyncio.get_event_loop().run_in_executor(None, _run_eval_sync, deployment_id, eval_run_id)
        except Exception:
            pass
        return resp
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e).lower()
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if code == "23505" or code == 409 or "23505" in err_str or "unique constraint" in err_str or "duplicate key" in err_str:
            raise HTTPException(
                status_code=409,
                detail="An endpoint with this slug already exists in this organization.",
            )
        raise HTTPException(status_code=500, detail=f"Error creating deployment: {str(e)}")


class PromoteOverridePayload(BaseModel):
    override_reason: Optional[str] = None

    class Config:
        extra = "ignore"


@router.post("/workflow-deployments/{deployment_id}/promote")
async def promote_deployment_override(deployment_id: str, payload: PromoteOverridePayload):
    """Admin override: promote a deployment despite failed eval. Sets status=promoted, promoted_at=now(), override_reason."""
    try:
        result = (
            supabase.table("workflow_deployments")
            .update({
                "status": "promoted",
                "promoted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "override_by": None,
                "override_reason": (payload.override_reason or "").strip() or None,
            })
            .eq("id", deployment_id)
            .execute()
        )
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="Deployment not found")
        row = result.data[0]
        org_id = str(row.get("org_id", ""))
        endpoint_slug = (row.get("endpoint_slug") or "").strip()
        if org_id and endpoint_slug:
            await asyncio.to_thread(_end_experiments_on_endpoint_sync, org_id, endpoint_slug, "new_deployment")
        return _deployment_row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow-runs")
async def list_workflow_runs(org_id: str, limit: int = 50):
    """List recent workflow runs for an organization (for Logs / observability)."""
    try:
        result = (
            supabase.table("workflow_runs")
            .select("id, workflow_id, org_id, user_id, input_text, final_output, node_results, total_cost, total_latency_ms, endpoint_slug, version, execution_mode, created_at")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .limit(max(1, min(limit, 200)))
            .execute()
        )
        return result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching workflow runs: {str(e)}")


@router.post("/execute-workflow")
async def api_execute_workflow(payload: ExecuteWorkflowPayload):
    """
    Execute a workflow. Draft vs production:
    - Draft: graph_json + org_id in body (Studio simulation). Logs execution_mode='draft'. Deployment table not used.
    - Production: no graph_json; workflow_id + optional version. Load graph from workflow_deployments only. Logs execution_mode='production'.
    """
    try:
        org_id = payload.org_id
        graph = payload.graph_json
        endpoint_slug = None
        version_num = None
        execution_mode = "draft"

        if graph is not None:
            # Draft: use current canvas
            if not org_id:
                raise HTTPException(status_code=400, detail="org_id required when graph_json is provided")
            if not graph.get("nodes"):
                raise HTTPException(status_code=400, detail="graph_json must contain nodes")
        else:
            # Production: load from workflow_deployments (version or latest)
            workflow_id = payload.workflow_id
            org_id_from_workflow = None
            if not org_id:
                wf_row = (
                    supabase.table("workflows")
                    .select("org_id")
                    .eq("id", workflow_id)
                    .single()
                    .execute()
                )
                if wf_row.data:
                    org_id_from_workflow = wf_row.data.get("org_id")
            lookup_org = org_id or org_id_from_workflow
            if not lookup_org:
                raise HTTPException(status_code=400, detail="org_id required when no graph_json (Production)")

            version_num = _parse_version(payload.version)
            if version_num is not None:
                dep_row = (
                    supabase.table("workflow_deployments")
                    .select(_DEP_COLS)
                    .eq("workflow_id", workflow_id)
                    .eq("org_id", lookup_org)
                    .eq("version", version_num)
                    .limit(1)
                    .execute()
                )
            else:
                dep_row = (
                    supabase.table("workflow_deployments")
                    .select(_DEP_COLS)
                    .eq("workflow_id", workflow_id)
                    .eq("org_id", lookup_org)
                    .order("version", desc=True)
                    .limit(1)
                    .execute()
                )
            if not dep_row.data or len(dep_row.data) == 0:
                raise HTTPException(status_code=400, detail="Workflow not published.")
            deployment = dep_row.data[0]
            org_id = deployment.get("org_id")
            graph = deployment.get("graph_json") or {"nodes": [], "edges": []}
            endpoint_slug = (deployment.get("endpoint_slug") or "").strip() or None
            version_num = deployment.get("version")
            execution_mode = "production"

        if not org_id:
            raise HTTPException(status_code=400, detail="Workflow has no org_id")

        input_text = ""
        variables = payload.variables
        if variables is not None and isinstance(variables, dict):
            wf = supabase.table("workflows").select("variables").eq("id", payload.workflow_id).single().execute()
            if wf.data and wf.data.get("variables") and isinstance(wf.data["variables"], list):
                from workflow_runtime import validate_workflow_variables
                variables = validate_workflow_variables(wf.data["variables"], variables)
            input_text = ""
        else:
            input_text = payload.input or ""

        result = execute_workflow(
            graph,
            input_text,
            str(org_id),
            payload.user_id or "",
            workflow_id=payload.workflow_id,
            endpoint_slug=endpoint_slug,
            version=version_num,
            execution_mode=execution_mode,
            variables=variables,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


async def _stream_execute_workflow_draft(
    graph: dict,
    input_text: str,
    org_id: str,
    workflow_id: str,
    user_id: str,
    variables: dict | None,
):
    """Run execute_workflow in thread for draft, then yield SSE events for progress."""
    loop = asyncio.get_event_loop()

    def run() -> dict:
        return execute_workflow(
            graph,
            input_text,
            str(org_id),
            user_id or "",
            workflow_id=workflow_id,
            endpoint_slug=None,
            version=None,
            execution_mode="draft",
            variables=variables,
        )

    result = await loop.run_in_executor(None, run)
    node_results = result.get("node_results") or []
    nodes_by_id = {n["id"]: n for n in (graph.get("nodes") or [])}

    yield _sse_line({"type": "workflow_info", "total_steps": len(node_results)})

    for nr in node_results:
        node_id = nr.get("node_id") or ""
        ntype = nr.get("type") or "node"
        node = nodes_by_id.get(node_id) or {}
        step_name = (node.get("data") or {}).get("label") or node_id

        yield _sse_line({"type": "step_start", "step": node_id, "step_name": step_name})
        payload = {
            "type": "step_end",
            "step": node_id,
            "step_name": step_name,
            "latency_ms": nr.get("latency_ms", 0),
            "cost_usd": nr.get("cost", 0),
        }
        if nr.get("content_type") is not None:
            payload["content_type"] = nr["content_type"]
        else:
            payload["content_type"] = "text"
        yield _sse_line(payload)

    total_latency = result.get("total_latency") or 0
    yield _sse_line({
        "type": "done",
        "final_output": result.get("final_output"),
        "content_type": result.get("content_type") or "text",
        "total_latency_ms": total_latency,
        "total_cost": result.get("total_cost"),
        "run_id": result.get("run_id"),
    })
    yield "data: [DONE]\n\n"


@router.post("/execute-workflow/stream")
async def api_execute_workflow_stream(payload: ExecuteWorkflowPayload):
    """
    Execute a workflow with SSE streaming (draft only). Emits workflow_info, step_start, step_end, done, [DONE].
    Requires graph_json and org_id. Use POST /execute-workflow for non-streaming or production.
    """
    graph = payload.graph_json
    if graph is None or not graph.get("nodes"):
        raise HTTPException(
            status_code=400,
            detail="Streaming requires graph_json with nodes (draft mode only).",
        )
    org_id = payload.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id required when graph_json is provided")

    input_text = ""
    variables = payload.variables
    if variables is not None and isinstance(variables, dict) and payload.workflow_id:
        wf = supabase.table("workflows").select("variables").eq("id", payload.workflow_id).single().execute()
        if wf.data and wf.data.get("variables") and isinstance(wf.data["variables"], list):
            from workflow_runtime import validate_workflow_variables
            variables = validate_workflow_variables(wf.data["variables"], variables)
    else:
        input_text = payload.input or ""

    async def generate():
        try:
            async for chunk in _stream_execute_workflow_draft(
                graph,
                input_text,
                str(org_id),
                payload.workflow_id,
                payload.user_id or "",
                variables,
            ):
                yield chunk
        except HTTPException:
            raise
        except Exception as e:
            yield _sse_line({"type": "error", "message": str(e)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class TestProductionCallPayload(BaseModel):
    """Body for test-production-call: run deployed endpoint as the org (no server key in request)."""
    org_id: str
    endpoint_slug: str
    input_text: Optional[str] = ""
    variables: Optional[dict] = None

    class Config:
        extra = "ignore"


@router.post("/test-production-call")
async def api_test_production_call(payload: TestProductionCallPayload):
    """
    Run the deployed production endpoint for the given org_id and endpoint_slug.
    Uses the same execution path as the public API but authorizes via request context (JWT).
    No rate limit. Returns the same shape as execute_workflow (final_output, total_latency_ms, etc.).
    """
    try:
        org_id = (payload.org_id or "").strip()
        endpoint_slug = (payload.endpoint_slug or "").strip()
        if not org_id or not endpoint_slug:
            raise HTTPException(status_code=400, detail="org_id and endpoint_slug required.")

        dep = (
            supabase.table("workflow_deployments")
            .select(_DEP_COLS)
            .eq("org_id", org_id)
            .eq("endpoint_slug", endpoint_slug)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        if not dep.data or len(dep.data) == 0:
            raise HTTPException(status_code=404, detail="Workflow not published.")

        deployment = dep.data[0]
        graph_json = deployment.get("graph_json") or {"nodes": [], "edges": []}
        workflow_id = deployment.get("workflow_id")
        dep_version = deployment.get("version")
        dep_slug = (deployment.get("endpoint_slug") or "").strip() or endpoint_slug

        variables = payload.variables if isinstance(payload.variables, dict) else None
        if variables is not None and workflow_id:
            from workflow_runtime import validate_workflow_variables
            wf = supabase.table("workflows").select("variables").eq("id", workflow_id).single().execute()
            if wf.data and wf.data.get("variables") and isinstance(wf.data["variables"], list):
                variables = validate_workflow_variables(wf.data["variables"], variables)

        input_text = (payload.input_text or "").strip() if payload.input_text is not None else ""

        result = execute_workflow(
            graph_json,
            input_text,
            org_id,
            user_id="",
            workflow_id=workflow_id,
            endpoint_slug=dep_slug,
            version=dep_version,
            execution_mode="production",
            variables=variables,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test production call failed: {str(e)}")


# ---------------------------------------------------------------------------
# Observability: workflow runs by endpoint_slug, version, execution_mode
# ---------------------------------------------------------------------------

_run_cols = "id, workflow_id, org_id, endpoint_slug, version, execution_mode, total_cost, total_latency_ms, node_results, created_at"


def _current_minute_count_by_slug(org_id: str) -> list[dict]:
    """Production runs in the last 60 seconds, count by endpoint_slug."""
    from datetime import datetime, timezone, timedelta
    try:
        since = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
        result = (
            supabase.table("workflow_runs")
            .select("endpoint_slug")
            .eq("org_id", org_id)
            .eq("execution_mode", "production")
            .gte("created_at", since)
            .execute()
        )
        rows = result.data or []
    except Exception:
        return []
    by_slug: dict[str, int] = {}
    for r in rows:
        slug = (r.get("endpoint_slug") or "").strip() or "_draft"
        by_slug[slug] = by_slug.get(slug, 0) + 1
    return [{"endpoint_slug": k, "current_minute_count": v} for k, v in by_slug.items()]


@router.get("/observability/summary")
async def get_workflow_observability_summary(org_id: str):
    """
    Summary: total production/draft requests, cost and request count by endpoint_slug,
    avg latency by endpoint_slug, version distribution, current_minute_count per slug.
    Scoped to the last 7 days so the numbers match the "Production health (7d)" UI label.
    """
    try:
        _since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat().replace("+00:00", "Z")
        result = (
            supabase.table("workflow_runs")
            .select(_run_cols)
            .eq("org_id", org_id)
            .gte("created_at", _since)
            .order("created_at", desc=True)
            .limit(5000)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching workflow runs: {str(e)}")

    current_minute_by_slug = _current_minute_count_by_slug(org_id)

    # Diagnostic: log error-bearing runs so we can verify persistence
    _error_runs = [
        r for r in rows
        if isinstance(r.get("node_results"), list) and any(
            _nr_has_error(nr) for nr in r["node_results"]
        )
    ]
    if _error_runs:
        logger.info(
            "observability/summary: org=%s total_rows=%d error_runs=%d  error_run_ids=%s",
            org_id, len(rows), len(_error_runs),
            [r.get("id") for r in _error_runs[:10]],
        )

    total_production = 0
    total_draft = 0
    by_slug: dict[str, dict] = {}
    version_dist_map: dict[tuple[str, int | None], int] = {}

    total_errors = 0          # errors across all runs (draft + production)
    total_production_errors = 0  # errors in production runs only
    for r in rows:
        _is_prod = (r.get("execution_mode") or "").strip() == "production"
        if _is_prod:
            total_production += 1
        else:
            total_draft += 1
        slug = (r.get("endpoint_slug") or "").strip() or "_draft"
        if slug not in by_slug:
            by_slug[slug] = {"request_count": 0, "total_cost": 0.0, "latency_sum": 0, "latency_count": 0, "error_count": 0}
        by_slug[slug]["request_count"] += 1
        by_slug[slug]["total_cost"] += float(r.get("total_cost") or 0)
        # Check if any node_result has an error or output quality warning
        _nr_list = r.get("node_results") or []
        _run_has_error = isinstance(_nr_list, list) and any(_nr_has_error(nr) for nr in _nr_list)
        if _run_has_error:
            by_slug[slug]["error_count"] += 1
            total_errors += 1
            if _is_prod:
                total_production_errors += 1
        lat = r.get("total_latency_ms")
        if lat is not None:
            by_slug[slug]["latency_sum"] += int(lat)
            by_slug[slug]["latency_count"] += 1
        if _is_prod:
            ver = r.get("version")
            key = (slug, ver)
            version_dist_map[key] = version_dist_map.get(key, 0) + 1

    version_dist = [{"endpoint_slug": k[0], "version": k[1], "request_count": c} for k, c in version_dist_map.items()]

    # Latest promoted version per endpoint (so UI can show "current" live version even if no runs yet on v6)
    latest_promoted = (
        supabase.table("workflow_deployments")
        .select("endpoint_slug, version")
        .eq("org_id", org_id)
        .eq("status", "promoted")
        .execute()
    )
    latest_by_slug: dict = {}
    for row in (latest_promoted.data or []):
        slug = (row.get("endpoint_slug") or "").strip()
        if not slug:
            continue
        ver = int(row.get("version") or 0)
        if slug not in latest_by_slug or ver > latest_by_slug[slug]:
            latest_by_slug[slug] = ver
    latest_promoted_per_endpoint = [{"endpoint_slug": s, "version": v} for s, v in latest_by_slug.items()]

    cost_by_endpoint = [{"endpoint_slug": k, "total_cost": v["total_cost"], "request_count": v["request_count"]} for k, v in by_slug.items()]
    request_count_by_endpoint = [{"endpoint_slug": k, "request_count": v["request_count"]} for k, v in by_slug.items()]
    avg_latency_by_endpoint = [
        {"endpoint_slug": k, "avg_latency_ms": round(v["latency_sum"] / v["latency_count"], 0) if v["latency_count"] else None}
        for k, v in by_slug.items()
    ]
    error_count_by_endpoint = [{"endpoint_slug": k, "error_count": v["error_count"]} for k, v in by_slug.items()]
    return {
        "total_production_requests": total_production,
        "total_draft_requests": total_draft,
        "total_error_count": total_errors,
        "production_error_count": total_production_errors,
        # error_rate = production errors / production requests (matches "Production health 7d" UI)
        "error_rate": round(total_production_errors / total_production * 100, 2) if total_production > 0 else 0,
        "cost_by_endpoint_slug": cost_by_endpoint,
        "request_count_by_endpoint_slug": request_count_by_endpoint,
        "avg_latency_by_endpoint_slug": avg_latency_by_endpoint,
        "error_count_by_endpoint_slug": error_count_by_endpoint,
        "version_distribution": version_dist,
        "latest_promoted_per_endpoint": latest_promoted_per_endpoint,
        "current_minute_count_by_slug": current_minute_by_slug,
    }


@router.get("/observability/by-endpoint")
async def get_workflow_observability_by_endpoint(org_id: str, slug: str):
    """
    Per-endpoint: version breakdown, last 50 runs, cost and latency per version. Explicit columns only.
    """
    try:
        result = (
            supabase.table("workflow_runs")
            .select(_run_cols)
            .eq("org_id", org_id)
            .eq("endpoint_slug", slug)
            .eq("execution_mode", "production")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching workflow runs: {str(e)}")

    by_version: dict[int | None, dict] = {}
    for r in rows:
        ver = r.get("version")
        if ver not in by_version:
            by_version[ver] = {"request_count": 0, "total_cost": 0.0, "latency_sum": 0, "latency_count": 0}
        by_version[ver]["request_count"] += 1
        by_version[ver]["total_cost"] += float(r.get("total_cost") or 0)
        lat = r.get("total_latency_ms")
        if lat is not None:
            by_version[ver]["latency_sum"] += int(lat)
            by_version[ver]["latency_count"] += 1

    version_breakdown = [
        {
            "version": v,
            "request_count": by_version[v]["request_count"],
            "total_cost": round(by_version[v]["total_cost"], 6),
            "avg_latency_ms": round(by_version[v]["latency_sum"] / by_version[v]["latency_count"], 0) if by_version[v]["latency_count"] else None,
        }
        for v in by_version
    ]
    last_50_runs = [
        {
            "id": r.get("id"),
            "workflow_id": r.get("workflow_id"),
            "version": r.get("version"),
            "total_cost": r.get("total_cost"),
            "total_latency_ms": r.get("total_latency_ms"),
            "created_at": r.get("created_at"),
        }
        for r in rows
    ]
    return {
        "endpoint_slug": slug,
        "version_breakdown": version_breakdown,
        "last_50_runs": last_50_runs,
        "cost_per_version": [{"version": v, "total_cost": by_version[v]["total_cost"]} for v in by_version],
        "latency_per_version": [
            {"version": v, "avg_latency_ms": round(by_version[v]["latency_sum"] / by_version[v]["latency_count"], 0) if by_version[v]["latency_count"] else None}
            for v in by_version
        ],
    }


# ---------------------------------------------------------------------------
# Golden inputs (test cases) CRUD — Phase 2 eval-gated deployments
# ---------------------------------------------------------------------------

_GOLDEN_INPUT_COLS = "id, org_id, workflow_id, name, input_text, variables, expected_output, source, source_run_id, created_at, updated_at"


def _golden_row_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "workflow_id": str(row["workflow_id"]),
        "name": row.get("name"),
        "input_text": row.get("input_text"),
        "variables": row.get("variables"),
        "expected_output": row.get("expected_output"),
        "source": row.get("source") or "manual",
        "source_run_id": str(row["source_run_id"]) if row.get("source_run_id") else None,
        "created_at": row["created_at"],
        "updated_at": row.get("updated_at"),
    }


class GoldenInputCreate(BaseModel):
    org_id: str
    workflow_id: str
    name: Optional[str] = None
    input_text: Optional[str] = None
    variables: Optional[dict] = None
    expected_output: Optional[str] = None
    source: str = "manual"
    source_run_id: Optional[str] = None

    class Config:
        extra = "ignore"


class GoldenInputUpdate(BaseModel):
    name: Optional[str] = None
    input_text: Optional[str] = None
    variables: Optional[dict] = None
    expected_output: Optional[str] = None

    class Config:
        extra = "ignore"


@router.get("/golden-inputs/{org_id}/{workflow_id}")
async def list_golden_inputs(org_id: str, workflow_id: str):
    """List golden inputs (test cases) for a workflow. Requires auth and org access."""
    try:
        result = (
            supabase.table("golden_inputs")
            .select(_GOLDEN_INPUT_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [_golden_row_to_response(row) for row in (result.data or [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching golden inputs: {str(e)}")


@router.post("/golden-inputs")
async def create_golden_input(payload: GoldenInputCreate):
    """Create a golden input (test case). Requires auth and org access."""
    try:
        insert = {
            "org_id": payload.org_id,
            "workflow_id": payload.workflow_id,
            "name": payload.name,
            "input_text": payload.input_text,
            "variables": payload.variables,
            "expected_output": payload.expected_output,
            "source": payload.source or "manual",
            "source_run_id": payload.source_run_id,
        }
        result = supabase.table("golden_inputs").insert(insert).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Create failed")
        return _golden_row_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating golden input: {str(e)}")


@router.put("/golden-inputs/{golden_input_id}")
async def update_golden_input(golden_input_id: str, payload: GoldenInputUpdate):
    """Update a golden input. Requires auth and org access."""
    try:
        update = {k: v for k, v in payload.model_dump(exclude_unset=True).items()}
        if not update:
            # Fetch existing and return
            result = supabase.table("golden_inputs").select(_GOLDEN_INPUT_COLS).eq("id", golden_input_id).single().execute()
            if not result.data:
                raise HTTPException(status_code=404, detail="Golden input not found")
            return _golden_row_to_response(result.data)
        update["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        result = supabase.table("golden_inputs").update(update).eq("id", golden_input_id).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="Golden input not found")
        return _golden_row_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating golden input: {str(e)}")


@router.delete("/golden-inputs/{golden_input_id}", status_code=204)
async def delete_golden_input(golden_input_id: str):
    """Delete a golden input. Requires auth and org access."""
    try:
        supabase.table("golden_inputs").delete().eq("id", golden_input_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting golden input: {str(e)}")


class ImportFromProductionPayload(BaseModel):
    org_id: str
    workflow_id: str
    run_id: str
    name: Optional[str] = None

    class Config:
        extra = "ignore"


@router.post("/golden-inputs/import-from-production")
async def import_golden_input_from_production(payload: ImportFromProductionPayload):
    """Create a golden input from a workflow run (e.g. production run). Requires auth and org access."""
    try:
        run = (
            supabase.table("workflow_runs")
            .select("id, workflow_id, org_id, input_text, final_output, node_results")
            .eq("id", payload.run_id)
            .eq("org_id", payload.org_id)
            .eq("workflow_id", payload.workflow_id)
            .single()
            .execute()
        )
        if not run.data:
            raise HTTPException(status_code=404, detail="Run not found")
        r = run.data
        insert = {
            "org_id": r["org_id"],
            "workflow_id": r["workflow_id"],
            "name": payload.name or f"From run {payload.run_id[:8]}",
            "input_text": r.get("input_text"),
            "variables": None,
            "expected_output": r.get("final_output"),
            "source": "imported_from_production",
            "source_run_id": r["id"],
        }
        result = supabase.table("golden_inputs").insert(insert).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Import failed")
        return _golden_row_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing golden input: {str(e)}")


# ---------------------------------------------------------------------------
# Eval suites (check configuration per workflow) — Phase 3
# ---------------------------------------------------------------------------

_EVAL_SUITE_COLS = "id, org_id, workflow_id, name, checks, enabled, created_at, updated_at"

DEFAULT_EVAL_CHECKS = [
    {"type": "deterministic", "name": "Output match", "enabled": True, "config": {"normalize_whitespace": True}},
    {"type": "regression", "name": "Regression vs production", "enabled": True, "config": {"max_latency_delta_pct": 50, "max_cost_delta_pct": 50}},
    {"type": "model_graded", "name": "Quality (model-graded)", "enabled": False, "config": {}},
]


def _eval_suite_row_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "workflow_id": str(row["workflow_id"]),
        "name": row.get("name") or "Default Suite",
        "checks": row.get("checks") if isinstance(row.get("checks"), list) else [],
        "enabled": row.get("enabled", True),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


@router.get("/eval-suites/{org_id}/{workflow_id}")
async def get_eval_suite(org_id: str, workflow_id: str):
    """Get eval suite for a workflow. Creates one with default checks if none exists."""
    try:
        result = (
            supabase.table("eval_suites")
            .select(_EVAL_SUITE_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .limit(1)
            .execute()
        )
        if result.data and len(result.data) > 0:
            return _eval_suite_row_to_response(result.data[0])
        # Create default suite
        insert = {
            "org_id": org_id,
            "workflow_id": workflow_id,
            "name": "Default Suite",
            "checks": DEFAULT_EVAL_CHECKS,
            "enabled": True,
        }
        created = supabase.table("eval_suites").insert(insert).execute()
        if not created.data or len(created.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create default eval suite")
        return _eval_suite_row_to_response(created.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching eval suite: {str(e)}")


class EvalSuiteUpdate(BaseModel):
    name: Optional[str] = None
    checks: Optional[List[dict]] = None
    enabled: Optional[bool] = None

    class Config:
        extra = "ignore"


@router.put("/eval-suites/{org_id}/{workflow_id}")
async def put_eval_suite(org_id: str, workflow_id: str, payload: EvalSuiteUpdate):
    """Create or update eval suite for a workflow."""
    try:
        existing = (
            supabase.table("eval_suites")
            .select(_EVAL_SUITE_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .limit(1)
            .execute()
        )
        update = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}
        if not update:
            if existing.data and len(existing.data) > 0:
                return _eval_suite_row_to_response(existing.data[0])
            # Create with defaults
            insert = {
                "org_id": org_id,
                "workflow_id": workflow_id,
                "name": "Default Suite",
                "checks": DEFAULT_EVAL_CHECKS,
                "enabled": True,
            }
            created = supabase.table("eval_suites").insert(insert).execute()
            if not created.data or len(created.data) == 0:
                raise HTTPException(status_code=500, detail="Failed to create eval suite")
            return _eval_suite_row_to_response(created.data[0])
        if existing.data and len(existing.data) > 0:
            suite_id = existing.data[0]["id"]
            update["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            result = supabase.table("eval_suites").update(update).eq("id", suite_id).execute()
            if not result.data or len(result.data) == 0:
                raise HTTPException(status_code=404, detail="Eval suite not found")
            return _eval_suite_row_to_response(result.data[0])
        # No existing suite: create with payload
        insert = {
            "org_id": org_id,
            "workflow_id": workflow_id,
            "name": (payload.name if payload.name is not None else "Default Suite"),
            "checks": payload.checks if payload.checks is not None else DEFAULT_EVAL_CHECKS,
            "enabled": payload.enabled if payload.enabled is not None else True,
        }
        created = supabase.table("eval_suites").insert(insert).execute()
        if not created.data or len(created.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create eval suite")
        return _eval_suite_row_to_response(created.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating eval suite: {str(e)}")


# ---------------------------------------------------------------------------
# Eval engine: run eval for a deployment — Phase 4
# ---------------------------------------------------------------------------

_DEP_COLS_FOR_EVAL = "id, workflow_id, org_id, version, endpoint_slug, graph_json"
_DEP_COLS_WITH_STATUS = _DEP_COLS_FOR_EVAL + ", status"


def _normalize_output(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())


def _run_eval_sync(deployment_id: str, eval_run_id: str) -> None:
    """Run eval for a deployment: execute each golden input on candidate (and production for regression), run checks, write results."""
    try:
        dep_row = (
            supabase.table("workflow_deployments")
            .select(_DEP_COLS_FOR_EVAL)
            .eq("id", deployment_id)
            .single()
            .execute()
        )
        if not dep_row.data:
            supabase.table("eval_runs").update({
                "status": "error",
                "summary": {"error": "Deployment not found"},
                "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }).eq("id", eval_run_id).execute()
            return
        deployment = dep_row.data
        org_id = str(deployment["org_id"])
        workflow_id = str(deployment["workflow_id"])
        graph = deployment.get("graph_json") or {"nodes": [], "edges": []}
        endpoint_slug = (deployment.get("endpoint_slug") or "").strip()

        suite_row = (
            supabase.table("eval_suites")
            .select(_EVAL_SUITE_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .limit(1)
            .execute()
        )
        checks = []
        if suite_row.data and len(suite_row.data) > 0:
            raw = suite_row.data[0].get("checks") or []
            checks = [c for c in raw if isinstance(c, dict) and c.get("enabled", True)]

        golden_rows = (
            supabase.table("golden_inputs")
            .select(_GOLDEN_INPUT_COLS)
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .execute()
        )
        golden_inputs = golden_rows.data or []

        production_dep = None
        try:
            prod_rows = (
                supabase.table("workflow_deployments")
                .select(_DEP_COLS_FOR_EVAL)
                .eq("org_id", org_id)
                .eq("endpoint_slug", endpoint_slug)
                .eq("status", "promoted")
                .order("version", desc=True)
                .limit(1)
                .execute()
            )
            if prod_rows.data and len(prod_rows.data) > 0 and str(prod_rows.data[0]["id"]) != str(deployment_id):
                production_dep = prod_rows.data[0]
        except Exception:
            pass

        total_checks = 0
        passed_checks = 0
        start_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        for gi in golden_inputs:
            gi_id = gi.get("id")
            input_text = (gi.get("input_text") or "")[:5000]
            variables = gi.get("variables") if isinstance(gi.get("variables"), dict) else None
            expected_output = gi.get("expected_output")

            candidate_output = None
            candidate_latency_ms = None
            candidate_cost = None
            try:
                out = execute_workflow(
                    graph,
                    input_text,
                    org_id,
                    "",
                    workflow_id=workflow_id,
                    endpoint_slug=endpoint_slug or None,
                    version=deployment.get("version"),
                    execution_mode="eval",
                    variables=variables,
                )
                candidate_output = (out.get("final_output") or "")[:10000]
                candidate_latency_ms = out.get("total_latency_ms")
                candidate_cost = out.get("total_cost")
            except Exception as e:
                candidate_output = None
                candidate_latency_ms = None
                candidate_cost = None
                for ch in checks:
                    total_checks += 1
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": ch.get("name") or ch.get("type", "check"),
                        "check_type": ch.get("type", "deterministic"),
                        "passed": False,
                        "candidate_output": None,
                        "failure_reason": str(e)[:500],
                    }).execute()
                continue

            production_output = None
            production_latency_ms = None
            production_cost = None
            if production_dep:
                try:
                    prod_graph = production_dep.get("graph_json") or {"nodes": [], "edges": []}
                    prod_out = execute_workflow(
                        prod_graph,
                        input_text,
                        org_id,
                        "",
                        workflow_id=workflow_id,
                        endpoint_slug=endpoint_slug or None,
                        version=production_dep.get("version"),
                        execution_mode="eval",
                        variables=variables,
                    )
                    production_output = (prod_out.get("final_output") or "")[:10000]
                    production_latency_ms = prod_out.get("total_latency_ms")
                    production_cost = prod_out.get("total_cost")
                except Exception:
                    pass

            for ch in checks:
                check_type = ch.get("type") or "deterministic"
                check_name = ch.get("name") or check_type
                config = ch.get("config") or {}
                passed = True
                failure_reason = None
                latency_delta_pct = None
                cost_delta_pct = None
                output_changed = None

                if check_type == "deterministic":
                    if expected_output is not None and expected_output != "":
                        passed = _normalize_output(candidate_output) == _normalize_output(expected_output)
                        if not passed:
                            failure_reason = "Output did not match expected"
                    else:
                        passed = True
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": check_name,
                        "check_type": check_type,
                        "passed": passed,
                        "candidate_output": candidate_output,
                        "candidate_latency_ms": candidate_latency_ms,
                        "candidate_cost": candidate_cost,
                        "failure_reason": failure_reason,
                    }).execute()

                elif check_type == "structural":
                    if config.get("expect_json"):
                        try:
                            import json
                            if candidate_output and candidate_output.strip():
                                json.loads(candidate_output)
                                passed = True
                            else:
                                passed = False
                                failure_reason = "Output is empty or not valid JSON"
                        except (ValueError, TypeError):
                            passed = False
                            failure_reason = "Output is not valid JSON"
                    else:
                        passed = True
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": check_name,
                        "check_type": check_type,
                        "passed": passed,
                        "candidate_output": candidate_output,
                        "candidate_latency_ms": candidate_latency_ms,
                        "candidate_cost": candidate_cost,
                        "failure_reason": failure_reason,
                    }).execute()

                elif check_type == "format":
                    pattern = (config.get("pattern") or "").strip()
                    if not pattern:
                        passed = True
                        failure_reason = "No pattern configured"
                    else:
                        import re
                        try:
                            re_pat = re.compile(pattern)
                            passed = bool(re_pat.search(candidate_output or ""))
                            if not passed:
                                failure_reason = f"Output did not match pattern: {pattern[:50]}..."
                        except re.error:
                            passed = False
                            failure_reason = f"Invalid regex: {pattern[:50]}"
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": check_name,
                        "check_type": check_type,
                        "passed": passed,
                        "candidate_output": candidate_output,
                        "candidate_latency_ms": candidate_latency_ms,
                        "candidate_cost": candidate_cost,
                        "failure_reason": failure_reason,
                    }).execute()

                elif check_type == "regression":
                    if production_output is None:
                        passed = True
                        failure_reason = "No production deployment to compare"
                    else:
                        output_changed = _normalize_output(candidate_output) != _normalize_output(production_output)
                        if candidate_latency_ms and production_latency_ms and production_latency_ms > 0:
                            latency_delta_pct = round((candidate_latency_ms - production_latency_ms) / production_latency_ms * 100, 2)
                        if candidate_cost is not None and production_cost is not None and production_cost > 0:
                            cost_delta_pct = round((float(candidate_cost) - float(production_cost)) / float(production_cost) * 100, 2)
                        max_latency = config.get("max_latency_delta_pct", 50)
                        max_cost = config.get("max_cost_delta_pct", 50)
                        if latency_delta_pct is not None and abs(latency_delta_pct) > max_latency:
                            passed = False
                            failure_reason = f"Latency delta {latency_delta_pct}% exceeds {max_latency}%"
                        if cost_delta_pct is not None and abs(cost_delta_pct) > max_cost:
                            passed = False
                            failure_reason = (failure_reason or "") + f" Cost delta {cost_delta_pct}% exceeds {max_cost}%"
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": check_name,
                        "check_type": check_type,
                        "passed": passed,
                        "candidate_output": candidate_output,
                        "candidate_latency_ms": candidate_latency_ms,
                        "candidate_cost": candidate_cost,
                        "production_output": production_output,
                        "production_latency_ms": production_latency_ms,
                        "production_cost": production_cost,
                        "latency_delta_pct": latency_delta_pct,
                        "cost_delta_pct": cost_delta_pct,
                        "output_changed": output_changed,
                        "failure_reason": failure_reason,
                    }).execute()

                elif check_type == "model_graded":
                    passed = False
                    failure_reason = "AI-graded check not yet implemented"
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": check_name,
                        "check_type": check_type,
                        "passed": passed,
                        "candidate_output": candidate_output,
                        "candidate_latency_ms": candidate_latency_ms,
                        "candidate_cost": candidate_cost,
                        "failure_reason": failure_reason,
                    }).execute()
                else:
                    passed = True
                    supabase.table("eval_run_results").insert({
                        "eval_run_id": eval_run_id,
                        "golden_input_id": gi_id,
                        "check_name": check_name,
                        "check_type": check_type,
                        "passed": passed,
                        "candidate_output": candidate_output,
                        "candidate_latency_ms": candidate_latency_ms,
                        "candidate_cost": candidate_cost,
                        "failure_reason": None,
                    }).execute()

                total_checks += 1
                if passed:
                    passed_checks += 1

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        duration_ms = end_ms - start_ms
        status = "passed" if (total_checks == 0 or passed_checks == total_checks) else "failed"
        supabase.table("eval_runs").update({
            "status": status,
            "summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": total_checks - passed_checks,
                "duration_ms": duration_ms,
            },
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }).eq("id", eval_run_id).execute()
        dep_status = "promoted" if status == "passed" else "failed"
        dep_update = {"eval_run_id": eval_run_id, "status": dep_status}
        if dep_status == "promoted":
            dep_update["promoted_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        supabase.table("workflow_deployments").update(dep_update).eq("id", deployment_id).execute()
        if dep_status == "promoted":
            _end_experiments_on_endpoint_sync(org_id, endpoint_slug, "new_deployment")
    except Exception as e:
        supabase.table("eval_runs").update({
            "status": "error",
            "summary": {"error": str(e)[:500]},
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }).eq("id", eval_run_id).execute()
        supabase.table("workflow_deployments").update({
            "status": "failed",
            "eval_run_id": eval_run_id,
        }).eq("id", deployment_id).execute()


class EvalRunStartPayload(BaseModel):
    deployment_id: str

    class Config:
        extra = "ignore"


@router.post("/eval/run")
async def start_eval_run(payload: EvalRunStartPayload):
    """Start an eval run for a deployment. Returns eval_run id; run executes and status can be polled via GET."""
    try:
        dep = (
            supabase.table("workflow_deployments")
            .select("id, org_id, workflow_id")
            .eq("id", payload.deployment_id)
            .single()
            .execute()
        )
        if not dep.data:
            raise HTTPException(status_code=404, detail="Deployment not found")
        org_id = str(dep.data["org_id"])
        workflow_id = str(dep.data["workflow_id"])

        suite_row = (
            supabase.table("eval_suites")
            .select("id")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .limit(1)
            .execute()
        )
        eval_suite_id = suite_row.data[0]["id"] if suite_row.data and len(suite_row.data) > 0 else None

        insert_row = {
            "org_id": org_id,
            "deployment_id": payload.deployment_id,
            "eval_suite_id": eval_suite_id,
            "status": "running",
        }
        result = supabase.table("eval_runs").insert(insert_row).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create eval run")
        eval_run = result.data[0]
        eval_run_id = str(eval_run["id"])

        asyncio.get_event_loop().run_in_executor(None, _run_eval_sync, payload.deployment_id, eval_run_id)
        return {
            "id": eval_run_id,
            "deployment_id": payload.deployment_id,
            "status": eval_run.get("status", "running"),
            "started_at": eval_run.get("started_at"),
            "summary": eval_run.get("summary") or {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/runs/{eval_run_id}")
async def get_eval_run(eval_run_id: str):
    """Get eval run status and results (for polling)."""
    try:
        run_row = (
            supabase.table("eval_runs")
            .select("id, org_id, deployment_id, eval_suite_id, status, results, summary, started_at, completed_at, created_at")
            .eq("id", eval_run_id)
            .single()
            .execute()
        )
        if not run_row.data:
            raise HTTPException(status_code=404, detail="Eval run not found")
        r = run_row.data
        results_rows = (
            supabase.table("eval_run_results")
            .select("id, golden_input_id, check_name, check_type, passed, candidate_output, candidate_latency_ms, candidate_cost, production_output, production_latency_ms, production_cost, latency_delta_pct, cost_delta_pct, output_changed, failure_reason")
            .eq("eval_run_id", eval_run_id)
            .execute()
        )
        results = results_rows.data or []
        return {
            "id": str(r["id"]),
            "deployment_id": str(r["deployment_id"]),
            "eval_suite_id": str(r["eval_suite_id"]) if r.get("eval_suite_id") else None,
            "status": r.get("status", "running"),
            "summary": r.get("summary") if isinstance(r.get("summary"), dict) else {},
            "started_at": r.get("started_at"),
            "completed_at": r.get("completed_at"),
            "created_at": r.get("created_at"),
            "results": [
                {
                    "id": str(x["id"]),
                    "golden_input_id": str(x["golden_input_id"]) if x.get("golden_input_id") else None,
                    "check_name": x.get("check_name"),
                    "check_type": x.get("check_type"),
                    "passed": x.get("passed", False),
                    "candidate_output": x.get("candidate_output"),
                    "candidate_latency_ms": x.get("candidate_latency_ms"),
                    "candidate_cost": x.get("candidate_cost"),
                    "production_output": x.get("production_output"),
                    "production_latency_ms": x.get("production_latency_ms"),
                    "production_cost": x.get("production_cost"),
                    "latency_delta_pct": x.get("latency_delta_pct"),
                    "cost_delta_pct": x.get("cost_delta_pct"),
                    "output_changed": x.get("output_changed"),
                    "failure_reason": x.get("failure_reason"),
                }
                for x in results
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Promote override (admin) — Phase 5
# ---------------------------------------------------------------------------

class PromoteOverridePayload(BaseModel):
    override_reason: Optional[str] = None

    class Config:
        extra = "ignore"


@router.post("/workflow-deployments/{deployment_id}/promote")
async def promote_deployment_override(deployment_id: str, payload: PromoteOverridePayload = None):
    """Admin override: promote a deployment despite failed eval. Sets status=promoted, promoted_at, override_reason."""
    try:
        payload = payload or PromoteOverridePayload()
        dep = (
            supabase.table("workflow_deployments")
            .select("id, status")
            .eq("id", deployment_id)
            .single()
            .execute()
        )
        if not dep.data:
            raise HTTPException(status_code=404, detail="Deployment not found")
        update = {
            "status": "promoted",
            "promoted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "override_reason": (payload.override_reason or "").strip() or None,
        }
        result = (
            supabase.table("workflow_deployments")
            .update(update)
            .eq("id", deployment_id)
            .execute()
        )
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Update failed")
        row = result.data[0]
        oid = row.get("org_id")
        slug = (row.get("endpoint_slug") or "").strip()
        if oid and slug:
            await asyncio.to_thread(_end_experiments_on_endpoint_sync, str(oid), slug, "new_deployment")
        return _deployment_row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Traffic routing policies (canary / weighted) — Phase 3
# ---------------------------------------------------------------------------

_ROUTING_POLICY_COLS = "id, org_id, endpoint_slug, policy_type, rules, active, created_by, created_at, updated_at"


def _routing_policy_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "endpoint_slug": row.get("endpoint_slug") or "",
        "policy_type": row.get("policy_type") or "latest",
        "rules": row.get("rules") if isinstance(row.get("rules"), list) else [],
        "active": row.get("active", True),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


class RoutingPolicyCreate(BaseModel):
    org_id: str
    endpoint_slug: str
    policy_type: str = "weighted"
    rules: List[dict]

    class Config:
        extra = "ignore"


@router.get("/routing-policies/{org_id}/{endpoint_slug}")
async def get_routing_policy(org_id: str, endpoint_slug: str):
    """Get active routing policy for an endpoint, if any."""
    try:
        result = (
            supabase.table("routing_policies")
            .select(_ROUTING_POLICY_COLS)
            .eq("org_id", org_id)
            .eq("endpoint_slug", endpoint_slug.strip())
            .eq("active", True)
            .limit(1)
            .execute()
        )
        if not result.data or len(result.data) == 0:
            return None
        return _routing_policy_to_response(result.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/routing-policies")
async def create_or_update_routing_policy(payload: RoutingPolicyCreate):
    """Create or update routing policy. Only one active policy per endpoint; weights must sum to 100."""
    try:
        slug = (payload.endpoint_slug or "").strip()
        if not slug:
            raise HTTPException(status_code=400, detail="endpoint_slug required")
        rules = payload.rules or []
        if payload.policy_type == "weighted":
            if len(rules) < 2:
                raise HTTPException(status_code=400, detail="Weighted policy needs at least 2 versions")
            total = sum(r.get("weight", 0) for r in rules)
            if total != 100:
                raise HTTPException(status_code=400, detail="Weights must sum to 100")
            versions = [r.get("version") for r in rules if r.get("version") is not None]
            for v in versions:
                dep = (
                    supabase.table("workflow_deployments")
                    .select("id")
                    .eq("org_id", payload.org_id)
                    .eq("endpoint_slug", slug)
                    .eq("version", v)
                    .eq("status", "promoted")
                    .limit(1)
                    .execute()
                )
                if not dep.data or len(dep.data) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Version v{v} is not promoted. Only promoted deployments can be used in routing. The deployment may still be evaluating or have failed eval.",
                    )
        existing = (
            supabase.table("routing_policies")
            .select("id")
            .eq("org_id", payload.org_id)
            .eq("endpoint_slug", slug)
            .eq("active", True)
            .execute()
        )
        if existing.data and len(existing.data) > 0:
            for row in existing.data:
                supabase.table("routing_policies").update({"active": False}).eq("id", row["id"]).execute()
        insert = {
            "org_id": payload.org_id,
            "endpoint_slug": slug,
            "policy_type": payload.policy_type or "weighted",
            "rules": rules,
            "active": True,
        }
        result = supabase.table("routing_policies").insert(insert).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create routing policy")
        return _routing_policy_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/routing-policies/{policy_id}", status_code=204)
async def delete_routing_policy(policy_id: str):
    """Delete (deactivate) a routing policy; traffic reverts to latest promoted."""
    try:
        supabase.table("routing_policies").update({"active": False}).eq("id", policy_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Per-version metrics (for canary / experiment dashboards) — Phase 3
# ---------------------------------------------------------------------------

_WINDOW_MAP = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}


@router.get("/metrics/by-version/{org_id}/{endpoint_slug}")
async def get_metrics_by_version(
    org_id: str,
    endpoint_slug: str,
    versions: str,
    window: str = "1h",
):
    """Aggregate workflow_runs by served_version for the given versions and time window."""
    try:
        from datetime import timedelta
        version_list = [int(x.strip()) for x in versions.split(",") if x.strip().isdigit()]
        if not version_list:
            return {"versions": {}}
        hours = (7 * 24) if window == "7d" else (1 if window == "1h" else (6 if window == "6h" else 24))
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        since_str = since.isoformat().replace("+00:00", "Z")
        select_with_created = (
            supabase.table("workflow_runs")
            .select("id, served_version, total_latency_ms, total_cost, node_results, created_at")
            .eq("org_id", org_id)
            .eq("endpoint_slug", endpoint_slug.strip())
            .in_("served_version", version_list)
            .eq("execution_mode", "production")
            .gte("created_at", since_str)
            .execute()
        )
        rows = select_with_created.data or []
        by_version = {}
        for r in rows:
            v = r.get("served_version")
            if v is None:
                continue
            key = str(v)
            if key not in by_version:
                by_version[key] = {"requests": 0, "errors": 0, "latencies": [], "costs": []}
            by_version[key]["requests"] += 1
            node_results = r.get("node_results") or []
            has_error = any(_nr_has_error(nr) for nr in node_results)
            if has_error:
                by_version[key]["errors"] += 1
            lat = r.get("total_latency_ms")
            if lat is not None:
                by_version[key]["latencies"].append(int(lat))
            c = r.get("total_cost")
            if c is not None:
                by_version[key]["costs"].append(float(c))
        out = {}
        for key in by_version:
            d = by_version[key]
            n = d["requests"]
            errs = d["errors"]
            latencies = sorted(d["latencies"]) if d["latencies"] else []
            costs = d["costs"]
            p50 = latencies[len(latencies) // 2] if latencies else 0
            p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else (latencies[0] if latencies else 0)
            out[key] = {
                "requests": n,
                "errors": errs,
                "error_rate": round(errs / n * 100, 2) if n > 0 else 0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "avg_cost": round(sum(costs) / len(costs), 6) if costs else 0,
                "total_cost": round(sum(costs), 6),
            }
        return {"versions": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Experiments (A/B testing) — Phase 4
# ---------------------------------------------------------------------------

_EXPERIMENT_COLS = (
    "id, org_id, endpoint_slug, name, description, status, variants, primary_metric, max_error_rate, min_sample_size,"
    " confidence_level, mde, power, sequential_testing, auto_conclude,"
    " results, winner_version, concluded_at, concluded_reason, created_at, updated_at"
)


def _experiment_to_response(row: dict) -> dict:
    out = {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "endpoint_slug": row.get("endpoint_slug") or "",
        "name": row.get("name") or "",
        "description": row.get("description"),
        "status": row.get("status") or "draft",
        "variants": row.get("variants") if isinstance(row.get("variants"), list) else [],
        "primary_metric": row.get("primary_metric") or "error_rate",
        "max_error_rate": float(row["max_error_rate"]) if row.get("max_error_rate") is not None else 10.0,
        "min_sample_size": int(row["min_sample_size"]) if row.get("min_sample_size") is not None else 50,
        "results": row.get("results") if isinstance(row.get("results"), dict) else {},
        "winner_version": row.get("winner_version"),
        "concluded_at": row.get("concluded_at"),
        "concluded_reason": row.get("concluded_reason"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    out["confidence_level"] = float(row["confidence_level"]) if row.get("confidence_level") is not None else 95.0
    out["mde"] = float(row["mde"]) if row.get("mde") is not None else 10.0
    out["power"] = float(row["power"]) if row.get("power") is not None else 80.0
    out["sequential_testing"] = bool(row["sequential_testing"]) if row.get("sequential_testing") is not None else True
    out["auto_conclude"] = bool(row["auto_conclude"]) if row.get("auto_conclude") is not None else False
    return out


def _variant_zero_row() -> dict:
    return {
        "requests": 0,
        "error_rate": 0.0,
        "avg_latency_ms": 0,
        "p50_latency_ms": 0,
        "p95_latency_ms": 0,
        "avg_cost": 0.0,
        "total_cost": 0.0,
    }


def _get_custom_metrics_for_org(org_id: str) -> List[dict]:
    """Return custom metric definitions for org (id, name, key, metric_type, direction)."""
    if not org_id:
        return []
    try:
        r = (
            supabase.table("custom_metrics")
            .select("id, name, key, metric_type, direction")
            .eq("org_id", org_id)
            .execute()
        )
        return list(r.data or [])
    except Exception:
        return []


def _norm_cdf(z: float) -> float:
    """Standard normal CDF using error function. P(Z <= z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _chance_to_win(diff: float, se: float) -> float:
    """P(true difference > 0) under normal approximation. Frontend inverts for lower-is-better."""
    if se <= 0:
        return 0.5
    return _norm_cdf(diff / se)


def _ci_proportion_relative(
    p1: float, n1: int, p2: float, n2: int, confidence_level: float
) -> tuple:
    """CI for relative change (candidate vs control) in proportion metrics. Returns (ci_low_pct, ci_high_pct, significant)."""
    if n1 < 1 or n2 < 1 or p1 <= 0:
        return (0.0, 0.0, False)
    alpha = 1.0 - (confidence_level / 100.0)
    z = abs(_norm_ppf(alpha / 2.0))
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    ci_low = d - z * se
    ci_high = d + z * se
    rel_low = (ci_low / p1) * 100.0
    rel_high = (ci_high / p1) * 100.0
    significant = (ci_low > 0) or (ci_high < 0)
    return (rel_low, rel_high, significant)


def _ci_means_relative(
    mean1: float, var1: float, n1: int,
    mean2: float, var2: float, n2: int,
    confidence_level: float,
) -> tuple:
    """CI for relative change in mean metrics (control as baseline). Returns (ci_low_pct, ci_high_pct, significant)."""
    if n1 < 1 or n2 < 1 or mean1 == 0:
        return (0.0, 0.0, False)
    alpha = 1.0 - (confidence_level / 100.0)
    z = abs(_norm_ppf(alpha / 2.0))
    se_diff = math.sqrt(var1 / n1 + var2 / n2)
    diff = mean2 - mean1
    ci_low = diff - z * se_diff
    ci_high = diff + z * se_diff
    rel_low = (ci_low / mean1) * 100.0
    rel_high = (ci_high / mean1) * 100.0
    significant = (ci_low > 0) or (ci_high < 0)
    return (rel_low, rel_high, significant)


def _compute_experiment_results(
    experiment_id: str,
    min_sample_size: int,
    expected_variant_names: Optional[List[str]] = None,
    org_id: Optional[str] = None,
    confidence_level: float = 95.0,
) -> dict:
    """Compute results from workflow_runs for this experiment. Always includes an entry for each expected_variant_names so UI has control/candidate.
    When org_id is set, also aggregates custom_metrics from api_request_log and merges into variant results.
    Computes per-metric CIs and significance when two variants are present."""
    runs = (
        supabase.table("workflow_runs")
        .select("variant_name, total_latency_ms, total_cost, node_results")
        .eq("experiment_id", experiment_id)
        .execute()
    )
    from collections import defaultdict
    by_variant = defaultdict(list)
    for r in runs.data or []:
        vname = (r.get("variant_name") or "").strip() or "unknown"
        by_variant[vname].append(r)
    results = {}
    raw_builtin: dict = defaultdict(dict)
    for name, variant_runs in by_variant.items():
        n = len(variant_runs)
        errors = sum(1 for r in variant_runs if any(
            _nr_has_error(nr) for nr in (r.get("node_results") or [])
        ))
        latencies = [r["total_latency_ms"] for r in variant_runs if r.get("total_latency_ms") is not None]
        costs = [float(r["total_cost"]) for r in variant_runs if r.get("total_cost") is not None]
        lat_sorted = sorted(latencies) if latencies else []
        p_err = (errors / n) if n > 0 else 0.0
        raw_builtin[name]["error_rate"] = (p_err, n)
        if latencies:
            m_lat = sum(latencies) / len(latencies)
            v_lat = sum((x - m_lat) ** 2 for x in latencies) / len(latencies) if latencies else 0.0
            raw_builtin[name]["avg_latency_ms"] = (m_lat, v_lat, len(latencies))
        if costs:
            m_cost = sum(costs) / len(costs)
            v_cost = sum((x - m_cost) ** 2 for x in costs) / len(costs) if costs else 0.0
            raw_builtin[name]["avg_cost"] = (m_cost, v_cost, len(costs))
        results[name] = {
            "requests": n,
            "error_rate": round(errors / n * 100, 2) if n > 0 else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
            "p50_latency_ms": lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0,
            "p95_latency_ms": lat_sorted[int(len(lat_sorted) * 0.95)] if len(lat_sorted) > 1 else (lat_sorted[0] if lat_sorted else 0),
            "avg_cost": round(sum(costs) / len(costs), 6) if costs else 0,
            "total_cost": round(sum(costs), 6),
        }
    raw_custom: dict = defaultdict(dict)
    # Custom metrics from api_request_log (feedback)
    if org_id:
        custom_defs = _get_custom_metrics_for_org(org_id)
        if custom_defs:
            try:
                log_rows = (
                    supabase.table("api_request_log")
                    .select("variant_name, custom_metrics")
                    .eq("experiment_id", experiment_id)
                    .execute()
                )
            except Exception:
                log_rows = type("R", (), {"data": []})()
            by_var_logs: dict = defaultdict(list)
            for row in log_rows.data or []:
                vn = (row.get("variant_name") or "").strip()
                if not vn:
                    continue
                by_var_logs[vn].append(row.get("custom_metrics") or {})
            for metric_def in custom_defs:
                key = metric_def.get("key")
                metric_type = (metric_def.get("metric_type") or "number").lower()
                if not key:
                    continue
                for vname, logs in by_var_logs.items():
                    if vname not in results:
                        continue
                    values = [
                        m.get(key) for m in logs
                        if m is not None and isinstance(m, dict) and m.get(key) is not None
                    ]
                    if not values:
                        continue
                    if metric_type == "number":
                        try:
                            nums = [float(v) for v in values]
                            mean_n = sum(nums) / len(nums)
                            var_n = sum((x - mean_n) ** 2 for x in nums) / len(nums) if nums else 0.0
                            results[vname][f"custom_{key}"] = round(mean_n, 4)
                            raw_custom[vname][f"custom_{key}"] = ("mean", mean_n, var_n, len(nums))
                        except (TypeError, ValueError):
                            pass
                    elif metric_type == "boolean":
                        trues = sum(1 for v in values if v is True or v in (1, "true", "1"))
                        p_b = trues / len(values)
                        results[vname][f"custom_{key}"] = round(p_b * 100, 2)
                        raw_custom[vname][f"custom_{key}"] = ("prop", p_b, len(values))
                    else:
                        results[vname][f"custom_{key}"] = sum(1 for _ in values) / len(values)  # category: just average count for now
                    results[vname][f"custom_{key}_count"] = len(values)

    raw_graded: dict = defaultdict(dict)
    # Auto-graded metrics from auto_grade_results (joined with api_request_log by experiment_id)
    if org_id and experiment_id:
        try:
            log_rows = (
                supabase.table("api_request_log")
                .select("id, variant_name")
                .eq("experiment_id", experiment_id)
                .execute()
            )
            log_id_to_variant = {str(r["id"]): (r.get("variant_name") or "").strip() for r in (log_rows.data or []) if (r.get("variant_name") or "").strip()}
            if log_id_to_variant:
                ids = list(log_id_to_variant.keys())
                agr_rows = (
                    supabase.table("auto_grade_results")
                    .select("request_log_id, metric_key, score, binary_result, category_result")
                    .in_("request_log_id", ids)
                    .execute()
                )
                by_var_key: dict = defaultdict(lambda: defaultdict(list))
                for row in agr_rows.data or []:
                    rlid = str(row.get("request_log_id") or "")
                    vn = log_id_to_variant.get(rlid)
                    if not vn or vn not in results:
                        continue
                    key = (row.get("metric_key") or "").strip()
                    if not key:
                        continue
                    by_var_key[vn][key].append(row)
                for vn, keys in by_var_key.items():
                    for key, rows in keys.items():
                        scores = [float(r["score"]) for r in rows if r.get("score") is not None]
                        bins = [r["binary_result"] for r in rows if r.get("binary_result") is not None]
                        if scores:
                            mean_s = sum(scores) / len(scores)
                            var_s = sum((x - mean_s) ** 2 for x in scores) / len(scores) if scores else 0.0
                            results[vn][f"graded_{key}"] = round(mean_s, 4)
                            results[vn][f"graded_{key}_count"] = len(scores)
                            raw_graded[vn][f"graded_{key}"] = ("mean", mean_s, var_s, len(scores))
                        elif bins:
                            p_b = sum(1 for b in bins if b) / len(bins)
                            results[vn][f"graded_{key}"] = round(p_b * 100, 2)
                            results[vn][f"graded_{key}_count"] = len(bins)
                            raw_graded[vn][f"graded_{key}"] = ("prop", p_b, len(bins))
        except Exception:
            pass

    # Ensure we have an entry for each expected variant name (from experiment.variants) so frontend always has control + candidate
    if expected_variant_names:
        for name in expected_variant_names:
            if name and name not in results:
                results[name] = _variant_zero_row()
    min_sample_reached = all(results[v]["requests"] >= min_sample_size for v in results) if results else False

    # Per-metric CIs and significance (relative %; control = baseline)
    metric_ci: dict = {}
    if expected_variant_names and len(expected_variant_names) >= 2:
        ctrl_name = expected_variant_names[0]
        cand_name = expected_variant_names[1]
        if ctrl_name in results and cand_name in results:
            for key in ("error_rate", "avg_latency_ms", "avg_cost"):
                r1 = raw_builtin.get(ctrl_name, {}).get(key)
                r2 = raw_builtin.get(cand_name, {}).get(key)
                if not r1 or not r2:
                    continue
                if key == "error_rate":
                    p1, n1 = r1[0], r1[1]
                    p2, n2 = r2[0], r2[1]
                    lo, hi, sig = _ci_proportion_relative(p1, n1, p2, n2, confidence_level)
                    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
                    chance = _chance_to_win(p2 - p1, se)
                else:
                    m1, v1, n1 = r1[0], r1[1], r1[2]
                    m2, v2, n2 = r2[0], r2[1], r2[2]
                    lo, hi, sig = _ci_means_relative(m1, v1, n1, m2, v2, n2, confidence_level)
                    se_diff = math.sqrt(v1 / n1 + v2 / n2)
                    chance = _chance_to_win(m2 - m1, se_diff)
                metric_ci[key] = {"ci_low_pct": round(lo, 2), "ci_high_pct": round(hi, 2), "significant": sig, "chance_to_win": round(chance, 4)}
            for ckey in set(raw_custom.get(ctrl_name, {})) & set(raw_custom.get(cand_name, {})):
                r1 = raw_custom[ctrl_name][ckey]
                r2 = raw_custom[cand_name][ckey]
                if r1[0] == "prop":
                    p1, n1 = r1[1], r1[2]
                    p2, n2 = r2[1], r2[2]
                    lo, hi, sig = _ci_proportion_relative(p1, n1, p2, n2, confidence_level)
                    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
                    chance = _chance_to_win(p2 - p1, se)
                else:
                    m1, v1, n1 = r1[1], r1[2], r1[3]
                    m2, v2, n2 = r2[1], r2[2], r2[3]
                    if m1 == 0:
                        continue
                    lo, hi, sig = _ci_means_relative(m1, v1, n1, m2, v2, n2, confidence_level)
                    se_diff = math.sqrt(v1 / n1 + v2 / n2)
                    chance = _chance_to_win(m2 - m1, se_diff)
                metric_ci[ckey] = {"ci_low_pct": round(lo, 2), "ci_high_pct": round(hi, 2), "significant": sig, "chance_to_win": round(chance, 4)}
            for gkey in set(raw_graded.get(ctrl_name, {})) & set(raw_graded.get(cand_name, {})):
                r1 = raw_graded[ctrl_name][gkey]
                r2 = raw_graded[cand_name][gkey]
                if r1[0] == "prop":
                    p1, n1 = r1[1], r1[2]
                    p2, n2 = r2[1], r2[2]
                    lo, hi, sig = _ci_proportion_relative(p1, n1, p2, n2, confidence_level)
                    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
                    chance = _chance_to_win(p2 - p1, se)
                else:
                    m1, v1, n1 = r1[1], r1[2], r1[3]
                    m2, v2, n2 = r2[1], r2[2], r2[3]
                    if m1 == 0:
                        continue
                    lo, hi, sig = _ci_means_relative(m1, v1, n1, m2, v2, n2, confidence_level)
                    se_diff = math.sqrt(v1 / n1 + v2 / n2)
                    chance = _chance_to_win(m2 - m1, se_diff)
                metric_ci[gkey] = {"ci_low_pct": round(lo, 2), "ci_high_pct": round(hi, 2), "significant": sig, "chance_to_win": round(chance, 4)}

    return {
        "variants": results,
        "min_sample_reached": min_sample_reached,
        "significant": min_sample_reached,
        "metric_ci": metric_ci,
    }


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (percent point function). Abramowitz and Stegun 26.2.23."""
    if p <= 0 or p >= 1:
        return 0.0
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _required_sample_size(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Required sample size per variant (two-proportion z-test normal approximation)."""
    delta = baseline_rate * mde_relative
    if delta <= 0:
        return 10000
    p1 = baseline_rate
    p2 = baseline_rate - delta
    p_avg = (p1 + p2) / 2
    z_alpha = abs(_norm_ppf(alpha / 2))
    z_beta = abs(_norm_ppf(1 - power))
    numerator = (
        z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = delta ** 2
    return max(100, math.ceil(numerator / denominator))


class EstimateSampleBody(BaseModel):
    baseline_error_rate: float = 2.0
    mde: float = 10.0
    confidence_level: float = 95.0
    power: float = 80.0
    endpoint_slug: Optional[str] = None
    org_id: Optional[str] = None

    class Config:
        extra = "ignore"


@router.post("/experiments/estimate-sample")
async def estimate_experiment_sample(body: EstimateSampleBody):
    """Estimate required sample size per variant and total; optional estimated_days from recent traffic."""
    try:
        alpha = 1.0 - (body.confidence_level / 100.0)
        power = body.power / 100.0
        baseline_rate = body.baseline_error_rate / 100.0
        mde_rel = body.mde / 100.0
        per_variant = _required_sample_size(baseline_rate, mde_rel, alpha=alpha, power=power)
        total = per_variant * 2
        estimated_days: Optional[int] = None
        if body.org_id and body.endpoint_slug:
            try:
                since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
                r = (
                    supabase.table("api_request_log")
                    .select("id")
                    .eq("org_id", body.org_id)
                    .eq("endpoint_slug", body.endpoint_slug.strip())
                    .gte("created_at", since)
                    .limit(50000)
                    .execute()
                )
                count = len(r.data or [])
                if count > 0:
                    per_day = count / 7.0
                    if per_day >= 1:
                        estimated_days = max(1, int(total / per_day))
            except Exception:
                pass
        return {
            "required_per_variant": per_variant,
            "total_required": total,
            "estimated_days": estimated_days,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExperimentCreate(BaseModel):
    org_id: str
    endpoint_slug: str
    name: str
    description: Optional[str] = None
    variants: List[dict]
    primary_metric: str = "error_rate"
    min_sample_size: int = 50
    max_error_rate: float = 10.0
    confidence_level: Optional[float] = 95.0
    mde: Optional[float] = 10.0
    power: Optional[float] = 80.0
    sequential_testing: Optional[bool] = True
    auto_conclude: Optional[bool] = False

    class Config:
        extra = "ignore"


@router.post("/experiments")
async def create_experiment(payload: ExperimentCreate):
    """Create a new experiment (draft)."""
    try:
        slug = (payload.endpoint_slug or "").strip()
        if not slug or not (payload.name or "").strip():
            raise HTTPException(status_code=400, detail="endpoint_slug and name required")
        variants = payload.variants or []
        if len(variants) < 2:
            raise HTTPException(status_code=400, detail="At least 2 variants required")
        total_weight = sum(v.get("weight", 0) for v in variants)
        if total_weight != 100:
            raise HTTPException(status_code=400, detail="Variant weights must sum to 100")
        for v in variants:
            ver = v.get("version")
            dep = (
                supabase.table("workflow_deployments")
                .select("id")
                .eq("org_id", payload.org_id)
                .eq("endpoint_slug", slug)
                .eq("version", ver)
                .eq("status", "promoted")
                .limit(1)
                .execute()
            )
            if not dep.data or len(dep.data) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Version v{ver} is not promoted. Only promoted deployments can be used in experiments. The deployment may still be evaluating or have failed eval.",
                )
        insert = {
            "org_id": payload.org_id,
            "endpoint_slug": slug,
            "name": (payload.name or "").strip(),
            "description": (payload.description or "").strip() or None,
            "status": "draft",
            "variants": variants,
            "primary_metric": payload.primary_metric or "error_rate",
            "min_sample_size": payload.min_sample_size or 50,
            "max_error_rate": payload.max_error_rate,
            "confidence_level": payload.confidence_level if payload.confidence_level is not None else 95.0,
            "mde": payload.mde if payload.mde is not None else 10.0,
            "power": payload.power if payload.power is not None else 80.0,
            "sequential_testing": payload.sequential_testing if payload.sequential_testing is not None else True,
            "auto_conclude": payload.auto_conclude if payload.auto_conclude is not None else False,
        }
        result = supabase.table("experiments").insert(insert).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create experiment")
        return _experiment_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(org_id: str, endpoint_slug: Optional[str] = None):
    """List experiments. If endpoint_slug is omitted, returns all experiments for the org."""
    try:
        q = (
            supabase.table("experiments")
            .select(_EXPERIMENT_COLS)
            .eq("org_id", org_id)
            .order("created_at", desc=True)
        )
        if endpoint_slug and endpoint_slug.strip():
            q = q.eq("endpoint_slug", endpoint_slug.strip())
        result = q.execute()
        return [_experiment_to_response(r) for r in (result.data or [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment by id with computed results."""
    try:
        result = (
            supabase.table("experiments")
            .select(_EXPERIMENT_COLS)
            .eq("id", experiment_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        row = result.data
        min_sample = int(row.get("min_sample_size") or 50)
        confidence_level = float(row.get("confidence_level") or 95.0)
        variants = row.get("variants") or []
        expected_names = [v.get("name") or f"v{v.get('version', 0)}" for v in variants if isinstance(v, dict)]
        computed = _compute_experiment_results(
            experiment_id, min_sample, expected_variant_names=expected_names, org_id=row.get("org_id"),
            confidence_level=confidence_level,
        )
        out = _experiment_to_response(row)
        out["results"] = computed
        if (row.get("status") or "") == "running" and row.get("auto_conclude"):
            asyncio.create_task(_maybe_auto_conclude(experiment_id))
        try:
            log_ids = (
                supabase.table("api_request_log")
                .select("id")
                .eq("experiment_id", experiment_id)
                .execute()
            )
            ids = [str(r["id"]) for r in (log_ids.data or []) if r.get("id")]
            if ids:
                agr = (
                    supabase.table("auto_grade_results")
                    .select("grading_cost")
                    .in_("request_log_id", ids)
                    .execute()
                )
                total = sum(
                    float(r["grading_cost"]) for r in (agr.data or [])
                    if r.get("grading_cost") is not None
                )
                out["total_grading_cost"] = round(total, 4)
            else:
                out["total_grading_cost"] = 0.0
        except Exception:
            out["total_grading_cost"] = 0.0
        if "total_grading_cost" not in out:
            out["total_grading_cost"] = 0.0
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Custom metrics (definitions per org; values come from feedback endpoint)
# ---------------------------------------------------------------------------

class CustomMetricCreate(BaseModel):
    org_id: str
    name: str
    key: str
    description: Optional[str] = None
    metric_type: str = "number"
    direction: str = "higher_is_better"

    class Config:
        extra = "ignore"


@router.get("/custom-metrics")
async def list_custom_metrics(org_id: str):
    """List custom metric definitions for an org."""
    try:
        result = (
            supabase.table("custom_metrics")
            .select("id, org_id, name, key, description, metric_type, direction, created_at")
            .eq("org_id", org_id)
            .order("created_at", desc=False)
            .execute()
        )
        return list(result.data or [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-metrics")
async def create_custom_metric(payload: CustomMetricCreate):
    """Create a custom metric definition. key must be unique per org."""
    try:
        key = (payload.key or "").strip()
        name = (payload.name or "").strip()
        if not key or not name or not payload.org_id:
            raise HTTPException(status_code=400, detail="org_id, name, and key required.")
        metric_type = (payload.metric_type or "number").strip().lower()
        if metric_type not in ("number", "boolean", "category"):
            metric_type = "number"
        direction = (payload.direction or "higher_is_better").strip().lower()
        if direction not in ("higher_is_better", "lower_is_better"):
            direction = "higher_is_better"
        row = {
            "org_id": payload.org_id,
            "name": name,
            "key": key,
            "description": (payload.description or "").strip() or None,
            "metric_type": metric_type,
            "direction": direction,
        }
        result = supabase.table("custom_metrics").insert(row).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create custom metric")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/custom-metrics/{metric_id}")
async def delete_custom_metric(metric_id: str):
    """Delete a custom metric definition. Does not remove existing values from api_request_log."""
    try:
        supabase.table("custom_metrics").delete().eq("id", metric_id).execute()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Auto-graded metrics (AI evaluates every production response)
# ---------------------------------------------------------------------------

def _get_provider_for_model(model: str) -> str:
    """Map grading model name to provider for API key lookup."""
    if not model:
        return "openai"
    m = (model or "").lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m:
        return "openai"
    if "gemini" in m:
        return "gemini"
    return "openai"


def _org_has_provider_key(org_id: str, provider: str) -> bool:
    """Return True if org has at least one api_keys row for this provider (case-insensitive)."""
    try:
        r = (
            supabase.table("api_keys")
            .select("id, provider")
            .eq("org_id", org_id)
            .execute()
        )
        want = (provider or "").lower()
        for row in (r.data or []):
            if (row.get("provider") or "").lower() == want:
                return True
        return False
    except Exception:
        return False


class AutoGradedMetricCreate(BaseModel):
    org_id: str
    name: str
    key: str
    description: Optional[str] = None
    metric_type: str = "score"
    rubric: str = ""
    categories: Optional[List[str]] = None
    direction: str = "higher_is_better"
    grading_model: str = "gpt-4o-mini"
    endpoint_slugs: Optional[List[str]] = None
    enabled: bool = True
    sample_rate: float = 25.0

    class Config:
        extra = "ignore"


class AutoGradedMetricUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metric_type: Optional[str] = None
    rubric: Optional[str] = None
    categories: Optional[List[str]] = None
    direction: Optional[str] = None
    grading_model: Optional[str] = None
    endpoint_slugs: Optional[List[str]] = None
    enabled: Optional[bool] = None
    sample_rate: Optional[float] = None

    class Config:
        extra = "ignore"


@router.get("/auto-graded-metrics")
async def list_auto_graded_metrics(org_id: str):
    """List auto-graded metric definitions for an org."""
    try:
        result = (
            supabase.table("auto_graded_metrics")
            .select("id, org_id, name, key, description, metric_type, rubric, categories, direction, grading_model, endpoint_slugs, enabled, sample_rate, created_at, updated_at")
            .eq("org_id", org_id)
            .order("created_at", desc=False)
            .execute()
        )
        return list(result.data or [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-graded-metrics")
async def create_auto_graded_metric(payload: AutoGradedMetricCreate):
    """Create an auto-graded metric definition."""
    try:
        key = (payload.key or "").strip().lower().replace(" ", "_")
        name = (payload.name or "").strip()
        if not key or not name or not payload.org_id:
            raise HTTPException(status_code=400, detail="org_id, name, and key required.")
        metric_type = (payload.metric_type or "score").strip().lower()
        if metric_type not in ("score", "binary", "category"):
            metric_type = "score"
        direction = (payload.direction or "higher_is_better").strip().lower()
        if direction not in ("higher_is_better", "lower_is_better"):
            direction = "higher_is_better"
        row = {
            "org_id": payload.org_id,
            "name": name,
            "key": key,
            "description": (payload.description or "").strip() or None,
            "metric_type": metric_type,
            "rubric": (payload.rubric or "").strip() or "Evaluate the response.",
            "categories": payload.categories or [],
            "direction": direction,
            "grading_model": grading_model,
            "endpoint_slugs": payload.endpoint_slugs or [],
            "enabled": bool(payload.enabled),
            "sample_rate": min(100.0, max(0.01, float(payload.sample_rate if payload.sample_rate is not None else 25))),
        }
        result = supabase.table("auto_graded_metrics").insert(row).execute()
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create auto-graded metric")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/auto-graded-metrics/{metric_id}")
async def update_auto_graded_metric(metric_id: str, payload: AutoGradedMetricUpdate):
    """Update an auto-graded metric definition."""
    try:
        updates = {}
        if payload.name is not None:
            updates["name"] = payload.name.strip()
        if payload.description is not None:
            updates["description"] = payload.description.strip() or None
        if payload.metric_type is not None:
            updates["metric_type"] = payload.metric_type.strip().lower()
        if payload.rubric is not None:
            updates["rubric"] = payload.rubric.strip()
        if payload.categories is not None:
            updates["categories"] = payload.categories
        if payload.direction is not None:
            updates["direction"] = payload.direction.strip().lower()
        if payload.grading_model is not None:
            new_model = payload.grading_model.strip()
            provider = _get_provider_for_model(new_model)
            existing = supabase.table("auto_graded_metrics").select("org_id").eq("id", metric_id).single().execute()
            org_id = (existing.data or {}).get("org_id")
            if org_id and not _org_has_provider_key(org_id, provider):
                raise HTTPException(
                    status_code=400,
                    detail=f"No {provider} API key configured. Add one in settings to use {new_model} for grading.",
                )
            updates["grading_model"] = new_model
        if payload.endpoint_slugs is not None:
            updates["endpoint_slugs"] = payload.endpoint_slugs
        if payload.enabled is not None:
            updates["enabled"] = payload.enabled
        if payload.sample_rate is not None:
            updates["sample_rate"] = min(100.0, max(0.01, float(payload.sample_rate)))
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        supabase.table("auto_graded_metrics").update(updates).eq("id", metric_id).execute()
        updated = supabase.table("auto_graded_metrics").select("*").eq("id", metric_id).single().execute()
        return updated.data if updated.data else {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/auto-graded-metrics/{metric_id}")
async def delete_auto_graded_metric(metric_id: str):
    """Delete an auto-graded metric definition."""
    try:
        supabase.table("auto_graded_metrics").delete().eq("id", metric_id).execute()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/timeseries")
async def get_experiment_timeseries(
    experiment_id: str,
    metric: str = "error_rate",
    bucket: str = "1h",
):
    """
    Bucketed metrics per variant for experiment. bucket: 15m, 1h, 6h, 1d.
    Returns buckets with control/candidate request counts and metric values.
    """
    from datetime import datetime, timezone, timedelta
    try:
        exp = (
            supabase.table("experiments")
            .select("id, created_at, variants")
            .eq("id", experiment_id)
            .single()
            .execute()
        )
        if not exp.data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        variants = exp.data.get("variants") or []
        if len(variants) < 2:
            raise HTTPException(status_code=400, detail="Experiment has no variants")
        created = exp.data.get("created_at")
        since = created
        if not since:
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        runs = (
            supabase.table("workflow_runs")
            .select("variant_name, total_latency_ms, total_cost, node_results, created_at")
            .eq("experiment_id", experiment_id)
            .gte("created_at", since)
            .order("created_at", desc=False)
            .execute()
        )
        rows = runs.data or []

        # Map bucket interval to PostgreSQL date_trunc
        trunc_map = {"15m": "hour", "1h": "hour", "6h": "hour", "1d": "day"}
        trunc = trunc_map.get(bucket, "hour")

        from collections import defaultdict
        by_bucket: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            ts = r.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if trunc == "day":
                    key = dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                else:
                    key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
            except Exception:
                continue
            vname = (r.get("variant_name") or "").strip() or "unknown"
            by_bucket[key][vname].append(r)

        control_name = variants[0].get("name") or f"v{variants[0].get('version', 0)}"
        candidate_name = (variants[1].get("name") or f"v{variants[1].get('version', 0)}") if len(variants) > 1 else "candidate"

        buckets_out = []
        for key in sorted(by_bucket.keys()):
            agg = by_bucket[key]
            control_runs = agg.get(control_name, [])
            candidate_runs = agg.get(candidate_name, [])

            def _stats(run_list):
                n = len(run_list)
                if n == 0:
                    return {"requests": 0, "error_rate": 0, "avg_latency_ms": 0, "avg_cost": 0}
                errs = sum(1 for r in run_list if any(
                    _nr_has_error(nr) for nr in (r.get("node_results") or [])
                ))
                lats = [r["total_latency_ms"] for r in run_list if r.get("total_latency_ms") is not None]
                costs = [float(r["total_cost"]) for r in run_list if r.get("total_cost") is not None]
                return {
                    "requests": n,
                    "error_rate": round(errs / n * 100, 2) if n > 0 else 0,
                    "avg_latency_ms": round(sum(lats) / len(lats), 0) if lats else 0,
                    "avg_cost": round(sum(costs) / len(costs), 6) if costs else 0,
                }

            control_stat = _stats(control_runs)
            candidate_stat = _stats(candidate_runs)
            buckets_out.append({
                "time": key,
                "control": control_stat,
                "candidate": candidate_stat,
            })
        return {"buckets": buckets_out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Set experiment to running and create weighted routing policy from variants."""
    try:
        result = (
            supabase.table("experiments")
            .select(_EXPERIMENT_COLS)
            .eq("id", experiment_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        row = result.data
        if (row.get("status") or "") != "draft":
            raise HTTPException(status_code=400, detail="Only draft experiments can be started")
        org_id = str(row["org_id"])
        slug = (row.get("endpoint_slug") or "").strip()
        variants = row.get("variants") or []
        existing = (
            supabase.table("routing_policies")
            .select("id")
            .eq("org_id", org_id)
            .eq("endpoint_slug", slug)
            .eq("active", True)
            .execute()
        )
        for r in existing.data or []:
            supabase.table("routing_policies").update({"active": False}).eq("id", r["id"]).execute()
        rules = [{"version": v.get("version"), "weight": v.get("weight", 0)} for v in variants]
        supabase.table("routing_policies").insert({
            "org_id": org_id,
            "endpoint_slug": slug,
            "policy_type": "weighted",
            "rules": rules,
            "active": True,
        }).execute()
        supabase.table("experiments").update({
            "status": "running",
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }).eq("id", experiment_id).execute()
        updated = supabase.table("experiments").select(_EXPERIMENT_COLS).eq("id", experiment_id).single().execute()
        return _experiment_to_response(updated.data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConcludeBody(BaseModel):
    winner_version: Optional[int] = None

    class Config:
        extra = "ignore"


async def _conclude_experiment_internal(experiment_id: str, row: dict, winner_version: int, reason: str) -> None:
    """Shared conclude logic: deactivate routing, promote winner if needed, update experiment."""
    org_id = str(row["org_id"])
    slug = (row.get("endpoint_slug") or "").strip()
    existing = (
        supabase.table("routing_policies")
        .select("id")
        .eq("org_id", org_id)
        .eq("endpoint_slug", slug)
        .eq("active", True)
        .execute()
    )
    for r in existing.data or []:
        supabase.table("routing_policies").update({"active": False}).eq("id", r["id"]).execute()
    latest = await get_latest_promoted_deployment(org_id, slug)
    if latest and int(latest.get("version", 0)) != winner_version:
        winner_dep = await get_promoted_deployment_by_version(org_id, slug, winner_version)
        if winner_dep and winner_dep.get("graph_json") is not None:
            workflow_id = str(winner_dep["workflow_id"])
            project_id = winner_dep.get("project_id")
            graph_json = winner_dep.get("graph_json") or {"nodes": [], "edges": []}
            next_ver_result = (
                supabase.table("workflow_deployments")
                .select("version")
                .eq("org_id", org_id)
                .eq("endpoint_slug", slug)
                .order("version", desc=True)
                .limit(1)
                .execute()
            )
            next_version = 1
            if next_ver_result.data and len(next_ver_result.data) > 0:
                next_version = int(next_ver_result.data[0]["version"]) + 1
            data = {
                "workflow_id": workflow_id,
                "org_id": org_id,
                "version": next_version,
                "endpoint_slug": slug,
                "graph_json": graph_json,
                "status": "candidate",
            }
            if project_id is not None:
                data["project_id"] = project_id
            insert_result = supabase.table("workflow_deployments").insert(data).execute()
            if insert_result.data:
                new_dep_id = str(insert_result.data[0]["id"])
                supabase.table("workflow_deployments").update({
                    "status": "promoted",
                    "promoted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "override_reason": "experiment_conclusion",
                }).eq("id", new_dep_id).execute()
    update = {
        "status": "concluded",
        "winner_version": winner_version,
        "concluded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "concluded_reason": reason,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    supabase.table("experiments").update(update).eq("id", experiment_id).execute()


async def _maybe_auto_conclude(experiment_id: str) -> None:
    """If experiment has auto_conclude, min sample reached, and primary metric significant, conclude with the better variant."""
    try:
        result = (
            supabase.table("experiments")
            .select(_EXPERIMENT_COLS)
            .eq("id", experiment_id)
            .single()
            .execute()
        )
        if not result.data:
            return
        row = result.data
        if (row.get("status") or "") != "running":
            return
        if not row.get("auto_conclude"):
            return
        variants = row.get("variants") or []
        if len(variants) < 2:
            return
        min_sample = int(row.get("min_sample_size") or 50)
        confidence_level = float(row.get("confidence_level") or 95.0)
        expected_names = [v.get("name") or f"v{v.get('version', 0)}" for v in variants if isinstance(v, dict)]
        computed = _compute_experiment_results(
            experiment_id, min_sample, expected_variant_names=expected_names, org_id=row.get("org_id"),
            confidence_level=confidence_level,
        )
        if not computed.get("min_sample_reached"):
            return
        primary_key = row.get("primary_metric") or "error_rate"
        primary_ci = (computed.get("metric_ci") or {}).get(primary_key)
        if not primary_ci or not primary_ci.get("significant"):
            return
        sorted_variants = sorted(variants, key=lambda v: int(v.get("version", 0)))
        control_version = int(sorted_variants[0].get("version", 0))
        candidate_version = int(sorted_variants[1].get("version", 0))
        primary_lower_is_better = primary_key in ("error_rate", "avg_latency_ms", "p95_latency_ms", "avg_cost")
        ctw = primary_ci.get("chance_to_win") or 0.5
        prob_candidate_better = (1 - ctw) if primary_lower_is_better else ctw
        winner_version = candidate_version if prob_candidate_better >= 0.5 else control_version
        await _conclude_experiment_internal(experiment_id, row, winner_version, "auto_significance")
    except Exception:
        pass


@router.put("/experiments/{experiment_id}/conclude")
async def conclude_experiment(experiment_id: str, body: Optional[ConcludeBody] = None):
    """Conclude experiment: set winner, deactivate routing policy. If winner is not latest promoted, create new deployment with winner's graph so it serves 100%."""
    winner_version = body.winner_version if body else None
    if winner_version is None:
        raise HTTPException(status_code=400, detail="winner_version is required")
    try:
        result = (
            supabase.table("experiments")
            .select(_EXPERIMENT_COLS)
            .eq("id", experiment_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        row = result.data
        if (row.get("status") or "") != "running":
            raise HTTPException(status_code=400, detail="Only running experiments can be concluded")
        variants = row.get("variants") or []
        variant_versions = [int(v["version"]) for v in variants if v.get("version") is not None]
        if winner_version not in variant_versions:
            raise HTTPException(status_code=400, detail=f"Version {winner_version} is not a variant in this experiment")
        await _conclude_experiment_internal(experiment_id, row, winner_version, "manual")
        updated = supabase.table("experiments").select(_EXPERIMENT_COLS).eq("id", experiment_id).single().execute()
        return _experiment_to_response(updated.data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _end_experiments_on_endpoint_sync(org_id: str, endpoint_slug: str, reason: str) -> None:
    """Cancel all running experiments on this endpoint and deactivate their routing. Used when deploying or rolling back."""
    now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    running = (
        supabase.table("experiments")
        .select("id")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("status", "running")
        .execute()
    )
    for exp in (running.data or []):
        supabase.table("experiments").update({
            "status": "cancelled",
            "concluded_at": now_str,
            "concluded_reason": reason,
            "updated_at": now_str,
        }).eq("id", exp["id"]).execute()
    policies = (
        supabase.table("routing_policies")
        .select("id")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("active", True)
        .execute()
    )
    for p in (policies.data or []):
        supabase.table("routing_policies").update({"active": False}).eq("id", p["id"]).execute()


@router.put("/experiments/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str):
    """Cancel experiment and deactivate its routing policy."""
    try:
        result = (
            supabase.table("experiments")
            .select(_EXPERIMENT_COLS)
            .eq("id", experiment_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        row = result.data
        if (row.get("status") or "") not in ("draft", "running"):
            raise HTTPException(status_code=400, detail="Experiment already concluded or cancelled")
        org_id = str(row["org_id"])
        slug = (row.get("endpoint_slug") or "").strip()
        existing = (
            supabase.table("routing_policies")
            .select("id")
            .eq("org_id", org_id)
            .eq("endpoint_slug", slug)
            .eq("active", True)
            .execute()
        )
        for r in existing.data or []:
            supabase.table("routing_policies").update({"active": False}).eq("id", r["id"]).execute()
        now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        supabase.table("experiments").update({
            "status": "cancelled",
            "concluded_at": now_str,
            "concluded_reason": "manual",
            "updated_at": now_str,
        }).eq("id", experiment_id).execute()
        updated = supabase.table("experiments").select(_EXPERIMENT_COLS).eq("id", experiment_id).single().execute()
        return _experiment_to_response(updated.data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Rollback rules (automatic rollback) — Phase 5
# ---------------------------------------------------------------------------

_ROLLBACK_RULE_COLS = "id, org_id, endpoint_slug, enabled, conditions, action, last_triggered_at, last_checked_at, created_at, updated_at"


def _rollback_rule_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "endpoint_slug": row.get("endpoint_slug") or "",
        "enabled": bool(row.get("enabled", True)),
        "conditions": row.get("conditions") if isinstance(row.get("conditions"), list) else [],
        "action": row.get("action") or "rollback",
        "last_triggered_at": row.get("last_triggered_at"),
        "last_checked_at": row.get("last_checked_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


class RollbackConditionModel(BaseModel):
    metric: str  # error_rate | latency_p95 | cost_per_request | sample_size
    operator: str  # gt | gte | lt | lte
    threshold: float
    window_minutes: Optional[int] = 60

    class Config:
        extra = "ignore"


class RollbackRuleCreate(BaseModel):
    org_id: str
    endpoint_slug: str
    enabled: bool = True
    conditions: List[dict]  # list of {metric, operator, threshold, window_minutes}
    action: str = "rollback"  # rollback | alert_only | pause_traffic

    class Config:
        extra = "ignore"


class RollbackRuleUpdate(BaseModel):
    enabled: Optional[bool] = None
    conditions: Optional[List[dict]] = None
    action: Optional[str] = None

    class Config:
        extra = "ignore"


@router.get("/rollback-rules/{org_id}/{endpoint_slug}")
async def list_rollback_rules(org_id: str, endpoint_slug: str):
    """List rollback rules for an endpoint."""
    try:
        result = (
            supabase.table("rollback_rules")
            .select(_ROLLBACK_RULE_COLS)
            .eq("org_id", org_id)
            .eq("endpoint_slug", endpoint_slug.strip())
            .order("created_at", desc=True)
            .execute()
        )
        return [_rollback_rule_to_response(r) for r in (result.data or [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback-rules/detail/{rule_id}")
async def get_rollback_rule(rule_id: str):
    """Get a single rollback rule by id."""
    try:
        result = (
            supabase.table("rollback_rules")
            .select(_ROLLBACK_RULE_COLS)
            .eq("id", rule_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Rollback rule not found")
        return _rollback_rule_to_response(result.data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback-rules")
async def create_rollback_rule(payload: RollbackRuleCreate):
    """Create a rollback rule."""
    try:
        if not payload.conditions:
            raise HTTPException(status_code=400, detail="At least one condition is required")
        data = {
            "org_id": payload.org_id,
            "endpoint_slug": payload.endpoint_slug.strip(),
            "enabled": payload.enabled,
            "conditions": payload.conditions,
            "action": payload.action or "rollback",
        }
        result = supabase.table("rollback_rules").insert(data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Insert failed")
        return _rollback_rule_to_response(result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rollback-rules/{rule_id}")
async def update_rollback_rule(rule_id: str, payload: RollbackRuleUpdate):
    """Update a rollback rule."""
    try:
        result = (
            supabase.table("rollback_rules")
            .select(_ROLLBACK_RULE_COLS)
            .eq("id", rule_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Rollback rule not found")
        update = {"updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        if payload.enabled is not None:
            update["enabled"] = payload.enabled
        if payload.conditions is not None:
            update["conditions"] = payload.conditions
        if payload.action is not None:
            update["action"] = payload.action
        out = (
            supabase.table("rollback_rules")
            .update(update)
            .eq("id", rule_id)
            .execute()
        )
        return _rollback_rule_to_response(out.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rollback-rules/{rule_id}", status_code=204)
async def delete_rollback_rule(rule_id: str):
    """Delete a rollback rule."""
    try:
        supabase.table("rollback_rules").delete().eq("id", rule_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_metrics_for_version_in_window_sync(
    org_id: str, endpoint_slug: str, version: int, window_minutes: int
) -> dict:
    """Aggregate workflow_runs for one version in the last window_minutes. Returns dict with error_rate (0-100), latency_p95, cost_per_request, sample_size."""
    from datetime import timedelta
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    since_str = since.isoformat().replace("+00:00", "Z")
    rows = (
        supabase.table("workflow_runs")
        .select("id, served_version, version, total_latency_ms, total_cost, node_results")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug.strip())
        .eq("execution_mode", "production")
        .gte("created_at", since_str)
        .execute()
    ).data or []
    # Filter by version: prefer served_version, fallback to version
    filtered = [
        r for r in rows
        if (r.get("served_version") == version or (r.get("served_version") is None and r.get("version") == version))
    ]
    n = len(filtered)
    errors = 0
    latencies = []
    costs = []
    for r in filtered:
        node_results = r.get("node_results") or []
        if any(_nr_has_error(nr) for nr in node_results):
            errors += 1
        lat = r.get("total_latency_ms")
        if lat is not None:
            latencies.append(int(lat))
        c = r.get("total_cost")
        if c is not None:
            costs.append(float(c))
    lat_sorted = sorted(latencies) if latencies else []
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if len(lat_sorted) > 1 else (lat_sorted[0] if lat_sorted else 0)
    total_cost = sum(costs)
    avg_cost = total_cost / n if n > 0 else 0
    error_rate_pct = (errors / n * 100) if n > 0 else 0
    return {
        "error_rate": error_rate_pct,
        "latency_p95": p95,
        "cost_per_request": avg_cost,
        "sample_size": n,
    }


def _evaluate_rollback_conditions_sync(rule: dict) -> bool:
    """Returns True if any condition in the rule is breached (so rollback should trigger)."""
    org_id = str(rule["org_id"])
    endpoint_slug = (rule.get("endpoint_slug") or "").strip()
    conditions = rule.get("conditions") or []
    if not conditions:
        return False
    # Current version to evaluate = latest promoted (the one we might roll back from)
    latest = (
        supabase.table("workflow_deployments")
        .select("version, workflow_id")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("status", "promoted")
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    if not latest.data or len(latest.data) == 0:
        return False
    current_version = int(latest.data[0]["version"])
    # Use max window from conditions so we have one metric set
    window_minutes = max((c.get("window_minutes") or 60 for c in conditions), default=60)
    metrics = _get_metrics_for_version_in_window_sync(org_id, endpoint_slug, current_version, window_minutes)
    for c in conditions:
        metric = (c.get("metric") or "").strip()
        op = (c.get("operator") or "gt").strip()
        threshold = float(c.get("threshold", 0))
        val = None
        if metric == "error_rate":
            val = metrics["error_rate"]
        elif metric == "latency_p95":
            val = metrics["latency_p95"]
        elif metric == "cost_per_request":
            val = metrics["cost_per_request"]
        elif metric == "sample_size":
            val = metrics["sample_size"]
        else:
            continue
        breached = False
        if op == "gt":
            breached = val > threshold
        elif op == "gte":
            breached = val >= threshold
        elif op == "lt":
            breached = val < threshold
        elif op == "lte":
            breached = val <= threshold
        if breached:
            return True
    return False


def _execute_automatic_rollback_sync(rule: dict) -> None:
    """Create a new deployment from the previous promoted version and promote it; deactivate routing for endpoint."""
    org_id = str(rule["org_id"])
    endpoint_slug = (rule.get("endpoint_slug") or "").strip()
    rule_id = str(rule["id"])
    # Get two latest promoted versions (current and previous)
    promoted = (
        supabase.table("workflow_deployments")
        .select("id, version, workflow_id, project_id, graph_json")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("status", "promoted")
        .order("version", desc=True)
        .limit(2)
        .execute()
    )
    if not promoted.data or len(promoted.data) < 2:
        return  # No previous version to roll back to
    current = promoted.data[0]
    previous = promoted.data[1]
    workflow_id = str(previous["workflow_id"])
    project_id = previous.get("project_id")
    graph_json = previous.get("graph_json") or {"nodes": [], "edges": []}
    current_version = int(current["version"])
    # Next version number
    next_ver_result = (
        supabase.table("workflow_deployments")
        .select("version")
        .eq("workflow_id", workflow_id)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    next_version = 1
    if next_ver_result.data and len(next_ver_result.data) > 0:
        next_version = int(next_ver_result.data[0]["version"]) + 1
    data = {
        "workflow_id": workflow_id,
        "org_id": org_id,
        "version": next_version,
        "endpoint_slug": endpoint_slug,
        "graph_json": graph_json,
        "status": "candidate",
        "rolled_back_from_version": current_version,
    }
    if project_id is not None:
        data["project_id"] = project_id
    insert_result = supabase.table("workflow_deployments").insert(data).execute()
    if not insert_result.data:
        return
    new_dep = insert_result.data[0]
    new_dep_id = str(new_dep["id"])
    # Promote immediately (override) so traffic switches
    supabase.table("workflow_deployments").update({
        "status": "promoted",
        "promoted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "override_reason": "automatic_rollback",
    }).eq("id", new_dep_id).execute()
    # Deactivate any active routing policy for this endpoint
    policies = (
        supabase.table("routing_policies")
        .select("id")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("active", True)
        .execute()
    )
    for p in (policies.data or []):
        supabase.table("routing_policies").update({"active": False}).eq("id", p["id"]).execute()
    _end_experiments_on_endpoint_sync(org_id, endpoint_slug, "rollback")
    # Update rule last_triggered_at and last_checked_at
    now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    supabase.table("rollback_rules").update({
        "last_triggered_at": now_str,
        "last_checked_at": now_str,
        "updated_at": now_str,
    }).eq("id", rule_id).execute()


async def run_rollback_monitor_cycle() -> dict:
    """Run one cycle: evaluate all enabled rollback rules and execute rollback when action=rollback. Returns summary."""
    result = {"checked": 0, "triggered": 0, "errors": []}
    try:
        rules_result = (
            supabase.table("rollback_rules")
            .select(_ROLLBACK_RULE_COLS)
            .eq("enabled", True)
            .execute()
        )
        rules = rules_result.data or []
        result["checked"] = len(rules)
        for rule in rules:
            try:
                triggered = await asyncio.to_thread(_evaluate_rollback_conditions_sync, rule)
                now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                supabase.table("rollback_rules").update({"last_checked_at": now_str}).eq("id", rule["id"]).execute()
                if triggered and (rule.get("action") or "rollback") == "rollback":
                    await asyncio.to_thread(_execute_automatic_rollback_sync, rule)
                    result["triggered"] += 1
            except Exception as e:
                result["errors"].append({"rule_id": str(rule.get("id")), "error": str(e)})
    except Exception as e:
        result["errors"].append({"cycle": str(e)})
    return result


@router.post("/rollback-monitor/run")
async def trigger_rollback_monitor():
    """Trigger one rollback monitor cycle (e.g. from cron every 60s). Returns summary."""
    summary = await run_rollback_monitor_cycle()
    return summary


# ---------------------------------------------------------------------------
# Control plane (Phase 6) — endpoint inventory + routing/experiments/safety summary
# ---------------------------------------------------------------------------

@router.get("/control-plane/endpoints")
async def control_plane_endpoints(org_id: str):
    """List endpoints for the org with routing, experiments, and rollback summary. For control plane UI."""
    try:
        # All promoted deployments for org (we'll take latest per endpoint_slug)
        dep_result = (
            supabase.table("workflow_deployments")
            .select("id, workflow_id, endpoint_slug, version, status")
            .eq("org_id", org_id)
            .eq("status", "promoted")
            .execute()
        )
        rows = dep_result.data or []
        # Latest version per endpoint (by endpoint_slug, max version)
        by_slug: dict = {}
        for r in rows:
            slug = (r.get("endpoint_slug") or "").strip()
            if not slug:
                continue
            ver = int(r.get("version") or 0)
            if slug not in by_slug or ver > by_slug[slug]["version"]:
                by_slug[slug] = {"workflow_id": str(r["workflow_id"]), "endpoint_slug": slug, "version": ver, "deployment_id": str(r["id"])}
        workflow_ids = list({e["workflow_id"] for e in by_slug.values()})
        # Workflow names
        names_by_wf: dict = {}
        if workflow_ids:
            wf_result = supabase.table("workflows").select("id, name").in_("id", workflow_ids).execute()
            for w in (wf_result.data or []):
                names_by_wf[str(w["id"])] = w.get("name") or "Untitled"
        # Active routing per endpoint
        routing_result = (
            supabase.table("routing_policies")
            .select("endpoint_slug, policy_type, rules, active")
            .eq("org_id", org_id)
            .eq("active", True)
            .execute()
        )
        routing_by_slug = {}
        for r in (routing_result.data or []):
            slug = (r.get("endpoint_slug") or "").strip()
            routing_by_slug[slug] = {"policy_type": r.get("policy_type") or "weighted", "rules": r.get("rules") or []}
        # Running experiments count per endpoint
        exp_result = (
            supabase.table("experiments")
            .select("endpoint_slug, status")
            .eq("org_id", org_id)
            .eq("status", "running")
            .execute()
        )
        experiments_by_slug: dict = {}
        for e in (exp_result.data or []):
            slug = (e.get("endpoint_slug") or "").strip()
            experiments_by_slug[slug] = experiments_by_slug.get(slug, 0) + 1
        # Enabled rollback rules count per endpoint
        rr_result = (
            supabase.table("rollback_rules")
            .select("endpoint_slug, enabled")
            .eq("org_id", org_id)
            .eq("enabled", True)
            .execute()
        )
        rollback_by_slug: dict = {}
        for r in (rr_result.data or []):
            slug = (r.get("endpoint_slug") or "").strip()
            rollback_by_slug[slug] = rollback_by_slug.get(slug, 0) + 1
        # Build response
        endpoints = []
        for slug, e in by_slug.items():
            endpoints.append({
                "endpoint_slug": slug,
                "workflow_id": e["workflow_id"],
                "workflow_name": names_by_wf.get(e["workflow_id"], "Untitled"),
                "version": e["version"],
                "deployment_id": e["deployment_id"],
                "routing": routing_by_slug.get(slug),
                "running_experiments": experiments_by_slug.get(slug, 0),
                "rollback_rules": rollback_by_slug.get(slug, 0),
            })
        endpoints.sort(key=lambda x: (x["workflow_name"], x["endpoint_slug"]))
        return {"org_id": org_id, "endpoints": endpoints}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
