"""
Public production endpoint: POST /api/public/{org_slug}/{endpoint_slug}

Enterprise org-scoped routing: org_slug identifies the tenant; API key must belong to that org.
- Validate Authorization Bearer (service API key).
- Resolve org_slug → org_id; enforce key.org_id == org_id (403 if mismatch).
- Look up deployment by (key org_id, endpoint_slug); 404 if not found.
- No graph_json from client; only deployed workflow. Optional ?version=vN.
- Every request (success or failure) is logged to api_request_log for metrics.
"""
import asyncio
import logging
import re
import time
import uuid
from fastapi import APIRouter, HTTPException, Header, Request

logger = logging.getLogger(__name__)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from supabase_client import supabase
from api_key_validation import validate_api_key
from rate_limiting import check_and_increment_usage
from plan_enforcement import check_monthly_request_limit, increment_monthly_usage
from workflow_runtime import execute_workflow, validate_workflow_variables
from workflow_streaming import stream_workflow_async
from conversation_service import (
    get_or_create_conversation,
    load_conversation_turns,
    trim_history_to_fit,
    format_history_as_prefix,
    format_agent_history_as_prefix,
    compress_history_prefix,
    generate_history_summary,
    get_next_turn_number,
    save_conversation_turn,
    update_conversation_updated_at,
    AGENT_MAX_TOKENS,
    AGENT_MAX_TURNS,
)
from routing.resolver import resolve_version, resolve_version_and_deployment, get_promoted_deployment_by_version
from api_request_logger import log_api_request, get_api_request_log_sync, update_api_request_log_metrics
from auto_grading import grade_response
from synthetic_mind.observer import observe_request as sm_observe_request

router = APIRouter()


class FeedbackBody(BaseModel):
    """Body for POST /feedback: request_id from API response + custom metrics to merge."""
    request_id: str
    metrics: Optional[dict] = None

    class Config:
        extra = "ignore"


def _slugify(name: str) -> str:
    """Match frontend orgSlug fallback: trim, lower, \\s+ -> '-', [^a-z0-9-] -> '', -+ -> '-', strip('-')."""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


# In-memory TTL cache for org slug → org_id resolution.
# Avoids hitting the DB on every single API request for the same org.
import threading as _threading

_org_slug_cache: dict[str, tuple[str, float]] = {}
_org_slug_lock = _threading.Lock()
_ORG_SLUG_TTL = 300  # 5 minutes — orgs are rarely renamed


def get_org_id_from_slug(org_slug: str) -> Optional[str]:
    """
    Resolve org_slug to org id.
    1. Check in-memory cache (5-min TTL).
    2. Exact match on organizations.slug column (indexed).
    3. Fallback: match by slugified name — but only fetches id+name,
       no full table scan; limited to 200 rows as a safety cap.
    """
    if not org_slug or not org_slug.strip():
        return None
    want = org_slug.strip()

    now = time.time()
    with _org_slug_lock:
        entry = _org_slug_cache.get(want)
        if entry and now < entry[1]:
            return entry[0]

    try:
        # 1. Exact match on slug column (preferred, indexed)
        result = (
            supabase.table("organizations")
            .select("id")
            .eq("slug", want)
            .limit(1)
            .execute()
        )
        if result.data and len(result.data) > 0:
            org_id = result.data[0].get("id")
            with _org_slug_lock:
                _org_slug_cache[want] = (org_id, now + _ORG_SLUG_TTL)
            return org_id

        # 2. Fallback: match by slugified name.
        # Limit to 200 rows to prevent loading the entire org table.
        all_orgs = (
            supabase.table("organizations")
            .select("id, name")
            .limit(200)
            .execute()
        )
        if all_orgs.data:
            for row in all_orgs.data:
                name = (row.get("name") or "").strip() or row.get("id") or ""
                if _slugify(name) == want:
                    org_id = row.get("id")
                    with _org_slug_lock:
                        _org_slug_cache[want] = (org_id, now + _ORG_SLUG_TTL)
                    return org_id
    except Exception:
        pass
    return None


class PublicExecuteBody(BaseModel):
    """Request body: variables (object) or input_text. graph_json never accepted in production."""
    input_text: Optional[str] = ""
    variables: Optional[dict] = None
    stream: Optional[bool] = False
    conversation_id: Optional[str] = None

    class Config:
        extra = "ignore"


class ToolResultBody(BaseModel):
    """Body for POST /tool-result/{yield_id}: external client submitting a client-side tool result."""
    result: str
    latency_ms: Optional[int] = 0

    class Config:
        extra = "ignore"


@router.post("/tool-result/{yield_id}")
async def submit_tool_result(yield_id: str, body: ToolResultBody):
    """
    Resume a pending client-side tool execution.

    When a workflow agent calls a tool configured with ``execution: "client"``,
    the SSE stream emits a ``tool_yield`` event containing a ``yield_id``.
    The external client (e.g. an IDE like Cursor) executes the tool locally
    and POSTs the result here.  The agent loop then resumes with this result.
    """
    from workflow_runtime import resume_tool_yield
    success = resume_tool_yield(yield_id, body.result, body.latency_ms or 0)
    if not success:
        raise HTTPException(status_code=404, detail="No pending tool yield with this ID — it may have timed out or already been resumed.")
    return {"status": "resumed", "yield_id": yield_id}


def _resolve_user_selected_providers(
    graph_json: dict,
    variables: Optional[dict],
) -> dict:
    """
    Resolve nodes with provider set to "user_selected".

    When the workflow designer sets a node's provider to "user_selected", the
    API caller's end-user picks the provider/model through the designer's own
    UI.  The chosen values are passed as workflow variables named ``provider``
    and ``model``.  This function patches those nodes at execution time so the
    runtime sees a concrete provider/model.

    If no nodes use "user_selected", the graph is returned unchanged.
    """
    if not variables:
        return graph_json
    caller_provider = (variables.get("provider") or "").strip()
    caller_model = (variables.get("model") or "").strip()
    if not caller_provider and not caller_model:
        return graph_json

    ai_node_types = {
        "ai-step", "model", "agent", "tool_call",
        "vision_step", "image_gen_step", "tts_step", "stt_step", "embedding_step",
    }
    nodes = graph_json.get("nodes", [])
    patched = []
    changed = False
    for node in nodes:
        ntype = (node.get("type") or "").lower()
        data = node.get("data") or {}
        node_provider = (data.get("provider") or "").strip().lower()
        if ntype in ai_node_types and node_provider == "user_selected":
            data = dict(data)
            if caller_provider:
                data["provider"] = caller_provider
            if caller_model:
                data["modelName"] = caller_model
            patched.append({**node, "data": data})
            changed = True
        else:
            patched.append(node)
    return {**graph_json, "nodes": patched} if changed else graph_json


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


@router.post("/{org_slug}/{endpoint_slug}")
async def public_execute(
    org_slug: str,
    endpoint_slug: str,
    request: Request,
    body: PublicExecuteBody,
    version: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """
    Execute a deployed workflow by org_slug and endpoint_slug.
    Requires Authorization: Bearer <service_api_key>.
    Key must belong to the organization identified by org_slug (403 otherwise).
    Optional query: ?version=v3. Defaults to latest deployment.
    Every request is logged to api_request_log (fire-and-forget) for metrics.
    """
    request_start = time.time()
    request_id = str(uuid.uuid4())
    slug_clean = endpoint_slug.strip() if endpoint_slug else ""
    log_entry = {
        "id": request_id,
        "org_id": None,
        "endpoint_slug": slug_clean,
        "served_version": None,
        "experiment_id": None,
        "variant_name": None,
        "http_status": 500,
        "success": False,
        "error_type": None,
        "error_message": None,
        "total_latency_ms": None,
        "total_cost": None,
        "workflow_run_id": None,
    }

    # Track workflow context so error handlers can persist failed runs.
    # These are set after deployment resolution; remain None for early errors.
    _wf_id: str | None = None
    _dep_slug: str | None = None
    _dep_version: int | None = None
    _resolved_version: int | None = None
    _experiment_id: str | None = None
    _variant_name: str | None = None
    _execution_started = False  # True once execute_workflow() is called (it handles its own persistence)

    try:
        if not org_slug or not org_slug.strip():
            raise HTTPException(status_code=400, detail="Invalid org slug.")
        if not slug_clean:
            raise HTTPException(status_code=400, detail="Invalid endpoint slug.")

        # 1. Auth
        if not authorization or not authorization.startswith("Bearer "):
            log_entry["http_status"] = 401
            log_entry["error_type"] = "auth"
            log_entry["error_message"] = "Missing or invalid Authorization header"
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
        token = authorization.split(" ", 1)[1]
        ctx = validate_api_key(token)

        # 2. Resolve org_slug → org_id
        org_id_from_slug = get_org_id_from_slug(org_slug.strip())
        if not org_id_from_slug:
            raise HTTPException(status_code=404, detail="Organization not found.")
        log_entry["org_id"] = org_id_from_slug

        # 3. Enforce tenant boundary
        if org_id_from_slug != ctx.org_id:
            log_entry["http_status"] = 403
            log_entry["error_type"] = "auth"
            log_entry["error_message"] = "API key does not belong to this organization"
            raise HTTPException(
                status_code=403,
                detail="API key does not belong to this organization.",
            )

        # 4+5. Resolve version AND load deployment in one pass
        # (eliminates the redundant second DB query for the deployment)
        pinned_version = _parse_version(version)
        resolved_version, experiment_id, variant_name, deployment = await resolve_version_and_deployment(
            org_id=ctx.org_id,
            endpoint_slug=slug_clean,
            request_headers=dict(request.headers),
            pinned_version=pinned_version,
        )
        log_entry["served_version"] = resolved_version
        log_entry["experiment_id"] = str(experiment_id) if experiment_id else None
        log_entry["variant_name"] = variant_name

        graph_json = deployment.get("graph_json") or {"nodes": [], "edges": []}

        # Resolve "user_selected" provider nodes from caller variables
        variables_raw = body.variables if isinstance(body.variables, dict) else None
        graph_json = _resolve_user_selected_providers(graph_json, variables_raw)

        workflow_id = deployment.get("workflow_id")
        dep_version = deployment.get("version")
        dep_slug = (deployment.get("endpoint_slug") or "").strip() or slug_clean

        # Capture for error-handler persistence
        _wf_id = workflow_id
        _dep_slug = dep_slug
        _dep_version = dep_version
        _resolved_version = resolved_version
        _experiment_id = str(experiment_id) if experiment_id else None
        _variant_name = variant_name

        # 6. Rate limit (per-minute)
        check_and_increment_usage(ctx.org_id, dep_slug, ctx.rate_limit_per_minute)

        # 6b. Monthly request limit (plan enforcement)
        check_monthly_request_limit(ctx.org_id)
        increment_monthly_usage(ctx.org_id)

        # 7. Validate variables
        variables = body.variables if isinstance(body.variables, dict) else None
        if variables is not None and workflow_id:
            wf = supabase.table("workflows").select("variables").eq("id", workflow_id).single().execute()
            if wf.data and wf.data.get("variables") and isinstance(wf.data["variables"], list):
                variables = validate_workflow_variables(wf.data["variables"], variables)

        input_text = (body.input_text or "").strip() if body.input_text is not None else ""

        # 7c. SM scope key: resolve the context scope identifier from workflow config + variables
        # If the workflow builder configured a scope_key (e.g. "concert_id", "customer_id"),
        # extract the value from the caller's variables. This scopes SM consolidation per-entity.
        _sm_scope_key = None
        _sm_scope_value = None
        _gj_metadata = graph_json.get("metadata") or {}
        if isinstance(_gj_metadata, dict):
            _sm_scope_key = _gj_metadata.get("sm_scope_key") or _gj_metadata.get("scope_key")
        if not _sm_scope_key:
            # Also check top-level graph_json for scope_key
            _sm_scope_key = graph_json.get("sm_scope_key")
        if _sm_scope_key and variables and isinstance(variables, dict):
            _sm_scope_value = str(variables.get(_sm_scope_key, "")).strip() or None

        # 7d. SM Phase 4: replace large variable content with consolidated summary
        # When a scope_value is resolved and we have a consolidation for it,
        # swap out the raw variable data with the compact summary. This is where
        # token savings happen for non-KB, non-chat workflows (e.g. concert reviews).
        _sm_vars_replaced = False
        if _sm_scope_value and variables and isinstance(variables, dict):
            try:
                from synthetic_mind.memory_store import get_variable_consolidation, bump_variable_consolidation_hit
                _vc = get_variable_consolidation(ctx.org_id, dep_slug, _sm_scope_value)
                if _vc and _vc.get("consolidated_text"):
                    # Find the largest non-scope variable and replace it
                    _largest_var = None
                    _largest_len = 0
                    for vk, vv in variables.items():
                        if vk == _sm_scope_key or vk.startswith("_sm_"):
                            continue
                        vlen = len(str(vv))
                        if vlen > _largest_len and vlen > 200:
                            _largest_var = vk
                            _largest_len = vlen
                    if _largest_var:
                        _raw_tokens = max(1, _largest_len // 4)
                        _cons_tokens = max(1, len(_vc["consolidated_text"]) // 4)
                        # Only replace if it actually saves tokens (>30%)
                        if _cons_tokens < _raw_tokens * 0.7:
                            variables = dict(variables)
                            variables[_largest_var] = _vc["consolidated_text"]
                            _sm_vars_replaced = True
                            bump_variable_consolidation_hit(_vc["id"])
                            logger.info(
                                "SM variable consolidation: %s=%s, %d→%d tokens (%d%% saved)",
                                _sm_scope_key, _sm_scope_value[:20],
                                _raw_tokens, _cons_tokens,
                                int((1 - _cons_tokens / max(_raw_tokens, 1)) * 100),
                            )
            except Exception:
                pass  # Never block execution for SM failures

        # 7b. Conversation: load history and build prefix when conversation_id is set
        conversation_prefix = None
        conversation_id = None
        if body.conversation_id and (body.conversation_id or "").strip():
            conv = get_or_create_conversation(ctx.org_id, dep_slug, (body.conversation_id or "").strip())
            if conv:
                conversation_id = str(conv.get("id") or body.conversation_id)
                turns = load_conversation_turns(conversation_id)
                # Use richer formatting + higher limits for workflows with agent nodes
                _has_agent = any(
                    (n.get("type") or "").lower() == "agent"
                    for n in (graph_json.get("nodes") or [])
                )
                # SM v2: use compressed history (summary + recent turns) when available
                try:
                    if _has_agent:
                        conversation_prefix = compress_history_prefix(
                            turns, conversation_id,
                            max_tokens=AGENT_MAX_TOKENS, max_turns=AGENT_MAX_TURNS,
                            is_agent=True,
                        )
                    else:
                        conversation_prefix = compress_history_prefix(
                            turns, conversation_id,
                        )
                except Exception:
                    # Fallback to raw formatting if compression fails
                    if _has_agent:
                        trimmed = trim_history_to_fit(turns, max_tokens=AGENT_MAX_TOKENS, max_turns=AGENT_MAX_TURNS)
                        conversation_prefix = format_agent_history_as_prefix(trimmed)
                    else:
                        trimmed = trim_history_to_fit(turns)
                        conversation_prefix = format_history_as_prefix(trimmed)

        # 8. Stream or execute
        if body.stream:
            # Mark log_entry as streaming — the wrapper generator will update
            # the final status after the stream completes (success or error).
            log_entry["http_status"] = 200
            log_entry["success"] = True

            async def _streaming_wrapper():
                """Wrap stream_workflow_async to capture final status for logging."""
                saw_error = False
                try:
                    async for event in stream_workflow_async(
                        graph_json,
                        input_text,
                        ctx.org_id,
                        request_id,
                        resolved_version,
                        workflow_id,
                        dep_slug,
                        dep_version,
                        variables,
                        experiment_id,
                        variant_name,
                        conversation_prefix=conversation_prefix,
                        conversation_id=conversation_id,
                    ):
                        if event.startswith("event: error"):
                            saw_error = True
                            log_entry["http_status"] = 500
                            log_entry["success"] = False
                            log_entry["error_type"] = "execution"
                        yield event
                except Exception as exc:
                    saw_error = True
                    log_entry["http_status"] = 500
                    log_entry["success"] = False
                    log_entry["error_type"] = "execution"
                    log_entry["error_message"] = str(exc)[:500]
                    from workflow_streaming import _sse_event
                    yield _sse_event("error", {"message": str(exc), "request_id": request_id})
                    yield "data: [DONE]\n\n"
                finally:
                    log_entry["total_latency_ms"] = int((time.time() - request_start) * 1000)
                    if log_entry.get("org_id") is not None:
                        asyncio.create_task(log_api_request(log_entry))

            return StreamingResponse(
                _streaming_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 9. Execute (non-streaming)
        # Run synchronous workflow in a thread so we don't block the event loop
        # (provider calls can take seconds; blocking would stall all other requests).
        _execution_started = True
        # Inject SM scope value into variables for context_runtime to use
        _exec_vars = dict(variables) if variables else {}
        if _sm_scope_value:
            _exec_vars["_sm_scope_value"] = _sm_scope_value
        result = await asyncio.to_thread(
            execute_workflow,
            graph_json,
            input_text,
            ctx.org_id,
            "",  # user_id
            workflow_id,
            dep_slug,
            dep_version,
            "production",
            _exec_vars if _exec_vars else variables,
            experiment_id,
            variant_name,
            resolved_version,
            conversation_prefix,
            deployment.get("id") if deployment else None,  # deployment_id for context versioning
        )

        log_entry["http_status"] = 200
        log_entry["success"] = True
        log_entry["total_latency_ms"] = result.get("total_latency")
        log_entry["total_cost"] = result.get("total_cost")
        log_entry["workflow_run_id"] = str(result["run_id"]) if result.get("run_id") else None

        if conversation_id and result.get("final_output") is not None:
            n = get_next_turn_number(conversation_id)
            save_conversation_turn(conversation_id, n, "user", input_text, variables, request_id, resolved_version)
            # For agent workflows, store a summary of tool calls in the variables column
            assistant_vars = None
            for nr in (result.get("node_results") or []):
                if nr.get("type") == "agent" and nr.get("reasoning_steps"):
                    tool_summary = "; ".join(
                        f"{s.get('tool_name', '?')}({s.get('latency_ms', 0)}ms)"
                        for s in nr["reasoning_steps"] if s.get("type") == "act"
                    )
                    if tool_summary:
                        assistant_vars = {"agent_tool_summary": f"Tools used: {tool_summary}"}
                    break
            save_conversation_turn(conversation_id, n + 1, "assistant", result["final_output"], assistant_vars, request_id, resolved_version)
            update_conversation_updated_at(conversation_id)
            # SM v2: generate/update history summary async (for next request's compression)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, generate_history_summary, conversation_id)

        if result.get("final_output"):
            asyncio.create_task(
                grade_response(
                    request_log_id=request_id,
                    org_id=ctx.org_id,
                    endpoint_slug=slug_clean,
                    output=result["final_output"],
                    input_text=(body.input_text or "").strip() if body.input_text else "",
                )
            )

        # ── Synthetic Mind: observe this request/response asynchronously ──
        _node_results = result.get("node_results") or []
        _first_ai = next((n for n in _node_results if n.get("type") in ("ai-step", "model")), None)

        # SM Phase 4: capture the actual variable content for consolidation.
        # input_text is often just a command ("summarize"); the real data lives in variables.
        _sm_variable_content = None
        if _sm_scope_value and variables and isinstance(variables, dict):
            _largest_var = None
            _largest_len = 0
            for vk, vv in variables.items():
                if vk == _sm_scope_key or vk.startswith("_sm_"):
                    continue
                vlen = len(str(vv))
                if vlen > _largest_len and vlen > 200:
                    _largest_var = vk
                    _largest_len = vlen
            if _largest_var:
                _sm_variable_content = str(variables[_largest_var])[:5000]

        asyncio.create_task(
            sm_observe_request(
                request_id=request_id,
                org_id=ctx.org_id,
                workflow_id=workflow_id,
                endpoint_slug=slug_clean,
                input_text=input_text,
                output_text=result.get("final_output"),
                node_results=_node_results,
                total_cost=result.get("total_cost"),
                total_latency_ms=result.get("total_latency"),
                input_tokens=(_first_ai or {}).get("input_tokens", 0),
                output_tokens=(_first_ai or {}).get("tokens", 0),
                model=(_first_ai or {}).get("model"),
                provider=(_first_ai or {}).get("provider"),
                success=True,
                scope_key=_sm_scope_key,
                scope_value=_sm_scope_value,
                variable_content=_sm_variable_content,
            )
        )

        return {
            **result,
            "request_id": request_id,
            "served_version": resolved_version,
            "experiment_id": experiment_id,
            "variant_name": variant_name,
        }

    except HTTPException as e:
        log_entry["http_status"] = e.status_code
        log_entry["error_message"] = (e.detail if isinstance(e.detail, str) else str(e.detail or ""))[:500]
        if log_entry.get("error_type") is None:
            if e.status_code == 401:
                log_entry["error_type"] = "auth"
            elif e.status_code == 403:
                log_entry["error_type"] = "auth"
            elif e.status_code == 404:
                log_entry["error_type"] = "not_found"
            elif e.status_code == 429:
                log_entry["error_type"] = "rate_limit"
            else:
                log_entry["error_type"] = "execution"
        # ── Persist failed workflow_run so error appears in observability ──
        # Only if we resolved the workflow (have _wf_id) and the error is not
        # an auth/routing issue (those aren't workflow execution failures).
        _err_status = e.status_code
        if _wf_id and not _execution_started and _err_status not in (401, 403, 404, 429):
            try:
                _err_detail_str = (e.detail if isinstance(e.detail, str) else str(e.detail or ""))[:500]
                supabase.table("workflow_runs").insert({
                    "workflow_id": _wf_id,
                    "org_id": log_entry.get("org_id"),
                    "user_id": None,
                    "input_text": ((body.input_text or "")[:5000] if body.input_text else None),
                    "final_output": None,
                    "node_results": [{
                        "node_id": "pre_execution",
                        "type": "gateway",
                        "latency_ms": int((time.time() - request_start) * 1000),
                        "tokens": 0, "cost": 0, "cost_usd": 0,
                        "output": "",
                        "error": True,
                        "status": "error",
                        "error_detail": _err_detail_str,
                        "error_status": _err_status,
                    }],
                    "total_cost": 0,
                    "total_latency_ms": int((time.time() - request_start) * 1000),
                    "endpoint_slug": _dep_slug,
                    "version": _dep_version,
                    "execution_mode": "production",
                }).execute()
            except Exception:
                pass  # best-effort; api_request_log is the primary record
        raise
    except Exception as e:
        log_entry["http_status"] = 500
        log_entry["error_type"] = "execution"
        log_entry["error_message"] = str(e)[:500]
        # ── Persist failed workflow_run for non-HTTP errors too ──
        if _wf_id and not _execution_started:
            try:
                supabase.table("workflow_runs").insert({
                    "workflow_id": _wf_id,
                    "org_id": log_entry.get("org_id"),
                    "user_id": None,
                    "input_text": ((body.input_text or "")[:5000] if body.input_text else None),
                    "final_output": None,
                    "node_results": [{
                        "node_id": "pre_execution",
                        "type": "gateway",
                        "latency_ms": int((time.time() - request_start) * 1000),
                        "tokens": 0, "cost": 0, "cost_usd": 0,
                        "output": "",
                        "error": True,
                        "status": "error",
                        "error_detail": str(e)[:500],
                        "error_status": 500,
                    }],
                    "total_cost": 0,
                    "total_latency_ms": int((time.time() - request_start) * 1000),
                    "endpoint_slug": _dep_slug,
                    "version": _dep_version,
                    "execution_mode": "production",
                }).execute()
            except Exception:
                pass  # best-effort
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Streaming requests handle their own logging in _streaming_wrapper.
        if not body.stream:
            if log_entry.get("total_latency_ms") is None:
                log_entry["total_latency_ms"] = int((time.time() - request_start) * 1000)
            if log_entry.get("org_id") is not None:
                asyncio.create_task(log_api_request(log_entry))


@router.post("/{org_slug}/{endpoint_slug}/feedback")
async def public_feedback(
    org_slug: str,
    endpoint_slug: str,
    body: FeedbackBody,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Merge custom metrics into the api_request_log row for the given request_id.
    Requires Authorization: Bearer <service_api_key>. Used after API call to record feedback (e.g. helpfulness_score).
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    ctx = validate_api_key(token)
    org_id_from_slug = get_org_id_from_slug((org_slug or "").strip())
    if not org_id_from_slug:
        raise HTTPException(status_code=404, detail="Organization not found.")
    if str(org_id_from_slug) != str(ctx.org_id):
        raise HTTPException(status_code=403, detail="API key does not belong to this organization.")

    request_id = (body.request_id or "").strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id required.")
    metrics = body.metrics if isinstance(body.metrics, dict) else {}
    if not metrics:
        return {"status": "ok"}

    import asyncio
    loop = asyncio.get_event_loop()
    log_entry = await loop.run_in_executor(None, get_api_request_log_sync, request_id)
    if not log_entry:
        raise HTTPException(status_code=404, detail="Request not found.")
    if str(log_entry.get("org_id")) != str(ctx.org_id):
        raise HTTPException(status_code=404, detail="Request not found.")

    await update_api_request_log_metrics(request_id, metrics)
    return {"status": "ok"}
