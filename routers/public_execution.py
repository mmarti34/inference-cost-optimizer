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
import re
import time
import uuid
from fastapi import APIRouter, HTTPException, Header, Request
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
    get_next_turn_number,
    save_conversation_turn,
    update_conversation_updated_at,
)
from routing.resolver import resolve_version, resolve_version_and_deployment, get_promoted_deployment_by_version
from api_request_logger import log_api_request, get_api_request_log_sync, update_api_request_log_metrics
from auto_grading import grade_response

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

        # 7b. Conversation: load history and build prefix when conversation_id is set
        conversation_prefix = None
        conversation_id = None
        if body.conversation_id and (body.conversation_id or "").strip():
            conv = get_or_create_conversation(ctx.org_id, dep_slug, (body.conversation_id or "").strip())
            if conv:
                conversation_id = str(conv.get("id") or body.conversation_id)
                turns = load_conversation_turns(conversation_id)
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
            variables,
            experiment_id,
            variant_name,
            resolved_version,
            conversation_prefix,
        )

        log_entry["http_status"] = 200
        log_entry["success"] = True
        log_entry["total_latency_ms"] = result.get("total_latency")
        log_entry["total_cost"] = result.get("total_cost")
        log_entry["workflow_run_id"] = str(result["run_id"]) if result.get("run_id") else None

        if conversation_id and result.get("final_output") is not None:
            n = get_next_turn_number(conversation_id)
            save_conversation_turn(conversation_id, n, "user", input_text, variables, request_id, resolved_version)
            save_conversation_turn(conversation_id, n + 1, "assistant", result["final_output"], None, request_id, resolved_version)
            update_conversation_updated_at(conversation_id)

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
