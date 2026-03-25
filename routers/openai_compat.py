"""
OpenAI-compatible HTTP façade for deployed workflows.

Lets clients that expect POST /v1/chat/completions (e.g. OpenClaw, LiteLLM-style
proxies) run OptiML workflows using the same service API key as
POST /api/public/{org_slug}/{endpoint_slug}.

- Authorization: Bearer <service_api_key> (org inferred from key; no org in URL).
- model: workflow endpoint_slug (promoted deployment for that org).
- messages: mapped to a single input_text transcript for the workflow Input node.
- Optional header X-OptiML-Version: vN pins deployment version (same semantics as public API).

Streaming and tools are not supported in this v1 façade; use native OptiML APIs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api_key_validation import validate_api_key
from api_request_logger import log_api_request
from plan_enforcement import check_monthly_request_limit, increment_monthly_usage
from rate_limiting import check_and_increment_usage
from routing.resolver import resolve_version_and_deployment
from routers.public_execution import _resolve_user_selected_providers
from supabase_client import supabase
from workflow_runtime import execute_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])


def _openai_error(
    status: int,
    message: str,
    *,
    err_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }
    return JSONResponse(status_code=status, content=body)


def _parse_version_header(raw: Optional[str]) -> Optional[int]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.lower().startswith("v"):
        s = s[1:]
    try:
        return int(s)
    except ValueError:
        return None


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)


def messages_to_input_text(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages into one transcript for workflow input_text."""
    lines: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        text = _message_content_to_text(m.get("content"))
        if text:
            lines.append(f"{role}: {text}")
    return "\n\n".join(lines).strip()


def _sum_usage_tokens(node_results: Optional[list]) -> tuple[int, int, int]:
    """Returns (prompt_tokens, completion_tokens, total_tokens)."""
    prompt = 0
    completion = 0
    for nr in node_results or []:
        if not isinstance(nr, dict):
            continue
        pi = nr.get("input_tokens")
        po = nr.get("output_tokens")
        if pi is not None:
            try:
                prompt += int(pi)
            except (TypeError, ValueError):
                pass
        if po is not None:
            try:
                completion += int(po)
            except (TypeError, ValueError):
                pass
        if pi is None and po is None:
            t = nr.get("tokens")
            if t is not None:
                try:
                    completion += int(t)
                except (TypeError, ValueError):
                    pass
    total = prompt + completion
    return prompt, completion, total


class _ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""

    class Config:
        extra = "ignore"


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[_ChatMessage] = Field(default_factory=list)
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list] = None
    tool_choice: Any = None
    user: Optional[str] = None

    class Config:
        extra = "ignore"


def _normalize_model_id(model: str) -> str:
    """Allow optiml/<slug> or plain endpoint_slug."""
    m = (model or "").strip()
    if "/" in m:
        prefix, rest = m.split("/", 1)
        if prefix.lower() in ("optiml", "optiml-workflow") and rest.strip():
            return rest.strip()
    return m


@router.post("/chat/completions")
async def openai_chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    x_optiml_version: Optional[str] = Header(None, alias="X-OptiML-Version"),
):
    """
    OpenAI-compatible chat completions backed by a promoted workflow deployment.

    The service API key determines the organization. ``model`` is the
    deployment ``endpoint_slug`` for that org.
    """
    request_start = time.time()
    request_id = str(uuid.uuid4())
    log_entry: dict[str, Any] = {
        "id": request_id,
        "org_id": None,
        "endpoint_slug": "",
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
        "custom_metrics": {"interface": "openai_compat_v1"},
    }

    try:
        if not authorization or not authorization.startswith("Bearer "):
            log_entry["http_status"] = 401
            log_entry["error_type"] = "auth"
            log_entry["error_message"] = "Missing or invalid Authorization"
            asyncio.create_task(log_api_request(log_entry))
            return _openai_error(
                401,
                "Missing or invalid Authorization header.",
                err_type="authentication_error",
            )

        if body.stream:
            log_entry["http_status"] = 400
            log_entry["error_type"] = "unsupported"
            log_entry["error_message"] = "stream=true not supported on /v1/chat/completions; use POST /api/public/... with stream"
            asyncio.create_task(log_api_request(log_entry))
            return _openai_error(
                400,
                "Streaming is not supported on the OpenAI-compatible endpoint yet. "
                "Use POST /api/public/{org_slug}/{endpoint_slug} with stream: true.",
                code="stream_not_supported",
            )

        if body.tools:
            log_entry["http_status"] = 400
            log_entry["error_type"] = "unsupported"
            log_entry["error_message"] = "tools not supported on openai compat"
            asyncio.create_task(log_api_request(log_entry))
            return _openai_error(
                400,
                "The tools parameter is not supported on /v1/chat/completions. "
                "Use an OptiML workflow with tool_call or agent nodes via the native public API.",
                code="tools_not_supported",
            )

        token = authorization.split(" ", 1)[1].strip()
        ctx = validate_api_key(token)
        log_entry["org_id"] = ctx.org_id

        raw_messages = [m.model_dump() for m in body.messages]
        if not raw_messages:
            asyncio.create_task(log_api_request(log_entry))
            return _openai_error(400, "messages must be a non-empty array.")

        input_text_preview = messages_to_input_text(raw_messages)
        if not input_text_preview:
            asyncio.create_task(log_api_request(log_entry))
            return _openai_error(400, "No usable text content in messages.")

        endpoint_slug = _normalize_model_id(body.model)
        if not endpoint_slug:
            asyncio.create_task(log_api_request(log_entry))
            return _openai_error(400, "model must be a non-empty endpoint slug.")

        log_entry["endpoint_slug"] = endpoint_slug

        pinned = _parse_version_header(x_optiml_version)
        resolved_version, experiment_id, variant_name, deployment = await resolve_version_and_deployment(
            org_id=ctx.org_id,
            endpoint_slug=endpoint_slug,
            request_headers=dict(request.headers),
            pinned_version=pinned,
        )
        log_entry["served_version"] = resolved_version
        log_entry["experiment_id"] = str(experiment_id) if experiment_id else None
        log_entry["variant_name"] = variant_name

        input_text = input_text_preview

        graph_json = deployment.get("graph_json") or {"nodes": [], "edges": []}
        variables: Optional[dict[str, Any]] = None
        graph_json = _resolve_user_selected_providers(graph_json, variables)

        workflow_id = deployment.get("workflow_id")
        dep_version = deployment.get("version")
        dep_slug = (deployment.get("endpoint_slug") or "").strip() or endpoint_slug

        check_and_increment_usage(ctx.org_id, dep_slug, ctx.rate_limit_per_minute)
        check_monthly_request_limit(ctx.org_id)
        increment_monthly_usage(ctx.org_id)

        if workflow_id:
            wf = supabase.table("workflows").select("variables").eq("id", workflow_id).single().execute()
            schema = (wf.data or {}).get("variables") if wf.data else None
            if isinstance(schema, list) and any(
                isinstance(v, dict) and v.get("required") is True for v in schema
            ):
                asyncio.create_task(log_api_request(log_entry))
                return _openai_error(
                    400,
                    "This workflow has required named variables; the OpenAI-compatible endpoint only sends a single "
                    "transcript as input_text. Use POST /api/public/{org_slug}/{endpoint_slug} with a variables object, "
                    "or mark variables optional.",
                    code="variables_not_supported",
                )

        result = await asyncio.to_thread(
            execute_workflow,
            graph_json,
            input_text,
            ctx.org_id,
            "",
            workflow_id,
            dep_slug,
            dep_version,
            "production",
            variables,
            experiment_id,
            variant_name,
            resolved_version,
            None,
            deployment.get("id") if deployment else None,
        )

        final = result.get("final_output")
        content = "" if final is None else (final if isinstance(final, str) else json.dumps(final))

        pt, ct, tt = _sum_usage_tokens(result.get("node_results"))
        if tt == 0 and (pt > 0 or ct > 0):
            tt = pt + ct

        log_entry["http_status"] = 200
        log_entry["success"] = True
        log_entry["total_latency_ms"] = result.get("total_latency_ms") or result.get("total_latency")
        log_entry["total_cost"] = result.get("total_cost")
        log_entry["workflow_run_id"] = str(result["run_id"]) if result.get("run_id") else None
        asyncio.create_task(log_api_request(log_entry))

        created = int(time.time())
        return {
            "id": f"chatcmpl-{request_id.replace('-', '')[:24]}",
            "object": "chat.completion",
            "created": created,
            "model": endpoint_slug,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            },
            "optiml_request_id": request_id,
            "optiml_served_version": resolved_version,
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            status = e.status_code
            detail = e.detail if isinstance(e.detail, str) else str(e.detail or "error")
            log_entry["http_status"] = status
            log_entry["error_message"] = detail[:500]
            if status == 404:
                log_entry["error_type"] = "not_found"
            elif status == 429:
                log_entry["error_type"] = "rate_limit"
            elif status in (401, 403):
                log_entry["error_type"] = "auth"
            else:
                log_entry["error_type"] = "execution"
            asyncio.create_task(log_api_request(log_entry))
            etype = "invalid_request_error"
            if status == 401:
                etype = "authentication_error"
            elif status == 403:
                etype = "permission_error"
            elif status == 404:
                etype = "invalid_request_error"
            return _openai_error(status, detail, err_type=etype)

        logger.exception("openai_compat chat completions failed")
        log_entry["http_status"] = 500
        log_entry["error_type"] = "execution"
        log_entry["error_message"] = str(e)[:500]
        asyncio.create_task(log_api_request(log_entry))
        return _openai_error(500, "Workflow execution failed.", err_type="api_error")
