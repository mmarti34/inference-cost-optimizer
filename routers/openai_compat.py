"""
``POST /v1/chat/completions`` — OptiML's OpenAI-compatible surface.

This is the endpoint that makes "don't rebuild anything, put OptiML in the path
and we'll start learning" true.  An existing application changes one line::

    client = OpenAI(base_url="https://api.optiml.one/v1", api_key="<optiml_service_key>")

…and keeps working, while OptiML observes cost, latency, workload and strategy.

═══════════════════════════════════════════════════════════════════════════════
THE MODE RULE — decidable from the request alone, no precedence guessing
═══════════════════════════════════════════════════════════════════════════════

``model`` decides the mode, and nothing else does:

    model = "optiml/<endpoint_slug>"   →  MODE 2: OptiML deployed workflow
    model = anything else              →  MODE 1: direct inference

* The ``optiml/`` namespace is RESERVED.  ``optiml-workflow/<slug>`` is accepted
  as a legacy alias of the same thing.
* A bare model id — ``gpt-4o``, ``claude-sonnet-4-5``, ``meta-llama/Llama-3.3-70B-Instruct-Turbo``
  — ALWAYS means direct inference to that provider.  There is no fall-through:
  a bare id that does not resolve to a provider model is a 400, never a
  workflow lookup.  A customer's production request can therefore never
  silently become someone's Studio workflow because a string collided.
* Optional header ``X-OptiML-Mode: direct|workflow`` is an ASSERTION, not a
  selector.  The mode is still computed from ``model``; a header that disagrees
  is a 400.  It exists so a caller can fail loudly if their config drifts, not
  so there are two ways to say the same thing.

Provider selection inside direct inference is likewise explicit-first:
``anthropic/claude-sonnet-4-5`` names the provider; a bare id is resolved
through ``utils.pricing`` (registry, then documented prefix heuristics).
See ``direct_inference.resolve_direct_model``.

═══════════════════════════════════════════════════════════════════════════════
BACKWARD COMPATIBILITY — BREAKING CHANGE, stated loudly
═══════════════════════════════════════════════════════════════════════════════
Before this change a bare ``model`` was interpreted as a deployment
``endpoint_slug``.  It now means direct inference.  A bare model that fails
provider resolution but DOES match one of the org's deployments returns a 400
naming the exact replacement (``optiml/<slug>``) instead of a confusing
"unknown model" — so any caller that existed gets a one-line fix, not silence.
``optiml/<slug>`` worked before this change too, so callers already using the
prefix are unaffected.

═══════════════════════════════════════════════════════════════════════════════
OPTIML METADATA (optional — discovery must work without it)
═══════════════════════════════════════════════════════════════════════════════
``workload``, ``user_id``, ``conversation_id``, ``experiment_tags`` may be sent
as ``metadata.optiml`` in the body or as ``X-OptiML-*`` headers.  Headers matter
because several SDKs reject unknown top-level body fields.  Nothing here is
required: without it, workload identity is derived structurally. That identity
lives in ``optimization/workloads.py`` (``direct_inference_identity`` +
``resolve_workload``) — this router computes none of its own.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

import direct_inference
from api_key_validation import validate_api_key
from api_request_logger import log_api_request, log_api_request_sync
from direct_inference import DirectInferenceError
from optimization.attempts import sum_usage_tokens
from optimization_bridge import (
    DirectInferenceAttempt,
    describe_workload,
    record_direct_inference_attempt,
)
from plan_enforcement import check_monthly_request_limit, increment_monthly_usage
from rate_limiting import check_and_increment_usage
from routing.resolver import (
    resolve_version_and_deployment,
    NO_PROMOTED_DEPLOYMENT_DETAIL as _WORKFLOW_NOT_FOUND_DETAIL,
)
from routers.public_execution import _resolve_user_selected_providers
from supabase_client import supabase
from workflow_runtime import execute_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

MODE_DIRECT = "direct"
MODE_WORKFLOW = "workflow"

#: Reserved namespaces that route to an OptiML deployment. Nothing else does.
WORKFLOW_PREFIXES = ("optiml", "optiml-workflow")

#: Rate-limit bucket for direct inference. Distinct from any endpoint slug so a
#: customer's direct traffic cannot consume a workflow endpoint's budget.
_DIRECT_RATE_BUCKET = "_direct_inference"

REQUEST_ID_HEADER = "X-Request-Id"
OPTIML_REQUEST_ID_HEADER = "X-OptiML-Request-Id"
OPTIML_MODE_HEADER = "X-OptiML-Mode"


# ═════════════════════════════════════════════════════════════════════════════
# OpenAI error envelope
# ═════════════════════════════════════════════════════════════════════════════
def _error_body(
    message: str, err_type: str, code: Optional[str], param: Optional[str]
) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


def _openai_error(
    status: int,
    message: str,
    *,
    err_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    headers = _id_headers(request_id) if request_id else None
    return JSONResponse(
        status_code=status,
        content=_error_body(message, err_type, code, param),
        headers=headers,
    )


def _id_headers(request_id: str) -> dict[str, str]:
    return {REQUEST_ID_HEADER: request_id, OPTIML_REQUEST_ID_HEADER: request_id}


def _error_from_direct(err: DirectInferenceError, request_id: str) -> JSONResponse:
    return _openai_error(
        err.status_code,
        err.message,
        err_type=err.err_type,
        code=err.code,
        param=err.param,
        request_id=request_id,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Mode resolution
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ModeDecision:
    mode: str
    endpoint_slug: Optional[str]  # set only for MODE_WORKFLOW
    model: str  # the model string minus any optiml/ prefix


def resolve_mode(model: str, asserted_mode: Optional[str] = None) -> ModeDecision:
    """
    Decide direct-inference vs deployed-workflow from the request alone.

    Deterministic and total: every non-empty ``model`` maps to exactly one mode,
    with no lookups, no I/O and no precedence between competing signals.
    """
    raw = (model or "").strip()
    if not raw:
        raise DirectInferenceError(
            400, "model is required.", code="model_required", param="model"
        )

    decision = ModeDecision(MODE_DIRECT, None, raw)
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix.lower() in WORKFLOW_PREFIXES:
            slug = rest.strip()
            if not slug:
                raise DirectInferenceError(
                    400,
                    "model 'optiml/' must be followed by a deployment endpoint slug, "
                    "e.g. 'optiml/support-triage'.",
                    code="invalid_workflow_model",
                    param="model",
                )
            decision = ModeDecision(MODE_WORKFLOW, slug, slug)

    asserted = (asserted_mode or "").strip().lower()
    if asserted:
        if asserted not in (MODE_DIRECT, MODE_WORKFLOW):
            raise DirectInferenceError(
                400,
                f"{OPTIML_MODE_HEADER} must be 'direct' or 'workflow'.",
                code="invalid_mode_header",
            )
        if asserted != decision.mode:
            raise DirectInferenceError(
                400,
                (
                    f"{OPTIML_MODE_HEADER}: {asserted} contradicts model "
                    f"'{raw}', which is {decision.mode} inference. The model "
                    "string decides the mode; the header only asserts it. Use "
                    "'optiml/<endpoint_slug>' for a workflow, or a provider "
                    "model id for direct inference."
                ),
                code="mode_conflict",
                param="model",
            )
    return decision


# ═════════════════════════════════════════════════════════════════════════════
# OptiML metadata (body + headers)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class OptimlMetadata:
    workload: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    experiment_tags: list[str] = field(default_factory=list)
    source: str = "none"  # where the values came from, for debugging

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "experiment_tags": self.experiment_tags,
            "source": self.source,
        }


_META_FIELDS = ("workload", "user_id", "conversation_id", "experiment_tags")


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()][:20]
    if isinstance(value, (list, tuple)):
        return [str(t).strip() for t in value if str(t).strip()][:20]
    return []


def _meta_sources(body: dict[str, Any], headers: dict[str, str]) -> list[tuple[str, dict]]:
    """
    Ordered candidate sources. First source that supplies a field wins.

    1. ``metadata.optiml``          — the documented form (object, or a JSON
       string for clients that force ``metadata`` values to be strings).
    2. ``metadata.optiml_<field>``  — flat fallback for the same reason.
    3. top-level ``optiml``         — for non-OpenAI HTTP callers.
    4. ``X-OptiML-*`` headers       — for SDKs that reject unknown body fields.
    """
    out: list[tuple[str, dict]] = []
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("optiml")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except (ValueError, TypeError):
                nested = None
        if isinstance(nested, dict):
            out.append(("metadata.optiml", nested))
        flat = {
            k[len("optiml_"):]: v
            for k, v in metadata.items()
            if isinstance(k, str) and k.lower().startswith("optiml_")
        }
        if flat:
            out.append(("metadata.optiml_*", flat))
    if isinstance(body.get("optiml"), dict):
        out.append(("body.optiml", body["optiml"]))

    lowered = {k.lower(): v for k, v in headers.items()}
    header_meta = {
        "workload": lowered.get("x-optiml-workload"),
        "user_id": lowered.get("x-optiml-user-id"),
        "conversation_id": lowered.get("x-optiml-conversation-id"),
        "experiment_tags": lowered.get("x-optiml-experiment-tags"),
    }
    if any(v for v in header_meta.values()):
        out.append(("headers", header_meta))
    return out


def extract_optiml_metadata(body: dict[str, Any], headers: dict[str, str]) -> OptimlMetadata:
    """Collect optional OptiML request metadata. Absent metadata is normal."""
    resolved: dict[str, Any] = {}
    sources: list[str] = []
    for label, candidate in _meta_sources(body, headers):
        used = False
        for name in _META_FIELDS:
            if resolved.get(name):
                continue
            value = candidate.get(name)
            if value in (None, "", [], {}):
                continue
            resolved[name] = value
            used = True
        if used:
            sources.append(label)

    return OptimlMetadata(
        workload=str(resolved["workload"])[:200] if resolved.get("workload") else None,
        user_id=str(resolved["user_id"])[:200] if resolved.get("user_id") else None,
        conversation_id=(
            str(resolved["conversation_id"])[:200] if resolved.get("conversation_id") else None
        ),
        experiment_tags=_coerce_tags(resolved.get("experiment_tags")),
        source="+".join(sources) if sources else "none",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Request validation
# ═════════════════════════════════════════════════════════════════════════════
_VALID_ROLES = {"system", "developer", "user", "assistant", "tool", "function"}


def validate_messages(messages: Any) -> list[dict[str, Any]]:
    """Validate the ``messages`` array in OpenAI terms, with OpenAI-shaped errors."""
    if not isinstance(messages, list) or not messages:
        raise DirectInferenceError(
            400, "messages must be a non-empty array.", code="invalid_messages", param="messages"
        )
    out: list[dict[str, Any]] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise DirectInferenceError(
                400,
                f"messages[{index}] must be an object.",
                code="invalid_message",
                param="messages",
            )
        role = str(msg.get("role") or "").strip().lower()
        if role not in _VALID_ROLES:
            raise DirectInferenceError(
                400,
                f"messages[{index}].role must be one of "
                f"{', '.join(sorted(_VALID_ROLES))}; got {role or 'none'!r}.",
                code="invalid_role",
                param="messages",
            )
        if role == "tool" and not msg.get("tool_call_id"):
            raise DirectInferenceError(
                400,
                f"messages[{index}] has role 'tool' but no tool_call_id.",
                code="missing_tool_call_id",
                param="messages",
            )
        if (
            msg.get("content") in (None, "")
            and not msg.get("tool_calls")
            and role != "assistant"
        ):
            raise DirectInferenceError(
                400,
                f"messages[{index}].content is required for role '{role}'.",
                code="invalid_content",
                param="messages",
            )
        out.append(msg)
    return out


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


# NOTE: node_results / step-result parsing lives in exactly one place —
# optimization/attempts.py. Do not parse `node_results` here or anywhere else;
# see that module's header for why.


def _include_usage(body: dict[str, Any]) -> bool:
    opts = body.get("stream_options")
    return bool(isinstance(opts, dict) and opts.get("include_usage"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint
# ═════════════════════════════════════════════════════════════════════════════
@router.post("/chat/completions")
async def openai_chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_optiml_version: Optional[str] = Header(None, alias="X-OptiML-Version"),
    x_optiml_mode: Optional[str] = Header(None, alias=OPTIML_MODE_HEADER),
):
    request_start = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    log_entry: dict[str, Any] = {
        "id": str(uuid.uuid4()),
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
        "custom_metrics": {"interface": "openai_compat_v1", "request_id": request_id},
    }

    # ── Auth: service key → org. Unchanged, and never relaxed. ─────────────
    if not authorization or not authorization.startswith("Bearer "):
        log_entry["http_status"] = 401
        log_entry["error_type"] = "auth"
        log_entry["error_message"] = "Missing or invalid Authorization"
        return _openai_error(
            401,
            "Missing or invalid Authorization header.",
            err_type="authentication_error",
            request_id=request_id,
        )

    try:
        ctx = validate_api_key(authorization.split(" ", 1)[1].strip())
    except HTTPException as e:
        return _openai_error(
            e.status_code,
            str(e.detail),
            err_type="authentication_error" if e.status_code == 401 else "invalid_request_error",
            request_id=request_id,
        )
    log_entry["org_id"] = ctx.org_id

    try:
        body = await request.json()
    except Exception:
        return _openai_error(
            400, "Request body must be valid JSON.", code="invalid_json", request_id=request_id
        )
    if not isinstance(body, dict):
        return _openai_error(
            400, "Request body must be a JSON object.", code="invalid_json", request_id=request_id
        )

    headers = dict(request.headers)
    meta = extract_optiml_metadata(body, headers)
    log_entry["custom_metrics"]["optiml_metadata_source"] = meta.source

    try:
        decision = resolve_mode(body.get("model"), x_optiml_mode)
        messages = validate_messages(body.get("messages"))
    except DirectInferenceError as err:
        log_entry["http_status"] = err.status_code
        log_entry["error_type"] = "invalid_request"
        log_entry["error_message"] = err.message
        asyncio.create_task(log_api_request(log_entry))
        return _error_from_direct(err, request_id)

    log_entry["custom_metrics"]["mode"] = decision.mode
    stream = bool(body.get("stream"))

    # ── Rate limit + monthly quota apply to BOTH modes, before dispatch. ───
    # /v1/prompt skips the monthly quota entirely; that bug is not repeated here.
    # The monthly counter is INCREMENTED later, immediately before dispatch, so a
    # request rejected for a bad model or a missing deployment does not consume
    # the customer's quota. The per-minute limiter runs first regardless: it is
    # abuse protection, and rejected requests still cost us work.
    bucket = decision.endpoint_slug if decision.mode == MODE_WORKFLOW else _DIRECT_RATE_BUCKET
    try:
        check_and_increment_usage(ctx.org_id, bucket or _DIRECT_RATE_BUCKET, ctx.rate_limit_per_minute)
        check_monthly_request_limit(ctx.org_id)
    except HTTPException as e:
        log_entry["http_status"] = e.status_code
        log_entry["error_type"] = "rate_limit" if e.status_code == 429 else "quota"
        log_entry["error_message"] = str(e.detail)[:500]
        asyncio.create_task(log_api_request(log_entry))
        return _openai_error(
            e.status_code,
            str(e.detail),
            err_type="rate_limit_error" if e.status_code == 429 else "invalid_request_error",
            code="rate_limit_exceeded" if e.status_code == 429 else "quota_exceeded",
            request_id=request_id,
        )

    if decision.mode == MODE_DIRECT:
        return await _handle_direct(
            body=body,
            messages=messages,
            decision=decision,
            ctx=ctx,
            meta=meta,
            stream=stream,
            request_id=request_id,
            request_start=request_start,
            log_entry=log_entry,
        )

    return await _handle_workflow(
        request=request,
        body=body,
        messages=messages,
        decision=decision,
        ctx=ctx,
        meta=meta,
        stream=stream,
        request_id=request_id,
        request_start=request_start,
        log_entry=log_entry,
        pinned_version=_parse_version_header(x_optiml_version),
    )


# ═════════════════════════════════════════════════════════════════════════════
# MODE 1 — Direct inference
# ═════════════════════════════════════════════════════════════════════════════
def _system_prompt_of(messages: list[dict[str, Any]]) -> Optional[str]:
    parts = [
        _message_content_to_text(m.get("content"))
        for m in messages
        if str(m.get("role") or "").lower() in ("system", "developer")
    ]
    joined = "\n\n".join(p for p in parts if p).strip()
    return joined or None


def _build_attempt(
    *,
    request_id: str,
    ctx,
    resolved,
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    meta: OptimlMetadata,
    stream: bool,
) -> DirectInferenceAttempt:
    attempt = DirectInferenceAttempt(
        attempt_id=request_id,
        org_id=ctx.org_id,
        occurred_at=_now_iso(),
        provider=resolved.provider,
        model=resolved.model,
        requested_model=resolved.requested,
        system_prompt=_system_prompt_of(messages),
        tools=params.get("tools"),
        response_format=params.get("response_format"),
        explicit_workload=meta.workload,
        params={
            k: params[k]
            for k in ("temperature", "top_p", "max_tokens", "max_completion_tokens")
            if params.get(k) is not None
        },
        end_user_id=meta.user_id or (str(params.get("user")) if params.get("user") else None),
        conversation_id=meta.conversation_id,
        experiment_tags=meta.experiment_tags,
        streamed=stream,
    )
    # Pure, no I/O: safe to resolve identity on the request path so the response
    # and the request log can carry it. Durable resolution happens in
    # record_direct_inference_attempt, off the hot path.
    attempt.workload = describe_workload(attempt)
    return attempt


def _with_attempt(response: JSONResponse, attempt: DirectInferenceAttempt) -> JSONResponse:
    """
    Attach THE domain-model integration call site as a Starlette background task.

    It runs after the response bytes are sent, so its database I/O (workload
    resolution) never adds latency to a passthrough request — but Starlette
    still awaits it, so unlike a bare ``create_task`` it cannot be dropped when
    the request finishes.
    """
    response.background = BackgroundTask(record_direct_inference_attempt, attempt)
    return response


def _apply_cost(attempt: DirectInferenceAttempt, cost: Optional[float], estimated: bool, source: str) -> None:
    """
    Never present an estimate as measured spend.

    ``utils.pricing`` flags a model it has no real price for. When that happens
    the number goes in ``estimated_cost_usd`` and ``inference_cost_usd`` stays
    None, so cost aggregation can exclude or badge it rather than quietly
    treating a guess as revenue-grade data.
    """
    attempt.pricing_source = source
    attempt.cost_estimated = estimated
    if cost is None:
        return
    if estimated:
        attempt.estimated_cost_usd = cost
    else:
        attempt.inference_cost_usd = cost


def _finalize_log(
    log_entry: dict[str, Any],
    attempt: DirectInferenceAttempt,
    *,
    request_start: float,
) -> None:
    log_entry["http_status"] = attempt.http_status
    log_entry["success"] = attempt.success
    log_entry["error_type"] = attempt.error_type
    log_entry["error_message"] = attempt.error_message
    log_entry["total_latency_ms"] = attempt.duration_ms or int(
        (time.time() - request_start) * 1000
    )
    # total_cost carries MEASURED spend only. The estimate, when there is one,
    # lives in custom_metrics and is explicitly labelled.
    log_entry["total_cost"] = attempt.inference_cost_usd
    log_entry["endpoint_slug"] = log_entry.get("endpoint_slug") or ""
    log_entry["custom_metrics"].update(
        {
            "provider": attempt.provider,
            "model": attempt.model,
            "requested_model": attempt.requested_model,
            "workload_ref": attempt.workload.get("identity_ref"),
            "workload_identity_level": attempt.workload.get("identity_level"),
            "prompt_tokens": attempt.prompt_tokens,
            "completion_tokens": attempt.completion_tokens,
            "tokens_known": attempt.tokens_known,
            "cost_estimated": attempt.cost_estimated,
            "estimated_cost_usd": attempt.estimated_cost_usd,
            "pricing_source": attempt.pricing_source,
            "finish_reason": attempt.finish_reason,
            "streamed": attempt.streamed,
            "tool_call_count": attempt.tool_call_count,
        }
    )


async def _handle_direct(
    *,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    decision: ModeDecision,
    ctx,
    meta: OptimlMetadata,
    stream: bool,
    request_id: str,
    request_start: float,
    log_entry: dict[str, Any],
):
    try:
        resolved = direct_inference.resolve_direct_model(decision.model)
    except DirectInferenceError as err:
        err = _maybe_explain_workflow_collision(err, ctx.org_id, decision.model)
        log_entry["http_status"] = err.status_code
        log_entry["error_type"] = "invalid_request"
        log_entry["error_message"] = err.message
        asyncio.create_task(log_api_request(log_entry))
        return _error_from_direct(err, request_id)

    increment_monthly_usage(ctx.org_id)

    params = direct_inference.forwardable_params(body)
    log_entry["endpoint_slug"] = f"direct:{resolved.provider}"
    attempt = _build_attempt(
        request_id=request_id,
        ctx=ctx,
        resolved=resolved,
        params=params,
        messages=messages,
        meta=meta,
        stream=stream,
    )

    if stream:
        return _stream_direct(
            resolved=resolved,
            messages=messages,
            params=params,
            body=body,
            ctx=ctx,
            attempt=attempt,
            request_id=request_id,
            request_start=request_start,
            log_entry=log_entry,
        )

    try:
        result = await asyncio.to_thread(
            direct_inference.complete,
            resolved,
            messages,
            params,
            ctx.org_id,
            request_id=request_id,
        )
    except DirectInferenceError as err:
        attempt.success = False
        attempt.http_status = err.status_code
        attempt.error_type = err.err_type
        attempt.error_message = err.message[:500]
        attempt.duration_ms = int((time.time() - request_start) * 1000)
        _finalize_log(log_entry, attempt, request_start=request_start)
        asyncio.create_task(log_api_request(log_entry))
        return _with_attempt(_error_from_direct(err, request_id), attempt)
    except Exception as exc:
        logger.exception("direct inference failed")
        attempt.success = False
        attempt.http_status = 500
        attempt.error_type = "api_error"
        attempt.error_message = str(exc)[:500]
        attempt.duration_ms = int((time.time() - request_start) * 1000)
        _finalize_log(log_entry, attempt, request_start=request_start)
        asyncio.create_task(log_api_request(log_entry))
        return _with_attempt(
            _openai_error(
                500, "Direct inference failed.", err_type="api_error", request_id=request_id
            ),
            attempt,
        )

    attempt.prompt_tokens = result.prompt_tokens
    attempt.completion_tokens = result.completion_tokens
    attempt.tokens_known = result.tokens_known
    attempt.finish_reason = result.finish_reason
    attempt.provider_latency_ms = result.provider_latency_ms
    attempt.duration_ms = int((time.time() - request_start) * 1000)
    attempt.tool_call_count = result.tool_call_count
    _apply_cost(attempt, result.cost_usd, result.cost_estimated, result.pricing_source)

    _finalize_log(log_entry, attempt, request_start=request_start)
    asyncio.create_task(log_api_request(log_entry))

    payload = dict(result.body)
    payload["id"] = request_id
    payload["optiml"] = {
        "request_id": request_id,
        "mode": MODE_DIRECT,
        "provider": resolved.provider,
        "workload": attempt.workload.get("identity_ref"),
        "workload_identity_level": attempt.workload.get("identity_level"),
        "cost_usd": attempt.inference_cost_usd,
        "estimated_cost_usd": attempt.estimated_cost_usd,
        "cost_estimated": attempt.cost_estimated,
    }
    return _with_attempt(JSONResponse(content=payload, headers=_id_headers(request_id)), attempt)


def _maybe_explain_workflow_collision(
    err: DirectInferenceError, org_id: str, model: str
) -> DirectInferenceError:
    """
    Turn "unknown model" into a precise migration instruction when the string
    happens to be one of THIS org's deployment slugs.

    This is an error message only. The request is still refused — a bare model
    id never executes a workflow, which is the whole point of the mode rule.
    Scoped to the caller's own org, so it reveals nothing across the tenant
    boundary.
    """
    if err.code != "model_not_found":
        return err
    try:
        found = (
            supabase.table("workflow_deployments")
            .select("endpoint_slug")
            .eq("org_id", org_id)
            .eq("endpoint_slug", model)
            .limit(1)
            .execute()
        )
    except Exception:
        return err
    if not (found.data or []):
        return err
    return DirectInferenceError(
        400,
        (
            f"'{model}' is one of your OptiML deployments, but a bare model id "
            "always means direct inference on this endpoint — it never resolves "
            f"to a workflow. Use model='optiml/{model}' to call the deployment. "
            "(This changed: bare slugs used to mean 'workflow'. The reserved "
            "'optiml/' namespace makes the contract unambiguous.)"
        ),
        code="workflow_requires_optiml_prefix",
        param="model",
    )


def _stream_direct(
    *,
    resolved,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    body: dict[str, Any],
    ctx,
    attempt: DirectInferenceAttempt,
    request_id: str,
    request_start: float,
    log_entry: dict[str, Any],
) -> StreamingResponse:
    accounting = direct_inference.StreamAccounting()
    include_usage = _include_usage(body)

    def generate() -> Iterator[str]:
        try:
            for chunk in direct_inference.stream(
                resolved,
                messages,
                params,
                ctx.org_id,
                request_id=request_id,
                accounting=accounting,
                include_usage=include_usage,
            ):
                yield chunk
        finally:
            attempt.prompt_tokens = accounting.prompt_tokens
            attempt.completion_tokens = accounting.completion_tokens
            attempt.tokens_known = accounting.tokens_known
            attempt.finish_reason = accounting.finish_reason
            attempt.provider_latency_ms = accounting.provider_latency_ms
            attempt.duration_ms = int((time.time() - request_start) * 1000)
            attempt.tool_call_count = accounting.tool_call_count
            cost, estimated, source = direct_inference.price_call(
                resolved.provider,
                resolved.model,
                accounting.prompt_tokens,
                accounting.completion_tokens,
            )
            _apply_cost(attempt, cost, estimated, source)
            if accounting.error is not None:
                attempt.success = False
                attempt.http_status = accounting.error.status_code
                attempt.error_type = accounting.error.err_type
                attempt.error_message = accounting.error.message[:500]
            # THE domain-model integration call site (streaming path).
            record_direct_inference_attempt(attempt)
            _finalize_log(log_entry, attempt, request_start=request_start)
            log_api_request_sync(log_entry)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            **_id_headers(request_id),
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2 — OptiML deployed workflow
# ═════════════════════════════════════════════════════════════════════════════
async def _handle_workflow(
    *,
    request: Request,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    decision: ModeDecision,
    ctx,
    meta: OptimlMetadata,
    stream: bool,
    request_id: str,
    request_start: float,
    log_entry: dict[str, Any],
    pinned_version: Optional[int],
):
    endpoint_slug = decision.endpoint_slug or ""
    log_entry["endpoint_slug"] = endpoint_slug

    # A deployed workflow's execution parameters come from its graph.
    # workflow_runtime._execute_model_node sends only
    # (org_id, provider, model, prompt, prompt_id) to the router, so these
    # never reach a provider on this path. Accepting them would be false
    # compatibility: the caller would believe they took effect.
    dropped = [
        name
        for name in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "response_format", "stop", "seed")
        if body.get(name) is not None
    ]
    if dropped:
        err = DirectInferenceError(
            400,
            (
                f"{', '.join(dropped)} cannot be applied to an OptiML deployed "
                "workflow: a workflow's execution parameters are defined by its "
                "graph, and OptiML will not accept a parameter it would discard. "
                "Set them on the workflow's model node, or use a provider model "
                "id (direct inference), where they are passed through."
            ),
            code="param_not_supported_for_workflow",
            param=dropped[0],
        )
        log_entry["http_status"] = 400
        log_entry["error_type"] = "unsupported"
        log_entry["error_message"] = err.message
        asyncio.create_task(log_api_request(log_entry))
        return _error_from_direct(err, request_id)

    if body.get("tools"):
        # A deployed workflow's tools are part of the graph, chosen at design
        # time. Accepting caller-supplied tools would silently ignore them.
        err = DirectInferenceError(
            400,
            (
                "tools are not applicable to an OptiML deployed workflow: a "
                "workflow's tools are defined by its graph. Use a provider model "
                "id (direct inference) to pass your own tools, or add tool nodes "
                "to the workflow."
            ),
            code="tools_not_supported_for_workflow",
            param="tools",
        )
        log_entry["http_status"] = 400
        log_entry["error_type"] = "unsupported"
        log_entry["error_message"] = err.message
        asyncio.create_task(log_api_request(log_entry))
        return _error_from_direct(err, request_id)

    input_text = messages_to_input_text(messages)
    if not input_text:
        err = DirectInferenceError(
            400, "No usable text content in messages.", code="invalid_messages", param="messages"
        )
        asyncio.create_task(log_api_request(log_entry))
        return _error_from_direct(err, request_id)

    try:
        (
            resolved_version,
            experiment_id,
            variant_name,
            deployment,
        ) = await resolve_version_and_deployment(
            org_id=ctx.org_id,
            endpoint_slug=endpoint_slug,
            request_headers=dict(request.headers),
            pinned_version=pinned_version,
        )
    except HTTPException as e:
        log_entry["http_status"] = e.status_code
        log_entry["error_type"] = "not_found" if e.status_code == 404 else "execution"
        log_entry["error_message"] = str(e.detail)[:500]
        asyncio.create_task(log_api_request(log_entry))
        return _openai_error(
            e.status_code, str(e.detail), request_id=request_id, code="workflow_not_found"
        )

    # Tenant boundary: the resolver is org-scoped; assert it anyway.
    #
    # If this ever fires the resolver has a bug, but the ANSWER must still not
    # be a distinct one. A 403 saying "this deployment does not belong to your
    # organization" confirms that the slug names a real deployment in some
    # other tenant — the same enumeration oracle this surface closed on
    # /api/public/{org_slug}/{endpoint_slug}. Return the byte-identical 404 the
    # missing-deployment branch above returns; keep the specific reason in the
    # operator log, where it belongs.
    dep_org = deployment.get("org_id") if isinstance(deployment, dict) else None
    if dep_org and str(dep_org) != str(ctx.org_id):
        logger.error(
            "tenant boundary violation attempt on %s: resolver returned a "
            "deployment owned by org %s for a key scoped to org %s",
            endpoint_slug,
            dep_org,
            ctx.org_id,
        )
        log_entry["http_status"] = 404
        log_entry["error_type"] = "not_found"
        log_entry["error_message"] = _WORKFLOW_NOT_FOUND_DETAIL
        asyncio.create_task(log_api_request(log_entry))
        return _openai_error(
            404,
            _WORKFLOW_NOT_FOUND_DETAIL,
            request_id=request_id,
            code="workflow_not_found",
        )

    increment_monthly_usage(ctx.org_id)

    log_entry["served_version"] = resolved_version
    log_entry["experiment_id"] = str(experiment_id) if experiment_id else None
    log_entry["variant_name"] = variant_name
    log_entry["custom_metrics"]["workload_ref"] = endpoint_slug
    if meta.workload:
        log_entry["custom_metrics"]["optiml_workload"] = meta.workload

    graph_json = deployment.get("graph_json") or {"nodes": [], "edges": []}
    graph_json = _resolve_user_selected_providers(graph_json, None)
    workflow_id = deployment.get("workflow_id")
    dep_version = deployment.get("version")
    dep_slug = (deployment.get("endpoint_slug") or "").strip() or endpoint_slug

    if workflow_id:
        try:
            wf = supabase.table("workflows").select("variables").eq("id", workflow_id).single().execute()
            schema = (wf.data or {}).get("variables") if wf.data else None
        except Exception:
            schema = None
        if isinstance(schema, list) and any(
            isinstance(v, dict) and v.get("required") is True for v in schema
        ):
            err = DirectInferenceError(
                400,
                (
                    "This workflow has required named variables; the "
                    "OpenAI-compatible endpoint only sends a single transcript as "
                    "input_text. Use POST /api/public/{org_slug}/{endpoint_slug} "
                    "with a variables object, or mark the variables optional."
                ),
                code="variables_not_supported",
            )
            log_entry["http_status"] = 400
            log_entry["error_type"] = "unsupported"
            log_entry["error_message"] = err.message
            asyncio.create_task(log_api_request(log_entry))
            return _error_from_direct(err, request_id)

    try:
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
            None,
            experiment_id,
            variant_name,
            resolved_version,
            None,
            deployment.get("id") if deployment else None,
        )
    except HTTPException as e:
        log_entry["http_status"] = e.status_code
        log_entry["error_type"] = "execution"
        log_entry["error_message"] = str(e.detail)[:500]
        asyncio.create_task(log_api_request(log_entry))
        return _openai_error(e.status_code, str(e.detail), request_id=request_id)
    except Exception as exc:
        logger.exception("openai_compat workflow execution failed")
        log_entry["error_type"] = "execution"
        log_entry["error_message"] = str(exc)[:500]
        asyncio.create_task(log_api_request(log_entry))
        return _openai_error(
            500, "Workflow execution failed.", err_type="api_error", request_id=request_id
        )

    final = result.get("final_output")
    content = "" if final is None else (final if isinstance(final, str) else json.dumps(final))
    pt, ct, tt = sum_usage_tokens(result.get("node_results"))
    if tt == 0 and (pt > 0 or ct > 0):
        tt = pt + ct

    log_entry["http_status"] = 200
    log_entry["success"] = True
    log_entry["total_latency_ms"] = result.get("total_latency_ms") or result.get("total_latency")
    log_entry["total_cost"] = result.get("total_cost")
    log_entry["workflow_run_id"] = str(result["run_id"]) if result.get("run_id") else None
    log_entry["custom_metrics"]["streamed"] = stream

    created = int(time.time())
    model_label = f"optiml/{dep_slug}"

    if stream:
        # Workflow execution is not token-incremental: a graph produces its
        # final output when it finishes. The stream is therefore REAL but
        # buffered — the completed output is chunked into OpenAI-format SSE so
        # an OpenAI client works unmodified. Documented, not disguised.
        asyncio.create_task(log_api_request(log_entry))
        return StreamingResponse(
            _workflow_sse(
                content=content,
                request_id=request_id,
                created=created,
                model=model_label,
                usage=(pt, ct, tt),
                include_usage=_include_usage(body),
                served_version=resolved_version,
            ),
            media_type="text/event-stream",
            headers={
                **_id_headers(request_id),
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    asyncio.create_task(log_api_request(log_entry))
    return JSONResponse(
        content={
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_label,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
            "optiml": {
                "request_id": request_id,
                "mode": MODE_WORKFLOW,
                "endpoint_slug": dep_slug,
                "served_version": resolved_version,
                "experiment_id": str(experiment_id) if experiment_id else None,
                "variant_name": variant_name,
                "workflow_run_id": log_entry["workflow_run_id"],
                "cost_usd": result.get("total_cost"),
            },
            # Retained for callers written against the previous response shape.
            "optiml_request_id": request_id,
            "optiml_served_version": resolved_version,
        },
        headers=_id_headers(request_id),
    )


_WORKFLOW_CHUNK_CHARS = 400


def _workflow_sse(
    *,
    content: str,
    request_id: str,
    created: int,
    model: str,
    usage: tuple[int, int, int],
    include_usage: bool,
    served_version: Optional[int],
) -> Iterator[str]:
    def frame(delta: dict[str, Any], finish: Optional[str] = None) -> str:
        return "data: " + json.dumps(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            },
            separators=(",", ":"),
        ) + "\n\n"

    yield frame({"role": "assistant"})
    for i in range(0, len(content), _WORKFLOW_CHUNK_CHARS):
        yield frame({"content": content[i : i + _WORKFLOW_CHUNK_CHARS]})
    yield frame({}, "stop")
    if include_usage:
        pt, ct, tt = usage
        yield "data: " + json.dumps(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
                "optiml": {"served_version": served_version},
            },
            separators=(",", ":"),
        ) + "\n\n"
    yield "data: [DONE]\n\n"


# ═════════════════════════════════════════════════════════════════════════════
# GET /v1/models — an OpenAI client's first call in many SDKs
# ═════════════════════════════════════════════════════════════════════════════
@router.get("/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """
    List what this org can call: every priced provider model (direct inference)
    plus every promoted deployment as ``optiml/<slug>`` (workflow mode).

    The two modes are visibly distinct in the listing, which is the point.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return _openai_error(
            401, "Missing or invalid Authorization header.", err_type="authentication_error"
        )
    try:
        ctx = validate_api_key(authorization.split(" ", 1)[1].strip())
    except HTTPException as e:
        return _openai_error(e.status_code, str(e.detail), err_type="authentication_error")

    from utils.pricing import get_all_providers

    created = int(time.time())
    data: list[dict[str, Any]] = []
    try:
        for provider, cfg in (get_all_providers() or {}).items():
            if provider not in direct_inference.KNOWN_PROVIDERS:
                continue
            for model_id in (cfg.get("models") or {}):
                data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": created,
                        "owned_by": provider,
                        "optiml_mode": MODE_DIRECT,
                    }
                )
    except Exception:
        logger.warning("could not enumerate provider models", exc_info=True)

    try:
        deployments = (
            supabase.table("workflow_deployments")
            .select("endpoint_slug, version, status")
            .eq("org_id", ctx.org_id)
            .execute()
        )
        seen: set[str] = set()
        for row in deployments.data or []:
            slug = (row.get("endpoint_slug") or "").strip()
            if not slug or slug in seen:
                continue
            seen.add(slug)
            data.append(
                {
                    "id": f"optiml/{slug}",
                    "object": "model",
                    "created": created,
                    "owned_by": "optiml",
                    "optiml_mode": MODE_WORKFLOW,
                }
            )
    except Exception:
        logger.warning("could not enumerate deployments", exc_info=True)

    return {"object": "list", "data": data}
