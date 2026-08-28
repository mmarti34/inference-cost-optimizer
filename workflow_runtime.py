"""
Workflow execution engine: traverses graph_json, executes nodes in order,
passes outputs via context, supports branching (condition) and routing (router).
"""
from __future__ import annotations

import json
import logging
import queue as _queue_mod
import threading
import time
import uuid as _uuid_mod
from typing import Any
from fastapi import HTTPException

from routers import (
    openai_router,
    anthropic_router,
    mistral_router,
    cohere_router,
    gemini_router,
    groq_router,
    together_router,
    deepseek_router,
    fireworks_router,
)
from utils.pricing import get_pricing, suggest_model
from supabase_client import supabase
from custom_model_management import get_custom_model_by_id
from model_target import ModelTarget
from provider_resilience import call_with_resilience
from context_runtime import resolve_node_context, build_context_trace
from context_embeddings import search_similar as _search_kb
from evidence_redaction import capture_variables
# SM v1 prompt injection removed — v2 compresses context sources instead
# (prompt_assembler.py still used by the /synthetic-mind/stats dashboard)

_logger = logging.getLogger(__name__)


# ── Internal latency instrumentation ────────────────────────────────────────
# Writes one row per provider call into routing_latency_facts.
# Covers: model, ai-step, optimizer, vision_step, image_gen_step, tts_step,
# stt_step, embedding_step.
# Best-effort: failures are logged (WARNING) but never propagate to the caller.

def _record_latency_fact(
    *,
    org_id: str | None,
    workflow_run_id: str | None,
    workflow_id: str | None,
    node_id: str,
    node_type: str,
    target_type: str | None,
    provider_label: str,
    model_name: str | None,
    endpoint_id: str | None,
    resolved_base_url: str | None,
    endpoint_slug: str | None,
    version: int | None,
    execution_mode: str | None,
    total_latency_ms: int,
    provider_latency_ms: int | None,
    gateway_overhead_ms: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    success: bool,
    error_type: str | None = None,
    http_status: int | None = None,
) -> None:
    """Insert a row into routing_latency_facts. Best-effort; never raises."""
    if supabase is None:
        _logger.warning("routing_latency_facts: skipped — supabase client is None")
        return
    try:
        row: dict[str, Any] = {
            "org_id": org_id,
            "workflow_run_id": workflow_run_id,
            "workflow_id": workflow_id,
            "node_id": node_id,
            "node_type": node_type,
            "target_type": target_type,
            "provider_label": provider_label,
            "model_name": model_name,
            "endpoint_id": endpoint_id,
            "resolved_base_url": resolved_base_url,
            "endpoint_slug": endpoint_slug,
            "version": version,
            "execution_mode": execution_mode,
            "total_latency_ms": total_latency_ms,
            "provider_latency_ms": provider_latency_ms,
            "gateway_overhead_ms": gateway_overhead_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": success,
            "error_type": error_type,
            "http_status": http_status,
        }
        # Strip None-valued keys so PostgREST uses column defaults (e.g. gen_random_uuid())
        row = {k: v for k, v in row.items() if v is not None}
        _logger.info(
            "routing_latency_facts: inserting node_type=%s provider=%s model=%s total=%dms provider=%s overhead=%s success=%s",
            node_type, provider_label, model_name, total_latency_ms,
            f"{provider_latency_ms}ms" if provider_latency_ms is not None else "n/a",
            f"{gateway_overhead_ms}ms" if gateway_overhead_ms is not None else "n/a",
            success,
        )
        supabase.table("routing_latency_facts").insert(row).execute()
        _logger.info("routing_latency_facts: insert OK")
    except Exception as exc:
        _logger.warning(
            "routing_latency_facts: INSERT FAILED — %s: %s (node_type=%s, provider=%s, model=%s)",
            type(exc).__name__, str(exc)[:500], node_type, provider_label, model_name,
        )


def _classify_error(exc: Exception | None, detail: str | None = None) -> str | None:
    """Classify an exception into a short error_type label for the latency table."""
    if exc is None and not detail:
        return None
    text = (detail or str(exc or "")).lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "rate" in text and "limit" in text:
        return "rate_limit"
    if "api key" in text or "auth" in text or "401" in text or "403" in text:
        return "auth"
    if "404" in text or "not found" in text:
        return "not_found"
    if "connection" in text or "connect" in text:
        return "connection"
    return "provider_error"


def _nodes_by_id(graph: dict) -> dict[str, dict]:
    nodes = graph.get("nodes") or []
    return {n["id"]: n for n in nodes}


def _edges_out(graph: dict) -> dict[str, list[dict]]:
    edges = graph.get("edges") or []
    out: dict[str, list[dict]] = {}
    for e in edges:
        src = e.get("source")
        if src not in out:
            out[src] = []
        out[src].append(e)
    return out


def _find_input_node(nodes_by_id: dict) -> str | None:
    for nid, n in nodes_by_id.items():
        if (n.get("type") or "").lower() == "input":
            return nid
    return None


def _entry_point_ids(nodes_by_id: dict, edges_out: dict) -> list[str]:
    """Nodes with no incoming edges (valid when there is no input node)."""
    targets = set()
    for out_list in edges_out.values():
        for e in out_list:
            tid = e.get("target")
            if tid:
                targets.add(tid)
    return [nid for nid in nodes_by_id if nid not in targets]


def _get_previous_output(context: dict, from_node_id: str) -> str:
    """Get the text output of a node for use as input to the next node."""
    out = context.get(from_node_id)
    if out is None:
        return ""
    if isinstance(out, dict):
        return out.get("output") or out.get("response") or str(out)
    return str(out)


class _PromptPayload:
    def __init__(self, org_id: str, provider: str, model: str, prompt: str, prompt_id: str):
        self.org_id = org_id
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.prompt_id = prompt_id


_ROUTER_MAP = {
    "openai": openai_router,
    "anthropic": anthropic_router,
    "mistral": mistral_router,
    "cohere": cohere_router,
    "gemini": gemini_router,
    "groq": groq_router,
    "together": together_router,
    "deepseek": deepseek_router,
    "fireworks": fireworks_router,
}


def _resolve_model_target(data: dict, org_id: str, prompt_text: str) -> tuple[ModelTarget | None, str, str, str]:
    """
    Resolve node data into (ModelTarget | None, provider, model, updated_prompt_text).

    Resolution priority:
      1. model_registry_id → full ModelTarget from DB (custom endpoints or registered provider models)
      2. customModelId → legacy custom_models table (preset configs on known providers)
      3. provider + modelName → direct provider_model (backwards-compatible default)

    Returns (target, provider, model, prompt_text).
    target is None for cases 2 and 3 (legacy paths that don't need custom routing).
    """
    # Path 1: model_registry_id (new system)
    # IMPORTANT: if registry ID is present but resolution fails, raise immediately.
    # Never silently fall back to provider+modelName — that makes debugging miserable.
    registry_id = data.get("modelRegistryId") or data.get("model_registry_id")
    if registry_id:
        from model_registry_management import resolve_model_target as _resolve_from_db
        try:
            target = _resolve_from_db(str(registry_id), org_id)
        except HTTPException:
            raise  # preserve the 404 with its detail message
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to resolve model registry entry {registry_id}: {exc}",
            ) from exc
        return target, target.provider, target.model_name, prompt_text

    # Path 2: customModelId (legacy preset configs)
    custom_model_id = data.get("customModelId") or data.get("custom_model_id")
    if custom_model_id:
        custom = get_custom_model_by_id(str(custom_model_id), org_id)
        if not custom:
            raise HTTPException(
                status_code=404,
                detail=f"Custom model not found. (custom_model_id={custom_model_id})",
            )
        provider = (custom.get("provider") or "OpenAI").strip() or "OpenAI"
        model = (custom.get("base_model") or "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
        system_prefix = (custom.get("system_prefix") or "").strip()
        if system_prefix:
            prompt_text = system_prefix + "\n\n" + prompt_text
        return None, provider, model, prompt_text

    # Path 3: direct provider + modelName (backwards-compatible)
    provider = (data.get("provider") or "OpenAI").strip() or "OpenAI"
    model = (data.get("modelName") or "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
    return None, provider, model, prompt_text


def _execute_model_node(node_id: str, node: dict, prompt_text: str, org_id: str) -> dict:
    data = node.get("data") or {}
    target, provider, model, prompt_text = _resolve_model_target(data, org_id, prompt_text)

    provider_lower = provider.lower()
    payload = _PromptPayload(org_id=org_id, provider=provider_lower, model=model, prompt=prompt_text, prompt_id=f"workflow-{node_id}")

    start = time.perf_counter()

    if target and target.target_type == "openai_compatible_endpoint":
        # Route through openai_router with target (base_url + auth injected inside)
        result = call_with_resilience(
            lambda: openai_router.handle_prompt(payload, target=target),
            context_label=f"custom-endpoint/{model} (node {node_id})",
        )
    else:
        # Standard provider routing (covers both registered provider_model and legacy paths)
        router = _ROUTER_MAP.get(provider_lower)
        if not router:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
        # call_with_resilience adds retry (exponential backoff on 429/5xx/timeouts)
        # and per-attempt timeout (120s default). Non-retryable errors (400/401/403/404)
        # are raised immediately without retry.
        result = call_with_resilience(
            lambda: router.handle_prompt(payload),
            context_label=f"{provider_lower}/{model} (node {node_id})",
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    out_text = result.get("response") or result.get("output") or ""

    # ── Output quality validation ──────────────────────────────────
    # Detect empty, refusal, or truncated outputs and flag them so
    # they surface in error-rate calculations and the trace viewer.
    _output_warning: str | None = None
    _stripped = (out_text or "").strip()
    if not _stripped:
        _output_warning = "empty_output"
    elif len(_stripped) < 4 and not any(c.isalnum() for c in _stripped):
        # Pure punctuation / whitespace-like responses (e.g. "..." or "—")
        _output_warning = "empty_output"
    elif _stripped.lower().startswith(("i'm sorry", "i cannot", "i can't", "as an ai")):
        _output_warning = "refusal"
    # Check for provider-level error signals that didn't raise an exception
    if result.get("status") == "error" or result.get("error"):
        _output_warning = _output_warning or "provider_error"

    # Provider latency: populated by instrumented routers (openai_router first)
    provider_latency_ms = result.get("provider_latency_ms")  # int | None
    gateway_overhead_ms = None
    if provider_latency_ms is not None:
        gateway_overhead_ms = max(latency_ms - provider_latency_ms, 0)

    out = {
        "output": out_text,
        "latency_ms": latency_ms,
        "tokens": result.get("output_tokens") or result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens", 0),
        "cost": float(result.get("cost_usd") or 0),
        "model": model,
        "provider": provider,
        "provider_latency_ms": provider_latency_ms,
        "gateway_overhead_ms": gateway_overhead_ms,
    }
    if _output_warning:
        out["output_warning"] = _output_warning
    # Observability: add target provenance when resolved from model_registry
    if target:
        out["target_type"] = target.target_type
        out["model_registry_id"] = target.model_registry_id
        if target.endpoint_id:
            out["endpoint_id"] = target.endpoint_id
        if target.base_url:
            out["resolved_base_url"] = target.base_url
    return out


# ── Tool Call support ─────────────────────────────────────────────────────

# Private IPs to block (SSRF protection)
import ipaddress as _ipaddress
import re as _re_ssrf
from urllib.parse import urlparse as _urlparse


def _is_private_url(url: str) -> bool:
    """Return True if URL resolves to a private/loopback IP (SSRF protection)."""
    try:
        hostname = _urlparse(url).hostname or ""
        # Block obvious private hostnames
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1", ""):
            return True
        # Try to parse as IP
        try:
            ip = _ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local
        except ValueError:
            pass  # Not a literal IP — allow (DNS resolution is at request time)
        return False
    except Exception:
        return True  # If we can't parse, block it


_secrets_cache: dict[str, dict[str, str]] = {}  # org_id -> {NAME: decrypted_value}


def _resolve_secrets(tool_def: dict, org_id: str) -> dict:
    """
    Replace {{secrets.NAME}} placeholders in tool definition fields
    (url, headers, hmacSecret) with decrypted values from org_secrets table.
    Returns a new dict with resolved values. Does NOT mutate the original.
    """
    import re
    pattern = re.compile(r"\{\{secrets\.([A-Z0-9_]+)\}\}", re.IGNORECASE)
    fields_to_resolve = ("url", "headers", "hmacSecret", "hmac_secret")

    # Check if any field contains a secret reference
    needs_resolve = False
    for f in fields_to_resolve:
        val = tool_def.get(f)
        if isinstance(val, str) and "{{secrets." in val:
            needs_resolve = True
            break
    if not needs_resolve:
        return tool_def

    # Load secrets for this org (cached per execution)
    if org_id not in _secrets_cache:
        try:
            from utils.encryption import decrypt_api_key
            result = supabase.table("org_secrets").select("name, encrypted_value").eq("org_id", org_id).execute()
            secrets = {}
            for row in (result.data or []):
                try:
                    secrets[row["name"].upper()] = decrypt_api_key(row["encrypted_value"])
                except Exception:
                    pass  # Skip secrets that fail to decrypt
            _secrets_cache[org_id] = secrets
        except Exception:
            _secrets_cache[org_id] = {}

    org_secrets = _secrets_cache[org_id]

    def replacer(match: re.Match) -> str:
        secret_name = match.group(1).upper()
        return org_secrets.get(secret_name, match.group(0))  # Keep placeholder if not found

    resolved = dict(tool_def)
    for f in fields_to_resolve:
        val = resolved.get(f)
        if isinstance(val, str) and "{{secrets." in val:
            resolved[f] = pattern.sub(replacer, val)
    return resolved


def _execute_tool(tool_def: dict, name: str, arguments: dict) -> tuple[str, int]:
    """
    Execute a single tool call. Returns (result_str, latency_ms).

    Supported types:
      - "http" / "webhook": POST/GET to a URL with JSON body
      - "builtin": built-in tools (get_current_time)
    """
    import httpx
    from datetime import datetime, timezone

    tool_type = (tool_def.get("type") or "http").strip().lower()
    _t0 = time.perf_counter()

    try:
        if tool_type in ("http", "webhook"):
            url = (tool_def.get("url") or "").strip()
            if not url:
                return "Error: no URL configured for HTTP tool", 0
            if _is_private_url(url):
                return "Error: URL targets a private/internal address (blocked)", 0

            method = (tool_def.get("method") or "POST").strip().upper()
            headers_raw = tool_def.get("headers") or "{}"
            try:
                extra_headers = json.loads(headers_raw) if isinstance(headers_raw, str) else (headers_raw or {})
            except (json.JSONDecodeError, TypeError):
                extra_headers = {}
            extra_headers.setdefault("Content-Type", "application/json")

            body_bytes = json.dumps(arguments).encode("utf-8")

            # Optional HMAC signing
            hmac_secret = (tool_def.get("hmacSecret") or tool_def.get("hmac_secret") or "").strip()
            if hmac_secret:
                import hmac as _hmac
                import hashlib as _hashlib
                sig = _hmac.new(hmac_secret.encode("utf-8"), body_bytes, _hashlib.sha256).hexdigest()
                extra_headers["X-OptiML-Signature"] = sig

            with httpx.Client(timeout=30.0) as client:
                if method == "GET":
                    resp = client.get(url, headers=extra_headers, params=arguments)
                else:
                    resp = client.post(url, headers=extra_headers, content=body_bytes)
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return resp.text[:4000], latency_ms

        elif tool_type == "builtin":
            if name == "get_current_time":
                result = datetime.now(timezone.utc).isoformat()
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "current_date":
                fmt = arguments.get("format", "%Y-%m-%d")
                try:
                    result = datetime.now(timezone.utc).strftime(fmt)
                except Exception:
                    result = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "generate_random_id":
                import uuid
                result = str(uuid.uuid4())
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "json_parse":
                raw = arguments.get("text", arguments.get("json_string", ""))
                try:
                    parsed = json.loads(raw)
                    result = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError) as e:
                    result = f"Invalid JSON: {str(e)[:200]}"
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "math_eval":
                import ast
                expr = str(arguments.get("expression", ""))
                try:
                    tree = ast.parse(expr, mode="eval")
                    for node in ast.walk(tree):
                        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                                                  ast.FloorDiv, ast.USub, ast.UAdd)):
                            raise ValueError(f"Unsupported operation: {type(node).__name__}")
                    result = str(eval(compile(tree, "<math>", "eval")))
                except Exception as e:
                    result = f"Math error: {str(e)[:200]}"
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "base64_encode":
                import base64 as _b64
                text = str(arguments.get("text", ""))
                result = _b64.b64encode(text.encode("utf-8")).decode("ascii")
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "base64_decode":
                import base64 as _b64
                encoded = str(arguments.get("text", ""))
                try:
                    result = _b64.b64decode(encoded).decode("utf-8")
                except Exception as e:
                    result = f"Decode error: {str(e)[:200]}"
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            elif name == "url_encode":
                from urllib.parse import quote
                text = str(arguments.get("text", ""))
                result = quote(text, safe="")
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                return result, latency_ms
            return f"Unknown built-in tool: {name}", 0

        else:
            return f"Unknown tool type: {tool_type}", 0

    except Exception as e:
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        return f"Error: {str(e)[:500]}", latency_ms


_TOOL_CALL_PROVIDERS = {"openai", "anthropic", "groq", "together", "deepseek", "fireworks", "mistral", "gemini"}


def _execute_tool_call_node(node_id: str, node: dict, prompt_text: str, org_id: str) -> dict:
    """Execute an AI model call with tool use. Returns the standard result dict plus tool_call fields."""
    data = node.get("data") or {}
    target, provider, model, prompt_text = _resolve_model_target(data, org_id, prompt_text)

    provider_lower = provider.lower()
    if provider_lower not in _TOOL_CALL_PROVIDERS and not (target and target.target_type == "openai_compatible_endpoint"):
        raise HTTPException(
            status_code=400,
            detail=f"Tool calling is not yet supported for provider '{provider}'. Use OpenAI or Anthropic.",
        )

    tools_config = data.get("tools") or []
    generic_tools = [
        {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
        for t in tools_config if t.get("name")
    ]

    # Build tool executor that maps tool name → definition → _execute_tool
    tools_by_name = {t["name"]: t for t in tools_config if t.get("name")}

    def tool_executor(name: str, arguments: dict) -> tuple[str, int]:
        tool_def = tools_by_name.get(name, {"type": "builtin"})
        # Resolve {{secrets.NAME}} placeholders before execution
        resolved_def = _resolve_secrets(tool_def, org_id)
        return _execute_tool(resolved_def, name, arguments)

    def _tc_can_parallelize(tool_name: str) -> bool:
        tool_def = tools_by_name.get(tool_name, {})
        return tool_def.get("execution") != "client"

    sys_msg = (data.get("systemInstructions") or data.get("system_prefix") or "").strip()
    max_iters = int(data.get("maxIterations") or 5)
    max_iters = max(1, min(max_iters, 20))

    payload = _PromptPayload(org_id=org_id, provider=provider_lower, model=model, prompt=prompt_text, prompt_id=f"workflow-{node_id}")

    start = time.perf_counter()

    if target and target.target_type == "openai_compatible_endpoint":
        result = call_with_resilience(
            lambda: openai_router.handle_prompt_with_tools(
                payload, generic_tools, target=target, system_message=sys_msg,
                max_iterations=max_iters, tool_executor=tool_executor,
                can_parallelize_tool=_tc_can_parallelize,
            ),
            context_label=f"custom-endpoint/{model} (tool_call node {node_id})",
        )
    elif provider_lower == "anthropic":
        result = call_with_resilience(
            lambda: anthropic_router.handle_prompt_with_tools(
                payload, generic_tools, system_message=sys_msg,
                max_iterations=max_iters, tool_executor=tool_executor,
                can_parallelize_tool=_tc_can_parallelize,
            ),
            context_label=f"anthropic/{model} (tool_call node {node_id})",
        )
    elif provider_lower in ("groq", "together", "deepseek", "fireworks", "mistral", "gemini"):
        router_mod = _ROUTER_MAP[provider_lower]
        result = call_with_resilience(
            lambda: router_mod.handle_prompt_with_tools(
                payload, generic_tools, system_message=sys_msg,
                max_iterations=max_iters, tool_executor=tool_executor,
                can_parallelize_tool=_tc_can_parallelize,
            ),
            context_label=f"{provider_lower}/{model} (tool_call node {node_id})",
        )
    else:
        # OpenAI and OpenAI-compatible
        result = call_with_resilience(
            lambda: openai_router.handle_prompt_with_tools(
                payload, generic_tools, system_message=sys_msg,
                max_iterations=max_iters, tool_executor=tool_executor,
                can_parallelize_tool=_tc_can_parallelize,
            ),
            context_label=f"{provider_lower}/{model} (tool_call node {node_id})",
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    out_text = result.get("response") or result.get("output") or ""

    # Output quality validation (same as _execute_model_node)
    _output_warning: str | None = None
    _stripped = (out_text or "").strip()
    if not _stripped:
        _output_warning = "empty_output"
    elif _stripped.lower().startswith(("i'm sorry", "i cannot", "i can't", "as an ai")):
        _output_warning = "refusal"
    if result.get("status") == "error" or result.get("error"):
        _output_warning = _output_warning or "provider_error"

    provider_latency_ms = result.get("provider_latency_ms")
    gateway_overhead_ms = None
    if provider_latency_ms is not None:
        gateway_overhead_ms = max(latency_ms - provider_latency_ms, 0)

    out = {
        "output": out_text,
        "latency_ms": latency_ms,
        "tokens": result.get("output_tokens") or result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens", 0),
        "cost": float(result.get("cost_usd") or 0),
        "model": model,
        "provider": provider,
        "provider_latency_ms": provider_latency_ms,
        "gateway_overhead_ms": gateway_overhead_ms,
        "tool_calls": result.get("tool_calls", []),
        "tool_calls_count": result.get("tool_calls_count", 0),
        "iterations": result.get("iterations", 1),
    }
    if _output_warning:
        out["output_warning"] = _output_warning
    if target:
        out["target_type"] = target.target_type
        out["model_registry_id"] = target.model_registry_id
        if target.endpoint_id:
            out["endpoint_id"] = target.endpoint_id
        if target.base_url:
            out["resolved_base_url"] = target.base_url
    return out


# ─── Agent Node Execution ──────────────────────────────────────────────────────

_AGENT_SYSTEM_TEMPLATE = (
    "You are an autonomous AI agent. Solve the given task step by step.\n\n"
    "Guidelines:\n"
    "- Think carefully before taking any action\n"
    "- Use the available tools when you need information or need to perform actions\n"
    "- If a tool call fails or returns unexpected results, try a different approach\n"
    "- When you have enough information, provide your final answer\n"
    "- Be thorough but efficient — do not make unnecessary tool calls\n"
)


# ─── Client-Side Tool Execution Protocol ─────────────────────────────────────

class ToolYieldRequest:
    """Represents a pending client-side tool execution waiting for external result."""
    __slots__ = ("yield_id", "tool_name", "arguments", "event", "result", "latency_ms", "org_id")

    def __init__(self, yield_id: str, tool_name: str, arguments: dict, org_id: str | None = None):
        self.yield_id = yield_id
        self.tool_name = tool_name
        self.arguments = arguments
        # Tenant this yield belongs to. The resume endpoint refuses a caller
        # whose API key is scoped to a different org.
        self.org_id = org_id
        self.event = threading.Event()
        self.result: str = ""
        self.latency_ms: int = 0


# Module-level store for pending client tool yields (yield_id → request).
# In a single-process deployment this is sufficient; horizontal scaling
# would need Redis or similar shared state.
_pending_tool_yields: dict[str, ToolYieldRequest] = {}
_yield_lock = threading.Lock()

_TOOL_YIELD_TIMEOUT_SECONDS = 300  # 5 minutes


def resume_tool_yield(yield_id: str, result: str, latency_ms: int = 0, org_id: str | None = None) -> bool:
    """Resume a pending client-side tool yield with the external result.

    Called by the POST /api/public/tool-result/{yield_id} endpoint.
    Returns True if the yield_id was found and resumed.

    When ``org_id`` is supplied (the org the caller's API key is scoped to) it
    must match the org the yield was created for, so an authenticated caller
    from one tenant cannot feed a result into another tenant's paused agent.
    """
    with _yield_lock:
        req = _pending_tool_yields.get(yield_id)
    if not req:
        return False
    if org_id is not None and req.org_id is not None and str(req.org_id) != str(org_id):
        _logger.warning("Cross-tenant tool-result rejected for yield_id=%s", yield_id)
        return False
    req.result = result
    req.latency_ms = latency_ms
    req.event.set()
    return True


def get_pending_yield(yield_id: str) -> ToolYieldRequest | None:
    """Get a pending tool yield by ID (for introspection)."""
    with _yield_lock:
        return _pending_tool_yields.get(yield_id)


def _execute_agent_node(node_id: str, node: dict, prompt_text: str, org_id: str, event_queue: _queue_mod.Queue | None = None, context_text: str | None = None, workflow_id: str | None = None, scope_value: str | None = None) -> dict:
    """
    Execute an autonomous agent with reasoning trace.

    Reuses the tool_call provider infrastructure (handle_prompt_with_tools) but
    wraps it with a ReAct-style system prompt and a recording tool executor that
    captures each act/observe step for a structured reasoning trace.

    When ``event_queue`` is provided (streaming mode), reasoning steps and
    tool_yield events are pushed onto the queue in real time so the SSE
    generator can emit them before the agent finishes.  Tools with
    ``"execution": "client"`` in their config will yield to the caller via
    the queue and block until the external client POSTs the result back.

    Returns the standard result dict plus:
      reasoning_steps: list of {step, type, content?, tool_name?, tool_input?, latency_ms?}
      agent_steps_count: int
    """
    data = node.get("data") or {}
    target, provider, model, prompt_text = _resolve_model_target(data, org_id, prompt_text)

    provider_lower = provider.lower()
    if provider_lower not in _TOOL_CALL_PROVIDERS and not (target and target.target_type == "openai_compatible_endpoint"):
        raise HTTPException(
            status_code=400,
            detail=f"Agent requires tool calling support. Provider '{provider}' is not supported.",
        )

    # ── Build tools (same as tool_call) ─────────────────────────────────
    tools_config = data.get("tools") or []
    generic_tools = [
        {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
        for t in tools_config if t.get("name")
    ]
    tools_by_name = {t["name"]: t for t in tools_config if t.get("name")}

    # Inject RAG tools if context is enabled
    _ctx_config = data.get("contextConfig") or {}
    if _ctx_config.get("enabled"):
        _scoped_ids = [
            s["assetId"] for s in (_ctx_config.get("sources") or [])
            if s.get("type") == "knowledge_asset" and s.get("assetId")
        ] or None

        _kb_search_tool = {
            "name": "search_knowledge_base",
            "description": "Search the organization's knowledge base for relevant information using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        }
        _kb_get_tool = {
            "name": "get_knowledge_asset",
            "description": "Get the full content of a knowledge base asset by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "ID of the asset"},
                },
                "required": ["asset_id"],
            },
        }
        generic_tools = [_kb_search_tool, _kb_get_tool] + generic_tools

    # ── Recording tool executor ─────────────────────────────────────────
    reasoning_steps: list[dict] = []
    step_counter = [0]
    import threading as _threading
    _step_lock = _threading.Lock()

    def can_parallelize_tool(tool_name: str) -> bool:
        """Return True if this tool can be executed in parallel (server-side only)."""
        tool_def = tools_by_name.get(tool_name, {})
        return tool_def.get("execution") != "client"

    def agent_tool_executor(name: str, arguments: dict) -> tuple[str, int]:
        with _step_lock:
            step_counter[0] += 1
            current_step = step_counter[0]
            # Record the act step
            reasoning_steps.append({
                "step": current_step,
                "type": "act",
                "tool_name": name,
                "tool_input": arguments,
            })
        if event_queue:
            event_queue.put({"event": "agent_step", "data": {
                "node_id": node_id, "step_number": current_step, "step_type": "act",
                "content": "", "tool_name": name, "tool_input": arguments, "latency_ms": 0,
            }})

        if name == "search_knowledge_base":
            import json as _json_mod
            _q = arguments.get("query", "")
            _lim = arguments.get("limit", 5)
            _start = time.perf_counter()
            _results = _search_kb(_q, org_id, limit=_lim, asset_ids=_scoped_ids if '_scoped_ids' in dir() else None)
            _lat = int((time.perf_counter() - _start) * 1000)
            return _json_mod.dumps(_results, default=str), _lat

        if name == "get_knowledge_asset":
            import json as _json_mod
            _aid = arguments.get("asset_id", "")
            _start = time.perf_counter()
            try:
                _row = supabase.table("context_assets").select("id, name, content, asset_type, metadata").eq("id", _aid).eq("org_id", org_id).execute()
                if _row.data:
                    _a = _row.data[0]
                    _out = {"id": _a["id"], "name": _a["name"], "content": _a.get("content", ""), "asset_type": _a.get("asset_type"), "char_count": len(_a.get("content") or "")}
                else:
                    _out = {"error": "Asset not found"}
            except Exception as _e:
                _out = {"error": str(_e)}
            _lat = int((time.perf_counter() - _start) * 1000)
            return _json_mod.dumps(_out, default=str), _lat

        tool_def = tools_by_name.get(name, {"type": "builtin"})

        if tool_def.get("execution") == "client":
            # ── Client-side tool: yield to external caller ────────────────
            if not event_queue:
                # Non-streaming mode — client tools cannot work
                err = f"Tool '{name}' requires client-side execution (only available in streaming mode)"
                with _step_lock:
                    reasoning_steps.append({"step": current_step, "type": "observe", "content": err, "latency_ms": 0})
                return err, 0

            yield_id = str(_uuid_mod.uuid4())
            yield_req = ToolYieldRequest(yield_id, name, arguments, org_id=org_id)
            with _yield_lock:
                _pending_tool_yields[yield_id] = yield_req

            # Signal the SSE stream to emit a tool_yield event
            event_queue.put({"event": "tool_yield", "data": {
                "yield_id": yield_id, "node_id": node_id,
                "tool_name": name, "arguments": arguments,
            }})

            # Block this thread until the client POSTs the result back
            if not yield_req.event.wait(timeout=_TOOL_YIELD_TIMEOUT_SECONDS):
                with _yield_lock:
                    _pending_tool_yields.pop(yield_id, None)
                err = f"Client tool '{name}' timed out ({_TOOL_YIELD_TIMEOUT_SECONDS}s)"
                with _step_lock:
                    reasoning_steps.append({"step": current_step, "type": "observe", "content": err, "latency_ms": 0})
                if event_queue:
                    event_queue.put({"event": "agent_step", "data": {
                        "node_id": node_id, "step_number": current_step, "step_type": "observe",
                        "content": err, "tool_name": name, "tool_input": {}, "latency_ms": 0,
                    }})
                return err, 0

            with _yield_lock:
                _pending_tool_yields.pop(yield_id, None)
            result_str = yield_req.result
            latency_ms = yield_req.latency_ms
        else:
            # ── Server-side tool: execute immediately ─────────────────────
            resolved_def = _resolve_secrets(tool_def, org_id)
            result_str, latency_ms = _execute_tool(resolved_def, name, arguments)

        # Record the observe step (thread-safe)
        with _step_lock:
            reasoning_steps.append({
                "step": current_step,
                "type": "observe",
                "content": result_str[:2000],
                "tool_name": name,
                "latency_ms": latency_ms,
            })
        if event_queue:
            event_queue.put({"event": "agent_step", "data": {
                "node_id": node_id, "step_number": current_step, "step_type": "observe",
                "content": (result_str or "")[:500], "tool_name": name, "tool_input": {}, "latency_ms": latency_ms,
            }})
        return result_str, latency_ms

    # ── Build agent system prompt ───────────────────────────────────────
    user_sys = (data.get("systemInstructions") or data.get("system_prefix") or "").strip()
    agent_system = _AGENT_SYSTEM_TEMPLATE
    if context_text:
        agent_system = agent_system + "\n\n--- Context ---\n" + context_text
    # SM v2 Phase 3: inject prior tool knowledge from past agent runs
    try:
        from synthetic_mind.prompt_assembler import generate_agent_prior_knowledge
        _prior = generate_agent_prior_knowledge(org_id, workflow_id=workflow_id, scope_value=scope_value)
        if _prior:
            agent_system = agent_system + "\n\n" + _prior
    except Exception:
        pass  # Never block agent execution for SM failures
    if user_sys:
        agent_system = agent_system + "\n" + user_sys

    max_steps = int(data.get("maxSteps") or data.get("maxIterations") or 10)
    max_steps = max(1, min(max_steps, 100))

    payload = _PromptPayload(
        org_id=org_id, provider=provider_lower, model=model,
        prompt=prompt_text, prompt_id=f"workflow-{node_id}",
    )

    start = time.perf_counter()

    # ── Provider dispatch (same routing as tool_call) ───────────────────
    if target and target.target_type == "openai_compatible_endpoint":
        result = call_with_resilience(
            lambda: openai_router.handle_prompt_with_tools(
                payload, generic_tools, target=target, system_message=agent_system,
                max_iterations=max_steps, tool_executor=agent_tool_executor,
                can_parallelize_tool=can_parallelize_tool,
            ),
            context_label=f"custom-endpoint/{model} (agent node {node_id})",
        )
    elif provider_lower == "anthropic":
        result = call_with_resilience(
            lambda: anthropic_router.handle_prompt_with_tools(
                payload, generic_tools, system_message=agent_system,
                max_iterations=max_steps, tool_executor=agent_tool_executor,
                can_parallelize_tool=can_parallelize_tool,
            ),
            context_label=f"anthropic/{model} (agent node {node_id})",
        )
    elif provider_lower in ("groq", "together", "deepseek", "fireworks", "mistral", "gemini"):
        router_mod = _ROUTER_MAP[provider_lower]
        result = call_with_resilience(
            lambda: router_mod.handle_prompt_with_tools(
                payload, generic_tools, system_message=agent_system,
                max_iterations=max_steps, tool_executor=agent_tool_executor,
                can_parallelize_tool=can_parallelize_tool,
            ),
            context_label=f"{provider_lower}/{model} (agent node {node_id})",
        )
    else:
        result = call_with_resilience(
            lambda: openai_router.handle_prompt_with_tools(
                payload, generic_tools, system_message=agent_system,
                max_iterations=max_steps, tool_executor=agent_tool_executor,
                can_parallelize_tool=can_parallelize_tool,
            ),
            context_label=f"{provider_lower}/{model} (agent node {node_id})",
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    out_text = result.get("response") or result.get("output") or ""

    # ── Add final answer step ───────────────────────────────────────────
    reasoning_steps.append({
        "step": step_counter[0] + 1,
        "type": "answer",
        "content": out_text[:2000],
    })
    if event_queue:
        event_queue.put({"event": "agent_step", "data": {
            "node_id": node_id, "step_number": step_counter[0] + 1, "step_type": "answer",
            "content": out_text[:500], "tool_name": "", "tool_input": {}, "latency_ms": 0,
        }})
        # Signal that agent execution is complete
        event_queue.put({"event": "agent_done"})

    # ── Output quality validation ───────────────────────────────────────
    _output_warning: str | None = None
    _stripped = (out_text or "").strip()
    if not _stripped:
        _output_warning = "empty_output"
    elif _stripped.lower().startswith(("i'm sorry", "i cannot", "i can't", "as an ai")):
        _output_warning = "refusal"
    if result.get("status") == "error" or result.get("error"):
        _output_warning = _output_warning or "provider_error"

    provider_latency_ms = result.get("provider_latency_ms")
    gateway_overhead_ms = None
    if provider_latency_ms is not None:
        gateway_overhead_ms = max(latency_ms - provider_latency_ms, 0)

    out = {
        "output": out_text,
        "latency_ms": latency_ms,
        "tokens": result.get("output_tokens") or result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens", 0),
        "cost": float(result.get("cost_usd") or 0),
        "model": model,
        "provider": provider,
        "provider_latency_ms": provider_latency_ms,
        "gateway_overhead_ms": gateway_overhead_ms,
        "tool_calls": result.get("tool_calls", []),
        "tool_calls_count": result.get("tool_calls_count", 0),
        "iterations": result.get("iterations", 1),
        # Agent-specific fields
        "reasoning_steps": reasoning_steps,
        "agent_steps_count": len(reasoning_steps),
    }
    if _output_warning:
        out["output_warning"] = _output_warning
    if target:
        out["target_type"] = target.target_type
        out["model_registry_id"] = target.model_registry_id
        if target.endpoint_id:
            out["endpoint_id"] = target.endpoint_id
        if target.base_url:
            out["resolved_base_url"] = target.base_url
    return out


def _execute_model_node_safe(node_id: str, node: dict, prompt_text: str, org_id: str) -> dict:
    """
    Same as _execute_model_node but catches provider failures and returns
    an error result dict instead of raising. Used by the optimizer node so
    that failures are recorded in node_results (feeding the error-rate signal)
    rather than silently lost.
    """
    data = node.get("data") or {}
    # Pre-resolve for error-path metadata (provider/model may come from registry)
    try:
        _, provider, model, _ = _resolve_model_target(data, org_id, "")
    except Exception:
        provider = (data.get("provider") or "OpenAI").strip() or "OpenAI"
        model = (data.get("modelName") or "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
    start = time.perf_counter()
    try:
        return _execute_model_node(node_id, node, prompt_text, org_id)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        status_code = getattr(exc, "status_code", 500)
        detail = getattr(exc, "detail", str(exc))
        return {
            "output": "",
            "latency_ms": latency_ms,
            "tokens": 0,
            "input_tokens": 0,
            "cost": 0.0,
            "model": model,
            "provider": provider,
            "error": True,
            "error_status": status_code,
            "error_detail": str(detail)[:500],
        }


OPTIMIZER_THRESHOLD = 25


# ── Model performance history ───────────────────────────────────────────────
# These three helpers MOVED to optimization/evidence.py so there is exactly one
# implementation shared with the optimization layer's candidate generators.
# They are imported back here under their original private names, so every
# existing call site in this module keeps working unchanged.
#
# The import is local to avoid a cycle: optimization.benchmark imports
# execute_workflow from this module, so optimization/__init__.py is kept
# import-light and optimization.evidence imports nothing from here.
from optimization.evidence import (  # noqa: E402
    infer_provider as _infer_provider,
    get_model_performance_history as _get_model_performance_history,
    select_optimal_model as _select_optimal_model,
)


def _evaluate_condition(operator: str, value: str, previous_output: str) -> bool:
    prev = (previous_output or "").strip().lower()
    val = (value or "").strip().lower()
    if operator == "contains":
        return val in prev or value in (previous_output or "")
    if operator == "equals":
        return prev == val
    if operator == "starts_with":
        return (previous_output or "").strip().lower().startswith(val)
    if operator == "not_contains":
        return val not in prev
    return False


# Relative latency rank for the "fastest" router strategy (lower = faster).
#
# This replaces a hardcoded 13-entry table of 2024-era model names whose ranks
# were guesses and which had drifted out of date (it contained ids like
# "mistral-3.1-small" that nothing in the app ever produces, while every dated
# or newer model id fell through to the 999 bucket). The tier now comes from
# the `category` field in shared/providers.json, which is maintained alongside
# pricing, so adding a model keeps "fastest" working instead of silently
# degrading it to "pick the first edge".
#
# This is still a coarse prior, not a measurement. Real per-model latency is
# being collected in routing_latency_facts; when there is enough of it, rank
# from observed p50 for the org and treat this as the cold-start fallback.
_CATEGORY_LATENCY_RANK: dict[str, int] = {
    "fast": 1,
    "balanced": 2,
    "flagship": 4,
    "reasoning": 5,
}
_UNKNOWN_LATENCY_RANK = 999


def _latency_rank(provider: str, model: str) -> int:
    """Latency tier for (provider, model). Unknown models sort last."""
    name = (model or "").strip()
    if not name:
        return _UNKNOWN_LATENCY_RANK
    try:
        from utils.pricing import get_all_providers, get_provider_for_model
        prov = (provider or "").strip().lower() or (get_provider_for_model(name) or "")
        models = (get_all_providers().get(prov) or {}).get("models") or {}
        entry = models.get(name)
        if entry is None:
            lowered = name.lower()
            for mid, spec in models.items():
                if mid.lower() == lowered or lowered.startswith(mid.lower()):
                    entry = spec
                    break
        if entry is None:
            return _UNKNOWN_LATENCY_RANK
        return _CATEGORY_LATENCY_RANK.get(
            str(entry.get("category") or "").lower(), _UNKNOWN_LATENCY_RANK
        )
    except Exception:
        return _UNKNOWN_LATENCY_RANK


def _select_router_edge(edges: list[dict], nodes_by_id: dict, strategy: str, prompt_text: str) -> str | None:
    """Pick target node id for router. strategy: cheapest, fastest, balanced. Targets can be model or ai-step."""
    model_targets = []
    for e in edges:
        tid = e.get("target")
        if not tid:
            continue
        n = nodes_by_id.get(tid)
        nt = (n.get("type") or "").lower()
        if n and (nt == "model" or nt == "ai-step"):
            model_targets.append((tid, n))

    if not model_targets:
        return edges[0].get("target") if edges else None

    if strategy == "balanced":
        sug = suggest_model(prompt_text or "Hello")
        prov = (sug.get("provider") or "openai").lower()
        mod = (sug.get("model") or "gpt-3.5-turbo").lower()
        for tid, n in model_targets:
            d = n.get("data") or {}
            mn = (d.get("modelName") or d.get("base_model") or "gpt-3.5-turbo").lower()
            pv = (d.get("provider") or "openai").lower()
            if pv == prov and mn == mod:
                return tid
        return model_targets[0][0]

    if strategy == "cheapest":
        best = None
        best_cost = float("inf")
        for tid, n in model_targets:
            d = n.get("data") or {}
            mod = (d.get("modelName") or d.get("base_model") or "gpt-3.5-turbo").lower()
            p = get_pricing((d.get("provider") or "openai").lower(), mod)
            cost = (p.get("input") or 0) + (p.get("output") or 0)
            if cost < best_cost:
                best_cost = cost
                best = tid
        return best or model_targets[0][0]

    if strategy == "fastest":
        best = None
        best_rank = 999
        for tid, n in model_targets:
            d = n.get("data") or {}
            mod = (d.get("modelName") or d.get("base_model") or "").strip()
            pv = (d.get("provider") or "openai").strip().lower()
            r = _latency_rank(pv, mod)
            if r < best_rank:
                best_rank = r
                best = tid
        return best or model_targets[0][0]

    if strategy == "fallback":
        return model_targets[0][0]

    return model_targets[0][0]


def validate_workflow_variables(
    schema: list[dict],
    variables: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate variables against workflow schema. Strict type check; no eval.
    Returns normalized variables (e.g. JSON strings parsed to dict/list).
    Raises HTTPException 400 on missing required or type mismatch.
    """
    if not schema or not isinstance(schema, list):
        return variables
    schema_by_name = {v["name"]: v for v in schema if isinstance(v, dict) and v.get("name")}
    normalized: dict[str, Any] = {}
    for name, val in variables.items():
        if name not in schema_by_name:
            normalized[name] = val
            continue
        spec = schema_by_name[name]
        var_type = (spec.get("type") or "string").lower()
        required = bool(spec.get("required"))

        if val is None or (isinstance(val, str) and val.strip() == ""):
            if required:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required variable: {name}",
                )
            normalized[name] = val
            continue

        if var_type == "string":
            if not isinstance(val, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid type for variable: {name} (expected string)",
                )
            normalized[name] = val
        elif var_type == "number":
            if isinstance(val, bool):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid type for variable: {name} (expected number)",
                )
            if not isinstance(val, (int, float)):
                try:
                    normalized[name] = float(val) if val != "" else 0
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid type for variable: {name} (expected number)",
                    )
            else:
                normalized[name] = val
        elif var_type == "boolean":
            if not isinstance(val, bool):
                if isinstance(val, str):
                    low = val.strip().lower()
                    if low in ("true", "1", "yes"):
                        normalized[name] = True
                    elif low in ("false", "0", "no", ""):
                        normalized[name] = False
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid type for variable: {name} (expected boolean)",
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid type for variable: {name} (expected boolean)",
                    )
            else:
                normalized[name] = val
        elif var_type == "json":
            if isinstance(val, (dict, list)):
                normalized[name] = val
            elif isinstance(val, str):
                try:
                    normalized[name] = json.loads(val)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid type for variable: {name} (expected valid JSON)",
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid type for variable: {name} (expected json)",
                )
        else:
            normalized[name] = val

    for v in schema:
        if not isinstance(v, dict) or not v.get("required") or not v.get("name"):
            continue
        name = v["name"]
        if name not in variables or variables[name] is None or (isinstance(variables.get(name), str) and (variables[name] or "").strip() == ""):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required variable: {name}",
            )

    return normalized


def _apply_variables(
    template: str,
    variables: dict[str, Any] | None,
    prev_output: str | None = None,
) -> str:
    """
    Single-pass template replacement: {{varName}} -> variables[varName].
    If prev_output is provided, {{input}} is also replaced in the same pass.

    SECURITY: This is strictly single-pass. After replacement, the result is
    never scanned again for {{...}} patterns. This prevents template injection
    where a variable value like "{{input}}" could be expanded in a second pass.
    No eval, no recursion.
    """
    import re as _re

    if not template or not isinstance(template, str):
        return template or ""

    # Build a lookup dict for all known placeholders
    lookup: dict[str, str] = {}
    if variables and isinstance(variables, dict):
        for key, val in variables.items():
            if isinstance(key, str) and _re.match(r'^[a-zA-Z0-9_]+$', key):
                lookup[key] = str(val) if val is not None else ""
    if prev_output is not None:
        lookup["input"] = prev_output

    if not lookup:
        return template

    # Single-pass replacement using re.sub with a callback.
    # This ensures each placeholder is replaced exactly once and
    # values containing {{...}} are NOT re-expanded.
    def _replace_match(match: _re.Match) -> str:
        key = match.group(1)
        return lookup.get(key, match.group(0))  # leave unknown placeholders as-is

    return _re.sub(r'\{\{(\w+)\}\}', _replace_match, template)


def execute_workflow(
    graph: dict,
    input_text: str,
    org_id: str,
    user_id: str,
    workflow_id: str | None = None,
    endpoint_slug: str | None = None,
    version: int | None = None,
    execution_mode: str = "draft",
    variables: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    variant_name: str | None = None,
    served_version: int | None = None,
    conversation_prefix: str | None = None,
    deployment_id: str | None = None,
) -> dict:
    """
    Execute workflow graph. Returns final_output, node_results, total_cost, total_latency.
    If variables is provided, {{varName}} in AI Step/prompt templates are replaced; input_text can be empty.
    If conversation_prefix is set (multi-turn), it is prepended to the prompt for each AI/model step.
    """
    # Clear per-execution secrets cache
    _secrets_cache.clear()

    nodes_by_id = _nodes_by_id(graph)
    edges_out = _edges_out(graph)
    input_node_id = _find_input_node(nodes_by_id)
    entry_points = _entry_point_ids(nodes_by_id, edges_out) if not input_node_id else [input_node_id]
    if not entry_points:
        raise HTTPException(status_code=400, detail="Workflow has no entry point (add an Input node or connect from a root node)")

    if variables and input_text == "" and input_node_id:
        import json
        try:
            input_text = json.dumps(variables)
        except Exception:
            input_text = str(variables)

    context: dict[str, Any] = {}
    node_results: list[dict] = []
    total_cost = 0.0
    total_latency = 0
    executed_edges: set[tuple[str, str]] = set()
    last_content_type: str = "text"

    queue: list[tuple[str, str | None]] = [(nid, None) for nid in entry_points]

    # ── Evidence capture: the named inputs that actually drove this run ─────
    # `workflow_runs` stored `input_text` and nothing else, so a workflow driven
    # by named variables recorded NOTHING about its inputs and none of its
    # traffic could ever become an evaluation case. Capture is redacted at write
    # time (evidence_redaction) and is BEST-EFFORT AND TOTAL: it can never turn
    # into a customer-facing failure. `_raw_variables` is read before the
    # `or {}` normalisation below so that "caller sent nothing" and "caller sent
    # an empty object" stay distinguishable instead of collapsing into `{}`.
    _raw_variables = variables
    _capture_cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    def _capture_variables_once() -> tuple[Any, dict[str, Any]]:
        """Redact + summarise the run's variables, at most once, never raising."""
        if "v" not in _capture_cache:
            try:
                _capture_cache["v"] = capture_variables(_raw_variables)
            except BaseException as _cap_err:  # noqa: BLE001 — see above
                _logger.warning(
                    "evidence capture failed (%s); recording variables as unavailable",
                    type(_cap_err).__name__,
                )
                _capture_cache["v"] = (
                    None,
                    {
                        "status": "unavailable",
                        "reason": "capture_failed",
                        "error_type": type(_cap_err).__name__,
                        "redacted": False,
                        "truncated": False,
                    },
                )
        return _capture_cache["v"]

    variables = variables or {}

    # Track what's currently executing so we can record it in error node_results
    _cur_node_id: str = ""
    _cur_node_type: str = ""
    _cur_provider: str = ""
    _cur_model: str = ""

    try:
     while queue:
        node_id, from_node_id = queue.pop(0)
        if node_id in context:
            continue

        node = nodes_by_id.get(node_id)
        if not node:
            continue

        node_type = (node.get("type") or "").lower()
        data = node.get("data") or {}

        # Update tracking for error recording
        _cur_node_id = node_id
        _cur_node_type = node_type
        _cur_provider = (data.get("provider") or "").strip()
        _cur_model = (data.get("modelName") or data.get("model") or "").strip()

        if from_node_id is not None:
            executed_edges.add((from_node_id, node_id))

        if node_type == "input":
            context[node_id] = input_text
            node_results.append({
                "node_id": node_id,
                "type": "input",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "output": input_text[:200] + ("..." if len(input_text) > 200 else ""),
            })
        elif node_type == "prompt":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            template = data.get("template") or data.get("preview") or data.get("label") or "{{input}}"
            if not isinstance(template, str):
                template = str(template)
            formatted = _apply_variables(template, variables, prev_output=prev)
            context[node_id] = formatted
            node_results.append({
                "node_id": node_id,
                "type": "prompt",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "output": formatted[:200] + ("..." if len(formatted) > 200 else ""),
            })
        elif node_type == "ai-step":
            ctx_result = None
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            task = (data.get("taskDescription") or data.get("task") or "Respond to the user.").strip()
            if not isinstance(task, str):
                task = str(task)
            prompt_text = _apply_variables(task, variables, prev_output=prev)
            # If the task template doesn't contain {{input}}, the previous node
            # output was never interpolated — append it so context always flows.
            if "{{input}}" not in task and prev and prev.strip():
                prompt_text = prompt_text + "\n\n" + prev
            if conversation_prefix:
                prompt_text = conversation_prefix + prompt_text
            # --- Context injection ---
            ctx_result = resolve_node_context(node, context, variables, input_text, org_id, execution_mode, deployment_id=deployment_id)
            if ctx_result:
                _ctx_text = ctx_result["final_text"]
                _ctx_loc = ctx_result["injection_location"]
                if _ctx_loc == "append_to_prompt":
                    prompt_text = prompt_text + "\n\n" + _ctx_text
                elif _ctx_loc == "prepend_to_prompt":
                    prompt_text = _ctx_text + "\n\n" + prompt_text
                # prepend_to_system handled after sys_instructions below
            # --- End context injection (part 1) ---
            sys_instructions = (data.get("systemInstructions") or data.get("system_prefix") or "").strip()
            if sys_instructions:
                sys_instructions = _apply_variables(sys_instructions, variables)
                prompt_text = sys_instructions + "\n\n" + prompt_text
            # --- Context injection (part 2: prepend_to_system) ---
            if ctx_result and ctx_result["injection_location"] == "prepend_to_system":
                prompt_text = ctx_result["final_text"] + "\n\n" + prompt_text
            # --- End context injection ---
            model_node = {"id": node_id, "type": "model", "data": {**data, "modelName": data.get("modelName") or "gpt-3.5-turbo", "provider": data.get("provider") or "OpenAI"}}
            _t0_ai_step = time.perf_counter()
            try:
                result = _execute_model_node(node_id, model_node, prompt_text, org_id)
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_ai_step) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="ai-step",
                    target_type=None, provider_label=(data.get("provider") or "unknown").strip().lower(),
                    model_name=data.get("modelName"), endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No API key configured for this provider. Add a key in Settings → Integrations. (node_id={node_id})",
                    ) from e
                raise
            context[node_id] = result
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            out_tok = result.get("tokens") or 0
            in_tok = result.get("input_tokens") or 0
            nr_entry = {
                "node_id": node_id,
                "type": "ai-step",
                "latency_ms": result["latency_ms"],
                "tokens": out_tok,
                "tokens_output": out_tok,
                "input_tokens": in_tok,
                "tokens_input": in_tok,
                "cost": result["cost"],
                "cost_usd": result["cost"],
                "model": result.get("model"),
                "provider": result.get("provider"),
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
                "prompt_after_interpolation": prompt_text[:2000] + ("..." if len(prompt_text) > 2000 else ""),
            }
            # Append model target observability fields if present
            for _k in ("target_type", "model_registry_id", "endpoint_id", "resolved_base_url"):
                if result.get(_k):
                    nr_entry[_k] = result[_k]
            # Output quality: propagate warning as soft error
            if result.get("output_warning"):
                nr_entry["output_warning"] = result["output_warning"]
                nr_entry["status"] = "warning"
            nr_entry["context_trace"] = build_context_trace(ctx_result, data)
            node_results.append(nr_entry)
            # Internal latency instrumentation (best-effort, never raises)
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="ai-step",
                target_type=result.get("target_type"), provider_label=(result.get("provider") or "unknown").lower(),
                model_name=result.get("model"), endpoint_id=result.get("endpoint_id"),
                resolved_base_url=result.get("resolved_base_url"),
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=result["latency_ms"],
                provider_latency_ms=result.get("provider_latency_ms"),
                gateway_overhead_ms=result.get("gateway_overhead_ms"),
                input_tokens=in_tok or None, output_tokens=out_tok or None,
                success=True,
            )
        elif node_type == "tool_call":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            task = (data.get("taskDescription") or data.get("task") or "Respond to the user using the available tools.").strip()
            if not isinstance(task, str):
                task = str(task)
            prompt_text = _apply_variables(task, variables, prev_output=prev)
            # If the task template doesn't contain {{input}}, the previous node
            # output was never interpolated — append it so context always flows.
            if "{{input}}" not in task and prev and prev.strip():
                prompt_text = prompt_text + "\n\n" + prev
            if conversation_prefix:
                prompt_text = conversation_prefix + prompt_text
            # System instructions are passed separately to the tool-call handler (not prepended to prompt)
            _t0_tool_call = time.perf_counter()
            try:
                result = _execute_tool_call_node(node_id, node, prompt_text, org_id)
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_tool_call) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="tool_call",
                    target_type=None, provider_label=(data.get("provider") or "unknown").strip().lower(),
                    model_name=data.get("modelName"), endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No API key configured for this provider. Add a key in Settings → Integrations. (node_id={node_id})",
                    ) from e
                raise
            context[node_id] = result
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            out_tok = result.get("tokens") or 0
            in_tok = result.get("input_tokens") or 0
            nr_entry = {
                "node_id": node_id,
                "type": "tool_call",
                "latency_ms": result["latency_ms"],
                "tokens": out_tok,
                "tokens_output": out_tok,
                "input_tokens": in_tok,
                "tokens_input": in_tok,
                "cost": result["cost"],
                "cost_usd": result["cost"],
                "model": result.get("model"),
                "provider": result.get("provider"),
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
                "prompt_after_interpolation": prompt_text[:2000] + ("..." if len(prompt_text) > 2000 else ""),
                "tool_calls_count": result.get("tool_calls_count", 0),
                "tool_calls": result.get("tool_calls", []),
                "iterations": result.get("iterations", 1),
            }
            for _k in ("target_type", "model_registry_id", "endpoint_id", "resolved_base_url"):
                if result.get(_k):
                    nr_entry[_k] = result[_k]
            if result.get("output_warning"):
                nr_entry["output_warning"] = result["output_warning"]
                nr_entry["status"] = "warning"
            node_results.append(nr_entry)
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="tool_call",
                target_type=result.get("target_type"), provider_label=(result.get("provider") or "unknown").lower(),
                model_name=result.get("model"), endpoint_id=result.get("endpoint_id"),
                resolved_base_url=result.get("resolved_base_url"),
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=result["latency_ms"],
                provider_latency_ms=result.get("provider_latency_ms"),
                gateway_overhead_ms=result.get("gateway_overhead_ms"),
                input_tokens=in_tok or None, output_tokens=out_tok or None,
                success=True,
            )
        elif node_type == "agent":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            task = (data.get("taskDescription") or data.get("task") or "Solve the given task using the available tools.").strip()
            if not isinstance(task, str):
                task = str(task)
            prompt_text = _apply_variables(task, variables, prev_output=prev)
            if "{{input}}" not in task and prev and prev.strip():
                prompt_text = prompt_text + "\n\n" + prev
            if conversation_prefix:
                prompt_text = conversation_prefix + prompt_text
            # --- Context injection ---
            ctx_result = None
            _ctx_for_agent = None
            _ctx_resolved = resolve_node_context(node, context, variables, input_text, org_id, execution_mode, deployment_id=deployment_id)
            if _ctx_resolved:
                _ctx_text = _ctx_resolved["final_text"]
                _ctx_loc = _ctx_resolved["injection_location"]
                if _ctx_loc == "prepend_to_system":
                    _ctx_for_agent = _ctx_text
                elif _ctx_loc == "prepend_to_prompt":
                    prompt_text = _ctx_text + "\n\n" + prompt_text
                elif _ctx_loc == "append_to_prompt":
                    prompt_text = prompt_text + "\n\n" + _ctx_text
                ctx_result = _ctx_resolved
            # --- End context injection ---
            _t0_agent = time.perf_counter()
            try:
                result = _execute_agent_node(node_id, node, prompt_text, org_id, context_text=_ctx_for_agent, workflow_id=workflow_id, scope_value=(variables or {}).get("_sm_scope_value"))
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_agent) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="agent",
                    target_type=None, provider_label=(data.get("provider") or "unknown").strip().lower(),
                    model_name=data.get("modelName"), endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No API key configured for this provider. Add a key in Settings → Integrations. (node_id={node_id})",
                    ) from e
                raise
            context[node_id] = result
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            out_tok = result.get("tokens") or 0
            in_tok = result.get("input_tokens") or 0
            nr_entry = {
                "node_id": node_id,
                "type": "agent",
                "latency_ms": result["latency_ms"],
                "tokens": out_tok,
                "tokens_output": out_tok,
                "input_tokens": in_tok,
                "tokens_input": in_tok,
                "cost": result["cost"],
                "cost_usd": result["cost"],
                "model": result.get("model"),
                "provider": result.get("provider"),
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
                "prompt_after_interpolation": prompt_text[:2000] + ("..." if len(prompt_text) > 2000 else ""),
                "tool_calls_count": result.get("tool_calls_count", 0),
                "tool_calls": result.get("tool_calls", []),
                "iterations": result.get("iterations", 1),
                "reasoning_steps": result.get("reasoning_steps", []),
                "agent_steps_count": result.get("agent_steps_count", 0),
            }
            for _k in ("target_type", "model_registry_id", "endpoint_id", "resolved_base_url"):
                if result.get(_k):
                    nr_entry[_k] = result[_k]
            if result.get("output_warning"):
                nr_entry["output_warning"] = result["output_warning"]
                nr_entry["status"] = "warning"
            nr_entry["context_trace"] = build_context_trace(ctx_result, data)
            node_results.append(nr_entry)
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="agent",
                target_type=result.get("target_type"), provider_label=(result.get("provider") or "unknown").lower(),
                model_name=result.get("model"), endpoint_id=result.get("endpoint_id"),
                resolved_base_url=result.get("resolved_base_url"),
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=result["latency_ms"],
                provider_latency_ms=result.get("provider_latency_ms"),
                gateway_overhead_ms=result.get("gateway_overhead_ms"),
                input_tokens=in_tok or None, output_tokens=out_tok or None,
                success=True,
            )
        elif node_type == "loop":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            max_iter = int(data.get("maxIterations") or 5)
            exit_type = (data.get("exitConditionType") or "none").strip().lower()
            exit_value = (data.get("exitConditionValue") or "").strip()
            judge_prompt = (data.get("llmJudgePrompt") or "").strip()
            judge_provider = (data.get("llmJudgeProvider") or "").strip()
            judge_model = (data.get("llmJudgeModel") or "").strip()

            # Find the first downstream AI/tool_call node to loop over
            downstream_edges = edges_out.get(node_id, [])
            loop_body_node_id = None
            loop_body_node = None
            for edge in downstream_edges:
                target_id = edge.get("target")
                if not target_id:
                    continue
                dn = nodes_by_id.get(target_id)
                if dn and (dn.get("type") or "").lower() in ("ai-step", "tool_call"):
                    loop_body_node_id = target_id
                    loop_body_node = dn
                    break

            _t0_loop = time.perf_counter()
            loop_output = prev
            loop_cost = 0.0
            loop_latency = 0
            loop_tokens = 0
            iterations_run = 0

            for i in range(max_iter):
                iterations_run = i + 1

                if loop_body_node:
                    body_type = (loop_body_node.get("type") or "").lower()
                    body_data = loop_body_node.get("data") or {}

                    if body_type == "ai-step":
                        task = (body_data.get("taskDescription") or "Respond to the user.").strip()
                        body_prompt = _apply_variables(task, variables, prev_output=loop_output)
                        sys_instr = (body_data.get("systemInstructions") or "").strip()
                        if sys_instr:
                            sys_instr = _apply_variables(sys_instr, variables)
                            body_prompt = sys_instr + "\n\n" + body_prompt
                        body_model_node = {"id": loop_body_node_id, "type": "model", "data": {**body_data, "modelName": body_data.get("modelName") or "gpt-3.5-turbo", "provider": body_data.get("provider") or "OpenAI"}}
                        try:
                            step_result = _execute_model_node(loop_body_node_id, body_model_node, body_prompt, org_id)
                            loop_output = step_result.get("output") or step_result.get("response") or ""
                            loop_cost += step_result.get("cost", 0)
                            loop_latency += step_result.get("latency_ms", 0)
                            loop_tokens += step_result.get("tokens", 0)
                        except Exception as e:
                            _logger.warning("Loop body AI step failed at iteration %d: %s", i + 1, e)
                            break

                    elif body_type == "tool_call":
                        task = (body_data.get("taskDescription") or "Respond using the tools.").strip()
                        body_prompt = _apply_variables(task, variables, prev_output=loop_output)
                        try:
                            step_result = _execute_tool_call_node(loop_body_node_id, loop_body_node, body_prompt, org_id)
                            loop_output = step_result.get("output") or step_result.get("response") or ""
                            loop_cost += step_result.get("cost", 0)
                            loop_latency += step_result.get("latency_ms", 0)
                            loop_tokens += step_result.get("tokens", 0)
                        except Exception as e:
                            _logger.warning("Loop body tool_call failed at iteration %d: %s", i + 1, e)
                            break

                # Check exit condition
                if exit_type == "none":
                    continue
                elif exit_type == "contains":
                    if exit_value and exit_value.lower() in loop_output.lower():
                        break
                elif exit_type == "equals":
                    if loop_output.strip().lower() == exit_value.lower():
                        break
                elif exit_type == "not_contains":
                    if exit_value and exit_value.lower() not in loop_output.lower():
                        break
                elif exit_type == "llm_judge":
                    if judge_provider and judge_model and judge_prompt:
                        try:
                            judge_text = judge_prompt.replace("{{output}}", loop_output[:2000])
                            judge_payload = _PromptPayload(
                                org_id=org_id, provider=judge_provider, model=judge_model,
                                prompt=judge_text, prompt_id=f"loop-judge-{node_id}-{i}"
                            )
                            router_mod = _ROUTER_MAP.get(judge_provider.lower())
                            if router_mod:
                                judge_result = router_mod.handle_prompt(judge_payload)
                                judge_answer = (judge_result.get("response") or "").strip().lower()
                                loop_cost += judge_result.get("cost_usd", 0)
                                if any(w in judge_answer for w in ("yes", "true", "stop", "exit", "done")):
                                    break
                        except Exception as e:
                            _logger.warning("LLM judge failed at iteration %d: %s", i + 1, e)

            _loop_total_ms = int((time.perf_counter() - _t0_loop) * 1000)

            # Set the loop body node in context too so downstream nodes after the body can be skipped
            if loop_body_node_id:
                context[loop_body_node_id] = loop_output
            context[node_id] = loop_output
            total_cost += loop_cost
            total_latency += _loop_total_ms
            node_results.append({
                "node_id": node_id,
                "type": "loop",
                "latency_ms": _loop_total_ms,
                "tokens": loop_tokens,
                "cost": loop_cost,
                "output": loop_output[:200] + ("..." if len(loop_output) > 200 else ""),
                "iterations_run": iterations_run,
            })
        elif node_type == "human_review":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            reviewer_instructions = (data.get("reviewerInstructions") or data.get("instructions") or "").strip()

            _t0_review = time.perf_counter()

            if execution_mode == "draft":
                # In draft/simulation mode, pass through without pausing
                review_output = prev
                review_status = "passed_through"
            else:
                # In production mode, create a pending review
                try:
                    review_record = supabase.table("pending_reviews").insert({
                        "org_id": org_id,
                        "workflow_id": workflow_id or "",
                        "node_id": node_id,
                        "status": "pending",
                        "input_data": prev[:5000] if prev else "",
                        "reviewer_instructions": reviewer_instructions,
                    }).execute()
                    review_id = review_record.data[0]["id"] if review_record.data else "unknown"
                    review_output = f"[Awaiting human review: {review_id}]"
                    review_status = "pending"
                except Exception as e:
                    _logger.error("Failed to create pending review: %s", e)
                    review_output = prev
                    review_status = "error"

            _review_ms = int((time.perf_counter() - _t0_review) * 1000)
            context[node_id] = review_output
            node_results.append({
                "node_id": node_id,
                "type": "human_review",
                "latency_ms": _review_ms,
                "tokens": 0,
                "cost": 0,
                "output": review_output[:200],
                "review_status": review_status,
            })

        elif node_type == "model":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            if conversation_prefix:
                prev = conversation_prefix + prev
            _t0_model = time.perf_counter()
            try:
                result = _execute_model_node(node_id, node, prev, org_id)
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_model) * 1000)
                _m_data = node.get("data") or {}
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="model",
                    target_type=None, provider_label=(_m_data.get("provider") or "unknown").strip().lower(),
                    model_name=_m_data.get("modelName"), endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No API key configured for this provider. Add a key in Settings → Integrations. (node_id={node_id})",
                    ) from e
                raise
            context[node_id] = result
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            out_tok = result.get("tokens") or 0
            in_tok = result.get("input_tokens") or 0
            nr_entry = {
                "node_id": node_id,
                "type": "model",
                "latency_ms": result["latency_ms"],
                "tokens": out_tok,
                "tokens_output": out_tok,
                "input_tokens": in_tok,
                "tokens_input": in_tok,
                "cost": result["cost"],
                "cost_usd": result["cost"],
                "model": result.get("model"),
                "provider": result.get("provider"),
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
            }
            for _k in ("target_type", "model_registry_id", "endpoint_id", "resolved_base_url"):
                if result.get(_k):
                    nr_entry[_k] = result[_k]
            # Output quality: propagate warning as soft error
            if result.get("output_warning"):
                nr_entry["output_warning"] = result["output_warning"]
                nr_entry["status"] = "warning"
            node_results.append(nr_entry)
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="model",
                target_type=result.get("target_type"), provider_label=(result.get("provider") or "unknown").lower(),
                model_name=result.get("model"), endpoint_id=result.get("endpoint_id"),
                resolved_base_url=result.get("resolved_base_url"),
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=result["latency_ms"],
                provider_latency_ms=result.get("provider_latency_ms"),
                gateway_overhead_ms=result.get("gateway_overhead_ms"),
                input_tokens=in_tok or None, output_tokens=out_tok or None,
                success=True,
            )
        elif node_type == "vision_step":
            prompt = _apply_variables(str(data.get("prompt") or "Describe this image."), variables)
            image_source = (data.get("image_source") or data.get("imageSource") or "input").strip()
            image_var = str(data.get("image_variable") or data.get("imageVariable") or "image").strip()
            if image_source == "input":
                image_data = variables.get(image_var) or variables.get("image") or ""
            else:
                image_data = _get_previous_output(context, from_node_id or "") if from_node_id else ""
            if isinstance(image_data, dict):
                image_data = image_data.get("output") or image_data.get("url") or ""
            image_url = (str(image_data or "")).strip()
            if not image_url:
                raise HTTPException(status_code=400, detail=f"Vision step needs an image. Set variable '{image_var}' (or 'image') in the draft input and upload an image, or connect a node that outputs an image. (node_id={node_id})")
            provider = (data.get("provider") or "openai").strip().lower() or "openai"
            model = str(data.get("model") or data.get("modelName") or "").strip()
            if not model:
                model = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5-20250929", "gemini": "gemini-2.5-flash", "mistral": "mistral-small-3.2", "cohere": "command-a-vision-07-2025", "groq": "llama-3.2-90b-vision-preview", "together": "meta/llama-3.2-90b-vision-instruct-turbo", "deepseek": "deepseek-chat", "fireworks": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"}.get(provider, "gpt-4o-mini")
            _t0_vision = time.perf_counter()
            try:
                if provider == "openai":
                    v_result = openai_router.handle_vision(openai_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "anthropic":
                    v_result = anthropic_router.handle_vision(anthropic_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "gemini":
                    v_result = gemini_router.handle_vision(gemini_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "mistral":
                    v_result = mistral_router.handle_vision(mistral_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "cohere":
                    v_result = cohere_router.handle_vision(cohere_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "groq":
                    v_result = groq_router.handle_vision(groq_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "together":
                    v_result = together_router.handle_vision(together_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "deepseek":
                    v_result = deepseek_router.handle_vision(deepseek_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                elif provider == "fireworks":
                    v_result = fireworks_router.handle_vision(fireworks_router.VisionPayload(org_id=org_id, provider=provider, model=model, prompt=prompt, image_url=image_url, prompt_id=f"workflow-{node_id}"))
                else:
                    raise HTTPException(status_code=400, detail=f"Vision not supported for provider: {provider}. Use openai, anthropic, gemini, mistral, cohere, groq, together, deepseek, or fireworks.")
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_vision) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="vision_step",
                    target_type=None, provider_label=provider,
                    model_name=model, endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No API key for {provider}. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            _total_vision_ms = int((time.perf_counter() - _t0_vision) * 1000)
            _prov_vision_ms = v_result.get("provider_latency_ms") or v_result.get("latency_ms")
            _oh_vision_ms = max(_total_vision_ms - _prov_vision_ms, 0) if _prov_vision_ms is not None else None
            out_text = v_result.get("response") or v_result.get("output") or ""
            # Output quality check for vision
            _v_warning: str | None = None
            _v_stripped = (out_text or "").strip()
            if not _v_stripped:
                _v_warning = "empty_output"
            elif _v_stripped.lower().startswith(("i'm sorry", "i cannot", "i can't", "as an ai")):
                _v_warning = "refusal"
            result = {"output": out_text, "content_type": "text", "cost": v_result.get("cost_usd") or 0.0, "latency_ms": _total_vision_ms}
            context[node_id] = {"output": result["output"], "content_type": "text"}
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            _v_in_tok = v_result.get("input_tokens") or None
            _v_out_tok = v_result.get("output_tokens") or None
            _v_nr = {"node_id": node_id, "type": "vision_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": (out_text or "")[:200], "content_type": "text", "model": model, "provider": provider}
            if _v_warning:
                _v_nr["output_warning"] = _v_warning
                _v_nr["status"] = "warning"
            node_results.append(_v_nr)
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="vision_step",
                target_type=None, provider_label=provider,
                model_name=model, endpoint_id=None, resolved_base_url=None,
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=_total_vision_ms, provider_latency_ms=_prov_vision_ms,
                gateway_overhead_ms=_oh_vision_ms,
                input_tokens=_v_in_tok, output_tokens=_v_out_tok, success=True,
            )
        elif node_type == "image_gen_step":
            prompt_source = (data.get("prompt_source") or data.get("promptSource") or "static").strip().lower()
            if prompt_source == "previous_step" and from_node_id:
                prompt = _get_previous_output(context, from_node_id)
            elif prompt_source == "input":
                prompt_var = str(data.get("prompt_variable") or data.get("promptVariable") or "prompt").strip()
                prompt = str(variables.get(prompt_var) or "").strip()
            else:
                prompt = str(data.get("prompt") or "").strip()
            prompt = _apply_variables(prompt, variables) if prompt else "A beautiful image."
            provider = (data.get("provider") or "openai").strip().lower() or "openai"
            model = str(data.get("model") or data.get("modelName") or "").strip()
            size = str(data.get("size") or "1024x1024").strip() or "1024x1024"
            quality = str(data.get("quality") or "standard").strip().lower() or "standard"
            if not model:
                model = "dall-e-3" if provider == "openai" else "imagen-3.0-generate-002" if provider == "gemini" else "accounts/fireworks/models/flux-1-schnell"
            negative_prompt = (data.get("negative_prompt") or "").strip() or None
            _t0_imggen = time.perf_counter()
            try:
                if provider == "openai":
                    img_result = openai_router.handle_image_generation(openai_router.ImageGenerationPayload(
                        org_id=org_id, prompt=prompt, size=size, model=model, quality=quality, prompt_id=f"workflow-{node_id}", negative_prompt=negative_prompt,
                    ))
                elif provider == "gemini":
                    img_result = gemini_router.handle_image_generation(gemini_router.ImageGenerationPayload(
                        org_id=org_id, prompt=prompt, model=model, prompt_id=f"workflow-{node_id}", negative_prompt=negative_prompt,
                    ))
                elif provider == "fireworks":
                    img_result = fireworks_router.handle_image_generation(fireworks_router.ImageGenerationPayload(
                        org_id=org_id, prompt=prompt, model=model, prompt_id=f"workflow-{node_id}", negative_prompt=negative_prompt,
                    ))
                else:
                    raise HTTPException(status_code=400, detail=f"Image generation not supported for provider: {provider}. Use openai, gemini, or fireworks.")
                _total_imggen_ms = int((time.perf_counter() - _t0_imggen) * 1000)
                _prov_imggen_ms = img_result.get("provider_latency_ms") or img_result.get("latency_ms")
                _oh_imggen_ms = max(_total_imggen_ms - _prov_imggen_ms, 0) if _prov_imggen_ms is not None else None
                image_url = img_result.get("url") or ""
                result = {
                    "output": image_url,
                    "content_type": "image_url",
                    "cost": img_result.get("cost_usd") or 0.0,
                    "latency_ms": _total_imggen_ms,
                }
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_imggen) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="image_gen_step",
                    target_type=None, provider_label=provider,
                    model_name=model, endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No API key for {provider}. Add a key in Settings → Integrations. (node_id={node_id})",
                    ) from e
                raise
            context[node_id] = {"output": result["output"], "content_type": "image_url"}
            last_content_type = "image_url"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({
                "node_id": node_id, "type": "image_gen_step", "latency_ms": result["latency_ms"], "cost": result["cost"],
                "output": (result["output"] or "")[:200], "content_type": "image_url", "model": model, "provider": provider,
            })
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="image_gen_step",
                target_type=None, provider_label=provider,
                model_name=model, endpoint_id=None, resolved_base_url=None,
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=_total_imggen_ms, provider_latency_ms=_prov_imggen_ms,
                gateway_overhead_ms=_oh_imggen_ms,
                input_tokens=None, output_tokens=None, success=True,
            )
        elif node_type == "tts_step":
            text_source = (data.get("text_source") or data.get("textSource") or data.get("input_source") or "previous_step").strip().lower()
            if text_source == "previous_step" and from_node_id:
                text = _get_previous_output(context, from_node_id)
            elif text_source == "input":
                text_var = str(data.get("input_variable") or data.get("inputVariable") or "text").strip()
                text = str(variables.get(text_var) or "").strip()
            else:
                text = str(data.get("text") or data.get("prompt") or data.get("input_text") or "").strip()
            text = _apply_variables(text, variables) if text else " "
            provider = (data.get("provider") or "openai").strip().lower() or "openai"
            if provider == "openai":
                model = str(data.get("model") or data.get("modelName") or "tts-1").strip() or "tts-1"
                voice = str(data.get("voice") or "alloy").strip().lower() or "alloy"
            else:
                model = str(data.get("model") or data.get("modelName") or "canopylabs/orpheus-v1-english").strip() or "canopylabs/orpheus-v1-english"
                voice = str(data.get("voice") or "austin").strip().lower() or "austin"
            if provider not in ("openai", "groq"):
                raise HTTPException(status_code=400, detail=f"TTS supported for OpenAI or Groq. (node_id={node_id})")
            _t0_tts = time.perf_counter()
            try:
                if provider == "openai":
                    tts_result = openai_router.handle_tts(openai_router.TTSPayload(
                        org_id=org_id, text=text, model=model, voice=voice, prompt_id=f"workflow-{node_id}",
                    ))
                else:
                    tts_result = groq_router.handle_tts(groq_router.TTSPayload(
                        org_id=org_id, text=text, model=model, voice=voice, prompt_id=f"workflow-{node_id}",
                    ))
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_tts) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="tts_step",
                    target_type=None, provider_label=provider,
                    model_name=model, endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No {provider} API key. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            _total_tts_ms = int((time.perf_counter() - _t0_tts) * 1000)
            _prov_tts_ms = tts_result.get("provider_latency_ms") or tts_result.get("latency_ms")
            _oh_tts_ms = max(_total_tts_ms - _prov_tts_ms, 0) if _prov_tts_ms is not None else None
            result = {"output": tts_result.get("output") or "", "content_type": "audio_url", "cost": tts_result.get("cost_usd") or 0.0, "latency_ms": _total_tts_ms}
            context[node_id] = {"output": result["output"], "content_type": "audio_url"}
            last_content_type = "audio_url"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "tts_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": "[audio]", "content_type": "audio_url", "model": model, "provider": provider})
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="tts_step",
                target_type=None, provider_label=provider,
                model_name=model, endpoint_id=None, resolved_base_url=None,
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=_total_tts_ms, provider_latency_ms=_prov_tts_ms,
                gateway_overhead_ms=_oh_tts_ms,
                input_tokens=None, output_tokens=None, success=True,
            )
        elif node_type == "stt_step":
            audio_var = str(data.get("audio_variable") or "audio_file").strip()
            audio_input = variables.get(audio_var)
            if audio_input is None and from_node_id:
                audio_input = _get_previous_output(context, from_node_id)
            if not audio_input:
                raise HTTPException(status_code=400, detail=f"STT needs audio: set variable '{audio_var}' or connect previous node. (node_id={node_id})")
            if isinstance(audio_input, dict):
                audio_input = audio_input.get("output") or audio_input.get("data") or ""
            s = str(audio_input).strip()
            if s.startswith("data:"):
                idx = s.find("base64,")
                audio_base64 = s[idx + 7:] if idx >= 0 else ""
            else:
                audio_base64 = s
            if not audio_base64:
                raise HTTPException(status_code=400, detail="STT audio must be base64 or data URL. (node_id={node_id})")
            # Fail fast: Whisper has a 25 MB limit. ~25 MB in base64 is ~34 M chars.
            if len(audio_base64) > 35_000_000:
                raise HTTPException(
                    status_code=400,
                    detail=f"STT audio is too large (Whisper supports up to 25 MB). Use a shorter clip or extract audio from video. (node_id={node_id})",
                )
            provider = (data.get("provider") or "openai").strip().lower() or "openai"
            model = str(data.get("model") or data.get("modelName") or "").strip() or ("whisper-1" if provider == "openai" else "whisper-v3-turbo")
            if provider not in ("openai", "fireworks"):
                raise HTTPException(status_code=400, detail=f"STT supported for OpenAI or Fireworks. (node_id={node_id})")
            stt_prompt = (data.get("prompt") or "").strip() or None
            _t0_stt = time.perf_counter()
            try:
                if provider == "openai":
                    stt_result = openai_router.handle_stt(openai_router.STTPayload(org_id=org_id, audio_base64=audio_base64, model=model, prompt_id=f"workflow-{node_id}", prompt=stt_prompt))
                else:
                    stt_result = fireworks_router.handle_stt(fireworks_router.STTPayload(org_id=org_id, audio_base64=audio_base64, model=model, prompt_id=f"workflow-{node_id}", prompt=stt_prompt))
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_stt) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="stt_step",
                    target_type=None, provider_label=provider,
                    model_name=model, endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No {provider} API key. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            _total_stt_ms = int((time.perf_counter() - _t0_stt) * 1000)
            _prov_stt_ms = stt_result.get("provider_latency_ms") or stt_result.get("latency_ms")
            _oh_stt_ms = max(_total_stt_ms - _prov_stt_ms, 0) if _prov_stt_ms is not None else None
            result = {"output": stt_result.get("output") or "", "content_type": "text", "cost": stt_result.get("cost_usd") or 0.0, "latency_ms": _total_stt_ms}
            context[node_id] = {"output": result["output"], "content_type": "text"}
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "stt_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": (result["output"] or "")[:200], "content_type": "text", "model": model, "provider": provider})
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="stt_step",
                target_type=None, provider_label=provider,
                model_name=model, endpoint_id=None, resolved_base_url=None,
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=_total_stt_ms, provider_latency_ms=_prov_stt_ms,
                gateway_overhead_ms=_oh_stt_ms,
                input_tokens=None, output_tokens=None, success=True,
            )
        elif node_type == "embedding_step":
            text_source = (data.get("text_source") or data.get("textSource") or "previous_step").strip().lower()
            if text_source == "previous_step" and from_node_id:
                text = _get_previous_output(context, from_node_id)
            else:
                text = str(data.get("text") or data.get("prompt") or "").strip()
            text = _apply_variables(text, variables) if text else " "
            provider = (data.get("provider") or "openai").strip().lower() or "openai"
            _emb_defaults = {"openai": "text-embedding-3-small", "gemini": "text-embedding-005", "mistral": "mistral-embed", "cohere": "embed-english-v3.0", "together": "togethercomputer/m2-bert-80M-8k-retrieval", "deepseek": "deepseek-embedding-v2", "fireworks": "nomic-ai/nomic-embed-text-v1.5"}
            model = str(data.get("model") or data.get("modelName") or "").strip() or _emb_defaults.get(provider, "text-embedding-3-small")
            if provider not in ("openai", "gemini", "mistral", "cohere", "together", "deepseek", "fireworks"):
                raise HTTPException(status_code=400, detail=f"Embedding supported for openai, gemini, mistral, cohere, together, deepseek, fireworks. (node_id={node_id})")
            _t0_emb = time.perf_counter()
            try:
                if provider == "openai":
                    emb_result = openai_router.handle_embedding(openai_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
                elif provider == "gemini":
                    emb_result = gemini_router.handle_embedding(gemini_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
                elif provider == "mistral":
                    emb_result = mistral_router.handle_embedding(mistral_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
                elif provider == "cohere":
                    emb_result = cohere_router.handle_embedding(cohere_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
                elif provider == "together":
                    emb_result = together_router.handle_embedding(together_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
                elif provider == "deepseek":
                    emb_result = deepseek_router.handle_embedding(deepseek_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
                else:
                    emb_result = fireworks_router.handle_embedding(fireworks_router.EmbeddingPayload(org_id=org_id, text=text, model=model, prompt_id=f"workflow-{node_id}"))
            except HTTPException as e:
                _fail_ms = int((time.perf_counter() - _t0_emb) * 1000)
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="embedding_step",
                    target_type=None, provider_label=provider,
                    model_name=model, endpoint_id=None, resolved_base_url=None,
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=_fail_ms, provider_latency_ms=None, gateway_overhead_ms=None,
                    input_tokens=None, output_tokens=None, success=False,
                    error_type=_classify_error(e, e.detail), http_status=e.status_code,
                )
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No {provider} API key. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            _total_emb_ms = int((time.perf_counter() - _t0_emb) * 1000)
            _prov_emb_ms = emb_result.get("provider_latency_ms") or emb_result.get("latency_ms")
            _oh_emb_ms = max(_total_emb_ms - _prov_emb_ms, 0) if _prov_emb_ms is not None else None
            _emb_in_tok = emb_result.get("input_tokens") or None
            out_str = emb_result.get("output") or "[]"
            result = {"output": out_str, "content_type": "embedding", "cost": emb_result.get("cost_usd") or 0.0, "latency_ms": _total_emb_ms}
            context[node_id] = {"output": result["output"], "content_type": "embedding"}
            last_content_type = "embedding"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "embedding_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": f"[{len(emb_result.get('embedding') or [])}d]", "content_type": "embedding", "model": model, "provider": provider})
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="embedding_step",
                target_type=None, provider_label=provider,
                model_name=model, endpoint_id=None, resolved_base_url=None,
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=_total_emb_ms, provider_latency_ms=_prov_emb_ms,
                gateway_overhead_ms=_oh_emb_ms,
                input_tokens=_emb_in_tok, output_tokens=None, success=True,
            )
        elif node_type == "optimizer":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            prompt = (data.get("prompt") or data.get("taskDescription") or "Respond to the user.").strip()
            if not isinstance(prompt, str):
                prompt = str(prompt)
            prompt_text = _apply_variables(prompt, variables, prev_output=prev)
            if conversation_prefix:
                prompt_text = conversation_prefix + prompt_text
            sys_instructions = (data.get("systemInstructions") or "").strip()
            if sys_instructions:
                sys_instructions = _apply_variables(sys_instructions, variables)
                prompt_text = sys_instructions + "\n\n" + prompt_text
            priority = (data.get("priority") or "cheapest").lower()
            max_cost = data.get("maxCostPerCall")
            if max_cost is not None:
                max_cost = float(max_cost)
            max_latency = data.get("maxLatencyMs")
            if max_latency is not None:
                max_latency = int(max_latency)
            allowed_models = data.get("allowedModels")
            if allowed_models is not None and not isinstance(allowed_models, list):
                allowed_models = None
            excluded_models = list(data.get("excludedModels") or [])
            manual_model = (data.get("manualModel") or "gpt-4o-mini").strip() or "gpt-4o-mini"
            manual_provider = (data.get("manualProvider") or "openai").strip().lower() or "openai"

            history = _get_model_performance_history(org_id, workflow_id or "", limit=200)
            run_count = len(history)
            if run_count >= OPTIMIZER_THRESHOLD:
                selected_model, selected_provider, selection_reason = _select_optimal_model(
                    history, priority, max_cost, max_latency, allowed_models, excluded_models
                )
                mode = "auto"
                candidates_evaluated = len(set(h.get("model") for h in history if h.get("model")))
            else:
                selected_model = manual_model
                selected_provider = manual_provider
                selection_reason = f"manual (need {OPTIMIZER_THRESHOLD - run_count} more runs for auto)"
                mode = "manual"
                candidates_evaluated = 0

            synthetic_node = {
                "id": node_id,
                "type": "model",
                "data": {
                    "provider": selected_provider,
                    "modelName": selected_model,
                },
            }
            # Use _safe variant so provider failures are recorded in node_results
            # instead of being silently lost. This feeds the optimizer's error-rate
            # signal which filters out unreliable models.
            result = _execute_model_node_safe(node_id, synthetic_node, prompt_text, org_id)

            if result.get("error"):
                # Record the error in node_results so the optimizer learns from it,
                # but still raise so the workflow reports failure to the caller.
                node_results.append({
                    "node_id": node_id,
                    "type": "optimizer",
                    "latency_ms": result["latency_ms"],
                    "tokens": 0,
                    "cost": 0,
                    "cost_usd": 0,
                    "model": selected_model,
                    "provider": selected_provider,
                    "output": "",
                    "error": True,
                    "status": "error",
                    "error_detail": result.get("error_detail", ""),
                    "selection_reason": selection_reason,
                    "mode": mode,
                    "priority": priority,
                    "candidates_evaluated": candidates_evaluated,
                })
                _record_latency_fact(
                    org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                    node_id=node_id, node_type="optimizer",
                    target_type=result.get("target_type"), provider_label=selected_provider.lower(),
                    model_name=selected_model, endpoint_id=result.get("endpoint_id"),
                    resolved_base_url=result.get("resolved_base_url"),
                    endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                    total_latency_ms=result["latency_ms"],
                    provider_latency_ms=result.get("provider_latency_ms"),
                    gateway_overhead_ms=result.get("gateway_overhead_ms"),
                    input_tokens=None, output_tokens=None,
                    success=False,
                    error_type=_classify_error(None, result.get("error_detail")),
                    http_status=result.get("error_status"),
                )
                # Run persistence is now handled by the top-level except handler
                # around the while loop — no need to duplicate it here.

                error_status = result.get("error_status", 500)
                error_detail = result.get("error_detail", "Provider call failed")
                if error_status == 404 or "API key" in str(error_detail):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No API key configured for this provider. Add a key in Settings → Integrations. (node_id={node_id})",
                    )
                raise HTTPException(status_code=error_status, detail=error_detail)

            context[node_id] = result
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            out_tok = result.get("tokens") or 0
            in_tok = result.get("input_tokens") or 0
            node_results.append({
                "node_id": node_id,
                "type": "optimizer",
                "latency_ms": result["latency_ms"],
                "tokens": out_tok,
                "tokens_output": out_tok,
                "input_tokens": in_tok,
                "tokens_input": in_tok,
                "cost": result["cost"],
                "cost_usd": result["cost"],
                "model": selected_model,
                "provider": selected_provider,
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
                "selection_reason": selection_reason,
                "mode": mode,
                "priority": priority,
                "candidates_evaluated": candidates_evaluated,
            })
            _record_latency_fact(
                org_id=org_id, workflow_run_id=None, workflow_id=workflow_id,
                node_id=node_id, node_type="optimizer",
                target_type=result.get("target_type"), provider_label=selected_provider.lower(),
                model_name=selected_model, endpoint_id=result.get("endpoint_id"),
                resolved_base_url=result.get("resolved_base_url"),
                endpoint_slug=endpoint_slug, version=version, execution_mode=execution_mode,
                total_latency_ms=result["latency_ms"],
                provider_latency_ms=result.get("provider_latency_ms"),
                gateway_overhead_ms=result.get("gateway_overhead_ms"),
                input_tokens=in_tok or None, output_tokens=out_tok or None,
                success=True,
            )
        elif node_type == "condition":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            operator = (data.get("operator") or "contains").lower()
            value = str(data.get("value") or "")
            cond_result = _evaluate_condition(operator, value, prev)
            # Store both result and the incoming text so downstream model gets the prompt, not the dict
            context[node_id] = {"condition_result": cond_result, "output": prev}
            branch_taken = "true" if cond_result else "false"
            node_results.append({
                "node_id": node_id,
                "type": "condition",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "output": branch_taken,
                "branch_taken": branch_taken,
            })
            out_edges = edges_out.get(node_id) or []
            branch = "true" if cond_result else "false"
            for e in out_edges:
                sh = (e.get("sourceHandle") or e.get("data", {}).get("branch") or "").lower()
                if sh == branch or (not sh and len(out_edges) == 2 and branch == "true" and e == out_edges[0]) or (not sh and len(out_edges) == 2 and branch == "false" and e == out_edges[1]):
                    tid = e.get("target")
                    if tid:
                        queue.append((tid, node_id))
                    break
            else:
                if out_edges:
                    tid = out_edges[0].get("target") if cond_result and len(out_edges) > 0 else (out_edges[1].get("target") if len(out_edges) > 1 else None)
                    if tid:
                        queue.append((tid, node_id))
            continue
        elif node_type == "router":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            raw_strategy = (data.get("strategy") or data.get("primaryModel") or "balanced").lower()
            # Normalize aliases to backend strategy names
            if raw_strategy in ("lowest_cost",):
                strategy = "cheapest"
            elif raw_strategy in ("lowest_latency",):
                strategy = "fastest"
            elif raw_strategy == "fallback":
                strategy = "fallback"
            elif raw_strategy not in ("cheapest", "fastest", "balanced"):
                strategy = "balanced"
            else:
                strategy = raw_strategy
            out_edges = edges_out.get(node_id) or []
            chosen = _select_router_edge(out_edges, nodes_by_id, strategy, prev)
            # Build list of candidate node ids for trace
            candidate_ids = []
            for e in out_edges:
                tid = e.get("target")
                if not tid:
                    continue
                n = nodes_by_id.get(tid)
                nt = (n.get("type") or "").lower() if n else ""
                if n and (nt == "model" or nt == "ai-step"):
                    candidate_ids.append(tid)
            # Store incoming text as output so downstream model gets the prompt, not the routing dict
            context[node_id] = {"chosen_model_node": chosen, "output": prev}
            chosen_node = nodes_by_id.get(chosen, {}) if chosen else {}
            chosen_data = chosen_node.get("data") or {}
            selected_provider = (chosen_data.get("provider") or "").strip() or "openai"
            selected_model = (chosen_data.get("modelName") or chosen_data.get("base_model") or "").strip() or ""
            node_results.append({
                "node_id": node_id,
                "type": "router",
                "strategy": strategy,
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "output": chosen or "",
                "router_selected_node_id": chosen,
                "selected": chosen,
                "candidates": candidate_ids,
                "router_selected_provider": selected_provider,
                "router_selected_model": selected_model,
            })
            if chosen:
                queue.append((chosen, node_id))
            continue
        elif node_type == "output":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            context[node_id] = prev
            node_results.append({
                "node_id": node_id,
                "type": "output",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "output": prev[:500] + ("..." if len(prev) > 500 else ""),
            })
            out = {
                "final_output": prev,
                "content_type": last_content_type,
                "node_results": node_results,
                "total_cost": round(total_cost, 6),
                "total_latency": total_latency,
                "total_latency_ms": total_latency,
                "executed_edges": [{"source": a, "target": b} for a, b in executed_edges],
            }
            if workflow_id:
                try:
                    mode = (execution_mode or "draft").strip() or "draft"
                    if mode not in ("draft", "production", "eval"):
                        mode = "draft"
                    row = {
                        "workflow_id": workflow_id,
                        "org_id": org_id,
                        "user_id": user_id or None,
                        "input_text": input_text[:5000] if input_text else None,
                        "final_output": (prev or "")[:10000] if prev else None,
                        "node_results": node_results,
                        "total_cost": out["total_cost"],
                        "total_latency_ms": total_latency,
                        "endpoint_slug": endpoint_slug if endpoint_slug else None,
                        "version": version,
                        "execution_mode": mode,
                    }
                    _vars_value, _vars_capture = _capture_variables_once()
                    row["variables"] = _vars_value
                    row["variables_capture"] = _vars_capture
                    if experiment_id is not None:
                        row["experiment_id"] = experiment_id
                    if variant_name is not None:
                        row["variant_name"] = variant_name
                    if served_version is not None:
                        row["served_version"] = served_version
                    # Use .execute() only; Supabase returns inserted row(s) by default (no .select() after insert)
                    insert_result = supabase.table("workflow_runs").insert(row).execute()
                    data = insert_result.data
                    first = data[0] if isinstance(data, list) and len(data) > 0 else (data if isinstance(data, dict) else None)
                    if first and first.get("id") is not None:
                        out["run_id"] = str(first["id"])
                except Exception as e:
                    _logger.warning("workflow_runs insert failed: %s", e, exc_info=True)
            return out

        out_edges = edges_out.get(node_id) or []
        for e in out_edges:
            tid = e.get("target")
            if tid:
                queue.append((tid, node_id))

     raise HTTPException(status_code=400, detail="Workflow did not reach an output node")
    except HTTPException as _exec_err:
        # ── Persist failed runs so they appear in error-rate calculations ──
        # Without this, failed model/ai-step/vision/tts/stt/embedding runs
        # simply vanish — the workflow_run is never inserted and the error
        # is invisible to observability, experiments, and rollback monitoring.
        _err_detail = getattr(_exec_err, "detail", str(_exec_err))
        _err_status = getattr(_exec_err, "status_code", 500)
        # Only append an error entry if the failing node didn't already add one
        # (the optimizer node appends its own detailed error entry before raising).
        _already_recorded = (
            node_results
            and node_results[-1].get("status") == "error"
            and node_results[-1].get("node_id") == _cur_node_id
        )
        if not _already_recorded:
            node_results.append({
                "node_id": _cur_node_id or "unknown",
                "type": _cur_node_type or "unknown",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "cost_usd": 0,
                "output": "",
                "model": _cur_model or None,
                "provider": _cur_provider or None,
                "error": True,
                "status": "error",
                "error_detail": str(_err_detail)[:500],
                "error_status": _err_status,
            })
        if workflow_id:
            try:
                _mode = (execution_mode or "draft").strip() or "draft"
                if _mode not in ("draft", "production", "eval"):
                    _mode = "draft"
                _fail_row: dict[str, Any] = {
                    "workflow_id": workflow_id,
                    "org_id": org_id,
                    "user_id": user_id or None,
                    "input_text": (input_text[:5000] if input_text else None),
                    "final_output": None,
                    "node_results": node_results,
                    "total_cost": round(total_cost, 6),
                    "total_latency_ms": total_latency,
                    "endpoint_slug": endpoint_slug if endpoint_slug else None,
                    "version": version,
                    "execution_mode": _mode,
                }
                # Failing inputs are exactly the edge cases an evaluation set
                # needs most, so the error paths capture no less than the
                # success path does.
                _fail_vars_value, _fail_vars_capture = _capture_variables_once()
                _fail_row["variables"] = _fail_vars_value
                _fail_row["variables_capture"] = _fail_vars_capture
                if experiment_id is not None:
                    _fail_row["experiment_id"] = experiment_id
                if variant_name is not None:
                    _fail_row["variant_name"] = variant_name
                if served_version is not None:
                    _fail_row["served_version"] = served_version
                _logger.info(
                    "workflow_runs INSERT (error path): workflow_id=%s endpoint_slug=%s mode=%s node_results_count=%d error_node=%s",
                    workflow_id, endpoint_slug, _mode, len(node_results), _cur_node_id,
                )
                supabase.table("workflow_runs").insert(_fail_row).execute()
                _logger.info("workflow_runs INSERT (error path): OK")
            except Exception as _ins_err:
                _logger.warning(
                    "workflow_runs INSERT (error path) FAILED — %s: %s  (workflow_id=%s, endpoint_slug=%s)",
                    type(_ins_err).__name__, str(_ins_err)[:500], workflow_id, endpoint_slug,
                    exc_info=True,
                )
        raise
    except Exception as _generic_err:
        # ── Catch non-HTTP errors (TypeError, KeyError, ConnectionError, etc.) ──
        # These bypass the HTTPException handler above, so without this block
        # the workflow_run is never persisted and the error is invisible to
        # observability dashboards.
        _err_detail_generic = str(_generic_err)[:500]
        _logger.warning(
            "execute_workflow: non-HTTP error caught (%s): %s — persisting as failed run",
            type(_generic_err).__name__, _err_detail_generic,
        )
        _already_recorded_g = (
            node_results
            and node_results[-1].get("status") == "error"
            and node_results[-1].get("node_id") == _cur_node_id
        )
        if not _already_recorded_g:
            node_results.append({
                "node_id": _cur_node_id or "unknown",
                "type": _cur_node_type or "unknown",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0,
                "cost_usd": 0,
                "output": "",
                "model": _cur_model or None,
                "provider": _cur_provider or None,
                "error": True,
                "status": "error",
                "error_detail": _err_detail_generic,
                "error_status": 500,
            })
        if workflow_id:
            try:
                _mode_g = (execution_mode or "draft").strip() or "draft"
                if _mode_g not in ("draft", "production", "eval"):
                    _mode_g = "draft"
                _fail_row_g: dict[str, Any] = {
                    "workflow_id": workflow_id,
                    "org_id": org_id,
                    "user_id": user_id or None,
                    "input_text": (input_text[:5000] if input_text else None),
                    "final_output": None,
                    "node_results": node_results,
                    "total_cost": round(total_cost, 6),
                    "total_latency_ms": total_latency,
                    "endpoint_slug": endpoint_slug if endpoint_slug else None,
                    "version": version,
                    "execution_mode": _mode_g,
                }
                _g_vars_value, _g_vars_capture = _capture_variables_once()
                _fail_row_g["variables"] = _g_vars_value
                _fail_row_g["variables_capture"] = _g_vars_capture
                if experiment_id is not None:
                    _fail_row_g["experiment_id"] = experiment_id
                if variant_name is not None:
                    _fail_row_g["variant_name"] = variant_name
                if served_version is not None:
                    _fail_row_g["served_version"] = served_version
                _logger.info(
                    "workflow_runs INSERT (generic-error path): workflow_id=%s endpoint_slug=%s error=%s",
                    workflow_id, endpoint_slug, type(_generic_err).__name__,
                )
                supabase.table("workflow_runs").insert(_fail_row_g).execute()
                _logger.info("workflow_runs INSERT (generic-error path): OK")
            except Exception as _ins_err2:
                _logger.warning(
                    "workflow_runs INSERT (generic-error path) FAILED — %s: %s",
                    type(_ins_err2).__name__, str(_ins_err2)[:500],
                    exc_info=True,
                )
        # Re-raise as HTTPException so callers get a proper HTTP error response
        raise HTTPException(status_code=500, detail=_err_detail_generic) from _generic_err
