"""
SSE streaming for public workflow execution.
When stream=true, the handler runs the workflow (in thread for now) and emits
step_start, step_end, done, then data: [DONE]. For linear workflows with a single
OpenAI (or OpenAI-compatible) AI step, we stream tokens in real time.
"""
import asyncio
import json
import time
from typing import Any, AsyncGenerator

from supabase_client import supabase
from utils.encryption import decrypt_api_key
from utils.pricing import get_pricing
from utils.usage_logger import log_usage
from fastapi import HTTPException

from workflow_runtime import (
    execute_workflow,
    _apply_variables,
    _get_previous_output,
    _nodes_by_id,
    _edges_out,
    _find_input_node,
    _entry_point_ids,
)


def _sse_event(event: str, data: Any) -> str:
    """Format one SSE message: event line + data line (JSON) + blank line."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _get_api_key_sync(org_id: str, provider: str) -> str:
    """Fetch and decrypt API key for org_id + provider (cached). Raises HTTPException if not found."""
    from api_key_cache import get_provider_api_key
    return get_provider_api_key(org_id, provider)


async def _stream_ai_step_openai(
    org_id: str,
    model: str,
    messages: list[dict],
    node_id: str,
) -> AsyncGenerator[dict, None]:
    """Stream OpenAI chat completion; yield token deltas then usage."""
    from openai import AsyncOpenAI

    import httpx

    api_key = await asyncio.to_thread(_get_api_key_sync, org_id, "openai")
    # Timeout: 30s for connection, 120s total for the initial response (not the stream)
    client = AsyncOpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    stream = await client.chat.completions.create(
        model=model, messages=messages, stream=True,
        stream_options={"include_usage": True},
    )
    full_content = ""
    input_tokens = 0
    output_tokens = 0
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            full_content += delta
            yield {"type": "token", "delta": delta}
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens or 0
            output_tokens = chunk.usage.completion_tokens or 0
    pricing = get_pricing("openai", model)
    cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
    log_usage(
        org_id,
        "OpenAI",
        model,
        messages[0].get("content", "") if messages else "",
        full_content,
        f"workflow-{node_id}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost_usd,
    )
    yield {"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}


def _is_linear_single_ai_workflow(graph: dict) -> tuple[bool, list[str] | None]:
    """
    If the workflow is linear (input -> prompt? -> ai-step -> output) with exactly one ai-step
    and no condition/router/optimizer, return (True, ordered node ids). Else (False, None).
    """
    nodes_by_id = _nodes_by_id(graph)
    edges_out = _edges_out(graph)
    input_node_id = _find_input_node(nodes_by_id)
    entry_points = _entry_point_ids(nodes_by_id, edges_out) if not input_node_id else [input_node_id]
    if len(entry_points) != 1:
        return False, None
    path: list[str] = []
    visited = set()
    queue = [entry_points[0]]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = nodes_by_id.get(nid)
        if not node:
            continue
        ntype = (node.get("type") or "").lower()
        if ntype in ("condition", "router", "optimizer"):
            return False, None
        path.append(nid)
        if ntype == "output":
            break
        for e in edges_out.get(nid) or []:
            tid = e.get("target")
            if tid and tid not in visited:
                queue.append(tid)
                break
        else:
            for e in edges_out.get(nid) or []:
                tid = e.get("target")
                if tid:
                    queue.append(tid)
                    break
    ai_count = sum(1 for nid in path if (nodes_by_id.get(nid) or {}).get("type", "").lower() == "ai-step")
    if ai_count != 1:
        return False, None
    has_output = any((nodes_by_id.get(nid) or {}).get("type", "").lower() == "output" for nid in path)
    if not has_output:
        return False, None
    return True, path


async def stream_workflow_async(
    graph_json: dict,
    input_text: str,
    org_id: str,
    request_id: str,
    served_version: int,
    workflow_id: str | None,
    endpoint_slug: str,
    dep_version: int,
    variables: dict[str, Any] | None,
    experiment_id: str | None,
    variant_name: str | None,
    conversation_prefix: str | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE payloads: step_start, token (for streamable AI step), step_end, done, then data: [DONE].
    For linear workflows with one OpenAI AI step, streams tokens; otherwise runs execute_workflow in a thread.
    """
    variables = variables or {}
    nodes_by_id = _nodes_by_id(graph_json)
    edges_out = _edges_out(graph_json)
    start = time.perf_counter()

    linear_ok, path = _is_linear_single_ai_workflow(graph_json)
    use_linear = bool(linear_ok and path)
    if use_linear:
        context: dict[str, Any] = {}
        node_results: list[dict] = []
        total_cost = 0.0
        total_latency = 0
        last_content_type = "text"
        from_node_id: str | None = None
        try:
            for node_id in path:
                node = nodes_by_id.get(node_id)
                if not node:
                    continue
                ntype = (node.get("type") or "").lower()
                data = node.get("data") or {}
                yield _sse_event("step_start", {"node_id": node_id, "type": ntype})

                if ntype == "input":
                    context[node_id] = input_text
                    node_results.append({"node_id": node_id, "type": "input", "latency_ms": 0, "cost": 0, "output": input_text[:200]})
                elif ntype == "prompt":
                    prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
                    formatted = _apply_variables(str(data.get("template") or data.get("preview") or "{{input}}"), variables, prev_output=prev)
                    context[node_id] = formatted
                    node_results.append({"node_id": node_id, "type": "prompt", "latency_ms": 0, "cost": 0, "output": formatted[:200]})
                elif ntype == "ai-step":
                    provider = (data.get("provider") or "OpenAI").strip().lower()
                    if provider != "openai":
                        use_linear = False
                        break
                    prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
                    prompt_text = _apply_variables(str(data.get("taskDescription") or data.get("task") or "Respond to the user."), variables, prev_output=prev)
                    if conversation_prefix:
                        prompt_text = conversation_prefix + prompt_text
                    sys_instructions = _apply_variables(str(data.get("systemInstructions") or data.get("system_prefix") or ""), variables).strip()
                    messages = [{"role": "user", "content": prompt_text}]
                    if sys_instructions:
                        messages = [{"role": "system", "content": sys_instructions}, {"role": "user", "content": prompt_text}]
                    model = (data.get("modelName") or "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
                    step_start = time.perf_counter()
                    full_output = ""
                    cost_usd = 0.0
                    in_tok = out_tok = 0
                    async for ev in _stream_ai_step_openai(org_id, model, messages, node_id):
                        if ev.get("type") == "token":
                            yield _sse_event("token", {"delta": ev.get("delta", "")})
                            full_output += ev.get("delta", "")
                        elif ev.get("type") == "usage":
                            in_tok = ev.get("input_tokens", 0)
                            out_tok = ev.get("output_tokens", 0)
                            cost_usd = ev.get("cost_usd", 0)
                    latency_ms = int((time.perf_counter() - step_start) * 1000)
                    context[node_id] = full_output
                    total_cost += cost_usd
                    total_latency += latency_ms
                    last_content_type = "text"
                    node_results.append({
                        "node_id": node_id, "type": "ai-step", "latency_ms": latency_ms, "cost": cost_usd,
                        "output": full_output[:200], "tokens": out_tok, "input_tokens": in_tok,
                        "model": model, "provider": "openai",
                    })
                elif ntype == "output":
                    prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
                    context[node_id] = prev
                    node_results.append({"node_id": node_id, "type": "output", "latency_ms": 0, "cost": 0, "output": prev[:500]})
                    total_latency += int((time.perf_counter() - start) * 1000)
                    if conversation_id:
                        from conversation_service import (
                            get_next_turn_number,
                            save_conversation_turn,
                            update_conversation_updated_at,
                        )
                        n = get_next_turn_number(conversation_id)
                        save_conversation_turn(conversation_id, n, "user", input_text, variables, request_id, served_version)
                        save_conversation_turn(conversation_id, n + 1, "assistant", prev, None, request_id, served_version)
                        update_conversation_updated_at(conversation_id)
                    yield _sse_event("step_end", {"node_id": node_id, "type": ntype, "latency_ms": 0, "cost": 0, "output": prev[:500]})
                    yield _sse_event("done", {
                        "request_id": request_id, "served_version": served_version,
                        "final_output": prev, "content_type": last_content_type,
                        "total_latency_ms": total_latency, "total_cost": round(total_cost, 6), "run_id": None,
                    })
                    yield "data: [DONE]\n\n"
                    return
                else:
                    use_linear = False
                    break

                yield _sse_event("step_end", {
                    "node_id": node_id, "type": ntype,
                    "latency_ms": node_results[-1].get("latency_ms", 0),
                    "cost": node_results[-1].get("cost", 0),
                    "output": (node_results[-1].get("output") or "")[:500],
                })
                from_node_id = node_id
        except Exception as e:
            yield _sse_event("error", {"message": str(e), "request_id": request_id})
            yield "data: [DONE]\n\n"
            return

    # Fallback: run full workflow in thread (non-linear or non-OpenAI ai-step)
    loop = asyncio.get_event_loop()

    def run() -> dict:
        return execute_workflow(
            graph_json,
            input_text,
            org_id,
            user_id="",
            workflow_id=workflow_id,
            endpoint_slug=endpoint_slug,
            version=dep_version,
            execution_mode="production",
            variables=variables,
            experiment_id=experiment_id,
            variant_name=variant_name,
            served_version=served_version,
            conversation_prefix=conversation_prefix,
        )

    try:
        result = await loop.run_in_executor(None, run)
    except Exception as e:
        yield _sse_event("error", {"message": str(e), "request_id": request_id})
        yield "data: [DONE]\n\n"
        return

    node_results = result.get("node_results") or []
    for nr in node_results:
        node_id = nr.get("node_id") or ""
        ntype = nr.get("type") or "node"
        yield _sse_event("step_start", {"node_id": node_id, "type": ntype})
        yield _sse_event(
            "step_end",
            {
                "node_id": node_id,
                "type": ntype,
                "latency_ms": nr.get("latency_ms", 0),
                "cost": nr.get("cost", 0),
                "output": (nr.get("output") or "")[:500],
            },
        )

    total_latency = int((time.perf_counter() - start) * 1000)
    if conversation_id and result.get("final_output") is not None:
        from conversation_service import (
            get_next_turn_number,
            save_conversation_turn,
            update_conversation_updated_at,
        )
        n = get_next_turn_number(conversation_id)
        save_conversation_turn(conversation_id, n, "user", input_text, variables, request_id, served_version)
        save_conversation_turn(conversation_id, n + 1, "assistant", result["final_output"], None, request_id, served_version)
        update_conversation_updated_at(conversation_id)
    yield _sse_event(
        "done",
        {
            "request_id": request_id,
            "served_version": served_version,
            "final_output": result.get("final_output"),
            "content_type": result.get("content_type") or "text",
            "total_latency_ms": result.get("total_latency") or total_latency,
            "total_cost": result.get("total_cost"),
            "run_id": result.get("run_id"),
        },
    )
    yield "data: [DONE]\n\n"
