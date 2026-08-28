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
    _execute_tool,
)
from evidence_redaction import capture_variables, persist_input_text, persist_node_results
from context_runtime import resolve_node_context, build_context_trace


def _insert_workflow_run_linear(
    org_id: str,
    workflow_id: str | None,
    endpoint_slug: str,
    version: int,
    input_text: str,
    final_output: str,
    node_results: list,
    total_cost: float,
    total_latency_ms: int,
    experiment_id: str | None = None,
    variant_name: str | None = None,
    served_version: int | None = None,
    variables: dict[str, Any] | None = None,
    variables_supplied: bool = False,
) -> str | None:
    """Insert a workflow run for the linear streaming path so /usage and observability see it.

    `variables_supplied` distinguishes "this call site does not know the run's
    variables" from "the run genuinely had none", so an un-wired caller can
    never be misread as a run without inputs.
    """
    if not workflow_id:
        return None
    try:
        row = {
            "workflow_id": workflow_id,
            "org_id": org_id,
            "user_id": None,
            # input_text and node_results are set below, at the persist
            # boundary.
            "final_output": (final_output or "")[:10000] or None,
            "total_cost": round(total_cost, 6),
            "total_latency_ms": total_latency_ms,
            "endpoint_slug": endpoint_slug or None,
            "version": version,
            "execution_mode": "production",
        }
        # Evidence capture, best-effort: never allowed to break the stream.
        try:
            if variables_supplied:
                _vars_value, _vars_capture = capture_variables(variables)
            else:
                _vars_value, _vars_capture = None, {
                    "status": "unavailable",
                    "reason": "not_captured_at_this_call_site",
                    "redacted": False,
                    "truncated": False,
                }
        except BaseException:  # noqa: BLE001
            _vars_value, _vars_capture = None, {
                "status": "unavailable",
                "reason": "capture_failed",
                "redacted": False,
                "truncated": False,
            }
        row["variables"] = _vars_value
        # ── Redaction boundary for input_text ─────────────────────────────
        # Same boundary, same ruleset, same module as `variables` above; the
        # streamed value the caller received is untouched, only what is
        # written here is redacted. persist_input_text never raises.
        row["input_text"], _vars_capture = persist_input_text(input_text, _vars_capture)
        # Same boundary for the trace. The list is not modified, so the steps
        # already streamed to the caller are unaffected.
        row["node_results"], _vars_capture = persist_node_results(node_results, _vars_capture)
        row["variables_capture"] = _vars_capture
        if experiment_id is not None:
            row["experiment_id"] = experiment_id
        if variant_name is not None:
            row["variant_name"] = variant_name
        if served_version is not None:
            row["served_version"] = served_version
        insert_result = supabase.table("workflow_runs").insert(row).execute()
        data = insert_result.data
        first = data[0] if isinstance(data, list) and len(data) > 0 else (data if isinstance(data, dict) else None)
        if first and first.get("id") is not None:
            return str(first["id"])
    except Exception:
        pass
    return None


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
    provider: str = "openai",
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
    # Price with the node's actual provider, not a hardcoded "openai".
    pricing = get_pricing(provider or "openai", model)
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


def _is_linear_streamable_workflow(graph: dict) -> tuple[bool, list[str] | None]:
    """
    If the workflow is linear (input -> prompt? -> ai-step/tool_call/agent/loop -> output) with exactly one
    ai-step, tool_call, agent, or loop node and no condition/router/optimizer, return (True, ordered node ids).
    Else (False, None).
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
    # Accept exactly one ai-step, tool_call, agent, or loop (not multiple).
    # When a loop node is present, its downstream body node (ai-step/tool_call) is
    # part of the loop and should not be counted separately.
    loop_body_ids: set[str] = set()
    for nid in path:
        n = nodes_by_id.get(nid)
        if n and (n.get("type") or "").lower() == "loop":
            for e in edges_out.get(nid) or []:
                tid = e.get("target")
                if tid:
                    tn = nodes_by_id.get(tid)
                    if tn and (tn.get("type") or "").lower() in ("ai-step", "tool_call", "agent"):
                        loop_body_ids.add(tid)
    ai_count = sum(
        1 for nid in path
        if nid not in loop_body_ids
        and (nodes_by_id.get(nid) or {}).get("type", "").lower() in ("ai-step", "tool_call", "agent", "loop")
    )
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
    # Read before the `or {}` normalisation so "no variables supplied" and
    # "empty object supplied" stay distinguishable for evidence capture.
    _raw_variables = variables
    variables = variables or {}
    nodes_by_id = _nodes_by_id(graph_json)
    edges_out = _edges_out(graph_json)
    start = time.perf_counter()

    linear_ok, path = _is_linear_streamable_workflow(graph_json)
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
                # Skip nodes already executed (e.g. loop body nodes set in context by loop handler)
                if node_id in context:
                    from_node_id = node_id
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
                    async for ev in _stream_ai_step_openai(org_id, model, messages, node_id, provider=provider):
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
                elif ntype == "tool_call":
                    # Stream tool call iterations — run synchronously in thread, emit SSE per iteration
                    prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
                    prompt_text = _apply_variables(str(data.get("taskDescription") or data.get("task") or "Respond to the user."), variables, prev_output=prev)
                    if conversation_prefix:
                        prompt_text = conversation_prefix + prompt_text
                    sys_instructions = _apply_variables(str(data.get("systemInstructions") or data.get("system_prefix") or ""), variables).strip()
                    provider = (data.get("provider") or "openai").strip().lower()
                    model = (data.get("modelName") or "gpt-4o-mini").strip() or "gpt-4o-mini"
                    max_iters = int(data.get("maxIterations") or 5)
                    max_iters = max(1, min(max_iters, 20))
                    tools_config = data.get("tools") or []
                    generic_tools = [
                        {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
                        for t in tools_config if t.get("name")
                    ]
                    tools_by_name = {t["name"]: t for t in tools_config if t.get("name")}

                    # Run the tool call node in a thread and collect results
                    step_start = time.perf_counter()

                    def _run_tool_call():
                        from workflow_runtime import _execute_tool_call_node
                        return _execute_tool_call_node(node_id, node, prompt_text, org_id)

                    tool_result = await asyncio.to_thread(_run_tool_call)
                    latency_ms = int((time.perf_counter() - step_start) * 1000)

                    full_output = tool_result.get("output", "")
                    cost_usd = float(tool_result.get("cost", 0))
                    in_tok = tool_result.get("input_tokens", 0)
                    out_tok = tool_result.get("tokens", 0)
                    tool_calls = tool_result.get("tool_calls", [])
                    iterations = tool_result.get("iterations", 1)

                    # Emit tool call events for each tool that was called
                    for tc in tool_calls:
                        yield _sse_event("tool_call_start", {"node_id": node_id, "tool_name": tc.get("name", ""), "arguments": tc.get("arguments", {})})
                        yield _sse_event("tool_call_end", {"node_id": node_id, "tool_name": tc.get("name", ""), "result": (tc.get("result", ""))[:500], "latency_ms": tc.get("latency_ms", 0)})

                    yield _sse_event("iteration", {"node_id": node_id, "iterations": iterations, "tool_calls_count": len(tool_calls)})

                    context[node_id] = full_output
                    total_cost += cost_usd
                    total_latency += latency_ms
                    last_content_type = "text"
                    node_results.append({
                        "node_id": node_id, "type": "tool_call", "latency_ms": latency_ms, "cost": cost_usd,
                        "output": full_output[:200], "tokens": out_tok, "input_tokens": in_tok,
                        "model": model, "provider": provider,
                        "tool_calls": tool_calls, "iterations": iterations,
                    })
                elif ntype == "agent":
                    # Stream agent execution with real-time events via queue.
                    # This enables client-side tool execution: when the agent
                    # calls a tool with execution="client", a tool_yield event
                    # is emitted and the agent blocks until the client POSTs
                    # the result to /api/public/tool-result/{yield_id}.
                    prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
                    prompt_text = _apply_variables(str(data.get("taskDescription") or data.get("task") or "Respond to the user."), variables, prev_output=prev)
                    if conversation_prefix:
                        prompt_text = conversation_prefix + prompt_text
                    provider = (data.get("provider") or "openai").strip().lower()
                    model = (data.get("modelName") or "gpt-4o-mini").strip() or "gpt-4o-mini"

                    # --- Context injection ---
                    _ctx_for_agent = None
                    _ctx_resolved = resolve_node_context(node, context, variables, input_text, org_id, "production")
                    if _ctx_resolved:
                        _ctx_text = _ctx_resolved["final_text"]
                        _ctx_loc = _ctx_resolved["injection_location"]
                        if _ctx_loc == "prepend_to_system":
                            _ctx_for_agent = _ctx_text
                        elif _ctx_loc == "prepend_to_prompt":
                            prompt_text = _ctx_text + "\n\n" + prompt_text
                        elif _ctx_loc == "append_to_prompt":
                            prompt_text = prompt_text + "\n\n" + _ctx_text
                    # --- End context injection ---

                    step_start = time.perf_counter()

                    import queue as _q
                    import concurrent.futures

                    event_queue: _q.Queue = _q.Queue()

                    # Capture context_text for closure
                    _agent_context_text = _ctx_for_agent

                    def _run_agent():
                        from workflow_runtime import _execute_agent_node
                        return _execute_agent_node(node_id, node, prompt_text, org_id, event_queue=event_queue, context_text=_agent_context_text, workflow_id=workflow_id, scope_value=(variables or {}).get("_sm_scope_value"))

                    # Run agent in a thread; read events from queue in real time
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_run_agent)
                        agent_done = False
                        while not agent_done:
                            if future.done():
                                # Drain remaining events
                                while not event_queue.empty():
                                    try:
                                        ev = event_queue.get_nowait()
                                        if ev["event"] == "agent_step":
                                            yield _sse_event("agent_step", ev["data"])
                                        elif ev["event"] == "tool_yield":
                                            yield _sse_event("tool_yield", ev["data"])
                                    except _q.Empty:
                                        break
                                agent_done = True
                            else:
                                try:
                                    ev = event_queue.get(timeout=0.2)
                                    if ev["event"] == "agent_step":
                                        yield _sse_event("agent_step", ev["data"])
                                    elif ev["event"] == "tool_yield":
                                        yield _sse_event("tool_yield", ev["data"])
                                    elif ev["event"] == "agent_done":
                                        agent_done = True
                                except _q.Empty:
                                    await asyncio.sleep(0)  # Yield control to event loop
                                    continue

                        agent_result = future.result()

                    latency_ms = int((time.perf_counter() - step_start) * 1000)

                    full_output = agent_result.get("output", "")
                    cost_usd = float(agent_result.get("cost", 0))
                    in_tok = agent_result.get("input_tokens", 0)
                    out_tok = agent_result.get("tokens", 0)
                    reasoning_steps = agent_result.get("reasoning_steps", [])
                    tool_calls = agent_result.get("tool_calls", [])
                    iterations = agent_result.get("iterations", 1)

                    # agent_step events already emitted in real-time above.
                    # Emit tool_call_start/end for backward compatibility.
                    for tc in tool_calls:
                        yield _sse_event("tool_call_start", {"node_id": node_id, "tool_name": tc.get("name", ""), "arguments": tc.get("arguments", {})})
                        yield _sse_event("tool_call_end", {"node_id": node_id, "tool_name": tc.get("name", ""), "result": (tc.get("result", ""))[:500], "latency_ms": tc.get("latency_ms", 0)})

                    context[node_id] = full_output
                    total_cost += cost_usd
                    total_latency += latency_ms
                    last_content_type = "text"
                    _agent_nr = {
                        "node_id": node_id, "type": "agent", "latency_ms": latency_ms, "cost": cost_usd,
                        "output": full_output[:200], "tokens": out_tok, "input_tokens": in_tok,
                        "model": model, "provider": provider,
                        "tool_calls": tool_calls, "iterations": iterations,
                        "reasoning_steps": reasoning_steps, "agent_steps_count": len(reasoning_steps),
                    }
                    _agent_nr["context_trace"] = build_context_trace(_ctx_resolved, data)
                    node_results.append(_agent_nr)
                elif ntype == "loop":
                    # Run loop node in a thread — emit iteration events as SSE
                    prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
                    step_start = time.perf_counter()

                    # Capture closure variables for the thread
                    _loop_node_id = node_id
                    _loop_data = data
                    _loop_prev = prev
                    _loop_variables = variables
                    _loop_org_id = org_id
                    _loop_nodes_by_id = nodes_by_id
                    _loop_edges_out = edges_out

                    def _run_loop():
                        import time as _time
                        from workflow_runtime import (
                            _execute_model_node,
                            _execute_tool_call_node,
                            _PromptPayload,
                            _ROUTER_MAP,
                        )
                        max_iter = int(_loop_data.get("maxIterations") or 5)
                        exit_type = (_loop_data.get("exitConditionType") or "none").strip().lower()
                        exit_value = (_loop_data.get("exitConditionValue") or "").strip()
                        judge_prompt = (_loop_data.get("llmJudgePrompt") or "").strip()
                        judge_provider = (_loop_data.get("llmJudgeProvider") or "").strip()
                        judge_model = (_loop_data.get("llmJudgeModel") or "").strip()

                        downstream_edges = _loop_edges_out.get(_loop_node_id, [])
                        loop_body_node_id = None
                        loop_body_node = None
                        for edge in downstream_edges:
                            target_id = edge.get("target")
                            if not target_id:
                                continue
                            dn = _loop_nodes_by_id.get(target_id)
                            if dn and (dn.get("type") or "").lower() in ("ai-step", "tool_call", "agent"):
                                loop_body_node_id = target_id
                                loop_body_node = dn
                                break

                        loop_output = _loop_prev
                        loop_cost = 0.0
                        loop_tokens = 0
                        iterations_run = 0

                        for i in range(max_iter):
                            iterations_run = i + 1
                            if loop_body_node:
                                body_type = (loop_body_node.get("type") or "").lower()
                                body_data = loop_body_node.get("data") or {}
                                if body_type == "ai-step":
                                    task = (body_data.get("taskDescription") or "Respond to the user.").strip()
                                    body_prompt = _apply_variables(task, _loop_variables, prev_output=loop_output)
                                    sys_instr = (body_data.get("systemInstructions") or "").strip()
                                    if sys_instr:
                                        sys_instr = _apply_variables(sys_instr, _loop_variables)
                                        body_prompt = sys_instr + "\n\n" + body_prompt
                                    body_model_node = {"id": loop_body_node_id, "type": "model", "data": {**body_data, "modelName": body_data.get("modelName") or "gpt-3.5-turbo", "provider": body_data.get("provider") or "OpenAI"}}
                                    try:
                                        step_result = _execute_model_node(loop_body_node_id, body_model_node, body_prompt, _loop_org_id)
                                        loop_output = step_result.get("output") or step_result.get("response") or ""
                                        loop_cost += step_result.get("cost", 0)
                                        loop_tokens += step_result.get("tokens", 0)
                                    except Exception:
                                        break
                                elif body_type == "tool_call":
                                    task = (body_data.get("taskDescription") or "Respond using the tools.").strip()
                                    body_prompt = _apply_variables(task, _loop_variables, prev_output=loop_output)
                                    try:
                                        step_result = _execute_tool_call_node(loop_body_node_id, loop_body_node, body_prompt, _loop_org_id)
                                        loop_output = step_result.get("output") or step_result.get("response") or ""
                                        loop_cost += step_result.get("cost", 0)
                                        loop_tokens += step_result.get("tokens", 0)
                                    except Exception:
                                        break

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
                                            org_id=_loop_org_id, provider=judge_provider, model=judge_model,
                                            prompt=judge_text, prompt_id=f"loop-judge-{_loop_node_id}-{i}"
                                        )
                                        router_mod = _ROUTER_MAP.get(judge_provider.lower())
                                        if router_mod:
                                            judge_result = router_mod.handle_prompt(judge_payload)
                                            judge_answer = (judge_result.get("response") or "").strip().lower()
                                            loop_cost += judge_result.get("cost_usd", 0)
                                            if any(w in judge_answer for w in ("yes", "true", "stop", "exit", "done")):
                                                break
                                    except Exception:
                                        pass

                        return {
                            "output": loop_output,
                            "cost": loop_cost,
                            "tokens": loop_tokens,
                            "iterations_run": iterations_run,
                            "loop_body_node_id": loop_body_node_id,
                        }

                    loop_result = await asyncio.to_thread(_run_loop)
                    latency_ms = int((time.perf_counter() - step_start) * 1000)

                    full_output = loop_result.get("output", "")
                    cost_usd = float(loop_result.get("cost", 0))
                    loop_tokens_total = loop_result.get("tokens", 0)
                    iterations_run = loop_result.get("iterations_run", 0)
                    loop_body_nid = loop_result.get("loop_body_node_id")

                    yield _sse_event("iteration", {"node_id": node_id, "iterations": iterations_run})

                    context[node_id] = full_output
                    # Also set body node in context so it gets skipped in path traversal
                    if loop_body_nid:
                        context[loop_body_nid] = full_output
                    total_cost += cost_usd
                    total_latency += latency_ms
                    last_content_type = "text"
                    node_results.append({
                        "node_id": node_id, "type": "loop", "latency_ms": latency_ms, "cost": cost_usd,
                        "output": full_output[:200], "tokens": loop_tokens_total,
                        "iterations_run": iterations_run,
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
                        # For agent workflows, store tool call summary in variables column
                        _assistant_vars = None
                        for _nr in node_results:
                            if _nr.get("type") == "agent" and _nr.get("reasoning_steps"):
                                _tool_summary = "; ".join(
                                    f"{s.get('tool_name', '?')}({s.get('latency_ms', 0)}ms)"
                                    for s in _nr["reasoning_steps"] if s.get("type") == "act"
                                )
                                if _tool_summary:
                                    _assistant_vars = {"agent_tool_summary": f"Tools used: {_tool_summary}"}
                                break
                        save_conversation_turn(conversation_id, n + 1, "assistant", prev, _assistant_vars, request_id, served_version)
                        update_conversation_updated_at(conversation_id)
                    yield _sse_event("step_end", {"node_id": node_id, "type": ntype, "latency_ms": 0, "cost": 0, "output": prev[:500]})
                    run_id = _insert_workflow_run_linear(
                        org_id=org_id,
                        workflow_id=workflow_id,
                        endpoint_slug=endpoint_slug,
                        version=dep_version,
                        input_text=input_text,
                        final_output=prev,
                        node_results=node_results,
                        total_cost=total_cost,
                        total_latency_ms=total_latency,
                        experiment_id=experiment_id,
                        variant_name=variant_name,
                        served_version=served_version,
                        variables=_raw_variables,
                        variables_supplied=True,
                    )
                    yield _sse_event("done", {
                        "request_id": request_id, "served_version": served_version,
                        "final_output": prev, "content_type": last_content_type,
                        "total_latency_ms": total_latency, "total_cost": round(total_cost, 6), "run_id": run_id,
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
        _assistant_vars2 = None
        for _nr2 in (result.get("node_results") or []):
            if _nr2.get("type") == "agent" and _nr2.get("reasoning_steps"):
                _ts2 = "; ".join(
                    f"{s.get('tool_name', '?')}({s.get('latency_ms', 0)}ms)"
                    for s in _nr2["reasoning_steps"] if s.get("type") == "act"
                )
                if _ts2:
                    _assistant_vars2 = {"agent_tool_summary": f"Tools used: {_ts2}"}
                break
        save_conversation_turn(conversation_id, n + 1, "assistant", result["final_output"], _assistant_vars2, request_id, served_version)
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
