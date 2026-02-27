"""
Workflow execution engine: traverses graph_json, executes nodes in order,
passes outputs via context, supports branching (condition) and routing (router).
"""
from __future__ import annotations

import json
import time
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
from provider_resilience import call_with_resilience


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


def _execute_model_node(node_id: str, node: dict, prompt_text: str, org_id: str) -> dict:
    data = node.get("data") or {}
    custom_model_id = data.get("customModelId") or data.get("custom_model_id")
    if custom_model_id:
        custom = get_custom_model_by_id(str(custom_model_id), org_id)
        if not custom:
            raise HTTPException(
                status_code=404,
                detail=f"Custom model not found. (node_id={node_id})",
            )
        provider = (custom.get("provider") or "OpenAI").strip() or "OpenAI"
        model = (custom.get("base_model") or "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
        system_prefix = (custom.get("system_prefix") or "").strip()
        if system_prefix:
            prompt_text = system_prefix + "\n\n" + prompt_text
    else:
        provider = (data.get("provider") or "OpenAI").strip() or "OpenAI"
        model = (data.get("modelName") or "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
    provider_lower = provider.lower()
    payload = _PromptPayload(org_id=org_id, provider=provider_lower, model=model, prompt=prompt_text, prompt_id=f"workflow-{node_id}")

    router = _ROUTER_MAP.get(provider_lower)
    if not router:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    start = time.perf_counter()
    # call_with_resilience adds retry (exponential backoff on 429/5xx/timeouts)
    # and per-attempt timeout (120s default). Non-retryable errors (400/401/403/404)
    # are raised immediately without retry.
    result = call_with_resilience(
        lambda: router.handle_prompt(payload),
        context_label=f"{provider_lower}/{model} (node {node_id})",
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    out_text = result.get("response") or result.get("output") or ""
    return {
        "output": out_text,
        "latency_ms": latency_ms,
        "tokens": result.get("output_tokens") or result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens", 0),
        "cost": float(result.get("cost_usd") or 0),
        "model": model,
        "provider": provider,
    }


def _execute_model_node_safe(node_id: str, node: dict, prompt_text: str, org_id: str) -> dict:
    """
    Same as _execute_model_node but catches provider failures and returns
    an error result dict instead of raising. Used by the optimizer node so
    that failures are recorded in node_results (feeding the error-rate signal)
    rather than silently lost.
    """
    data = node.get("data") or {}
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


def _infer_provider(model: str) -> str:
    """Infer provider from model name."""
    m = (model or "").strip().lower()
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("mistral") or m.startswith("mixtral"):
        return "mistral"
    if m.startswith("command"):
        return "cohere"
    return "openai"


def _get_model_performance_history(org_id: str, workflow_id: str, limit: int = 200) -> list[dict]:
    """
    Get historical AI step results for this workflow from workflow_runs.node_results.
    Returns list of { "model", "provider", "cost", "latency_ms", "error" }.
    """
    if not workflow_id or not org_id:
        return []
    history: list[dict] = []
    try:
        resp = (
            supabase.table("workflow_runs")
            .select("node_results,created_at")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception:
        return []
    data = getattr(resp, "data", None) or []
    for run in data:
        node_results = run.get("node_results") or []
        if not isinstance(node_results, list):
            continue
        for step in node_results:
            if not isinstance(step, dict):
                continue
            stype = (step.get("type") or "").lower()
            if stype not in ("ai-step", "model", "optimizer"):
                continue
            model = step.get("model")
            if not model:
                continue
            cost = float(step.get("cost_usd") or step.get("cost") or 0)
            latency_ms = int(step.get("latency_ms") or 0)
            provider = (step.get("provider") or _infer_provider(model)).strip().lower()
            err = step.get("error") or (step.get("status") == "error")
            history.append({
                "model": str(model).strip(),
                "provider": provider,
                "cost": cost,
                "latency_ms": latency_ms,
                "error": bool(err),
            })
    return history


def _select_optimal_model(
    history: list[dict],
    priority: str,
    max_cost: float | None,
    max_latency: int | None,
    allowed_models: list[str] | None,
    excluded_models: list[str] | None,
) -> tuple[str, str, str]:
    """
    Given historical performance and constraints, pick the best model.
    Returns (model, provider, reason).
    """
    from collections import defaultdict

    stats: dict[str, dict] = defaultdict(lambda: {"costs": [], "latencies": [], "errors": 0, "total": 0})
    for h in history:
        model = h.get("model") or ""
        if not model:
            continue
        stats[model]["costs"].append(h.get("cost") or 0)
        stats[model]["latencies"].append(h.get("latency_ms") or 0)
        stats[model]["total"] += 1
        if h.get("error"):
            stats[model]["errors"] += 1

    candidates = []
    for model, s in stats.items():
        if allowed_models and model not in allowed_models:
            continue
        if excluded_models and model in excluded_models:
            continue
        avg_cost = sum(s["costs"]) / len(s["costs"]) if s["costs"] else 999.0
        avg_latency = int(sum(s["latencies"]) / len(s["latencies"])) if s["latencies"] else 99999
        error_rate = (s["errors"] / s["total"]) if s["total"] > 0 else 1.0
        if max_cost is not None and avg_cost > max_cost:
            continue
        if max_latency is not None and avg_latency > max_latency:
            continue
        if error_rate > 0.2:
            continue
        candidates.append({
            "model": model,
            "provider": _infer_provider(model),
            "avg_cost": avg_cost,
            "avg_latency": avg_latency,
            "error_rate": error_rate,
            "runs": s["total"],
        })

    if not candidates:
        return ("gpt-4o-mini", "openai", "fallback: no candidates met constraints")

    priority = (priority or "cheapest").lower()
    if priority == "cheapest":
        candidates.sort(key=lambda c: c["avg_cost"])
        winner = candidates[0]
        return (winner["model"], winner["provider"], f"cheapest at ${winner['avg_cost']:.4f}/call avg")
    if priority == "fastest":
        candidates.sort(key=lambda c: c["avg_latency"])
        winner = candidates[0]
        return (winner["model"], winner["provider"], f"fastest at {winner['avg_latency']}ms avg")
    if priority == "quality":
        candidates.sort(key=lambda c: (c["error_rate"], c["avg_latency"]))
        winner = candidates[0]
        return (
            winner["model"],
            winner["provider"],
            f"highest quality at {int((1 - winner['error_rate']) * 100)}% success rate",
        )
    winner = candidates[0]
    return (winner["model"], winner["provider"], "default selection")


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


# Relative latency rank for "fastest" strategy (lower = faster). Unknown models get 999.
_LATENCY_RANK: dict[str, int] = {
    "gpt-3.5-turbo": 1,
    "gpt-4o-mini": 2,
    "claude-3-haiku": 1,
    "claude-3-5-haiku": 1,
    "gemini-1.5-flash": 1,
    "mistral-3.1-small": 2,
    "gpt-4o": 4,
    "gpt-4-turbo": 3,
    "claude-3-sonnet": 3,
    "claude-3-5-sonnet": 3,
    "claude-3-opus": 5,
    "gpt-4": 4,
    "gemini-1.5-pro": 3,
}


def _latency_rank(provider: str, model: str) -> int:
    key = (model or "").strip().lower()
    return _LATENCY_RANK.get(key, 999)


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
) -> dict:
    """
    Execute workflow graph. Returns final_output, node_results, total_cost, total_latency.
    If variables is provided, {{varName}} in AI Step/prompt templates are replaced; input_text can be empty.
    If conversation_prefix is set (multi-turn), it is prepended to the prompt for each AI/model step.
    """
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
    variables = variables or {}

    while queue:
        node_id, from_node_id = queue.pop(0)
        if node_id in context:
            continue

        node = nodes_by_id.get(node_id)
        if not node:
            continue

        node_type = (node.get("type") or "").lower()
        data = node.get("data") or {}

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
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            task = (data.get("taskDescription") or data.get("task") or "Respond to the user.").strip()
            if not isinstance(task, str):
                task = str(task)
            prompt_text = _apply_variables(task, variables, prev_output=prev)
            if conversation_prefix:
                prompt_text = conversation_prefix + prompt_text
            sys_instructions = (data.get("systemInstructions") or data.get("system_prefix") or "").strip()
            if sys_instructions:
                sys_instructions = _apply_variables(sys_instructions, variables)
                prompt_text = sys_instructions + "\n\n" + prompt_text
            model_node = {"id": node_id, "type": "model", "data": {**data, "modelName": data.get("modelName") or "gpt-3.5-turbo", "provider": data.get("provider") or "OpenAI"}}
            try:
                result = _execute_model_node(node_id, model_node, prompt_text, org_id)
            except HTTPException as e:
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
            node_results.append({
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
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
                "prompt_after_interpolation": prompt_text[:2000] + ("..." if len(prompt_text) > 2000 else ""),
            })
        elif node_type == "model":
            prev = _get_previous_output(context, from_node_id or "") if from_node_id else (input_text or "")
            if conversation_prefix:
                prev = conversation_prefix + prev
            try:
                result = _execute_model_node(node_id, node, prev, org_id)
            except HTTPException as e:
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
            node_results.append({
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
                "output": (result.get("output") or "")[:200] + ("..." if len(result.get("output") or "") > 200 else ""),
            })
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
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No API key for {provider}. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            out_text = v_result.get("response") or v_result.get("output") or ""
            result = {"output": out_text, "content_type": "text", "cost": v_result.get("cost_usd") or 0.0, "latency_ms": v_result.get("latency_ms") or 0}
            context[node_id] = {"output": result["output"], "content_type": "text"}
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "vision_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": (out_text or "")[:200], "content_type": "text"})
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
                image_url = img_result.get("url") or ""
                result = {
                    "output": image_url,
                    "content_type": "image_url",
                    "cost": img_result.get("cost_usd") or 0.0,
                    "latency_ms": img_result.get("latency_ms") or 0,
                }
            except HTTPException as e:
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
                "output": (result["output"] or "")[:200], "content_type": "image_url",
            })
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
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No {provider} API key. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            result = {"output": tts_result.get("output") or "", "content_type": "audio_url", "cost": tts_result.get("cost_usd") or 0.0, "latency_ms": tts_result.get("latency_ms") or 0}
            context[node_id] = {"output": result["output"], "content_type": "audio_url"}
            last_content_type = "audio_url"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "tts_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": "[audio]", "content_type": "audio_url"})
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
            try:
                if provider == "openai":
                    stt_result = openai_router.handle_stt(openai_router.STTPayload(org_id=org_id, audio_base64=audio_base64, model=model, prompt_id=f"workflow-{node_id}", prompt=stt_prompt))
                else:
                    stt_result = fireworks_router.handle_stt(fireworks_router.STTPayload(org_id=org_id, audio_base64=audio_base64, model=model, prompt_id=f"workflow-{node_id}", prompt=stt_prompt))
            except HTTPException as e:
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No {provider} API key. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            result = {"output": stt_result.get("output") or "", "content_type": "text", "cost": stt_result.get("cost_usd") or 0.0, "latency_ms": stt_result.get("latency_ms") or 0}
            context[node_id] = {"output": result["output"], "content_type": "text"}
            last_content_type = "text"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "stt_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": (result["output"] or "")[:200], "content_type": "text"})
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
                if e.status_code == 404 or "API key" in (e.detail or ""):
                    raise HTTPException(status_code=404, detail=f"No {provider} API key. Add in Settings → Integrations. (node_id={node_id})") from e
                raise
            out_str = emb_result.get("output") or "[]"
            result = {"output": out_str, "content_type": "embedding", "cost": emb_result.get("cost_usd") or 0.0, "latency_ms": emb_result.get("latency_ms") or 0}
            context[node_id] = {"output": result["output"], "content_type": "embedding"}
            last_content_type = "embedding"
            total_cost += result["cost"]
            total_latency += result["latency_ms"]
            node_results.append({"node_id": node_id, "type": "embedding_step", "latency_ms": result["latency_ms"], "cost": result["cost"], "output": f"[{len(emb_result.get('embedding') or [])}d]", "content_type": "embedding"})
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
                # Still persist the run with the error node_results so history captures the failure
                if workflow_id:
                    try:
                        _row = {
                            "workflow_id": workflow_id,
                            "org_id": org_id,
                            "user_id": user_id or None,
                            "input_text": (input_text[:5000] if input_text else None),
                            "final_output": None,
                            "node_results": node_results,
                            "total_cost": round(total_cost, 6),
                            "total_latency_ms": total_latency + result["latency_ms"],
                            "endpoint_slug": endpoint_slug,
                            "version": version,
                            "execution_mode": (execution_mode or "draft").strip() or "draft",
                        }
                        if experiment_id is not None:
                            _row["experiment_id"] = experiment_id
                        if variant_name is not None:
                            _row["variant_name"] = variant_name
                        if served_version is not None:
                            _row["served_version"] = served_version
                        supabase.table("workflow_runs").insert(_row).execute()
                    except Exception:
                        pass  # best-effort — don't mask the real error

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
                    import logging
                    logging.getLogger(__name__).warning("workflow_runs insert failed: %s", e, exc_info=True)
            return out

        out_edges = edges_out.get(node_id) or []
        for e in out_edges:
            tid = e.get("target")
            if tid:
                queue.append((tid, node_id))

    raise HTTPException(status_code=400, detail="Workflow did not reach an output node")
