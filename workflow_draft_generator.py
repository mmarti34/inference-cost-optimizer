"""Generate draft workflow from plain-English description using AI."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

import openai
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth_dependency import require_org_member, AuthenticatedUser

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workflow-draft"])

_ALLOWED_NODE_TYPES = {"input", "output", "ai-step", "agent", "tool_call", "condition", "prompt", "router", "loop"}

_SYSTEM_PROMPT = """You are a workflow architect for OptiML, an AI workflow platform. Given a plain-English description of what a workflow should do, generate a draft workflow configuration.

Return ONLY valid JSON matching this schema:
{
  "name": "Short workflow name (2-5 words)",
  "description": "One sentence describing what this workflow does",
  "graph_json": {
    "nodes": [
      {
        "id": "unique_id",
        "type": "node_type",
        "position": { "x": number, "y": number },
        "data": { ... node-specific data ... }
      }
    ],
    "edges": [
      { "id": "edge_id", "source": "node_id", "target": "node_id" }
    ]
  },
  "variables": [
    { "name": "var_name", "type": "var_type", "required": true, "description": "..." }
  ]
}

VALID NODE TYPES:
- "input": Entry point. data: {}
- "output": Terminal node. data: { "response_format": "text"|"json"|"markdown"|"auto" }
- "ai-step": Single LLM call. data: { "taskDescription": "prompt text with {{variable}} placeholders", "systemInstructions": "system prompt", "provider": "OpenAI", "modelName": "gpt-4o" }
- "agent": Multi-step tool-using agent. data: { "taskDescription": "task", "systemInstructions": "system prompt", "provider": "OpenAI", "modelName": "gpt-4o", "maxSteps": 10, "tools": [{ "name": "tool_name", "description": "what it does", "type": "http", "url": "https://...", "method": "GET", "parameters": {} }] }
- "tool_call": Single LLM call with tools. data: same as agent but for single-turn tool use.
- "condition": Branch on text predicate. data: { "conditionType": "contains", "conditionValue": "text" }
- "prompt": Pure template (no LLM). data: { "template": "text with {{variables}}" }

VALID VARIABLE TYPES: string, textarea, number, json, boolean, date, image, audio, file

CONTEXT CONFIG (optional, add when user mentions docs/knowledge/files):
Add to ai-step or agent data:
"contextConfig": {
  "enabled": true,
  "mode": "prepacked",
  "sources": [{ "type": "inline_text", "label": "Instructions", "value": "placeholder for business context" }],
  "packaging": { "strategy": "concat", "maxChars": 8000, "includeSourceLabels": true },
  "injection": { "location": "prepend_to_system" }
}

POSITION LAYOUT:
- Input node: x=100, y=200
- Processing nodes: x=400, y=200 (stack vertically with y+=200 for multiple)
- Output node: x=700, y=200

RULES:
1. Keep graphs SMALL. Default: input -> ai-step -> output (3 nodes)
2. Use ai-step for: summarization, extraction, classification, rewriting, drafting, Q&A, analysis
3. Use agent ONLY when the task genuinely needs iterative tool use (web browsing, multi-step research)
4. Use tool_call when the task needs ONE tool call (API lookup, single search)
5. Include contextConfig when user mentions "docs", "knowledge", "playbook", "policies", "uploaded files"
6. Always include an input node and an output node
7. Generate practical, starter-quality prompts — not overly complex
8. Use {{input}} or {{variable_name}} placeholders in taskDescription
9. Default provider: OpenAI, default model: gpt-4o
10. Return ONLY the JSON object, no markdown, no explanation"""


class GenerateRequest(BaseModel):
    description: str


@router.post("/generate-workflow-draft")
async def generate_workflow_draft(
    body: GenerateRequest,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    if not body.description or not body.description.strip():
        raise HTTPException(400, "Description is required")

    if len(body.description) > 5000:
        raise HTTPException(400, "Description too long (max 5000 characters)")

    api_key = os.environ.get("SYSTEM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "AI generation not available — no OpenAI API key configured")

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": body.description.strip()},
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
    except Exception:
        logger.exception("Workflow draft generation failed")
        return _fallback_workflow(body.description.strip())

    try:
        draft = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse AI response as JSON, using fallback")
        return _fallback_workflow(body.description.strip())

    # Validate and sanitize
    return _validate_draft(draft, body.description.strip())


def _validate_draft(draft: dict, description: str) -> dict:
    """Validate and fix the AI-generated draft."""
    name = draft.get("name") or "Untitled Workflow"
    desc = draft.get("description") or description
    graph = draft.get("graph_json") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    variables = draft.get("variables") or []

    # Validate node types
    valid_nodes = []
    node_ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        ntype = (node.get("type") or "").lower()
        if ntype not in _ALLOWED_NODE_TYPES:
            continue
        nid = node.get("id") or f"node_{uuid.uuid4().hex[:8]}"
        node["id"] = nid
        node_ids.add(nid)
        # Ensure position
        if "position" not in node or not isinstance(node.get("position"), dict):
            node["position"] = {"x": 400, "y": 200}
        # Ensure data
        if "data" not in node:
            node["data"] = {}
        valid_nodes.append(node)

    # Ensure we have input and output
    has_input = any(n.get("type") == "input" for n in valid_nodes)
    has_output = any(n.get("type") == "output" for n in valid_nodes)

    if not has_input:
        valid_nodes.insert(0, {
            "id": "input_1", "type": "input",
            "position": {"x": 100, "y": 200}, "data": {}
        })
        node_ids.add("input_1")
    if not has_output:
        valid_nodes.append({
            "id": "output_1", "type": "output",
            "position": {"x": 700, "y": 200}, "data": {"response_format": "text"}
        })
        node_ids.add("output_1")

    # Validate edges
    valid_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src in node_ids and tgt in node_ids:
            eid = edge.get("id") or f"e_{src}_{tgt}"
            valid_edges.append({"id": eid, "source": src, "target": tgt})

    # If no valid edges, create a linear chain
    if not valid_edges and len(valid_nodes) >= 2:
        for i in range(len(valid_nodes) - 1):
            valid_edges.append({
                "id": f"e_{valid_nodes[i]['id']}_{valid_nodes[i+1]['id']}",
                "source": valid_nodes[i]["id"],
                "target": valid_nodes[i+1]["id"],
            })

    # If we still have nothing useful, use fallback
    if len(valid_nodes) < 2:
        return _fallback_workflow(description)

    return {
        "name": name[:100],
        "description": desc[:500],
        "graph_json": {"nodes": valid_nodes, "edges": valid_edges},
        "variables": variables[:20],
    }


def _fallback_workflow(description: str) -> dict:
    """Return a safe minimal workflow when generation fails."""
    return {
        "name": "Draft Workflow",
        "description": description[:200],
        "graph_json": {
            "nodes": [
                {"id": "input_1", "type": "input", "position": {"x": 100, "y": 200}, "data": {}},
                {"id": "ai_step_1", "type": "ai-step", "position": {"x": 400, "y": 200}, "data": {
                    "taskDescription": description,
                    "systemInstructions": "Complete the task described by the user.",
                    "provider": "OpenAI",
                    "modelName": "gpt-4o",
                }},
                {"id": "output_1", "type": "output", "position": {"x": 700, "y": 200}, "data": {"response_format": "text"}},
            ],
            "edges": [
                {"id": "e_input_ai", "source": "input_1", "target": "ai_step_1"},
                {"id": "e_ai_output", "source": "ai_step_1", "target": "output_1"},
            ],
        },
        "variables": [
            {"name": "input", "type": "textarea", "required": True, "description": "Input for the workflow"},
        ],
    }
