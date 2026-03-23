"""Tests for context_runtime module."""

import sys
import types
from unittest.mock import patch, MagicMock
import pytest

# Mock supabase before importing
_mock_supabase_mod = types.ModuleType("supabase_client")
_mock_supabase_mod.supabase = MagicMock()
sys.modules.setdefault("supabase_client", _mock_supabase_mod)

# Mock openai (needed for summarize strategy)
_mock_openai = types.ModuleType("openai")
_mock_openai.OpenAI = MagicMock()
sys.modules.setdefault("openai", _mock_openai)

from context_runtime import resolve_node_context, build_context_trace, _collect_sources, _package_context


def test_returns_none_when_no_context_config():
    node = {"data": {"taskDescription": "Do stuff"}}
    result = resolve_node_context(node, {}, {}, "hello", "org-1", "draft")
    assert result is None


def test_returns_none_when_context_disabled():
    node = {"data": {"contextConfig": {"enabled": False, "sources": []}}}
    result = resolve_node_context(node, {}, {}, "hello", "org-1", "draft")
    assert result is None


def test_returns_none_when_sources_empty():
    node = {"data": {"contextConfig": {"enabled": True, "sources": []}}}
    result = resolve_node_context(node, {}, {}, "hello", "org-1", "draft")
    assert result is None


def test_inline_text_source():
    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "mode": "prepacked",
                "sources": [
                    {"type": "inline_text", "label": "Instructions", "value": "Be helpful and kind."}
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            }
        }
    }
    result = resolve_node_context(node, {}, {}, "hello", "org-1", "draft")
    assert result is not None
    assert result["final_text"] == "Be helpful and kind."
    assert result["total_chars"] == 20
    assert result["truncated"] is False
    assert result["mode"] == "prepacked"
    assert result["injection_location"] == "prepend_to_system"
    assert len(result["items_used"]) == 1
    assert result["items_used"][0]["source_type"] == "inline_text"


def test_previous_node_output_source():
    execution_context = {
        "prompt_1": {"output": "Customer is VIP tier, account since 2020."}
    }
    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "mode": "prepacked",
                "sources": [
                    {"type": "previous_node_output", "nodeId": "prompt_1", "label": "Profile"}
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_prompt"},
            }
        }
    }
    result = resolve_node_context(node, execution_context, {}, "hello", "org-1", "draft")
    assert result is not None
    assert "VIP tier" in result["final_text"]
    assert result["injection_location"] == "prepend_to_prompt"


def test_previous_node_output_string_context():
    execution_context = {"input_1": "raw input text"}
    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "sources": [
                    {"type": "previous_node_output", "nodeId": "input_1", "label": "Input"}
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            }
        }
    }
    result = resolve_node_context(node, execution_context, {}, "", "org-1", "draft")
    assert result is not None
    assert result["final_text"] == "raw input text"


@patch("context_runtime.supabase")
def test_knowledge_asset_source(mock_sb):
    chain = MagicMock()
    chain.select.return_value = chain
    chain.in_.return_value = chain
    chain.eq.return_value = chain
    chain.execute.return_value = MagicMock(data=[
        {"id": "asset-1", "name": "FAQ", "content": "Q: What is OptiML?\nA: A platform.", "asset_type": "text", "status": "active"}
    ])
    mock_sb.table.return_value = chain

    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "sources": [
                    {"type": "knowledge_asset", "assetId": "asset-1", "label": "FAQ"}
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            }
        }
    }
    result = resolve_node_context(node, {}, {}, "", "org-1", "draft")
    assert result is not None
    assert "What is OptiML" in result["final_text"]


def test_multiple_sources_concat():
    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "sources": [
                    {"type": "inline_text", "label": "Rule 1", "value": "Always be polite."},
                    {"type": "inline_text", "label": "Rule 2", "value": "Never share secrets."},
                ],
                "packaging": {"strategy": "concat", "includeSourceLabels": True},
                "injection": {"location": "prepend_to_system"},
            }
        }
    }
    result = resolve_node_context(node, {}, {}, "", "org-1", "draft")
    assert result is not None
    assert "## Rule 1" in result["final_text"]
    assert "## Rule 2" in result["final_text"]
    assert "Always be polite." in result["final_text"]
    assert "Never share secrets." in result["final_text"]


def test_max_chars_trimming():
    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "sources": [
                    {"type": "inline_text", "label": "Long", "value": "x" * 1000},
                ],
                "packaging": {"strategy": "concat", "maxChars": 100},
                "injection": {"location": "prepend_to_system"},
            }
        }
    }
    result = resolve_node_context(node, {}, {}, "", "org-1", "draft")
    assert result is not None
    assert result["truncated"] is True
    assert result["total_chars"] == 100


def test_per_source_max_chars():
    node = {
        "data": {
            "contextConfig": {
                "enabled": True,
                "sources": [
                    {"type": "inline_text", "label": "Long", "value": "x" * 500, "maxChars": 50},
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            }
        }
    }
    result = resolve_node_context(node, {}, {}, "", "org-1", "draft")
    assert result is not None
    assert result["total_chars"] == 50


def test_trace_disabled():
    trace = build_context_trace(None, {})
    assert trace == {"enabled": False}


def test_trace_enabled():
    resolved = {
        "mode": "prepacked",
        "items_used": [
            {"source_type": "inline_text", "label": "Rules", "estimated_chars": 100}
        ],
        "truncated": False,
        "total_chars": 100,
    }
    node_data = {"contextConfig": {"enabled": True}}
    trace = build_context_trace(resolved, node_data)
    assert trace["enabled"] is True
    assert trace["source_count"] == 1
    assert trace["total_chars"] == 100


def test_package_empty():
    result = _package_context([], {})
    assert result["final_text"] == ""
    assert result["truncated"] is False


def test_package_with_labels():
    candidates = [
        {"source_type": "inline_text", "label": "A", "raw_text": "Hello", "estimated_chars": 5},
        {"source_type": "inline_text", "label": "B", "raw_text": "World", "estimated_chars": 5},
    ]
    result = _package_context(candidates, {"includeSourceLabels": True})
    assert "## A\nHello" in result["final_text"]
    assert "## B\nWorld" in result["final_text"]
