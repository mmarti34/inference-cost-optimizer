"""
Tests for POST /v1/chat/completions — OptiML's OpenAI-compatible surface.

Covers the four things that must not regress:
  1. THE MODE RULE. A bare model id is ALWAYS direct inference; only the
     reserved ``optiml/`` namespace reaches a deployment. No fall-through.
  2. Streaming. Real OpenAI SSE chunk format, terminated by [DONE].
  3. Tools. Forwarded to the provider and returned to the CALLER, not executed
     server-side.
  4. Usage accounting. Real tokens, and an estimated price is never reported as
     measured cost.

Style follows test_public_execution.py, including the Crypto stub so the router
imports without pycryptodome.
"""
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub Crypto so the router imports without pycryptodome (auth is patched).
if "Crypto" not in sys.modules:
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        sys.modules["Crypto." + sub] = types.ModuleType("Crypto." + sub)
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import direct_inference
from direct_inference import DirectInferenceError
from routers import openai_compat
from routers.openai_compat import (
    MODE_DIRECT,
    MODE_WORKFLOW,
    extract_optiml_metadata,
    resolve_mode,
    router as openai_compat_router,
    validate_messages,
)

app = FastAPI()
app.include_router(openai_compat_router)

ORG_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ORG_ID = "22222222-2222-2222-2222-222222222222"
AUTH = {"Authorization": "Bearer sk_test_key"}


@pytest.fixture
def client():
    return TestClient(app)


async def _async_noop(*args, **kwargs):
    """log_api_request is awaited via asyncio.create_task; MagicMock is not awaitable."""
    return None


@pytest.fixture
def mock_ctx():
    from api_key_validation import OrgContext

    return OrgContext(org_id=ORG_ID, key_type="live", rate_limit_per_minute=60)


@pytest.fixture
def authed(mock_ctx):
    """Valid key, no rate-limit or quota interference."""
    with patch.object(openai_compat, "validate_api_key", return_value=mock_ctx), patch.object(
        openai_compat, "check_and_increment_usage"
    ), patch.object(openai_compat, "check_monthly_request_limit"), patch.object(
        openai_compat, "increment_monthly_usage"
    ), patch.object(
        openai_compat, "log_api_request", new=_async_noop
    ), patch.object(
        openai_compat, "log_api_request_sync"
    ):
        yield


# ═════════════════════════════════════════════════════════════════════════════
# 1. THE MODE RULE
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4-5-20250929",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "accounts/fireworks/models/deepseek-v3",
        "command-r",
        "some-slug-that-looks-like-an-endpoint",
    ],
)
def test_bare_model_is_always_direct_inference(model):
    """No bare model id — however slug-like — ever selects a workflow."""
    decision = resolve_mode(model)
    assert decision.mode == MODE_DIRECT
    assert decision.endpoint_slug is None
    assert decision.model == model


@pytest.mark.parametrize("prefix", ["optiml", "OptiML", "OPTIML", "optiml-workflow"])
def test_optiml_prefix_selects_workflow(prefix):
    """Only the reserved namespace routes to a deployment; matching is case-insensitive."""
    decision = resolve_mode(f"{prefix}/support-triage")
    assert decision.mode == MODE_WORKFLOW
    assert decision.endpoint_slug == "support-triage"


def test_optiml_prefix_without_slug_is_rejected():
    with pytest.raises(DirectInferenceError) as exc:
        resolve_mode("optiml/")
    assert exc.value.status_code == 400
    assert exc.value.code == "invalid_workflow_model"


def test_empty_model_is_rejected():
    with pytest.raises(DirectInferenceError) as exc:
        resolve_mode("")
    assert exc.value.code == "model_required"


def test_mode_header_is_an_assertion_not_a_selector():
    """Agreement passes; disagreement is a hard 400 — the model always decides."""
    assert resolve_mode("gpt-4o", "direct").mode == MODE_DIRECT
    assert resolve_mode("optiml/x", "workflow").mode == MODE_WORKFLOW

    with pytest.raises(DirectInferenceError) as exc:
        resolve_mode("gpt-4o", "workflow")
    assert exc.value.code == "mode_conflict"

    with pytest.raises(DirectInferenceError) as exc:
        resolve_mode("optiml/x", "direct")
    assert exc.value.code == "mode_conflict"


def test_invalid_mode_header_is_rejected():
    with pytest.raises(DirectInferenceError) as exc:
        resolve_mode("gpt-4o", "sideways")
    assert exc.value.code == "invalid_mode_header"


@pytest.mark.parametrize(
    "model,provider,native",
    [
        ("gpt-4o", "openai", "gpt-4o"),
        ("claude-sonnet-4-5-20250929", "anthropic", "claude-sonnet-4-5-20250929"),
        ("anthropic/claude-3-5-haiku-20241022", "anthropic", "claude-3-5-haiku-20241022"),
        ("meta-llama/Llama-3.3-70B-Instruct-Turbo", "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("together/openai/gpt-oss-120b", "together", "openai/gpt-oss-120b"),
        ("accounts/fireworks/models/anything-new", "fireworks", "accounts/fireworks/models/anything-new"),
        ("gemini-2.5-flash", "gemini", "gemini-2.5-flash"),
        ("deepseek-chat", "deepseek", "deepseek-chat"),
    ],
)
def test_direct_model_resolution(model, provider, native):
    resolved = direct_inference.resolve_direct_model(model)
    assert resolved.provider == provider
    assert resolved.model == native


def test_unknown_bare_model_is_400_not_a_workflow_lookup(client, authed):
    """The critical constraint: an unrecognised string fails, it never executes."""
    with patch.object(openai_compat.supabase, "table") as table:
        table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "totally-made-up", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["param"] == "model"


def test_bare_model_matching_a_deployment_gets_a_migration_error_not_execution(client, authed):
    """
    A string that IS one of the org's slugs still does not execute the workflow.
    The caller gets a precise instruction instead.
    """
    with patch.object(openai_compat.supabase, "table") as table:
        table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"endpoint_slug": "support-triage"}]
        )
        with patch.object(openai_compat, "execute_workflow") as execute:
            response = client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={"model": "support-triage", "messages": [{"role": "user", "content": "hi"}]},
            )
            execute.assert_not_called()
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "workflow_requires_optiml_prefix"
    assert "optiml/support-triage" in body["error"]["message"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Auth, rate limiting, quota
# ═════════════════════════════════════════════════════════════════════════════
def test_missing_authorization_is_401(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_invalid_key_is_401(client):
    with patch.object(
        openai_compat, "validate_api_key", side_effect=HTTPException(401, "Invalid API key.")
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 401


def test_monthly_quota_applies_to_direct_inference(client, mock_ctx):
    """/v1/prompt escapes the monthly quota. Direct inference must not."""
    with patch.object(openai_compat, "validate_api_key", return_value=mock_ctx), patch.object(
        openai_compat, "check_and_increment_usage"
    ), patch.object(
        openai_compat,
        "check_monthly_request_limit",
        side_effect=HTTPException(429, "Monthly request limit reached."),
    ), patch.object(
        openai_compat, "log_api_request", new=_async_noop
    ), patch.object(
        openai_compat, "increment_monthly_usage"
    ) as increment, patch.object(
        direct_inference, "complete"
    ) as complete:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
    complete.assert_not_called()
    increment.assert_not_called()


def test_rate_limit_applies_to_direct_inference(client, mock_ctx):
    with patch.object(openai_compat, "validate_api_key", return_value=mock_ctx), patch.object(
        openai_compat,
        "check_and_increment_usage",
        side_effect=HTTPException(429, "Rate limit exceeded"),
    ), patch.object(openai_compat, "check_monthly_request_limit"), patch.object(
        openai_compat, "increment_monthly_usage"
    ), patch.object(
        openai_compat, "log_api_request", new=_async_noop
    ), patch.object(
        direct_inference, "complete"
    ) as complete:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 429
    complete.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# 3. Message validation
# ═════════════════════════════════════════════════════════════════════════════
def test_messages_must_be_non_empty():
    with pytest.raises(DirectInferenceError) as exc:
        validate_messages([])
    assert exc.value.code == "invalid_messages"


def test_all_openai_roles_are_accepted():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
    ]
    assert len(validate_messages(messages)) == 4


def test_tool_message_without_tool_call_id_is_rejected():
    with pytest.raises(DirectInferenceError) as exc:
        validate_messages([{"role": "tool", "content": "42"}])
    assert exc.value.code == "missing_tool_call_id"


def test_unknown_role_is_rejected():
    with pytest.raises(DirectInferenceError) as exc:
        validate_messages([{"role": "wizard", "content": "hi"}])
    assert exc.value.code == "invalid_role"


# ═════════════════════════════════════════════════════════════════════════════
# 4. OptiML metadata (body and headers), all optional
# ═════════════════════════════════════════════════════════════════════════════
def test_metadata_from_nested_body_object():
    meta = extract_optiml_metadata(
        {"metadata": {"optiml": {"workload": "support-refund", "user_id": "u1", "experiment_tags": ["a", "b"]}}},
        {},
    )
    assert meta.workload == "support-refund"
    assert meta.user_id == "u1"
    assert meta.experiment_tags == ["a", "b"]


def test_metadata_from_json_string_in_metadata():
    """Some SDKs coerce every metadata value to a string."""
    meta = extract_optiml_metadata(
        {"metadata": {"optiml": json.dumps({"workload": "w", "conversation_id": "c"})}}, {}
    )
    assert meta.workload == "w"
    assert meta.conversation_id == "c"


def test_metadata_from_headers():
    meta = extract_optiml_metadata(
        {},
        {
            "X-OptiML-Workload": "support-refund",
            "X-OptiML-User-Id": "u9",
            "X-OptiML-Conversation-Id": "conv-1",
            "X-OptiML-Experiment-Tags": "beta, canary",
        },
    )
    assert meta.workload == "support-refund"
    assert meta.experiment_tags == ["beta", "canary"]
    assert meta.source == "headers"


def test_body_metadata_wins_over_headers():
    meta = extract_optiml_metadata(
        {"metadata": {"optiml": {"workload": "from-body"}}},
        {"X-OptiML-Workload": "from-header", "X-OptiML-User-Id": "u1"},
    )
    assert meta.workload == "from-body"
    assert meta.user_id == "u1"  # gaps still fill from headers


def test_metadata_is_optional():
    meta = extract_optiml_metadata({}, {})
    assert meta.workload is None
    assert meta.source == "none"


def test_optiml_metadata_never_reaches_the_provider():
    params = direct_inference.forwardable_params(
        {
            "model": "gpt-4o",
            "messages": [],
            "temperature": 0.2,
            "metadata": {"optiml": {"workload": "w"}, "customer_key": "keep-me"},
        }
    )
    assert "optiml" not in json.dumps(params)
    assert params["temperature"] == 0.2

    cleaned = direct_inference.strip_optiml_metadata(
        {"metadata": {"optiml": {"workload": "w"}, "customer_key": "keep-me"}}
    )
    assert cleaned["metadata"] == {"customer_key": "keep-me"}


# ═════════════════════════════════════════════════════════════════════════════
# 5. Direct inference: usage accounting and cost honesty
# ═════════════════════════════════════════════════════════════════════════════
def _provider_response(content="hello", prompt=11, completion=7, finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-provider",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-2024-08-06",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def test_direct_inference_returns_openai_shape_with_real_usage(client, authed):
    with patch.object(
        direct_inference, "_openai_complete", return_value=(_provider_response(), 123)
    ), patch.object(direct_inference, "get_org_provider_key", return_value="sk-provider"):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    # The caller's model string is echoed, not the provider's dated variant.
    assert body["model"] == "gpt-4o"
    assert body["optiml"]["mode"] == MODE_DIRECT
    assert body["optiml"]["cost_estimated"] is False
    assert body["optiml"]["cost_usd"] > 0
    # Request id is echoed in headers and body.
    assert response.headers["X-Request-Id"] == body["id"]
    assert response.headers["X-OptiML-Request-Id"] == body["id"]


def test_unknown_model_pricing_is_flagged_not_silently_defaulted(client, authed):
    """
    An unpriced model must not report a fabricated measured cost. The guess
    goes in estimated_cost_usd and cost_usd stays null.
    """
    unpriced = _provider_response()
    with patch.object(
        direct_inference, "_openai_complete", return_value=(unpriced, 10)
    ), patch.object(direct_inference, "get_org_provider_key", return_value="sk-provider"):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "openai/gpt-not-in-the-registry",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    optiml = response.json()["optiml"]
    assert optiml["cost_estimated"] is True
    assert optiml["cost_usd"] is None
    assert optiml["estimated_cost_usd"] is not None


@pytest.mark.parametrize(
    "status,body,expected_type",
    [
        (
            429,
            {"error": {"message": "Rate limit reached for gpt-4o", "type": "rate_limit_error"}},
            "rate_limit_error",
        ),
        (401, {"error": {"message": "Incorrect API key provided"}}, "authentication_error"),
        (
            400,
            {"error": {"message": "Unsupported parameter: seed", "param": "seed"}},
            "invalid_request_error",
        ),
        (503, {"error": {"message": "The engine is overloaded"}}, "api_error"),
    ],
)
def test_provider_error_parsing_preserves_status_type_and_message(status, body, expected_type):
    err = direct_inference._provider_error("openai", status, json.dumps(body))
    assert err.status_code == status
    assert err.err_type == expected_type
    assert body["error"]["message"] in err.message


def test_provider_errors_are_not_collapsed_into_500(client, authed):
    """A provider 429 reaches the caller as a 429 in the OpenAI error envelope."""
    provider_error = direct_inference._provider_error(
        "openai",
        429,
        json.dumps(
            {"error": {"message": "Rate limit reached for gpt-4o", "type": "rate_limit_error"}}
        ),
    )
    with patch.object(
        direct_inference, "_openai_complete", side_effect=provider_error
    ), patch.object(direct_inference, "get_org_provider_key", return_value="sk-provider"):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert "Rate limit reached for gpt-4o" in error["message"]


def test_provider_4xx_is_not_retried_but_5xx_is():
    """
    A proxy must not absorb the caller's 429: their own client owns backoff.
    Server-side faults, which the caller cannot see, still retry.
    """
    resolved = direct_inference.resolve_direct_model("gpt-4o")
    calls = []

    def raise_429(*args, **kwargs):
        calls.append("429")
        raise direct_inference._provider_error("openai", 429, '{"error":{"message":"slow down"}}')

    with patch.object(direct_inference, "get_org_provider_key", return_value="k"), patch.object(
        direct_inference, "_openai_complete", side_effect=raise_429
    ):
        with pytest.raises(DirectInferenceError) as exc:
            direct_inference.complete(resolved, [{"role": "user", "content": "hi"}], {}, "org", request_id="r")
    assert exc.value.status_code == 429
    assert len(calls) == 1, "a provider 4xx must be passed straight through"

    calls.clear()

    def raise_503(*args, **kwargs):
        calls.append("503")
        raise direct_inference._provider_error("openai", 503, '{"error":{"message":"overloaded"}}')

    with patch.object(direct_inference, "get_org_provider_key", return_value="k"), patch.object(
        direct_inference, "_openai_complete", side_effect=raise_503
    ), patch("provider_resilience.time.sleep"):
        with pytest.raises(DirectInferenceError) as exc:
            direct_inference.complete(resolved, [{"role": "user", "content": "hi"}], {}, "org", request_id="r")
    assert exc.value.status_code == 503
    assert len(calls) > 1, "a provider 5xx is transient and should be retried"


def test_missing_provider_key_is_a_clear_400(client, authed):
    with patch.object(
        direct_inference,
        "get_org_provider_key",
        side_effect=DirectInferenceError(
            400, "No openai API key is configured", code="provider_key_missing"
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "provider_key_missing"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Tools — forwarded, and returned to the CALLER
# ═════════════════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def test_tools_are_forwarded_and_tool_calls_returned(client, authed):
    """No 400, no server-side tool execution: the caller runs its own tools."""
    captured = {}

    def fake_complete(resolved, messages, params, api_key, timeout=None):
        captured["params"] = params
        return (
            _provider_response(
                content=None,
                finish="tool_calls",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            ),
            15,
        )

    with patch.object(direct_inference, "_openai_complete", side_effect=fake_complete), patch.object(
        direct_inference, "get_org_provider_key", return_value="sk-provider"
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        )
    assert response.status_code == 200
    assert captured["params"]["tools"] == TOOLS
    assert captured["params"]["tool_choice"] == "auto"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_tools_on_a_workflow_are_rejected_with_a_named_field(client, authed):
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "model": "optiml/support-triage",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "tools_not_supported_for_workflow"
    assert error["param"] == "tools"


def test_anthropic_unsupported_params_fail_loudly_naming_the_field():
    """Never silently ignored — the offending field is named."""
    resolved = direct_inference.resolve_direct_model("claude-sonnet-4-5-20250929")
    with pytest.raises(DirectInferenceError) as exc:
        direct_inference._anthropic_payload(
            resolved, [{"role": "user", "content": "hi"}],
            {"response_format": {"type": "json_object"}}, stream=False,
        )
    assert exc.value.param == "response_format"
    assert exc.value.code == "param_not_supported_by_provider"

    with pytest.raises(DirectInferenceError) as exc:
        direct_inference._anthropic_payload(
            resolved, [{"role": "user", "content": "hi"}], {"temperature": 1.8}, stream=False
        )
    assert exc.value.param == "temperature"


def test_anthropic_tool_round_trip_translation():
    """An existing OpenAI tool conversation survives translation to Anthropic."""
    resolved = direct_inference.resolve_direct_model("claude-sonnet-4-5-20250929")
    payload = direct_inference._anthropic_payload(
        resolved,
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
        ],
        {"tools": TOOLS, "tool_choice": "required", "max_tokens": 256},
        stream=False,
    )
    assert payload["system"] == "be terse"
    assert payload["max_tokens"] == 256
    assert payload["tools"][0]["name"] == "get_weather"
    assert "input_schema" in payload["tools"][0]
    assert payload["tool_choice"] == {"type": "any"}
    assistant = payload["messages"][1]
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["input"] == {"city": "Paris"}
    tool_result = payload["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"


def test_anthropic_response_maps_to_openai_finish_reasons():
    resolved = direct_inference.resolve_direct_model("claude-sonnet-4-5-20250929")
    body, finish, tool_calls = direct_inference._anthropic_to_openai_body(
        {
            "content": [
                {"type": "text", "text": "sure"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 12},
        },
        resolved,
        request_id="chatcmpl-x",
        created=1,
    )
    assert finish == "tool_calls"
    assert tool_calls == 1
    assert body["usage"]["total_tokens"] == 42
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


# ═════════════════════════════════════════════════════════════════════════════
# 7. Streaming — real OpenAI SSE
# ═════════════════════════════════════════════════════════════════════════════
def _sse_payloads(text: str) -> list:
    out = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        out.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return out


def test_direct_streaming_emits_openai_chunks_and_done(client, authed):
    def fake_stream(resolved, messages, params, api_key, **kwargs):
        accounting = kwargs["accounting"]
        rid, created = kwargs["request_id"], kwargs["created"]
        yield direct_inference._sse(
            direct_inference._chunk(rid, created, "gpt-4o", {"role": "assistant"})
        )
        for piece in ("Hel", "lo"):
            yield direct_inference._sse(
                direct_inference._chunk(rid, created, "gpt-4o", {"content": piece})
            )
        accounting.prompt_tokens = 5
        accounting.completion_tokens = 2
        accounting.finish_reason = "stop"
        yield direct_inference._sse(
            direct_inference._chunk(rid, created, "gpt-4o", {}, "stop")
        )

    with patch.object(
        direct_inference, "_stream_openai_dialect", side_effect=fake_stream
    ), patch.object(direct_inference, "get_org_provider_key", return_value="sk-provider"):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-OptiML-Request-Id"]
    payloads = _sse_payloads(response.text)
    assert payloads[-1] == "[DONE]"
    chunks = payloads[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices")
    ) == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_records_usage_for_accounting(client, authed):
    """Token accounting must survive the streaming path, not only the buffered one."""
    recorded = []

    def fake_stream(resolved, messages, params, api_key, **kwargs):
        accounting = kwargs["accounting"]
        accounting.prompt_tokens = 40
        accounting.completion_tokens = 9
        accounting.finish_reason = "stop"
        yield direct_inference._sse(
            direct_inference._chunk(kwargs["request_id"], 1, "gpt-4o", {"content": "x"})
        )

    with patch.object(
        direct_inference, "_stream_openai_dialect", side_effect=fake_stream
    ), patch.object(
        direct_inference, "get_org_provider_key", return_value="sk-provider"
    ), patch.object(
        openai_compat, "record_direct_inference_attempt", side_effect=recorded.append
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert response.text  # drain the stream so the finally block runs

    assert len(recorded) == 1
    attempt = recorded[0]
    assert attempt.prompt_tokens == 40
    assert attempt.completion_tokens == 9
    assert attempt.tokens_known is True
    assert attempt.streamed is True
    assert attempt.inference_cost_usd > 0
    assert attempt.cost_estimated is False


def test_stream_error_is_delivered_as_a_terminal_sse_event(client, authed):
    """Once the 200 is on the wire the status cannot change; the error still surfaces."""
    with patch.object(
        direct_inference,
        "get_org_provider_key",
        side_effect=DirectInferenceError(400, "no key", code="provider_key_missing"),
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    payloads = _sse_payloads(response.text)
    assert payloads[-1] == "[DONE]"
    assert payloads[0]["error"]["code"] == "provider_key_missing"


def test_anthropic_stream_translation():
    """Anthropic SSE events become OpenAI chunks, including tool-call deltas."""
    import httpx

    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_weather"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
    ]
    raw = "".join(f"data: {json.dumps(e)}\n\n" for e in events)

    class FakeStream:
        def __enter__(self):
            return httpx.Response(
                200, text=raw, request=httpx.Request("POST", "http://anthropic")
            )

        def __exit__(self, *args):
            return False

    resolved = direct_inference.resolve_direct_model("claude-sonnet-4-5-20250929")
    accounting = direct_inference.StreamAccounting()
    with patch("httpx.Client.stream", return_value=FakeStream()):
        chunks = list(
            direct_inference._stream_anthropic(
                resolved, [{"role": "user", "content": "hi"}], {}, "sk-ant",
                request_id="chatcmpl-x", created=1,
                accounting=accounting, include_usage=True, timeout=5,
            )
        )

    parsed = [json.loads(c[5:].strip()) for c in chunks]
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[1]["choices"][0]["delta"]["content"] == "Hi"
    tool_start = parsed[2]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_start["id"] == "tu_1"
    assert tool_start["function"]["name"] == "get_weather"
    arguments = "".join(
        p["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for p in parsed[3:5]
    )
    assert json.loads(arguments) == {"city": "Paris"}
    assert accounting.prompt_tokens == 12
    assert accounting.completion_tokens == 8
    assert accounting.finish_reason == "tool_calls"
    assert parsed[-1]["usage"]["total_tokens"] == 20


# ═════════════════════════════════════════════════════════════════════════════
# 8. Workflow mode still works
# ═════════════════════════════════════════════════════════════════════════════
DEPLOYMENT = {
    "id": "dep-id",
    "workflow_id": "wf-id",
    "org_id": ORG_ID,
    "version": 3,
    "endpoint_slug": "support-triage",
    "graph_json": {"nodes": [], "edges": []},
}


def _patch_workflow(result=None, deployment=None):
    async def resolver(**kwargs):
        return 3, None, None, (deployment if deployment is not None else DEPLOYMENT)

    table = MagicMock()
    table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"variables": []}
    )
    return (
        patch.object(openai_compat, "resolve_version_and_deployment", side_effect=resolver),
        patch.object(openai_compat, "execute_workflow", return_value=result or {
            "final_output": "triaged",
            "node_results": [{"input_tokens": 20, "output_tokens": 5}],
            "total_cost": 0.004,
            "total_latency_ms": 900,
            "run_id": "run-1",
        }),
        patch.object(openai_compat.supabase, "table", table),
        patch.object(openai_compat, "_resolve_user_selected_providers", side_effect=lambda g, v: g),
    )


def test_workflow_mode_executes_and_reports_usage(client, authed):
    p1, p2, p3, p4 = _patch_workflow()
    with p1, p2, p3, p4:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "optiml/support-triage", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "optiml/support-triage"
    assert body["choices"][0]["message"]["content"] == "triaged"
    assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
    assert body["optiml"]["mode"] == MODE_WORKFLOW
    assert body["optiml"]["served_version"] == 3
    # Legacy response fields retained for pre-existing callers.
    assert body["optiml_served_version"] == 3
    assert body["optiml_request_id"] == body["id"]


def test_workflow_mode_streams_in_openai_chunk_format(client, authed):
    p1, p2, p3, p4 = _patch_workflow()
    with p1, p2, p3, p4:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "optiml/support-triage",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    payloads = _sse_payloads(response.text)
    assert payloads[-1] == "[DONE]"
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(
        p["choices"][0]["delta"].get("content", "") for p in payloads[1:-1] if p.get("choices")
    ) == "triaged"
    assert payloads[-2]["usage"]["total_tokens"] == 25


def test_workflow_from_another_org_is_refused(client, authed):
    """Tenant boundary is asserted even though the resolver is org-scoped."""
    foreign = dict(DEPLOYMENT, org_id=OTHER_ORG_ID)
    p1, p2, p3, p4 = _patch_workflow(deployment=foreign)
    with p1, p2 as execute, p3, p4:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "optiml/support-triage", "messages": [{"role": "user", "content": "hi"}]},
        )
        execute.assert_not_called()
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "permission_error"


# ═════════════════════════════════════════════════════════════════════════════
# 9. Workload identity feeds the domain model
# ═════════════════════════════════════════════════════════════════════════════
def test_attempt_is_recorded_with_workload_strategy_and_cost(client, authed):
    recorded = []
    with patch.object(
        direct_inference, "_openai_complete", return_value=(_provider_response(), 42)
    ), patch.object(
        direct_inference, "get_org_provider_key", return_value="sk-provider"
    ), patch.object(
        openai_compat, "record_direct_inference_attempt", side_effect=recorded.append
    ):
        client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "you triage refunds"},
                    {"role": "user", "content": "hi"},
                ],
                "temperature": 0.3,
                "metadata": {"optiml": {"workload": "support-refund", "user_id": "u42"}},
            },
        )

    assert len(recorded) == 1
    attempt = recorded[0]
    assert attempt.org_id == ORG_ID
    assert attempt.surface == "direct_inference"
    assert attempt.workload["identity_level"] == "explicit"
    assert attempt.workload["external_key"] == "support-refund"
    assert attempt.provider == "openai"
    assert attempt.params["temperature"] == 0.3
    assert attempt.system_prompt == "you triage refunds"
    assert attempt.prompt_tokens == 11
    assert attempt.inference_cost_usd > 0
    assert attempt.end_user_id == "u42"
    assert attempt.duration_ms is not None


def test_workload_is_structural_when_the_customer_names_nothing(client, authed):
    """Discovery must work with zero cooperation from the caller."""
    recorded = []
    with patch.object(
        direct_inference, "_openai_complete", return_value=(_provider_response(), 42)
    ), patch.object(
        direct_inference, "get_org_provider_key", return_value="sk-provider"
    ), patch.object(
        openai_compat, "record_direct_inference_attempt", side_effect=recorded.append
    ):
        for model in ("gpt-4o", "gpt-4o-2024-08-06"):
            client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "you triage refunds"},
                        {"role": "user", "content": "anything at all"},
                    ],
                },
            )

    assert len(recorded) == 2
    assert all(a.workload["identity_level"] == "structural" for a in recorded)
    # Same job, dated model variant → ONE workload, not two.
    assert recorded[0].workload["identity_ref"] == recorded[1].workload["identity_ref"]
    assert recorded[0].workload["surface"] == "direct_inference"


def test_openai_dialect_stream_translation_and_usage_capture():
    """The passthrough stream rewrites object/model and captures usage for billing."""
    import httpx

    events = [
        {"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"id": "x", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]},
        {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "x", "choices": [], "usage": {"prompt_tokens": 6, "completion_tokens": 2}},
    ]
    raw = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    class FakeStream:
        def __enter__(self):
            return httpx.Response(200, text=raw, request=httpx.Request("POST", "http://p"))

        def __exit__(self, *args):
            return False

    resolved = direct_inference.resolve_direct_model("gpt-4o")
    accounting = direct_inference.StreamAccounting()
    with patch("httpx.Client.stream", return_value=FakeStream()):
        chunks = list(
            direct_inference._stream_openai_dialect(
                resolved, [{"role": "user", "content": "hi"}], {}, "sk",
                request_id="chatcmpl-x", created=1,
                accounting=accounting, include_usage=False, timeout=5,
            )
        )

    parsed = [json.loads(c[5:].strip()) for c in chunks]
    assert all(p["object"] == "chat.completion.chunk" for p in parsed)
    assert all(p["model"] == "gpt-4o" for p in parsed)
    # include_usage=False → the usage-only trailer is not forwarded to the caller,
    # but OptiML still recorded it.
    assert len(parsed) == 3
    assert accounting.prompt_tokens == 6
    assert accounting.completion_tokens == 2
    assert accounting.finish_reason == "stop"


def test_openai_dialect_forwards_unknown_params_verbatim():
    """
    A parameter OptiML has never heard of still reaches the provider. The
    provider is the authority on its own API; a hand-maintained allow-list would
    reject fields providers shipped last week.
    """
    resolved = direct_inference.resolve_direct_model("gpt-4o")
    payload = direct_inference._openai_dialect_payload(
        resolved,
        [{"role": "user", "content": "hi"}],
        {"temperature": 0.1, "response_format": {"type": "json_object"}, "some_new_param": 7},
        stream=False,
    )
    assert payload["model"] == "gpt-4o"
    assert payload["some_new_param"] == 7
    assert payload["response_format"] == {"type": "json_object"}
    assert "stream" not in payload


def test_stream_options_include_usage_only_sent_where_documented():
    """Sending stream_options to a provider that does not know it is a 400."""
    opted_in = direct_inference._openai_dialect_payload(
        direct_inference.resolve_direct_model("gpt-4o"), [], {}, stream=True
    )
    assert opted_in["stream_options"] == {"include_usage": True}

    not_opted_in = direct_inference._openai_dialect_payload(
        direct_inference.resolve_direct_model("gemini-2.5-flash"), [], {}, stream=True
    )
    assert "stream_options" not in not_opted_in


def test_quota_is_not_consumed_by_a_request_that_never_dispatches(client, mock_ctx):
    """A typo'd model must not burn a month's request allowance."""
    with patch.object(openai_compat, "validate_api_key", return_value=mock_ctx), patch.object(
        openai_compat, "check_and_increment_usage"
    ), patch.object(openai_compat, "check_monthly_request_limit"), patch.object(
        openai_compat, "log_api_request", new=_async_noop
    ), patch.object(
        openai_compat, "increment_monthly_usage"
    ) as increment, patch.object(
        openai_compat.supabase, "table"
    ) as table:
        table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "typo-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    increment.assert_not_called()


def test_quota_is_consumed_by_a_dispatched_request(client, mock_ctx):
    with patch.object(openai_compat, "validate_api_key", return_value=mock_ctx), patch.object(
        openai_compat, "check_and_increment_usage"
    ), patch.object(openai_compat, "check_monthly_request_limit"), patch.object(
        openai_compat, "log_api_request", new=_async_noop
    ), patch.object(
        openai_compat, "increment_monthly_usage"
    ) as increment, patch.object(
        direct_inference, "_openai_complete", return_value=(_provider_response(), 10)
    ), patch.object(
        direct_inference, "get_org_provider_key", return_value="sk-provider"
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    increment.assert_called_once_with(ORG_ID)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Parameters must actually reach the provider, or be refused
# ═════════════════════════════════════════════════════════════════════════════
def test_temperature_and_max_tokens_reach_the_openai_dialect_payload():
    """
    workflow_runtime drops temperature/max_tokens. Direct inference must not:
    if this surface accepts them, they have to land in the outbound body.
    """
    payload = direct_inference._openai_dialect_payload(
        direct_inference.resolve_direct_model("gpt-4o"),
        [{"role": "user", "content": "hi"}],
        {"temperature": 0.15, "max_tokens": 321, "top_p": 0.9},
        stream=False,
    )
    assert payload["temperature"] == 0.15
    assert payload["max_tokens"] == 321
    assert payload["top_p"] == 0.9


def test_temperature_and_max_tokens_reach_the_anthropic_payload():
    payload = direct_inference._anthropic_payload(
        direct_inference.resolve_direct_model("claude-sonnet-4-5-20250929"),
        [{"role": "user", "content": "hi"}],
        {"temperature": 0.15, "max_tokens": 321, "top_p": 0.9},
        stream=False,
    )
    assert payload["temperature"] == 0.15
    assert payload["max_tokens"] == 321
    assert payload["top_p"] == 0.9


def test_anthropic_max_completion_tokens_is_honoured():
    payload = direct_inference._anthropic_payload(
        direct_inference.resolve_direct_model("claude-sonnet-4-5-20250929"),
        [{"role": "user", "content": "hi"}],
        {"max_completion_tokens": 99},
        stream=False,
    )
    assert payload["max_tokens"] == 99


def test_params_reach_the_provider_end_to_end(client, authed):
    captured = {}

    def fake_complete(resolved, messages, params, api_key, timeout=None):
        captured.update(
            direct_inference._openai_dialect_payload(resolved, messages, params, stream=False)
        )
        return (_provider_response(), 5)

    with patch.object(direct_inference, "_openai_complete", side_effect=fake_complete), patch.object(
        direct_inference, "get_org_provider_key", return_value="sk-provider"
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.15,
                "max_tokens": 321,
            },
        )
    assert response.status_code == 200
    assert captured["temperature"] == 0.15
    assert captured["max_tokens"] == 321


@pytest.mark.parametrize(
    "param,value",
    [
        ("temperature", 0.5),
        ("max_tokens", 100),
        ("max_completion_tokens", 100),
        ("response_format", {"type": "json_object"}),
        ("seed", 7),
    ],
)
def test_workflow_mode_refuses_params_it_would_discard(client, authed, param, value):
    """
    A workflow's parameters come from its graph; workflow_runtime never forwards
    these. Accepting them would be false compatibility.
    """
    with patch.object(openai_compat, "execute_workflow") as execute:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "optiml/support-triage",
                "messages": [{"role": "user", "content": "hi"}],
                param: value,
            },
        )
        execute.assert_not_called()
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "param_not_supported_for_workflow"
    assert param in error["message"]


# ═════════════════════════════════════════════════════════════════════════════
# 11. Workload identity comes from the domain layer (one implementation)
# ═════════════════════════════════════════════════════════════════════════════
def test_workload_identity_is_the_domain_implementation():
    """There must be exactly one workload-identity implementation."""
    from optimization import workloads

    assert hasattr(workloads, "direct_inference_identity")

    a = workloads.direct_inference_identity(
        provider="openai",
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": "you triage refunds"}],
    )
    b = workloads.direct_inference_identity(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "system", "content": "you triage   refunds"}],
    )
    c = workloads.direct_inference_identity(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "system", "content": "you write poems"}],
    )
    # Dated variant + whitespace noise collapse to one workload.
    assert a["model_target"] == b["model_target"]
    # A different job is a different workload.
    assert a["model_target"] != c["model_target"]

    with_tools = workloads.direct_inference_identity(
        provider="openai", model="gpt-4o",
        messages=[{"role": "system", "content": "you triage refunds"}], tools=TOOLS,
    )
    assert with_tools["model_target"] != a["model_target"]


def test_no_second_workload_identity_module_exists():
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("workload_identity")


def test_attempt_recording_runs_as_a_background_task_not_on_the_hot_path():
    """
    The domain write does DB I/O. It must run after the response is sent, and it
    must not be a bare create_task that can be dropped when the request ends.
    """
    import inspect

    source = inspect.getsource(openai_compat._with_attempt)
    assert "BackgroundTask(record_direct_inference_attempt" in source
    assert "asyncio.create_task(asyncio.to_thread(record_direct_inference_attempt" not in inspect.getsource(
        openai_compat
    )


def test_bridge_consumes_the_domain_workload_resolver():
    """The bridge must call optimization.workloads, not carry its own copy."""
    import optimization_bridge
    from optimization import workloads

    attempt = optimization_bridge.DirectInferenceAttempt(
        attempt_id="a", org_id=ORG_ID, occurred_at="2026-01-01T00:00:00Z",
        provider="openai", model="gpt-4o", requested_model="gpt-4o",
        system_prompt="you triage refunds",
    )
    described = optimization_bridge.describe_workload(attempt)
    expected = workloads.direct_inference_identity(
        provider="openai", model="gpt-4o",
        messages=[{"role": "system", "content": "you triage refunds"}],
    )
    assert described["identity_ref"] == expected["model_target"]
    assert described["surface"] == "direct_inference"
    assert described["identity_level"] == "structural"

    attempt.explicit_workload = "support-refund"
    named = optimization_bridge.describe_workload(attempt)
    assert named["identity_level"] == "explicit"
    assert named["external_key"] == "support-refund"


def test_bridge_resolves_the_workload_through_the_domain_layer():
    """record_direct_inference_attempt must go through workloads.resolve_workload."""
    import optimization_bridge

    attempt = optimization_bridge.DirectInferenceAttempt(
        attempt_id="a", org_id=ORG_ID, occurred_at="2026-01-01T00:00:00Z",
        provider="openai", model="gpt-4o", requested_model="gpt-4o",
        system_prompt="you triage refunds",
    )
    with patch("optimization.workloads.resolve_workload") as resolve:
        resolve.return_value = {
            "id": "wl-1", "identity_kind": "model_endpoint",
            "identity_level": "structural", "name": "gpt-4o",
        }
        optimization_bridge.record_direct_inference_attempt(attempt)

    resolve.assert_called_once()
    assert resolve.call_args.kwargs["surface"] == "direct_inference"
    assert resolve.call_args.kwargs["model_target"].startswith("di_")
    assert attempt.workload_id == "wl-1"
    # Strategy comes from optimization.strategy, built for the no-deployment case.
    assert attempt.strategy["surface"] == "direct_inference"
    assert attempt.strategy["surface_binding"]["deployment_id"] is None
    assert attempt.strategy["steps"][0]["executor_ref"]["external_id"] == "gpt-4o"


def test_bridge_never_raises_into_the_request_path():
    """Observability must not be able to fail a customer's production request."""
    import optimization_bridge

    attempt = optimization_bridge.DirectInferenceAttempt(
        attempt_id="a", org_id=ORG_ID, occurred_at="2026-01-01T00:00:00Z",
        provider="openai", model="gpt-4o", requested_model="gpt-4o",
    )
    with patch("optimization.workloads.resolve_workload", side_effect=RuntimeError("db down")):
        optimization_bridge.record_direct_inference_attempt(attempt)  # must not raise
