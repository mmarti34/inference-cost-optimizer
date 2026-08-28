"""
Tenant opacity for POST /v1/prompt (main.py).

/v1/prompt is authenticated by a SERVER API KEY and takes a caller-supplied
``prompt_id``. It used to answer differently depending on whose prompt that id
named:

  403 "You do not have access to this prompt."  -> the id exists, in ANOTHER org
  500 "Error accessing prompt template."        -> the id exists nowhere
                                                   (`.single()` raised on zero
                                                   rows and the except clause
                                                   turned the 404 into a 500)

so a single valid key could test whether a prompt_template id belonged to
another tenant. Both are now one byte-identical 404. Same rule as
POST /v1/outcomes and POST /api/public/{org_slug}/{endpoint_slug}.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub Crypto so main imports without pycryptodome (auth is patched).
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

from fastapi.testclient import TestClient

import api_key_validation
import main
import rate_limiting

ORG_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ORG_ID = "22222222-2222-2222-2222-222222222222"
AUTH = {"Authorization": "Bearer sk_live_fake"}


@pytest.fixture
def client():
    return TestClient(main.app)


def _single(data):
    """A supabase chain whose terminal .execute() yields `data`."""
    chain = MagicMock()
    for attr in ("select", "eq", "gte", "order", "limit", "single"):
        getattr(chain, attr).return_value = chain
    chain.execute.return_value = MagicMock(data=data)
    return chain


def _supabase_for(prompt_rows):
    """
    Everything /v1/prompt reads before the prompt lookup, plus the prompt row.

    A Personal org on a paid plan clears the access checks without touching
    member limits, so the test exercises the prompt branch and nothing else.
    """
    tables = {
        "organizations": _single({
            "id": ORG_ID, "name": "Acme", "type": "Personal", "plan": "pro",
            "created_by": "u1", "logo": None, "created_at": "2025-01-01T00:00:00Z",
        }),
        "organization_members": _single({"user_id": "u1"}),
        "user_profiles": _single({
            "subscription_tier": "pro", "subscription_status": "active",
        }),
        "prompt_templates": _single(prompt_rows),
    }
    sb = MagicMock()
    sb.table.side_effect = lambda name: tables.get(name, _single([]))
    return sb


def _probe(client, prompt_rows, prompt_id="pt-1"):
    ctx = api_key_validation.OrgContext(
        org_id=ORG_ID, key_type="live", rate_limit_per_minute=60
    )
    with patch.object(api_key_validation, "validate_api_key", return_value=ctx), \
            patch.object(rate_limiting, "check_and_increment_usage"), \
            patch.object(main, "supabase", _supabase_for(prompt_rows)), \
            patch("utils.observability_integration.log_gateway_span"), \
            patch("utils.observability_integration.log_provider_call_span"), \
            patch("utils.observability_integration.log_request_with_observability"):
        return client.post(
            "/v1/prompt",
            headers=AUTH,
            json={"prompt_id": prompt_id, "input": "hi"},
        )


def _shape(response):
    return (
        response.status_code,
        response.content,
        {k.lower(): v for k, v in response.headers.items()
         if k.lower() not in ("x-request-id", "x-trace-id", "date")},
    )


def test_another_orgs_prompt_id_is_the_same_404_as_no_such_prompt(client):
    """THE test: a real foreign template and a nonexistent id are one response."""
    foreign = _probe(client, [{
        "id": "pt-1", "org_id": OTHER_ORG_ID, "prompt": "SECRET TEMPLATE {input}",
        "provider": "openai", "model": "gpt-4o", "name": "theirs",
        "project_id": "p1", "created_at": "2025-01-01T00:00:00Z",
    }])
    missing = _probe(client, [])

    assert foreign.status_code == 404
    assert _shape(foreign) == _shape(missing)


def test_the_refusal_leaks_neither_the_foreign_org_nor_the_template(client):
    body = _probe(client, [{
        "id": "pt-1", "org_id": OTHER_ORG_ID, "prompt": "SECRET TEMPLATE {input}",
        "provider": "openai", "model": "gpt-4o", "name": "theirs",
        "project_id": "p1", "created_at": "2025-01-01T00:00:00Z",
    }]).text
    assert OTHER_ORG_ID not in body
    assert "SECRET TEMPLATE" not in body
    assert "do not have access" not in body


def test_a_template_with_no_org_is_refused_not_served(client):
    """Fail closed, and fail opaque."""
    response = _probe(client, [{
        "id": "pt-1", "org_id": None, "prompt": "orphan {input}",
        "provider": "openai", "model": "gpt-4o", "name": "orphan",
        "project_id": "p1", "created_at": "2025-01-01T00:00:00Z",
    }])
    assert response.status_code == 404


def test_missing_key_is_still_401(client):
    response = client.post("/v1/prompt", json={"prompt_id": "pt-1", "input": "hi"})
    assert response.status_code == 401
