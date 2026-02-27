"""
Minimal tests for public execution route: POST /api/public/{org_slug}/{endpoint_slug}
- Valid org + valid key -> 200
- Valid key + wrong org_slug -> 403
- Valid org + wrong endpoint_slug -> 404
- Missing key -> 401
"""
import sys
import pytest
from unittest.mock import patch, MagicMock

# Stub Crypto so we can import the router without pycryptodome (tests patch validate_api_key).
if "Crypto" not in sys.modules:
    import types
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        _m = types.ModuleType("Crypto." + sub)
        sys.modules["Crypto." + sub] = _m
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.public_execution import router as public_execution_router

app = FastAPI()
app.include_router(public_execution_router, prefix="/api/public", tags=["public"])


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_org_id():
    return "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def other_org_id():
    return "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def mock_ctx(valid_org_id):
    from api_key_validation import OrgContext
    return OrgContext(org_id=valid_org_id, key_type="live", rate_limit_per_minute=60)


@pytest.fixture
def mock_deployment(valid_org_id):
    return {
        "id": "dep-id",
        "workflow_id": "wf-id",
        "project_id": "proj-id",
        "org_id": valid_org_id,
        "version": 1,
        "endpoint_slug": "my-endpoint",
        "graph_json": {"nodes": [], "edges": []},
        "created_at": "2025-01-01T00:00:00Z",
    }


def test_public_execute_missing_key_401(client):
    """Missing or invalid Authorization -> 401."""
    response = client.post(
        "/api/public/my-org/my-endpoint",
        json={"input_text": "hi"},
    )
    assert response.status_code == 401


def test_public_execute_invalid_key_401(client):
    """Invalid Bearer token -> 401 (validate_api_key raises)."""
    response = client.post(
        "/api/public/my-org/my-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_public_execute_valid_key_wrong_org_slug_403(
    mock_get_org_id, mock_validate, client, mock_ctx, other_org_id
):
    """Valid key but org_slug resolves to a different org -> 403."""
    mock_validate.return_value = mock_ctx
    mock_get_org_id.return_value = other_org_id  # key is for valid_org_id, slug resolves to other
    response = client.post(
        "/api/public/other-org/my-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )
    assert response.status_code == 403
    assert "does not belong" in response.json().get("detail", "")


@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_public_execute_org_not_found_404(mock_get_org_id, mock_validate, client, mock_ctx):
    """Valid key but org_slug does not resolve to any org -> 404."""
    mock_validate.return_value = mock_ctx
    mock_get_org_id.return_value = None
    response = client.post(
        "/api/public/nonexistent-org/my-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )
    assert response.status_code == 404
    assert "Organization not found" in response.json().get("detail", "")


@patch("routers.public_execution.execute_workflow")
@patch("routers.public_execution.check_and_increment_usage")
@patch("routers.public_execution.supabase")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_public_execute_valid_org_wrong_endpoint_slug_404(
    mock_get_org_id, mock_validate, mock_supabase, mock_rate, mock_execute,
    client, mock_ctx, valid_org_id
):
    """Valid key and org, but no deployment for endpoint_slug -> 404."""
    mock_validate.return_value = mock_ctx
    mock_get_org_id.return_value = valid_org_id
    # Deployment query returns empty (chain: select().eq().eq().order().limit().execute())
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=[])
    mock_table_dep = MagicMock()
    mock_table_dep.select.return_value = chain
    mock_supabase.table.return_value = mock_table_dep

    response = client.post(
        "/api/public/my-org/nonexistent-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )
    assert response.status_code == 404
    assert "not published" in response.json().get("detail", "")


@patch("routers.public_execution.execute_workflow")
@patch("routers.public_execution.check_and_increment_usage")
@patch("routers.public_execution.supabase")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_public_execute_valid_org_valid_key_200(
    mock_get_org_id, mock_validate, mock_supabase, mock_rate, mock_execute,
    client, mock_ctx, valid_org_id, mock_deployment
):
    """Valid org_slug + valid key + deployment exists -> 200."""
    mock_validate.return_value = mock_ctx
    mock_get_org_id.return_value = valid_org_id
    mock_execute.return_value = {"output": "ok"}

    # Deployment chain: select().eq().eq().order().limit().execute()
    dep_chain = MagicMock()
    dep_chain.eq.return_value = dep_chain
    dep_chain.order.return_value = dep_chain
    dep_chain.limit.return_value = dep_chain
    dep_chain.execute.return_value = MagicMock(data=[mock_deployment])
    mock_table_dep = MagicMock()
    mock_table_dep.select.return_value = dep_chain

    # Workflows table: select().eq().single().execute()
    wf_chain = MagicMock()
    wf_chain.eq.return_value = wf_chain
    wf_chain.single.return_value = wf_chain
    wf_chain.execute.return_value = MagicMock(data={"variables": None})
    mock_table_wf = MagicMock()
    mock_table_wf.select.return_value = wf_chain

    def table(name):
        if name == "workflow_deployments":
            return mock_table_dep
        if name == "workflows":
            return mock_table_wf
        return MagicMock()

    mock_supabase.table.side_effect = table

    response = client.post(
        "/api/public/my-org/my-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )
    assert response.status_code == 200
    assert response.json().get("output") == "ok"
