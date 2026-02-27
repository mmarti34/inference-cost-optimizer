"""
Tests for workflow deployment creation: 409 on duplicate (org_id, endpoint_slug).
"""
import pytest
from unittest.mock import patch, MagicMock

# Stub Crypto so router/workflow imports can load (same as test_public_execution.py)
import sys
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
from workflow_management import router as workflow_router

app = FastAPI()
app.include_router(workflow_router, prefix="/api")


@pytest.fixture
def client():
    return TestClient(app)


ORG_ID = "11111111-1111-1111-1111-111111111111"
WF_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
WF_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
ENDPOINT_SLUG = "my-endpoint"


@patch("workflow_management.supabase")
def test_deploy_same_endpoint_slug_returns_409(mock_supabase, client):
    """Deploy with endpoint_slug that already exists in org (another workflow) returns 409 (app-level check)."""
    def make_dep_chain(execute_data):
        c = MagicMock()
        c.eq.return_value = c
        c.order.return_value = c
        c.limit.return_value = c
        c.execute.return_value = MagicMock(data=execute_data)
        return c

    dep_table = MagicMock()
    # 1) existing for this workflow (WF_B) -> empty. 2) slug_collision -> slug taken. 3) all_deployments for plan count -> empty
    dep_table.select.side_effect = [
        make_dep_chain([]),
        make_dep_chain([{"id": "dep-existing", "workflow_id": WF_A, "org_id": ORG_ID, "version": 1, "endpoint_slug": ENDPOINT_SLUG}]),
        make_dep_chain([]),
    ]

    wf_table = MagicMock()
    wf_table.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"project_id": None, "org_id": ORG_ID}
    )

    org_table = MagicMock()
    org_chain = MagicMock()
    org_chain.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"plan_tier": "free", "plan": "free"}
    )
    org_table.select.return_value = org_chain

    def table(name):
        if name == "workflow_deployments":
            return dep_table
        if name == "workflows":
            return wf_table
        if name == "organizations":
            return org_table
        return MagicMock()
    mock_supabase.table.side_effect = table

    payload = {
        "workflow_id": WF_B,
        "org_id": ORG_ID,
        "endpoint_slug": ENDPOINT_SLUG,
        "graph_json": {"nodes": [], "edges": []},
    }
    r = client.post("/api/workflow-deployments", json=payload)
    assert r.status_code == 409
    assert r.json().get("detail") == "An endpoint with this slug already exists in this organization."


@patch("workflow_management.supabase")
def test_deploy_db_unique_violation_returns_409(mock_supabase, client):
    """When insert raises DB unique constraint violation (23505), return 409 with exact message."""
    def make_chain(execute_data):
        c = MagicMock()
        c.eq.return_value = c
        c.order.return_value = c
        c.limit.return_value = c
        c.single.return_value = c
        c.execute.return_value = MagicMock(data=execute_data)
        return c

    dep_table = MagicMock()
    dep_table.select.side_effect = [
        make_chain([]),   # existing for workflow: empty (first deploy)
        make_chain([]),   # slug_collision: empty
        make_chain([]),   # all_deployments for plan count
    ]
    dep_table.insert.return_value.execute.side_effect = Exception(
        'duplicate key value violates unique constraint "workflow_deployments_org_slug_unique" (23505)'
    )

    wf_table = MagicMock()
    wf_table.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"project_id": None, "org_id": ORG_ID}
    )

    org_table = MagicMock()
    org_chain = MagicMock()
    org_chain.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"plan_tier": "free", "plan": "free"}
    )
    org_table.select.return_value = org_chain

    def table(name):
        if name == "workflow_deployments":
            return dep_table
        if name == "workflows":
            return wf_table
        if name == "organizations":
            return org_table
        return MagicMock()
    mock_supabase.table.side_effect = table

    payload = {
        "workflow_id": WF_A,
        "org_id": ORG_ID,
        "endpoint_slug": ENDPOINT_SLUG,
        "graph_json": {"nodes": [], "edges": []},
    }
    r = client.post("/api/workflow-deployments", json=payload)
    assert r.status_code == 409
    assert r.json().get("detail") == "An endpoint with this slug already exists in this organization."


@patch("workflow_management.supabase")
def test_redeploy_different_slug_returns_400(mock_supabase, client):
    """Redeploy with a different endpoint_slug returns 400; slug is permanent after first deploy."""
    def make_dep_chain(execute_data):
        c = MagicMock()
        c.eq.return_value = c
        c.order.return_value = c
        c.limit.return_value = c
        c.execute.return_value = MagicMock(data=execute_data)
        return c

    existing_slug = "original-slug"
    dep_table = MagicMock()
    dep_table.select.side_effect = [
        make_dep_chain([{"version": 1, "endpoint_slug": existing_slug}]),
    ]

    wf_table = MagicMock()
    wf_table.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"project_id": None, "org_id": ORG_ID}
    )

    def table(name):
        if name == "workflow_deployments":
            return dep_table
        if name == "workflows":
            return wf_table
        return MagicMock()
    mock_supabase.table.side_effect = table

    payload = {
        "workflow_id": WF_A,
        "org_id": ORG_ID,
        "endpoint_slug": "different-slug",
        "graph_json": {"nodes": [], "edges": []},
    }
    r = client.post("/api/workflow-deployments", json=payload)
    assert r.status_code == 400
    assert "Cannot change endpoint_slug" in r.json().get("detail", "")
    assert existing_slug in r.json().get("detail", "")
