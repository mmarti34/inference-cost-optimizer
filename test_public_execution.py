"""
Minimal tests for public execution route: POST /api/public/{org_slug}/{endpoint_slug}
- Valid org + valid key -> 200
- Valid key + wrong org_slug -> the uniform tenant-opacity 404
- Valid org + wrong endpoint_slug -> the same uniform 404
- Missing key -> 401

See the TENANT OPACITY section at the end of this file for the isolation
contract these routes now hold to.
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


async def _async_noop(*args, **kwargs):
    """log_api_request is awaited inside asyncio.create_task; MagicMock is not awaitable."""
    return None


def _own_org_deployment_found(ctx):
    """
    Stand in for the deployment lookup, asserting it never escapes the key's org.

    Returning a deployment on the REJECTED paths is deliberate: the caller must
    get the same 404 even when their own org does have that endpoint.
    """
    async def _resolve(**kwargs):
        assert kwargs["org_id"] == ctx.org_id, (
            "deployment lookup escaped the authenticated key's org"
        )
        return (1, None, None, {
            "id": "dep-id", "workflow_id": "wf-id", "org_id": ctx.org_id,
            "version": 1, "endpoint_slug": "my-endpoint",
            "graph_json": {"nodes": [], "edges": []},
        })
    return _resolve


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


@patch("routers.public_execution.log_api_request")
@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_public_execute_valid_key_wrong_org_slug_403(
    mock_get_org_id, mock_validate, mock_resolve, mock_log, client, mock_ctx, other_org_id
):
    """
    Valid key, org_slug resolves to a DIFFERENT org -> the uniform 404.

    This used to assert 403 "does not belong", which was the enumeration
    oracle itself: 403 meant the org existed.
    """
    mock_validate.return_value = mock_ctx
    mock_get_org_id.return_value = other_org_id  # key is for valid_org_id, slug resolves to other
    # The deployment lookup now runs on the rejected path too (scoped to the
    # KEY's org), so that a refusal costs what an acceptance costs.
    mock_resolve.side_effect = _own_org_deployment_found(mock_ctx)
    mock_log.side_effect = _async_noop
    response = client.post(
        "/api/public/other-org/my-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )
    assert response.status_code == 404
    assert "does not belong" not in response.json().get("detail", "")


@patch("routers.public_execution.log_api_request")
@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_public_execute_org_not_found_404(
    mock_get_org_id, mock_validate, mock_resolve, mock_log, client, mock_ctx
):
    """
    Valid key, org_slug resolves to no org -> the uniform 404.

    "Organization not found." was the other half of the oracle: it confirmed
    absence just as clearly as the 403 confirmed presence.
    """
    mock_validate.return_value = mock_ctx
    mock_get_org_id.return_value = None
    mock_resolve.side_effect = _own_org_deployment_found(mock_ctx)
    mock_log.side_effect = _async_noop
    response = client.post(
        "/api/public/nonexistent-org/my-endpoint",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )
    assert response.status_code == 404
    assert "Organization not found" not in response.json().get("detail", "")


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


# ═════════════════════════════════════════════════════════════════════════════
# TENANT OPACITY
#
# A valid server API key must not be usable to enumerate which organizations
# exist. Before this suite, POST /api/public/{org_slug}/{endpoint_slug}
# answered three distinguishable shapes:
#
#   403 "API key does not belong to this organization."  -> the org EXISTS
#   404 "Organization not found."                        -> the org does NOT
#   404 "No promoted deployment found."                  -> your org, no slug
#
# The contract now: own org + existing resource works; EVERY other outcome an
# authenticated key can reach is one byte-identical 404.
# ═════════════════════════════════════════════════════════════════════════════

from routers.public_execution import _TENANT_OPAQUE_DETAIL


@pytest.fixture
def logged_requests():
    """Capture every api_request_log write instead of performing it."""
    captured = []

    async def _capture(entry):
        captured.append(dict(entry))

    with patch("routers.public_execution.log_api_request", new=_capture):
        yield captured


@pytest.fixture
def foreign_deployment(other_org_id):
    """A real, existing deployment — owned by an org that is NOT the key's."""
    return {
        "id": "foreign-dep-id",
        "workflow_id": "foreign-wf-id",
        "org_id": other_org_id,
        "version": 4,
        "endpoint_slug": "concert-reviews",
        "graph_json": {"nodes": [], "edges": []},
    }


def _tenant_probe(client, org_slug, endpoint_slug="my-endpoint", path_suffix=""):
    return client.post(
        f"/api/public/{org_slug}/{endpoint_slug}{path_suffix}",
        json={"input_text": "hi"},
        headers={"Authorization": "Bearer sk_live_fake"},
    )


def _wire_execute(mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
                  *, slug_map, deployment=None):
    """
    slug_map: org_slug -> org_id (or None for "no such org").
    deployment: what the OWN-ORG deployment lookup returns; None -> 404 miss.
    """
    from fastapi import HTTPException as _HTTPException
    mock_validate.return_value = mock_ctx
    mock_get_org_id.side_effect = lambda s: slug_map.get(s)

    async def _resolve(**kwargs):
        # The resolver must only ever be asked about the KEY's org.
        assert kwargs["org_id"] == mock_ctx.org_id, (
            "deployment lookup escaped the authenticated key's org"
        )
        if deployment is None:
            raise _HTTPException(status_code=404, detail="No promoted deployment found.")
        return (deployment["version"], None, None, deployment)

    mock_resolve.side_effect = _resolve


def _shape(response):
    """Everything a caller can observe, minus per-request noise."""
    return (
        response.status_code,
        response.content,
        {k.lower(): v for k, v in response.headers.items()},
    )


@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_foreign_org_and_nonexistent_org_are_byte_identical(
    mock_get_org_id, mock_validate, mock_resolve,
    client, mock_ctx, valid_org_id, other_org_id, mock_deployment, logged_requests,
):
    """
    THE test. A real foreign org and an org that does not exist must be
    indistinguishable — status, body AND headers.
    """
    _wire_execute(
        mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
        slug_map={"real-other-org": other_org_id, "no-such-org": None},
        deployment=mock_deployment,
    )

    foreign = _tenant_probe(client, "real-other-org", "concert-reviews")
    unknown = _tenant_probe(client, "no-such-org", "concert-reviews")

    assert foreign.status_code == 404
    assert _shape(foreign) == _shape(unknown)


@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_every_org_scoped_failure_collapses_to_one_response(
    mock_get_org_id, mock_validate, mock_resolve,
    client, mock_ctx, valid_org_id, other_org_id, mock_deployment, logged_requests,
):
    """
    Foreign org (resource present or absent), unknown org, and own org with no
    such deployment: four probes, one response.
    """
    slug_map = {
        "my-org": valid_org_id,
        "real-other-org": other_org_id,
        "no-such-org": None,
    }
    shapes = []
    for dep in (mock_deployment, None):
        _wire_execute(
            mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
            slug_map=slug_map, deployment=dep,
        )
        shapes.append(_shape(_tenant_probe(client, "real-other-org", "concert-reviews")))
        shapes.append(_shape(_tenant_probe(client, "no-such-org", "concert-reviews")))
    # Own org, missing resource — the third shape that used to say
    # "No promoted deployment found."
    _wire_execute(
        mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
        slug_map=slug_map, deployment=None,
    )
    shapes.append(_shape(_tenant_probe(client, "my-org", "concert-reviews")))

    assert all(s[0] == 404 for s in shapes)
    assert all(s == shapes[0] for s in shapes), "an org-scoped failure is still distinguishable"


@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_response_never_echoes_the_supplied_org_slug_or_key_ownership(
    mock_get_org_id, mock_validate, mock_resolve,
    client, mock_ctx, other_org_id, mock_deployment, logged_requests,
):
    """The body must not repeat what the caller sent, nor mention the key."""
    _wire_execute(
        mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
        slug_map={"acme-secret-workspace": other_org_id}, deployment=mock_deployment,
    )
    response = _tenant_probe(client, "acme-secret-workspace", "payroll-endpoint")
    body = response.text.lower()
    assert "acme-secret-workspace" not in body
    assert "payroll-endpoint" not in body
    assert "does not belong" not in body
    assert "organization" not in body
    assert response.json() == {"detail": _TENANT_OPAQUE_DETAIL}


@patch("routers.public_execution.increment_monthly_usage")
@patch("routers.public_execution.check_monthly_request_limit")
@patch("routers.public_execution.check_and_increment_usage")
@patch("routers.public_execution.execute_workflow")
@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_own_org_with_existing_resource_still_executes(
    mock_get_org_id, mock_validate, mock_resolve, mock_execute,
    mock_rate, mock_monthly, mock_incr,
    client, mock_ctx, valid_org_id, mock_deployment, logged_requests,
):
    """Closing the leak must not close the door: the legitimate call still runs."""
    _wire_execute(
        mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
        slug_map={"my-org": valid_org_id}, deployment=mock_deployment,
    )
    mock_execute.return_value = {"output": "ok", "run_id": None}

    response = _tenant_probe(client, "my-org", "my-endpoint")
    assert response.status_code == 200
    assert response.json().get("output") == "ok"


@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_cross_tenant_probe_writes_no_row_under_the_foreign_org(
    mock_get_org_id, mock_validate, mock_resolve,
    client, mock_ctx, valid_org_id, other_org_id, mock_deployment, logged_requests,
):
    """
    The second bug: log_entry["org_id"] was set from the PATH slug before the
    tenant check, so probing another org's endpoint inserted a row into THEIR
    api_request_log. A keyholder must not be able to write into another
    tenant's log.
    """
    _wire_execute(
        mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
        slug_map={"real-other-org": other_org_id, "no-such-org": None},
        deployment=mock_deployment,
    )
    _tenant_probe(client, "real-other-org", "concert-reviews")
    _tenant_probe(client, "no-such-org", "concert-reviews")

    assert logged_requests, "server-side observability was dropped, not just narrowed"
    assert all(e["org_id"] == valid_org_id for e in logged_requests), (
        "a rejected cross-tenant request was logged under an org that is not the key's"
    )
    assert not any(e["org_id"] == other_org_id for e in logged_requests)


@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_customer_visible_log_row_stays_uniform(
    mock_get_org_id, mock_validate, mock_resolve,
    client, mock_ctx, other_org_id, mock_deployment, logged_requests,
):
    """
    api_request_log is customer-visible and the row is now written under the
    CALLER's own org — so its error_message must not restate what the response
    withheld.
    """
    _wire_execute(
        mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
        slug_map={"real-other-org": other_org_id}, deployment=mock_deployment,
    )
    _tenant_probe(client, "real-other-org", "concert-reviews")

    row = logged_requests[-1]
    assert row["http_status"] == 404
    assert row["error_message"] == _TENANT_OPAQUE_DETAIL
    assert "real-other-org" not in str(row.get("error_message"))


@patch("routers.public_execution.resolve_version_and_deployment")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_internal_log_still_records_the_specific_reason(
    mock_get_org_id, mock_validate, mock_resolve,
    client, mock_ctx, valid_org_id, other_org_id, mock_deployment, logged_requests, caplog,
):
    """Uniform to the caller, specific to us: operators keep the real reason."""
    import logging as _logging

    slug_map = {"my-org": valid_org_id, "real-other-org": other_org_id, "no-such-org": None}

    with caplog.at_level(_logging.WARNING, logger="routers.public_execution"):
        _wire_execute(
            mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
            slug_map=slug_map, deployment=mock_deployment,
        )
        _tenant_probe(client, "real-other-org", "concert-reviews")
        _tenant_probe(client, "no-such-org", "concert-reviews")
        _wire_execute(
            mock_get_org_id, mock_validate, mock_resolve, mock_ctx,
            slug_map=slug_map, deployment=None,
        )
        _tenant_probe(client, "my-org", "concert-reviews")

    reasons = " | ".join(r.getMessage() for r in caplog.records)
    assert "org_slug_belongs_to_another_org" in reasons
    assert "org_slug_resolves_to_no_org" in reasons
    assert "no_promoted_deployment_for_endpoint_slug" in reasons


@patch("routers.public_execution.validate_api_key")
def test_invalid_key_is_still_401_not_the_uniform_404(mock_validate, client):
    """
    401 is unchanged. This fix is about what an AUTHENTICATED key can learn,
    not about hiding that a key is bad.
    """
    from fastapi import HTTPException as _HTTPException
    mock_validate.side_effect = _HTTPException(status_code=401, detail="Invalid API key.")
    response = _tenant_probe(client, "my-org")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}


# ── /feedback obeys the same rule ────────────────────────────────────────────

def _feedback_probe(client, org_slug, request_id="req-1"):
    return client.post(
        f"/api/public/{org_slug}/my-endpoint/feedback",
        json={"request_id": request_id, "metrics": {"helpfulness_score": 1}},
        headers={"Authorization": "Bearer sk_live_fake"},
    )


@patch("routers.public_execution.get_api_request_log_sync")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_feedback_foreign_and_nonexistent_org_are_byte_identical(
    mock_get_org_id, mock_validate, mock_get_log, client, mock_ctx, valid_org_id, other_org_id,
):
    mock_validate.return_value = mock_ctx
    mock_get_org_id.side_effect = lambda s: {
        "real-other-org": other_org_id, "no-such-org": None,
    }.get(s)
    mock_get_log.return_value = {"id": "req-1", "org_id": valid_org_id, "custom_metrics": {}}

    foreign = _feedback_probe(client, "real-other-org")
    unknown = _feedback_probe(client, "no-such-org")

    assert foreign.status_code == 404
    assert _shape(foreign) == _shape(unknown)
    assert foreign.json() == {"detail": _TENANT_OPAQUE_DETAIL}


@patch("routers.public_execution.get_api_request_log_sync")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_feedback_own_org_missing_request_id_is_the_same_404(
    mock_get_org_id, mock_validate, mock_get_log, client, mock_ctx, valid_org_id, other_org_id,
):
    mock_validate.return_value = mock_ctx
    mock_get_org_id.side_effect = lambda s: {
        "my-org": valid_org_id, "real-other-org": other_org_id,
    }.get(s)

    mock_get_log.return_value = None
    own_missing = _feedback_probe(client, "my-org", "no-such-request")

    mock_get_log.return_value = {"id": "req-1", "org_id": valid_org_id, "custom_metrics": {}}
    foreign = _feedback_probe(client, "real-other-org")

    assert own_missing.status_code == 404
    assert _shape(own_missing) == _shape(foreign)


@patch("routers.public_execution.get_api_request_log_sync")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_feedback_another_orgs_request_id_is_the_same_404(
    mock_get_org_id, mock_validate, mock_get_log, client, mock_ctx, valid_org_id, other_org_id,
):
    """A request id is obscurity, not access control."""
    mock_validate.return_value = mock_ctx
    mock_get_org_id.side_effect = lambda s: {"my-org": valid_org_id}.get(s)
    mock_get_log.return_value = {"id": "req-1", "org_id": other_org_id, "custom_metrics": {}}

    response = _feedback_probe(client, "my-org")
    assert response.status_code == 404
    assert response.json() == {"detail": _TENANT_OPAQUE_DETAIL}


@patch("routers.public_execution.update_api_request_log_metrics")
@patch("routers.public_execution.get_api_request_log_sync")
@patch("routers.public_execution.validate_api_key")
@patch("routers.public_execution.get_org_id_from_slug")
def test_feedback_own_org_own_request_still_works(
    mock_get_org_id, mock_validate, mock_get_log, mock_update,
    client, mock_ctx, valid_org_id,
):
    mock_validate.return_value = mock_ctx
    mock_get_org_id.side_effect = lambda s: {"my-org": valid_org_id}.get(s)
    mock_get_log.return_value = {"id": "req-1", "org_id": valid_org_id, "custom_metrics": {}}
    mock_update.side_effect = _async_noop

    response = _feedback_probe(client, "my-org")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
