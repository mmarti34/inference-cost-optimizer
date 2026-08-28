"""
Resource-ownership (tenant isolation) on the Tier-0 endpoints.

MECHANISM UNDER TEST
────────────────────
`supabase_client` uses the SERVICE-ROLE key, so RLS never applies: the query
filters ARE the authorization. `require_org_member` proves the caller belongs
to the org they named. It proves NOTHING about the other identifiers in the
same request — `workflow_id`, `deployment_id`, `assetId`. Every endpoint below
used to fetch or mutate one of those on `.eq("id", …)` alone, and several then
read `org_id` back out of the fetched row and executed as that org. No trick
required: an ordinary well-formed request from a member of any org.

The worst of them, POST /api/execute-workflow, ran the victim's deployment
under the victim's org, which makes `workflow_runtime._resolve_secrets` decrypt
the victim's `org_secrets` into tool URLs and headers and spend their provider
keys. That is a cross-tenant SECRET-USE primitive, not an information leak.

HOW THESE TESTS ARE BUILT
─────────────────────────
`_FakeDB` is a two-tenant in-memory Postgres double that ACTUALLY APPLIES the
filters a query declares. That is the point: it is not a recorder that can be
satisfied by a mock returning whatever the handler asked for. If a
`.eq("org_id", …)` is deleted from the source, the fake happily returns the
other tenant's row and these tests fail — which is what makes them evidence
rather than decoration. Each was mutation-tested that way.

Per resource class the shape is:

    own resource      -> expected success
    foreign resource  -> opaque failure
    unknown resource  -> BYTE-IDENTICAL failure   (no existence oracle)
    foreign mutation  -> ZERO mutation            (asserted against the DB)
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub Crypto so main/workflow imports load without pycryptodome.
if "Crypto" not in sys.modules:  # pragma: no cover - import shim only
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        sys.modules["Crypto." + sub] = types.ModuleType("Crypto." + sub)
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api_key_management  # noqa: E402
import main  # noqa: E402
import resource_access  # noqa: E402
import workflow_management  # noqa: E402
from auth_dependency import AuthenticatedUser  # noqa: E402

ORG_A = "11111111-1111-1111-1111-111111111111"   # the caller's org
ORG_B = "22222222-2222-2222-2222-222222222222"   # the victim
USER_A = "cccccccc-cccc-cccc-cccc-cccccccccccc"

WF_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
WF_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
DEP_A = "dddddddd-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DEP_B = "dddddddd-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
ASSET_A = "eeeeeeee-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
ASSET_B = "eeeeeeee-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
UNKNOWN = "00000000-0000-0000-0000-000000000000"

VICTIM_GRAPH_MARKER = "TENANT-B-PRODUCTION-GRAPH"
VICTIM_ASSET_MARKER = "TENANT-B-KNOWLEDGE-BASE-SECRET"
VICTIM_KEY_MARKER = "TENANT-B-ENCRYPTED-PROVIDER-KEY"


# ─── A two-tenant Postgres double that really filters ───────────────────────


class _Query:
    def __init__(self, db, table):
        self._db = db
        self._table = table
        self._op = "select"
        self._payload = None
        self._eq = []
        self._neq = []
        self._in = None
        self._single = False
        self._limit = None

    # builder ------------------------------------------------------------
    def select(self, *a, **k):
        return self

    def insert(self, data):
        self._op, self._payload = "insert", data
        return self

    def update(self, data):
        self._op, self._payload = "update", data
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._eq.append((col, val))
        return self

    def neq(self, col, val):
        self._neq.append((col, val))
        return self

    def in_(self, col, vals):
        self._in = (col, [str(v) for v in vals])
        return self

    def gte(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def single(self):
        self._single = True
        return self

    # execution ----------------------------------------------------------
    def _matches(self, row):
        for col, val in self._eq:
            if str(row.get(col)) != str(val):
                return False
        for col, val in self._neq:
            if str(row.get(col)) == str(val):
                return False
        if self._in is not None:
            col, vals = self._in
            if str(row.get(col)) not in vals:
                return False
        return True

    def execute(self):
        rows = self._db.rows.setdefault(self._table, [])
        if self._op == "insert":
            payload = self._payload if isinstance(self._payload, list) else [self._payload]
            created = []
            for item in payload:
                row = dict(item)
                row.setdefault("id", "gen-%d" % self._db.next_id())
                row.setdefault("created_at", "2026-01-01T00:00:00Z")
                rows.append(row)
                created.append(dict(row))
            self._db.writes.append(("insert", self._table, list(self._eq), created))
            return MagicMock(data=created)

        matched = [r for r in rows if self._matches(r)]

        if self._op == "update":
            for r in matched:
                r.update(self._payload)
            out = [dict(r) for r in matched]
            self._db.writes.append(("update", self._table, list(self._eq), out))
            return MagicMock(data=out)

        if self._op == "delete":
            for r in matched:
                rows.remove(r)
            out = [dict(r) for r in matched]
            self._db.writes.append(("delete", self._table, list(self._eq), out))
            return MagicMock(data=out)

        out = [dict(r) for r in matched]
        if self._limit is not None:
            out = out[: self._limit]
        if self._single:
            return MagicMock(data=(out[0] if out else None))
        return MagicMock(data=out)


class _FakeDB:
    def __init__(self, rows):
        self.rows = {k: [dict(r) for r in v] for k, v in rows.items()}
        self.writes = []
        self._n = 0

    def next_id(self):
        self._n += 1
        return self._n

    def table(self, name):
        return _Query(self, name)

    def mutating_writes(self):
        """Writes that actually touched a row (an update matching nothing is a no-op)."""
        return [w for w in self.writes if w[3]]


def _seed():
    return _FakeDB(
        {
            "organizations": [
                {"id": ORG_A, "plan": "enterprise"},
                {"id": ORG_B, "plan": "enterprise"},
            ],
            "workflows": [
                {"id": WF_A, "org_id": ORG_A, "project_id": "proj-a", "name": "A",
                 "slug": "a", "graph_json": {"nodes": [{"id": "n"}], "edges": []},
                 "variables": [], "created_at": "2026-01-01T00:00:00Z"},
                {"id": WF_B, "org_id": ORG_B, "project_id": "proj-b", "name": "B",
                 "slug": "b", "graph_json": {"nodes": [{"id": VICTIM_GRAPH_MARKER}], "edges": []},
                 "variables": [], "created_at": "2026-01-01T00:00:00Z"},
            ],
            "workflow_deployments": [
                {"id": DEP_A, "workflow_id": WF_A, "org_id": ORG_A, "project_id": "proj-a",
                 "version": 1, "endpoint_slug": "ep-a", "status": "candidate",
                 "graph_json": {"nodes": [{"id": "n"}], "edges": []},
                 "created_at": "2026-01-01T00:00:00Z"},
                {"id": DEP_B, "workflow_id": WF_B, "org_id": ORG_B, "project_id": "proj-b",
                 "version": 7, "endpoint_slug": "ep-b", "status": "candidate",
                 "graph_json": {"nodes": [{"id": VICTIM_GRAPH_MARKER}], "edges": []},
                 "created_at": "2026-01-01T00:00:00Z"},
            ],
            "context_assets": [
                {"id": ASSET_A, "org_id": ORG_A, "content": "tenant-a-notes", "metadata": {}},
                {"id": ASSET_B, "org_id": ORG_B, "content": VICTIM_ASSET_MARKER, "metadata": {}},
            ],
            "context_asset_snapshots": [],
            "api_keys": [
                {"id": "key-b", "org_id": ORG_B, "provider": "openai",
                 "api_key": VICTIM_KEY_MARKER, "name": "b", "user_id": None,
                 "created_at": "2026-01-01T00:00:00Z"},
            ],
            "service_api_keys": [],
            "eval_suites": [],
            "eval_runs": [],
        }
    )


# ─── App wiring ─────────────────────────────────────────────────────────────


def _member_of_org_a():
    user = AuthenticatedUser(user_id=USER_A, email="a@b.c")
    user._verified_org_id = ORG_A
    user._org_role = "admin"
    return user


@pytest.fixture
def wf_app():
    app = FastAPI()
    app.include_router(workflow_management.router, prefix="/api")
    app.dependency_overrides[workflow_management.require_org_member] = _member_of_org_a
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def db():
    return _seed()


@pytest.fixture
def wf_client(wf_app, db):
    """Workflow router wired to the fake DB, with execution/eval side effects stubbed."""
    with patch.object(workflow_management, "supabase", db), \
         patch.object(resource_access, "supabase", db), \
         patch.object(workflow_management, "execute_workflow") as exec_mock, \
         patch.object(workflow_management, "_run_eval_sync", MagicMock()), \
         patch.object(workflow_management, "_end_experiments_on_endpoint_sync", MagicMock()), \
         patch.object(workflow_management, "check_workflow_limit", MagicMock()):
        exec_mock.return_value = {"final_output": "ok", "node_results": [], "total_cost": 0}
        client = TestClient(wf_app)
        client.execute_mock = exec_mock
        yield client


@pytest.fixture
def main_client(db):
    main.app.dependency_overrides[main.require_org_member] = _member_of_org_a
    with patch.object(main, "supabase", db), \
         patch.object(resource_access, "supabase", db), \
         patch.object(main, "encrypt_api_key", lambda v: "enc:" + v):
        yield TestClient(main.app)
    main.app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: WORKFLOW  (PUT /api/workflows/{workflow_id})
# ═══════════════════════════════════════════════════════════════════════════

NEW_GRAPH = {"nodes": [{"id": "attacker-node"}], "edges": []}


def test_workflow_update_own_succeeds(wf_client, db):
    r = wf_client.put(f"/api/workflows/{WF_A}", json={"graph_json": NEW_GRAPH})
    assert r.status_code == 200, r.text
    assert r.json()["org_id"] == ORG_A
    assert db.rows["workflows"][0]["graph_json"] == NEW_GRAPH


def test_workflow_update_foreign_is_opaque_404(wf_client):
    r = wf_client.put(f"/api/workflows/{WF_B}", json={"graph_json": NEW_GRAPH})
    assert r.status_code == 404
    assert r.json() == {"detail": "Workflow not found"}
    assert VICTIM_GRAPH_MARKER not in r.text


def test_workflow_update_with_no_changes_does_not_disclose_a_foreign_workflow(wf_client):
    """
    The no-op branch RETURNS the row rather than writing, so the doubly-scoped
    UPDATE cannot cover it — only the ownership fetch can. This is the read
    path that a disclosure-before-authorization regression would reopen.
    """
    r = wf_client.put(f"/api/workflows/{WF_B}", json={})
    assert r.status_code == 404
    assert r.json() == {"detail": "Workflow not found"}
    assert VICTIM_GRAPH_MARKER not in r.text


def test_workflow_update_foreign_and_unknown_are_byte_identical(wf_client):
    foreign = wf_client.put(f"/api/workflows/{WF_B}", json={"graph_json": NEW_GRAPH})
    unknown = wf_client.put(f"/api/workflows/{UNKNOWN}", json={"graph_json": NEW_GRAPH})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_workflow_update_foreign_performs_zero_mutation(wf_client, db):
    before = [dict(r) for r in db.rows["workflows"]]
    wf_client.put(f"/api/workflows/{WF_B}", json={"graph_json": NEW_GRAPH})
    assert db.rows["workflows"] == before
    assert db.mutating_writes() == []


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: WORKFLOW EXECUTION  (POST /api/execute-workflow)
#
# The cross-tenant SECRET-USE primitive. `org_id` is omitted from the body, so
# the guard resolves and verifies ORG_A from X-Org-Id; the victim's workflow_id
# is named in its place.
# ═══════════════════════════════════════════════════════════════════════════


def _exec_body(workflow_id, **kw):
    body = {"workflow_id": workflow_id, "user_id": USER_A}
    body.update(kw)
    return body


def test_execute_own_workflow_runs_as_the_verified_org(wf_client):
    r = wf_client.post("/api/execute-workflow", json=_exec_body(WF_A))
    assert r.status_code == 200, r.text
    args = wf_client.execute_mock.call_args
    assert args[0][2] == ORG_A          # positional org_id passed to execute_workflow


def test_execute_foreign_workflow_never_reaches_the_runtime(wf_client):
    """The whole defect in one assertion: no execution, so no victim secret is decrypted."""
    r = wf_client.post("/api/execute-workflow", json=_exec_body(WF_B))
    assert r.status_code == 404
    assert r.json() == {"detail": "Workflow not found"}
    wf_client.execute_mock.assert_not_called()
    assert VICTIM_GRAPH_MARKER not in r.text


def test_execute_never_adopts_an_org_read_out_of_a_fetched_row(wf_client):
    """
    Even naming the victim's org explicitly cannot move execution into it: the
    org handed to the runtime is the guard's, not the deployment row's.
    """
    r = wf_client.post(
        "/api/execute-workflow",
        json=_exec_body(WF_A, org_id=ORG_B, graph_json={"nodes": [{"id": "n"}]}),
    )
    assert r.status_code == 200, r.text
    assert wf_client.execute_mock.call_args[0][2] == ORG_A


def test_execute_foreign_and_unknown_workflow_are_byte_identical(wf_client):
    foreign = wf_client.post("/api/execute-workflow", json=_exec_body(WF_B))
    unknown = wf_client.post("/api/execute-workflow", json=_exec_body(UNKNOWN))
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_execute_foreign_workflow_writes_nothing(wf_client, db):
    wf_client.post("/api/execute-workflow", json=_exec_body(WF_B))
    assert db.mutating_writes() == []


def test_stream_foreign_workflow_is_refused(wf_client):
    r = wf_client.post(
        "/api/execute-workflow/stream",
        json=_exec_body(WF_B, org_id=ORG_A, graph_json={"nodes": [{"id": "n"}]},
                        variables={"x": 1}),
    )
    assert r.status_code == 404
    assert r.json() == {"detail": "Workflow not found"}


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: DEPLOYMENT  (promote / activate / delete)
# ═══════════════════════════════════════════════════════════════════════════


def _dep_row(db, dep_id):
    return next(r for r in db.rows["workflow_deployments"] if r["id"] == dep_id)


def test_promote_own_deployment_succeeds(wf_client, db):
    r = wf_client.post(f"/api/workflow-deployments/{DEP_A}/promote", json={})
    assert r.status_code == 200, r.text
    assert _dep_row(db, DEP_A)["status"] == "promoted"


def test_promote_foreign_deployment_is_opaque_404_and_mutates_nothing(wf_client, db):
    r = wf_client.post(f"/api/workflow-deployments/{DEP_B}/promote", json={})
    assert r.status_code == 404
    assert r.json() == {"detail": "Deployment not found"}
    assert _dep_row(db, DEP_B)["status"] == "candidate"
    assert db.mutating_writes() == []


def test_promote_foreign_and_unknown_are_byte_identical(wf_client):
    foreign = wf_client.post(f"/api/workflow-deployments/{DEP_B}/promote", json={})
    unknown = wf_client.post(f"/api/workflow-deployments/{UNKNOWN}/promote", json={})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_activate_own_deployment_succeeds(wf_client, db):
    r = wf_client.post(f"/api/workflow-deployments/{DEP_A}/activate", json={})
    assert r.status_code == 200, r.text
    assert _dep_row(db, DEP_A)["status"] == "promoted"


def test_activate_foreign_deployment_discloses_nothing_and_mutates_nothing(wf_client, db):
    r = wf_client.post(f"/api/workflow-deployments/{DEP_B}/activate", json={})
    assert r.status_code == 404
    assert r.json() == {"detail": "Deployment not found"}
    # Not even the victim's endpoint_slug or graph leaks back.
    assert "ep-b" not in r.text and VICTIM_GRAPH_MARKER not in r.text
    assert _dep_row(db, DEP_B)["status"] == "candidate"
    assert db.mutating_writes() == []


def test_activate_foreign_and_unknown_are_byte_identical(wf_client):
    foreign = wf_client.post(f"/api/workflow-deployments/{DEP_B}/activate", json={})
    unknown = wf_client.post(f"/api/workflow-deployments/{UNKNOWN}/activate", json={})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_delete_own_deployment_succeeds(wf_client, db):
    r = wf_client.delete(f"/api/workflow-deployments/{DEP_A}")
    assert r.status_code == 200, r.text
    assert all(x["id"] != DEP_A for x in db.rows["workflow_deployments"])


def test_delete_foreign_deployment_is_opaque_404_and_deletes_nothing(wf_client, db):
    r = wf_client.delete(f"/api/workflow-deployments/{DEP_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Deployment not found"}
    assert any(x["id"] == DEP_B for x in db.rows["workflow_deployments"])
    assert db.mutating_writes() == []


def test_delete_foreign_and_unknown_are_byte_identical(wf_client):
    foreign = wf_client.delete(f"/api/workflow-deployments/{DEP_B}")
    unknown = wf_client.delete(f"/api/workflow-deployments/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: CONTEXT ASSET  (deployment snapshot)
# ═══════════════════════════════════════════════════════════════════════════


def _deploy_body(asset_ids):
    return {
        "workflow_id": WF_A,
        "org_id": ORG_A,
        "endpoint_slug": "ep-a",  # slug is permanent after first deploy
        "graph_json": {
            "nodes": [
                {
                    "id": "n1",
                    "data": {
                        "contextConfig": {
                            "enabled": True,
                            "sources": [
                                {"type": "knowledge_asset", "assetId": a} for a in asset_ids
                            ],
                        }
                    },
                }
            ],
            "edges": [],
        },
    }


def test_snapshot_copies_only_the_callers_own_assets(wf_client, db):
    r = wf_client.post("/api/workflow-deployments", json=_deploy_body([ASSET_A, ASSET_B]))
    assert r.status_code == 200, r.text
    snapped = {s["asset_id"] for s in db.rows["context_asset_snapshots"]}
    assert snapped == {ASSET_A}
    contents = " ".join(s["content"] for s in db.rows["context_asset_snapshots"])
    assert VICTIM_ASSET_MARKER not in contents


def test_snapshot_of_a_purely_foreign_asset_list_copies_nothing(wf_client, db):
    r = wf_client.post("/api/workflow-deployments", json=_deploy_body([ASSET_B]))
    assert r.status_code == 200, r.text
    assert db.rows["context_asset_snapshots"] == []


def test_deploying_a_foreign_workflow_is_refused(wf_client, db):
    body = _deploy_body([])
    body["workflow_id"] = WF_B
    body["endpoint_slug"] = "brand-new"  # a first deploy, so no slug conflict fires first
    r = wf_client.post("/api/workflow-deployments", json=body)
    assert r.status_code == 404
    assert r.json() == {"detail": "Workflow not found"}


def test_deployment_is_created_in_the_verified_org_not_the_body_org(wf_client, db):
    """Every query in the deploy handler, and the row it writes, use the guard's org."""
    body = _deploy_body([])
    body["org_id"] = ORG_B  # the body names the victim's org
    r = wf_client.post("/api/workflow-deployments", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["org_id"] == ORG_A
    # Version continuity also proves the "existing deployments" query ran
    # against ORG_A: DEP_A is v1, so this is v2, not a fresh v1.
    assert r.json()["version"] == 2
    assert not any(d["org_id"] == ORG_B and d["version"] == 2
                   for d in db.rows["workflow_deployments"])


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: PROVIDER CREDENTIAL  (POST /store-key, POST /test-prompt)
# ═══════════════════════════════════════════════════════════════════════════


def test_store_key_writes_into_the_verified_org_only(main_client, db):
    r = main_client.post(
        "/store-key",
        json={"org_id": ORG_A, "provider": "openai", "api_key": "sk-mine"},
    )
    assert r.status_code == 200, r.text
    mine = [k for k in db.rows["api_keys"] if k["org_id"] == ORG_A]
    assert len(mine) == 1


def test_store_key_cannot_overwrite_another_tenants_credential(main_client, db):
    """The body names the victim's org. The victim's row must be untouched."""
    r = main_client.post(
        "/store-key",
        json={"org_id": ORG_B, "provider": "openai", "api_key": "sk-attacker"},
    )
    assert r.status_code == 200, r.text
    victim = next(k for k in db.rows["api_keys"] if k["org_id"] == ORG_B)
    assert victim["api_key"] == VICTIM_KEY_MARKER
    assert all(w[1] != "api_keys" or w[0] == "insert" for w in db.mutating_writes())
    # ...and the new row landed in the caller's own org.
    assert any(k["org_id"] == ORG_A for k in db.rows["api_keys"])


def test_test_prompt_cannot_use_another_tenants_credential(main_client, db):
    """ORG_A holds no openai key; ORG_B does. Naming ORG_B must not reach it."""
    with patch("openai.OpenAI") as openai_mock:
        r = main_client.post(
            "/test-prompt",
            json={
                "user_id": USER_A, "provider": "openai", "model": "gpt-4o",
                "prompt": "hi", "prompt_id": "p1", "org_id": ORG_B,
            },
        )
    assert r.status_code == 404
    assert r.json() == {"detail": "API key not found for org/provider."}
    assert VICTIM_KEY_MARKER not in r.text
    openai_mock.assert_not_called()


def test_test_prompt_foreign_and_absent_credential_are_byte_identical(main_client, db):
    body = {"user_id": USER_A, "provider": "openai", "model": "gpt-4o",
            "prompt": "hi", "prompt_id": "p1"}
    foreign = main_client.post("/test-prompt", json=dict(body, org_id=ORG_B))
    absent = main_client.post("/test-prompt", json=dict(body, org_id=ORG_A))
    assert foreign.status_code == absent.status_code
    assert foreign.content == absent.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: SERVICE API KEY  (POST /api/service-api-keys)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def key_client(db):
    app = FastAPI()
    app.include_router(api_key_management.router, prefix="/api")
    app.dependency_overrides[api_key_management.require_org_member] = _member_of_org_a
    with patch.object(api_key_management, "supabase", db), \
         patch.object(api_key_management, "check_server_key_limit", MagicMock()), \
         patch.object(api_key_management, "encrypt_api_key", lambda v: "enc:" + v):
        yield TestClient(app)
    app.dependency_overrides.clear()


def test_service_key_is_minted_into_the_verified_org_not_the_body_org(key_client, db):
    r = key_client.post("/api/service-api-keys", json={"api_key": "sk_live_x", "org_id": ORG_B})
    assert r.status_code == 200, r.text
    assert r.json()["org_id"] == ORG_A
    assert [k["org_id"] for k in db.rows["service_api_keys"]] == [ORG_A]
    assert not any(k["org_id"] == ORG_B for k in db.rows["service_api_keys"])
