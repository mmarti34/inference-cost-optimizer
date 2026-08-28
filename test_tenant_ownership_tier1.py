"""
Resource-ownership (tenant isolation) on the Tier-1 endpoints.

MECHANISM UNDER TEST
────────────────────
Identical to `test_tenant_ownership.py`, one tier down in blast radius:
`supabase_client` uses the SERVICE-ROLE key, so RLS never applies and the query
filters ARE the authorization. Every endpoint below took a caller-supplied row
id straight to `.eq("id", …)` with no org constraint, and several then read
`org_id` back OUT of the fetched row and executed as that org.

The one that is NOT lower blast radius is `POST /api/eval/run`. Its
`deployment_id` is caller-supplied, its org was derived from the fetched
deployment row, and it was never compared to the verified org. Naming another
tenant's deployment id made `_run_eval_sync` execute THEIR graph (decrypting
their `org_secrets` and spending their provider keys), flip their deployment to
`promoted` with a fresh `promoted_at`, and cancel their running experiments.
That is a cross-tenant secret-use and live-traffic primitive — Tier-0 grade —
and `test_eval_run_on_a_foreign_deployment_*` below are its regression tests.

HOW THESE TESTS ARE BUILT
─────────────────────────
They reuse `_FakeDB` from `test_tenant_ownership` verbatim — the two-tenant
in-memory Postgres double that ACTUALLY APPLIES the filters a query declares.
That is the whole point: it is not a recorder that can be satisfied by a mock
returning whatever the handler asked for. Delete a `.eq("org_id", …)` from the
source and the fake cheerfully hands back the other tenant's row, so the test
fails. Each assertion below was mutation-tested that way.

Per resource class the shape is:

    own resource      -> expected success
    foreign resource  -> opaque failure
    unknown resource  -> BYTE-IDENTICAL failure   (no existence oracle)
    foreign mutation  -> ZERO mutation            (asserted against the DB)
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import resource_access
import workflow_management

# Reuse the harness rather than re-deriving it. `_FakeDB` really filters.
from test_tenant_ownership import (  # noqa: E402
    ORG_A,
    ORG_B,
    UNKNOWN,
    USER_A,
    WF_A,
    WF_B,
    _FakeDB,
    _member_of_org_a,
)

DEP_A = "dddddddd-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DEP_B = "dddddddd-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

GI_A = "9111aaaa-0000-0000-0000-000000000001"
GI_B = "9111bbbb-0000-0000-0000-000000000002"
EXP_A = "9222aaaa-0000-0000-0000-000000000001"
EXP_B = "9222bbbb-0000-0000-0000-000000000002"
RUN_A = "9333aaaa-0000-0000-0000-000000000001"
RUN_B = "9333bbbb-0000-0000-0000-000000000002"
POL_A = "9444aaaa-0000-0000-0000-000000000001"
POL_B = "9444bbbb-0000-0000-0000-000000000002"
CM_A = "9555aaaa-0000-0000-0000-000000000001"
CM_B = "9555bbbb-0000-0000-0000-000000000002"
AGM_A = "9666aaaa-0000-0000-0000-000000000001"
AGM_B = "9666bbbb-0000-0000-0000-000000000002"

# Markers that must never appear in a response to the attacker.
VICTIM_GOLDEN_MARKER = "TENANT-B-GOLDEN-INPUT-CUSTOMER-PII"
VICTIM_EXPERIMENT_MARKER = "TENANT-B-UNRELEASED-EXPERIMENT-NAME"
VICTIM_EVAL_MARKER = "TENANT-B-EVAL-CANDIDATE-OUTPUT"
VICTIM_METRIC_MARKER = "TENANT-B-PROPRIETARY-RUBRIC"
VICTIM_GRAPH_MARKER = "TENANT-B-PRODUCTION-GRAPH"

_VARIANTS = [
    {"name": "v1", "version": 1, "weight": 50},
    {"name": "v2", "version": 2, "weight": 50},
]


def _experiment(eid, org, slug, name, status="running"):
    return {
        "id": eid, "org_id": org, "endpoint_slug": slug, "name": name,
        "description": None, "status": status, "variants": _VARIANTS,
        "primary_metric": "error_rate", "max_error_rate": 10.0, "min_sample_size": 50,
        "confidence_level": 95.0, "mde": 10.0, "power": 80.0,
        "sequential_testing": True, "auto_conclude": False, "results": {},
        "winner_version": None, "concluded_at": None, "concluded_reason": None,
        "created_at": "2026-01-01T00:00:00Z", "updated_at": None,
    }


def _seed_tier1():
    return _FakeDB(
        {
            "organizations": [{"id": ORG_A, "plan": "enterprise"},
                              {"id": ORG_B, "plan": "enterprise"}],
            "workflows": [
                {"id": WF_A, "org_id": ORG_A, "name": "A", "slug": "a"},
                {"id": WF_B, "org_id": ORG_B, "name": "B", "slug": "b"},
            ],
            "workflow_deployments": [
                {"id": DEP_A, "workflow_id": WF_A, "org_id": ORG_A, "version": 1,
                 "endpoint_slug": "ep-a", "status": "candidate",
                 "graph_json": {"nodes": [], "edges": []},
                 "created_at": "2026-01-01T00:00:00Z"},
                {"id": DEP_B, "workflow_id": WF_B, "org_id": ORG_B, "version": 7,
                 "endpoint_slug": "ep-b", "status": "candidate",
                 "graph_json": {"nodes": [{"id": VICTIM_GRAPH_MARKER}], "edges": []},
                 "created_at": "2026-01-01T00:00:00Z"},
            ],
            "golden_inputs": [
                {"id": GI_A, "org_id": ORG_A, "workflow_id": WF_A, "name": "mine",
                 "input_text": "hello", "variables": {}, "expected_output": "hi",
                 "source": "manual", "source_run_id": None,
                 "created_at": "2026-01-01T00:00:00Z", "updated_at": None},
                {"id": GI_B, "org_id": ORG_B, "workflow_id": WF_B,
                 "name": VICTIM_GOLDEN_MARKER, "input_text": VICTIM_GOLDEN_MARKER,
                 "variables": {}, "expected_output": VICTIM_GOLDEN_MARKER,
                 "source": "manual", "source_run_id": None,
                 "created_at": "2026-01-01T00:00:00Z", "updated_at": None},
            ],
            "experiments": [
                _experiment(EXP_A, ORG_A, "ep-a", "mine"),
                _experiment(EXP_B, ORG_B, "ep-b", VICTIM_EXPERIMENT_MARKER),
            ],
            "eval_runs": [
                {"id": RUN_A, "org_id": ORG_A, "deployment_id": DEP_A,
                 "eval_suite_id": None, "status": "passed", "results": [],
                 "summary": {"total_checks": 1}, "started_at": None,
                 "completed_at": None, "created_at": "2026-01-01T00:00:00Z"},
                {"id": RUN_B, "org_id": ORG_B, "deployment_id": DEP_B,
                 "eval_suite_id": None, "status": "passed", "results": [],
                 "summary": {"leak": VICTIM_EVAL_MARKER}, "started_at": None,
                 "completed_at": None, "created_at": "2026-01-01T00:00:00Z"},
            ],
            "eval_run_results": [],
            "eval_suites": [],
            "routing_policies": [
                {"id": POL_A, "org_id": ORG_A, "endpoint_slug": "ep-a",
                 "policy_type": "weighted", "rules": [], "active": True,
                 "created_at": "2026-01-01T00:00:00Z"},
                {"id": POL_B, "org_id": ORG_B, "endpoint_slug": "ep-b",
                 "policy_type": "weighted", "rules": [], "active": True,
                 "created_at": "2026-01-01T00:00:00Z"},
            ],
            "custom_metrics": [
                {"id": CM_A, "org_id": ORG_A, "name": "mine", "created_at": "2026-01-01T00:00:00Z"},
                {"id": CM_B, "org_id": ORG_B, "name": VICTIM_METRIC_MARKER,
                 "created_at": "2026-01-01T00:00:00Z"},
            ],
            "auto_graded_metrics": [
                {"id": AGM_A, "org_id": ORG_A, "name": "mine", "rubric": "r",
                 "grading_model": "gpt-4o-mini", "enabled": True, "sample_rate": 100.0,
                 "created_at": "2026-01-01T00:00:00Z"},
                {"id": AGM_B, "org_id": ORG_B, "name": VICTIM_METRIC_MARKER,
                 "rubric": VICTIM_METRIC_MARKER, "grading_model": "gpt-4o-mini",
                 "enabled": True, "sample_rate": 100.0,
                 "created_at": "2026-01-01T00:00:00Z"},
            ],
            "workflow_runs": [],
            "api_request_log": [],
            "auto_grade_results": [],
        }
    )


@pytest.fixture
def db():
    return _seed_tier1()


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(workflow_management.router, prefix="/api")
    app.dependency_overrides[workflow_management.require_org_member] = _member_of_org_a
    with patch.object(workflow_management, "supabase", db), \
         patch.object(resource_access, "supabase", db), \
         patch.object(workflow_management, "_run_eval_sync") as eval_mock, \
         patch.object(workflow_management, "_org_has_provider_key", lambda o, p: True), \
         patch.object(workflow_management, "get_latest_promoted_deployment",
                      AsyncMock(return_value=None)), \
         patch.object(workflow_management, "get_promoted_deployment_by_version",
                      AsyncMock(return_value=None)):
        c = TestClient(app)
        c.eval_mock = eval_mock
        yield c
    app.dependency_overrides.clear()


def _row(db, table, rid):
    return next((r for r in db.rows[table] if r["id"] == rid), None)


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: GOLDEN INPUT  (PUT/DELETE /api/golden-inputs/{id})
#
# The eval corpus. An unscoped DELETE here erases another tenant's entire
# regression suite; an unscoped PUT rewrites the expected outputs their
# deployments are gated on.
# ═══════════════════════════════════════════════════════════════════════════


def test_golden_input_update_own_succeeds(client, db):
    r = client.put(f"/api/golden-inputs/{GI_A}", json={"name": "renamed"})
    assert r.status_code == 200, r.text
    assert _row(db, "golden_inputs", GI_A)["name"] == "renamed"


def test_golden_input_update_foreign_is_opaque_404(client):
    r = client.put(f"/api/golden-inputs/{GI_B}", json={"name": "pwned"})
    assert r.status_code == 404
    assert r.json() == {"detail": "Golden input not found"}
    assert VICTIM_GOLDEN_MARKER not in r.text


def test_golden_input_no_op_update_does_not_disclose_a_foreign_row(client):
    """The empty-payload branch RETURNS the row, so only the ownership fetch covers it."""
    r = client.put(f"/api/golden-inputs/{GI_B}", json={})
    assert r.status_code == 404
    assert VICTIM_GOLDEN_MARKER not in r.text


def test_golden_input_update_foreign_and_unknown_are_byte_identical(client):
    foreign = client.put(f"/api/golden-inputs/{GI_B}", json={"name": "x"})
    unknown = client.put(f"/api/golden-inputs/{UNKNOWN}", json={"name": "x"})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_golden_input_update_foreign_performs_zero_mutation(client, db):
    before = dict(_row(db, "golden_inputs", GI_B))
    client.put(f"/api/golden-inputs/{GI_B}", json={"name": "pwned"})
    assert _row(db, "golden_inputs", GI_B) == before
    assert db.mutating_writes() == []


def test_golden_input_delete_own_succeeds(client, db):
    r = client.delete(f"/api/golden-inputs/{GI_A}")
    assert r.status_code == 204, r.text
    assert _row(db, "golden_inputs", GI_A) is None


def test_golden_input_delete_foreign_is_opaque_404_and_deletes_nothing(client, db):
    r = client.delete(f"/api/golden-inputs/{GI_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Golden input not found"}
    assert _row(db, "golden_inputs", GI_B) is not None
    assert db.mutating_writes() == []


def test_golden_input_delete_foreign_and_unknown_are_byte_identical(client):
    foreign = client.delete(f"/api/golden-inputs/{GI_B}")
    unknown = client.delete(f"/api/golden-inputs/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: EXPERIMENT  (read / timeseries / start / conclude / cancel)
#
# The highest-impact class in this file. `start` INSERTS an active weighted
# routing policy and `conclude`/`cancel` DEACTIVATE one — all three move live
# production traffic on the owning org's endpoint.
# ═══════════════════════════════════════════════════════════════════════════


def test_experiment_read_own_succeeds(client):
    r = client.get(f"/api/experiments/{EXP_A}")
    assert r.status_code == 200, r.text
    assert r.json()["org_id"] == ORG_A


def test_experiment_read_foreign_is_opaque_404(client):
    r = client.get(f"/api/experiments/{EXP_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Experiment not found"}
    assert VICTIM_EXPERIMENT_MARKER not in r.text


def test_experiment_read_foreign_and_unknown_are_byte_identical(client):
    foreign = client.get(f"/api/experiments/{EXP_B}")
    unknown = client.get(f"/api/experiments/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_experiment_timeseries_foreign_is_opaque_404(client):
    r = client.get(f"/api/experiments/{EXP_B}/timeseries")
    assert r.status_code == 404
    assert r.json() == {"detail": "Experiment not found"}


def test_experiment_timeseries_foreign_and_unknown_are_byte_identical(client):
    foreign = client.get(f"/api/experiments/{EXP_B}/timeseries")
    unknown = client.get(f"/api/experiments/{UNKNOWN}/timeseries")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_experiment_start_own_succeeds(client, db):
    db.rows["experiments"][0]["status"] = "draft"
    r = client.put(f"/api/experiments/{EXP_A}/start")
    assert r.status_code == 200, r.text
    assert _row(db, "experiments", EXP_A)["status"] == "running"
    # The new weighted policy landed in the caller's own org.
    assert all(p["org_id"] == ORG_A
               for p in db.rows["routing_policies"] if p.get("policy_type") == "weighted"
               and p["id"] not in (POL_A, POL_B))


def test_experiment_start_foreign_is_opaque_404_and_mutates_nothing(client, db):
    db.rows["experiments"][1]["status"] = "draft"
    before_policies = [dict(p) for p in db.rows["routing_policies"]]
    r = client.put(f"/api/experiments/{EXP_B}/start")
    assert r.status_code == 404
    assert r.json() == {"detail": "Experiment not found"}
    assert _row(db, "experiments", EXP_B)["status"] == "draft"
    assert db.rows["routing_policies"] == before_policies
    assert db.mutating_writes() == []


def test_experiment_conclude_own_succeeds(client, db):
    r = client.put(f"/api/experiments/{EXP_A}/conclude", json={"winner_version": 2})
    assert r.status_code == 200, r.text
    assert _row(db, "experiments", EXP_A)["status"] == "concluded"
    assert _row(db, "routing_policies", POL_A)["active"] is False


def test_experiment_conclude_foreign_is_opaque_404_and_mutates_nothing(client, db):
    """Conclude retargets live traffic. A foreign id must not move a single row."""
    r = client.put(f"/api/experiments/{EXP_B}/conclude", json={"winner_version": 2})
    assert r.status_code == 404
    assert r.json() == {"detail": "Experiment not found"}
    assert _row(db, "experiments", EXP_B)["status"] == "running"
    assert _row(db, "routing_policies", POL_B)["active"] is True
    assert db.mutating_writes() == []


def test_experiment_conclude_foreign_and_unknown_are_byte_identical(client):
    foreign = client.put(f"/api/experiments/{EXP_B}/conclude", json={"winner_version": 2})
    unknown = client.put(f"/api/experiments/{UNKNOWN}/conclude", json={"winner_version": 2})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_experiment_cancel_own_succeeds(client, db):
    r = client.put(f"/api/experiments/{EXP_A}/cancel")
    assert r.status_code == 200, r.text
    assert _row(db, "experiments", EXP_A)["status"] == "cancelled"
    assert _row(db, "routing_policies", POL_A)["active"] is False


def test_experiment_cancel_foreign_is_opaque_404_and_mutates_nothing(client, db):
    r = client.put(f"/api/experiments/{EXP_B}/cancel")
    assert r.status_code == 404
    assert r.json() == {"detail": "Experiment not found"}
    assert _row(db, "experiments", EXP_B)["status"] == "running"
    assert _row(db, "routing_policies", POL_B)["active"] is True
    assert db.mutating_writes() == []


def test_experiment_cancel_foreign_and_unknown_are_byte_identical(client):
    foreign = client.put(f"/api/experiments/{EXP_B}/cancel")
    unknown = client.put(f"/api/experiments/{UNKNOWN}/cancel")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: EVAL RUN  (POST /api/eval/run, GET /api/eval/runs/{id})
#
# POST /api/eval/run is the Tier-0-grade one. Its org came out of the fetched
# deployment row and was never compared to the verified org, so a foreign
# `deployment_id` dispatched `_run_eval_sync` against the victim's deployment:
# executing their graph (decrypting their `org_secrets`, spending their
# provider keys), setting their deployment to `promoted`, and cancelling their
# running experiments.
# ═══════════════════════════════════════════════════════════════════════════


def test_eval_run_on_own_deployment_dispatches(client, db):
    r = client.post("/api/eval/run", json={"deployment_id": DEP_A})
    assert r.status_code == 200, r.text
    client.eval_mock.assert_called_once()
    assert [x["org_id"] for x in db.rows["eval_runs"] if x["deployment_id"] == DEP_A
            and x["id"] != RUN_A] == [ORG_A]


def test_eval_run_on_a_foreign_deployment_never_reaches_the_runtime(client):
    """The whole defect in one assertion: no eval dispatch, so no victim secret is used."""
    r = client.post("/api/eval/run", json={"deployment_id": DEP_B})
    assert r.status_code == 404
    assert r.json() == {"detail": "Deployment not found"}
    client.eval_mock.assert_not_called()
    assert VICTIM_GRAPH_MARKER not in r.text


def test_eval_run_on_a_foreign_deployment_writes_nothing(client, db):
    """No eval_runs row is minted into the victim's org, and their deployment stands."""
    before = [dict(r) for r in db.rows["eval_runs"]]
    client.post("/api/eval/run", json={"deployment_id": DEP_B})
    assert db.rows["eval_runs"] == before
    assert _row(db, "workflow_deployments", DEP_B)["status"] == "candidate"
    assert db.mutating_writes() == []


def test_eval_run_foreign_and_unknown_deployment_are_byte_identical(client):
    foreign = client.post("/api/eval/run", json={"deployment_id": DEP_B})
    unknown = client.post("/api/eval/run", json={"deployment_id": UNKNOWN})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_eval_run_read_own_succeeds(client):
    r = client.get(f"/api/eval/runs/{RUN_A}")
    assert r.status_code == 200, r.text
    assert r.json()["id"] == RUN_A


def test_eval_run_read_foreign_is_opaque_404(client):
    r = client.get(f"/api/eval/runs/{RUN_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Eval run not found"}
    assert VICTIM_EVAL_MARKER not in r.text


def test_eval_run_read_foreign_and_unknown_are_byte_identical(client):
    foreign = client.get(f"/api/eval/runs/{RUN_B}")
    unknown = client.get(f"/api/eval/runs/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: ROUTING POLICY  (DELETE /api/routing-policies/{id})
#
# Deactivating a policy REDIRECTS LIVE TRAFFIC: the endpoint falls back off its
# weighted split onto latest-promoted.
# ═══════════════════════════════════════════════════════════════════════════


def test_routing_policy_delete_own_succeeds(client, db):
    r = client.delete(f"/api/routing-policies/{POL_A}")
    assert r.status_code == 204, r.text
    assert _row(db, "routing_policies", POL_A)["active"] is False


def test_routing_policy_delete_foreign_is_opaque_404_and_leaves_traffic_alone(client, db):
    r = client.delete(f"/api/routing-policies/{POL_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Routing policy not found"}
    assert _row(db, "routing_policies", POL_B)["active"] is True
    assert db.mutating_writes() == []


def test_routing_policy_delete_foreign_and_unknown_are_byte_identical(client):
    foreign = client.delete(f"/api/routing-policies/{POL_B}")
    unknown = client.delete(f"/api/routing-policies/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: CUSTOM METRIC  (DELETE /api/custom-metrics/{id})
# ═══════════════════════════════════════════════════════════════════════════


def test_custom_metric_delete_own_succeeds(client, db):
    r = client.delete(f"/api/custom-metrics/{CM_A}")
    assert r.status_code == 200, r.text
    assert _row(db, "custom_metrics", CM_A) is None


def test_custom_metric_delete_foreign_is_opaque_404_and_deletes_nothing(client, db):
    r = client.delete(f"/api/custom-metrics/{CM_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Custom metric not found"}
    assert _row(db, "custom_metrics", CM_B) is not None
    assert db.mutating_writes() == []


def test_custom_metric_delete_foreign_and_unknown_are_byte_identical(client):
    foreign = client.delete(f"/api/custom-metrics/{CM_B}")
    unknown = client.delete(f"/api/custom-metrics/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: AUTO-GRADED METRIC  (PUT/DELETE /api/auto-graded-metrics/{id})
# ═══════════════════════════════════════════════════════════════════════════


def test_auto_graded_metric_update_own_succeeds(client, db):
    r = client.put(f"/api/auto-graded-metrics/{AGM_A}", json={"name": "renamed"})
    assert r.status_code == 200, r.text
    assert _row(db, "auto_graded_metrics", AGM_A)["name"] == "renamed"


def test_auto_graded_metric_update_foreign_is_opaque_404(client):
    r = client.put(f"/api/auto-graded-metrics/{AGM_B}", json={"name": "pwned"})
    assert r.status_code == 404
    assert r.json() == {"detail": "Auto-graded metric not found"}
    assert VICTIM_METRIC_MARKER not in r.text


def test_auto_graded_metric_update_foreign_performs_zero_mutation(client, db):
    before = dict(_row(db, "auto_graded_metrics", AGM_B))
    client.put(f"/api/auto-graded-metrics/{AGM_B}", json={"name": "pwned"})
    assert _row(db, "auto_graded_metrics", AGM_B) == before
    assert db.mutating_writes() == []


def test_auto_graded_metric_update_foreign_and_unknown_are_byte_identical(client):
    foreign = client.put(f"/api/auto-graded-metrics/{AGM_B}", json={"name": "x"})
    unknown = client.put(f"/api/auto-graded-metrics/{UNKNOWN}", json={"name": "x"})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_auto_graded_metric_grading_model_probe_never_targets_a_foreign_org(client, db):
    """
    The `grading_model` branch probed `_org_has_provider_key` with an org read
    OUT of the fetched row, turning the 400/200 split into an oracle for
    "does tenant B have a provider key configured?". It must probe the caller's
    own org, and a foreign id must not get that far at all.
    """
    seen = []
    with patch.object(workflow_management, "_org_has_provider_key",
                      lambda o, p: seen.append(o) or True):
        own = client.put(f"/api/auto-graded-metrics/{AGM_A}",
                         json={"grading_model": "gpt-4o"})
        foreign = client.put(f"/api/auto-graded-metrics/{AGM_B}",
                             json={"grading_model": "gpt-4o"})
    assert own.status_code == 200, own.text
    assert foreign.status_code == 404
    assert seen == [ORG_A]          # probed once, for the caller's own org
    assert ORG_B not in seen


def test_auto_graded_metric_delete_own_succeeds(client, db):
    r = client.delete(f"/api/auto-graded-metrics/{AGM_A}")
    assert r.status_code == 200, r.text
    assert _row(db, "auto_graded_metrics", AGM_A) is None


def test_auto_graded_metric_delete_foreign_is_opaque_404_and_deletes_nothing(client, db):
    r = client.delete(f"/api/auto-graded-metrics/{AGM_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Auto-graded metric not found"}
    assert _row(db, "auto_graded_metrics", AGM_B) is not None
    assert db.mutating_writes() == []


def test_auto_graded_metric_delete_foreign_and_unknown_are_byte_identical(client):
    foreign = client.delete(f"/api/auto-graded-metrics/{AGM_B}")
    unknown = client.delete(f"/api/auto-graded-metrics/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# The invariant itself: no helper has a parameter an org id can be passed into.
# ═══════════════════════════════════════════════════════════════════════════


def test_fetch_owned_row_refuses_a_caller_supplied_org_id():
    """`fetch_owned_row` is now called directly from this router. It must keep
    rejecting anything that is not a guard-produced identity, or a future edit
    could pass `payload.org_id` in and reintroduce a caller-controlled filter
    that merely LOOKS scoped."""
    with pytest.raises(TypeError):
        resource_access.fetch_owned_row("experiments", EXP_B, ORG_B, "id, org_id", "nope")
