"""
Tenant isolation and the audit trail on the last two unremediated files.

WHAT IS UNDER TEST
──────────────────
Two things that only look separate, and which live in the same two modules:

1. RESOURCE OWNERSHIP, same mechanism as `test_tenant_ownership.py` and
   `test_tenant_ownership_tier1.py`. `supabase_client` uses the SERVICE-ROLE
   key, so RLS never applies and the query filters ARE the authorization.
   `require_org_member` proves the caller belongs to the org they named and
   proves nothing about the other ids in the same request. What remained here:

     * `rollback_rules` — GET detail, PUT and DELETE were all `.eq("id", …)`
       alone, with the org read back out of the fetched row by
       `_rollback_rule_to_response`. `enabled: false` on another tenant's rule
       silently disarms the automatic rollback protecting their live endpoint.
     * `workflow_runs` reached by `experiment_id` with no org constraint. The
       table DOES carry `org_id`, so this was scopeable and is now scoped.
     * `POST /rollback-monitor/run` — an unscoped fan-out rather than a
       Mechanism-B read: `org_id` is an Optional QUERY parameter and None means
       EVERY organization, so satisfying the guard with `X-Org-Id` alone let one
       member of one org force a rollback sweep across every tenant.

2. THE AUDIT TRAIL on the ten call sites in these two files that can materially
   affect another user's production environment — provider credentials, server
   keys, and every path a deployment takes into or out of production.

WHY THE TESTS FOR BOTH ARE IN ONE FILE
──────────────────────────────────────
The audit assertions need a database double that ACTUALLY APPLIES the filters a
query declares, for the same reason the ownership assertions do: several audit
rows are written precisely BECAUSE an org-scoped statement matched nothing, and
`activate_deployment` records one `rolled_back` row per version its demote
statement returns — which a recorder that ignores `.neq()` would get wrong by
exactly one row. `test_tenant_ownership._FakeDB` applies `.eq`, `.neq` and
`.in_`; the double in `test_audit_log.py` does not implement `.neq`. So the
harness is imported from there and both kinds of assertion are made against it.

Per resource class the shape is the established one:

    own resource      -> success
    foreign resource  -> opaque failure
    unknown resource  -> BYTE-IDENTICAL failure   (no existence oracle)
    foreign mutation  -> ZERO side effects        (victim's rows compared)

And per audit site:

    exactly one row, with the right action, org, actor, resource and outcome
    refusals recorded, and filed under the VICTIM's org where the resource
        belongs to another tenant
    no secret, credential or prompt anywhere in the SERIALISED row
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import api_key_management
import audit
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
DEP_A2 = "dddddddd-aaaa-aaaa-aaaa-aaaaaaaaaaab"
DEP_B = "dddddddd-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

RR_A = "7111aaaa-0000-0000-0000-000000000001"
RR_B = "7111bbbb-0000-0000-0000-000000000002"
EXP_A = "7222aaaa-0000-0000-0000-000000000001"
RUN_A = "7333aaaa-0000-0000-0000-000000000001"
RUN_B = "7333bbbb-0000-0000-0000-000000000002"
PKEY_A = "7444aaaa-0000-0000-0000-000000000001"
PKEY_B = "7444bbbb-0000-0000-0000-000000000002"
SKEY_A = "7555aaaa-0000-0000-0000-000000000001"
SKEY_B = "7555bbbb-0000-0000-0000-000000000002"

# Markers that must never reach the attacker, and must never reach an audit row.
VICTIM_RULE_MARKER = "tenant-b-production-endpoint"
VICTIM_RUN_MARKER = "TENANT-B-PRODUCTION-RUN-OUTPUT"
REAL_KEY = "sk-proj-Ab3dEfGh1JkLmN0pQrStUvWxYz0123456789AbCdEfGhIjKl"
REAL_PROMPT = "Summarise the attached patient intake note and list red flags."

_VARIANTS = [
    {"name": "v1", "version": 1, "weight": 50},
    {"name": "v2", "version": 2, "weight": 50},
]


def _rule(rid, org, slug, enabled=True):
    return {
        "id": rid, "org_id": org, "endpoint_slug": slug, "enabled": enabled,
        "conditions": [{"metric": "error_rate", "operator": "gt", "threshold": 5.0,
                        "window_minutes": 60}],
        "action": "rollback", "last_triggered_at": None, "last_checked_at": None,
        "created_at": "2026-01-01T00:00:00Z", "updated_at": None,
    }


def _run(rid, org, experiment_id, variant, latency, cost):
    return {
        "id": rid, "org_id": org, "workflow_id": WF_A if org == ORG_A else WF_B,
        "experiment_id": experiment_id, "variant_name": variant,
        "total_latency_ms": latency, "total_cost": cost, "node_results": [],
        "endpoint_slug": "ep-a", "execution_mode": "production",
        "final_output": None if org == ORG_A else VICTIM_RUN_MARKER,
        "created_at": "2026-06-01T00:00:00Z",
    }


def _deployment(did, org, version, slug, status):
    return {
        "id": did, "workflow_id": WF_A if org == ORG_A else WF_B, "org_id": org,
        "project_id": None, "version": version, "endpoint_slug": slug,
        "graph_json": {"nodes": [], "edges": []}, "status": status,
        "promoted_at": None, "rolled_back_from_version": None,
        "created_at": "2026-01-01T00:00:00Z",
    }


def _seed_final():
    return _FakeDB(
        {
            "organizations": [{"id": ORG_A, "plan": "enterprise"},
                              {"id": ORG_B, "plan": "enterprise"}],
            "workflows": [
                {"id": WF_A, "org_id": ORG_A, "name": "A", "slug": "a"},
                {"id": WF_B, "org_id": ORG_B, "name": "B", "slug": "b"},
            ],
            "workflow_deployments": [
                _deployment(DEP_A, ORG_A, 1, "ep-a", "promoted"),
                _deployment(DEP_A2, ORG_A, 2, "ep-a", "candidate"),
                _deployment(DEP_B, ORG_B, 7, "ep-b", "promoted"),
            ],
            "rollback_rules": [
                _rule(RR_A, ORG_A, "ep-a"),
                _rule(RR_B, ORG_B, VICTIM_RULE_MARKER),
            ],
            "experiments": [{
                "id": EXP_A, "org_id": ORG_A, "endpoint_slug": "ep-a", "name": "mine",
                "description": None, "status": "running", "variants": _VARIANTS,
                "primary_metric": "error_rate", "max_error_rate": 10.0,
                "min_sample_size": 1, "confidence_level": 95.0, "mde": 10.0,
                "power": 80.0, "sequential_testing": True, "auto_conclude": False,
                "results": {}, "winner_version": None, "concluded_at": None,
                "concluded_reason": None, "created_at": "2026-01-01T00:00:00Z",
                "updated_at": None,
            }],
            # THE POINT OF THIS ROW: it carries the CALLER's experiment_id but
            # the VICTIM's org. Only an org filter on `workflow_runs` keeps it
            # out of ORG_A's aggregation.
            "workflow_runs": [
                _run(RUN_A, ORG_A, EXP_A, "v1", 100, 0.001),
                _run(RUN_B, ORG_B, EXP_A, "v2", 9000, 9.0),
            ],
            "routing_policies": [],
            "custom_metrics": [],
            "auto_graded_metrics": [],
            "api_request_log": [],
            "auto_grade_results": [],
            "api_keys": [
                {"id": PKEY_A, "org_id": ORG_A, "provider": "openai",
                 "api_key": "enc:a", "created_at": "2026-01-01T00:00:00Z"},
                {"id": PKEY_B, "org_id": ORG_B, "provider": "anthropic",
                 "api_key": "enc:" + REAL_KEY, "created_at": "2026-01-01T00:00:00Z"},
            ],
            "service_api_keys": [
                {"id": SKEY_A, "org_id": ORG_A, "api_key": "enc:a",
                 "name": "Mine", "status": "active", "rate_limit_per_minute": 60,
                 "created_at": "2026-01-01T00:00:00Z"},
                {"id": SKEY_B, "org_id": ORG_B, "api_key": "enc:" + REAL_KEY,
                 "name": "Production", "status": "active",
                 "rate_limit_per_minute": 60, "created_at": "2026-01-01T00:00:00Z"},
            ],
            audit.AUDIT_TABLE: [],
        }
    )


@pytest.fixture
def db():
    return _seed_final()


# ─── Audit assertions over the fake ─────────────────────────────────────────


def _audit_rows(db, action=None):
    rows = db.rows.setdefault(audit.AUDIT_TABLE, [])
    if action is None:
        return list(rows)
    return [r for r in rows if r["action"] == action]


def _only(db, action):
    """The single row for `action`. Fails loudly on zero or on duplicates."""
    rows = _audit_rows(db, action)
    assert len(rows) == 1, (
        "expected exactly one %r row, got %d (all actions: %r)"
        % (action, len(rows), [r["action"] for r in _audit_rows(db)])
    )
    return rows[0]


def _assert_no_content_anywhere(db):
    """Nothing sensitive in ANY audit row, asserted over the whole serialised row.

    Field-by-field assertions miss the case this is guarding: a future call site
    that puts a credential somewhere nobody thought to check.
    """
    blob = json.dumps(_audit_rows(db), default=str)
    for forbidden in (REAL_KEY, REAL_PROMPT, VICTIM_RUN_MARKER, "enc:" + REAL_KEY):
        assert forbidden not in blob, "audit row carried %r" % forbidden[:24]


def _row(db, table, rid):
    return next((r for r in db.rows[table] if r["id"] == rid), None)


# ─── App wiring ─────────────────────────────────────────────────────────────


@pytest.fixture
def wf_client(db):
    """Workflow router on the fake DB, with the audit writer pointed at it too."""
    app = FastAPI()
    app.include_router(workflow_management.router, prefix="/api")
    app.dependency_overrides[workflow_management.require_org_member] = _member_of_org_a
    with patch.object(workflow_management, "supabase", db), \
         patch.object(resource_access, "supabase", db), \
         patch.object(audit, "supabase", db), \
         patch.object(workflow_management, "_run_eval_sync", MagicMock()), \
         patch.object(workflow_management, "_end_experiments_on_endpoint_sync", MagicMock()), \
         patch.object(workflow_management, "_org_has_provider_key", lambda o, p: True), \
         patch.object(workflow_management, "get_latest_promoted_deployment",
                      AsyncMock(return_value=None)), \
         patch.object(workflow_management, "get_promoted_deployment_by_version",
                      AsyncMock(return_value=None)):
        yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def key_client(db):
    app = FastAPI()
    app.include_router(api_key_management.router, prefix="/api")
    app.dependency_overrides[api_key_management.require_org_member] = _member_of_org_a
    with patch.object(api_key_management, "supabase", db), \
         patch.object(audit, "supabase", db), \
         patch.object(api_key_management, "check_server_key_limit", MagicMock()), \
         patch.object(api_key_management, "encrypt_api_key", lambda v: "enc:" + v):
        yield TestClient(app)
    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Resource class: ROLLBACK RULE
#
# A rollback rule is the automatic protection on a live endpoint. Reading one
# discloses the tenant's production topology and their tolerance for failure;
# writing one disarms or weaponises it. All three verbs were id-only.
# ═══════════════════════════════════════════════════════════════════════════


def test_the_rollback_rule_detail_route_is_shadowed_and_never_dispatched(wf_client):
    """
    A FINDING, not a fix. `GET /rollback-rules/{org_id}/{endpoint_slug}` is
    declared BEFORE `GET /rollback-rules/detail/{rule_id}`, and FastAPI
    dispatches to the first matching route — so every request to the detail
    path is served by `list_rollback_rules` with `org_id="detail"`, and
    `get_rollback_rule` never runs. Its id-only read was therefore a LATENT
    Mechanism B, not a live disclosure.

    This test exists so that reordering the routes (which would make it live)
    fails loudly here rather than silently reopening the hole. The handler
    itself is exercised directly below.
    """
    r = wf_client.get(f"/api/rollback-rules/detail/{RR_B}")
    assert r.status_code == 200
    assert r.json() == []                      # list_rollback_rules, org_id="detail"
    assert VICTIM_RULE_MARKER not in r.text


def test_rollback_rule_detail_handler_returns_its_own_rule(db):
    """The shadowed handler, called directly — the route cannot reach it."""
    with patch.object(workflow_management, "supabase", db), \
         patch.object(resource_access, "supabase", db):
        out = asyncio.run(workflow_management.get_rollback_rule(RR_A, _member_of_org_a()))
    assert out["id"] == RR_A
    assert out["org_id"] == ORG_A


def test_rollback_rule_detail_handler_refuses_a_foreign_rule_opaquely(db):
    with patch.object(workflow_management, "supabase", db), \
         patch.object(resource_access, "supabase", db):
        with pytest.raises(HTTPException) as foreign:
            asyncio.run(workflow_management.get_rollback_rule(RR_B, _member_of_org_a()))
        with pytest.raises(HTTPException) as unknown:
            asyncio.run(workflow_management.get_rollback_rule(UNKNOWN, _member_of_org_a()))
    # Foreign and unknown are indistinguishable: no existence oracle.
    assert foreign.value.status_code == unknown.value.status_code == 404
    assert foreign.value.detail == unknown.value.detail
    assert foreign.value.detail == workflow_management.ROLLBACK_RULE_NOT_FOUND


def test_rollback_rule_update_own_succeeds(wf_client, db):
    r = wf_client.put(f"/api/rollback-rules/{RR_A}", json={"enabled": False})
    assert r.status_code == 200, r.text
    assert _row(db, "rollback_rules", RR_A)["enabled"] is False


def test_rollback_rule_update_foreign_is_opaque_404(wf_client):
    r = wf_client.put(f"/api/rollback-rules/{RR_B}", json={"enabled": False})
    assert r.status_code == 404
    assert r.json() == {"detail": workflow_management.ROLLBACK_RULE_NOT_FOUND}
    assert VICTIM_RULE_MARKER not in r.text


def test_rollback_rule_update_foreign_and_unknown_are_byte_identical(wf_client):
    foreign = wf_client.put(f"/api/rollback-rules/{RR_B}", json={"enabled": False})
    unknown = wf_client.put(f"/api/rollback-rules/{UNKNOWN}", json={"enabled": False})
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_disarming_a_foreign_rollback_rule_performs_zero_mutation(wf_client, db):
    """The defect in one assertion: their automatic rollback stays armed."""
    before = dict(_row(db, "rollback_rules", RR_B))
    wf_client.put(f"/api/rollback-rules/{RR_B}", json={"enabled": False})
    assert _row(db, "rollback_rules", RR_B) == before
    assert _row(db, "rollback_rules", RR_B)["enabled"] is True
    assert db.mutating_writes() == []


def test_rollback_rule_delete_own_succeeds(wf_client, db):
    r = wf_client.delete(f"/api/rollback-rules/{RR_A}")
    assert r.status_code == 204, r.text
    assert _row(db, "rollback_rules", RR_A) is None


def test_rollback_rule_delete_foreign_deletes_nothing(wf_client, db):
    before = [dict(x) for x in db.rows["rollback_rules"]]
    r = wf_client.delete(f"/api/rollback-rules/{RR_B}")
    # The endpoint answered 204 before and after: opacity was never the gap
    # here, the missing filter was.
    assert r.status_code == 204
    assert db.rows["rollback_rules"] == before
    assert db.mutating_writes() == []


def test_rollback_rule_delete_foreign_and_unknown_are_byte_identical(wf_client):
    foreign = wf_client.delete(f"/api/rollback-rules/{RR_B}")
    unknown = wf_client.delete(f"/api/rollback-rules/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# workflow_runs, reached by experiment_id
#
# `workflow_runs` carries `org_id`, so a run row is org-scopeable and now is.
# The seeded RUN_B carries the CALLER's experiment_id and the VICTIM's org: it
# is the row that only an org filter keeps out of the aggregate.
# ═══════════════════════════════════════════════════════════════════════════


def test_experiment_results_aggregate_only_the_verified_orgs_runs(wf_client):
    r = wf_client.get(f"/api/experiments/{EXP_A}")
    assert r.status_code == 200, r.text
    variants = r.json()["results"]["variants"]
    # ORG_A contributed one run to v1. ORG_B's run is tagged v2 and must not
    # appear at all — v2 is present only because the variant list demands it.
    assert variants["v1"]["requests"] == 1
    assert variants["v2"]["requests"] == 0
    assert variants["v2"]["avg_cost"] == 0
    assert VICTIM_RUN_MARKER not in r.text


def test_experiment_timeseries_aggregates_only_the_verified_orgs_runs(wf_client):
    r = wf_client.get(f"/api/experiments/{EXP_A}/timeseries")
    assert r.status_code == 200, r.text
    buckets = r.json()["buckets"]
    assert buckets, "expected at least one bucket from the caller's own run"
    assert sum(b["candidate"]["requests"] for b in buckets) == 0
    assert sum(b["control"]["requests"] for b in buckets) == 1
    assert VICTIM_RUN_MARKER not in r.text


# ═══════════════════════════════════════════════════════════════════════════
# POST /rollback-monitor/run — the unscoped fan-out
#
# NOT a Mechanism-B read. `org_id` is an Optional query parameter and None
# means EVERY organization, so absence — which the guard cannot see, because
# absence is not a conflicting value — was a cross-tenant action trigger.
# ═══════════════════════════════════════════════════════════════════════════


def test_rollback_monitor_runs_only_for_the_verified_org(wf_client):
    with patch.object(workflow_management, "run_rollback_monitor_cycle",
                      AsyncMock(return_value={"checked": 0, "triggered": 0, "errors": []})) as cycle:
        r = wf_client.post("/api/rollback-monitor/run")
        assert r.status_code == 200, r.text
    cycle.assert_awaited_once()
    assert cycle.await_args.kwargs["org_id"] == ORG_A


def test_rollback_monitor_cannot_be_widened_to_every_tenant(wf_client):
    """
    The exploit shape: satisfy the guard without ever putting `org_id` in the
    query string. It must still be impossible to reach the all-orgs sweep.
    """
    with patch.object(workflow_management, "run_rollback_monitor_cycle",
                      AsyncMock(return_value={"checked": 0, "triggered": 0, "errors": []})) as cycle:
        wf_client.post("/api/rollback-monitor/run", headers={"X-Org-Id": ORG_A})
    assert cycle.await_args.kwargs["org_id"] is not None
    assert cycle.await_args.kwargs["org_id"] == ORG_A


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT — deployments into and out of production
# ═══════════════════════════════════════════════════════════════════════════


def test_promote_override_writes_exactly_one_promoted_row(wf_client, db):
    r = wf_client.post(f"/api/workflow-deployments/{DEP_A2}/promote",
                       json={"override_reason": "ship it"})
    assert r.status_code == 200, r.text

    row = _only(db, audit.DEPLOYMENT_PROMOTED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, DEP_A2)
    assert row["resource_type"] == audit.RESOURCE_DEPLOYMENT
    assert row["metadata"]["outcome"] == "success"
    assert row["metadata"]["new_status"] == "promoted"
    _assert_no_content_anywhere(db)


def test_promote_override_of_a_foreign_deployment_records_nothing_and_changes_nothing(wf_client, db):
    before = dict(_row(db, "workflow_deployments", DEP_B))
    r = wf_client.post(f"/api/workflow-deployments/{DEP_B}/promote", json={})
    assert r.status_code == 404
    assert _row(db, "workflow_deployments", DEP_B) == before
    # No promotion happened, so no promotion may be recorded. An audit trail
    # that logs refused promotions as promotions is worse than none.
    assert _audit_rows(db, audit.DEPLOYMENT_PROMOTED) == []


def test_activate_records_the_promotion_and_every_version_it_displaced(wf_client, db):
    """
    The demote statement's `.data` was discarded, so the versions taken OUT of
    production were the one part of a rollback nothing recorded. DEP_A is the
    live version on ep-a; activating DEP_A2 must produce one rolled_back row
    for DEP_A and one promoted row for DEP_A2.
    """
    r = wf_client.post(f"/api/workflow-deployments/{DEP_A2}/activate", json={})
    assert r.status_code == 200, r.text

    rolled = _only(db, audit.DEPLOYMENT_ROLLED_BACK)
    assert (rolled["org_id"], rolled["actor_id"], rolled["resource_id"]) == (ORG_A, USER_A, DEP_A)
    assert rolled["metadata"]["prior_status"] == "promoted"
    assert rolled["metadata"]["new_status"] == "rolled_back"

    promoted = _only(db, audit.DEPLOYMENT_PROMOTED)
    assert promoted["resource_id"] == DEP_A2
    assert promoted["metadata"]["outcome"] == "success"
    _assert_no_content_anywhere(db)


def test_activate_of_a_foreign_deployment_records_nothing(wf_client, db):
    before = dict(_row(db, "workflow_deployments", DEP_B))
    r = wf_client.post(f"/api/workflow-deployments/{DEP_B}/activate", json={})
    assert r.status_code == 404
    assert _row(db, "workflow_deployments", DEP_B) == before
    assert _audit_rows(db) == []


def test_delete_deployment_writes_exactly_one_deleted_row(wf_client, db):
    r = wf_client.delete(f"/api/workflow-deployments/{DEP_A2}")
    assert r.status_code == 200, r.text
    assert _row(db, "workflow_deployments", DEP_A2) is None

    row = _only(db, audit.DEPLOYMENT_DELETED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, DEP_A2)
    assert row["resource_type"] == audit.RESOURCE_DEPLOYMENT
    assert row["metadata"]["outcome"] == "success"


def test_delete_of_a_foreign_deployment_records_nothing_and_deletes_nothing(wf_client, db):
    r = wf_client.delete(f"/api/workflow-deployments/{DEP_B}")
    assert r.status_code == 404
    assert _row(db, "workflow_deployments", DEP_B) is not None
    assert _audit_rows(db, audit.DEPLOYMENT_DELETED) == []


def test_automatic_rollback_records_a_server_derived_row_with_no_human_actor(db):
    """
    The machine path into production. Nobody is on the request, so `actor_id`
    is NULL and the org is the one the SERVER read off the `rollback_rules`
    row — not anything a client sent.
    """
    db.rows["workflow_deployments"] = [
        _deployment(DEP_A, ORG_A, 2, "ep-a", "promoted"),
        _deployment(DEP_A2, ORG_A, 1, "ep-a", "promoted"),
    ]
    with patch.object(workflow_management, "supabase", db), \
         patch.object(audit, "supabase", db), \
         patch.object(workflow_management, "_end_experiments_on_endpoint_sync", MagicMock()):
        workflow_management._execute_automatic_rollback_sync(_rule(RR_A, ORG_A, "ep-a"))

    row = _only(db, audit.DEPLOYMENT_ROLLED_BACK)
    assert row["org_id"] == ORG_A
    assert row["actor_id"] is None          # no human decided this
    assert row["resource_type"] == audit.RESOURCE_DEPLOYMENT
    assert row["metadata"]["outcome"] == "success"
    assert row["metadata"]["new_status"] == "promoted"


def test_experiment_conclusion_records_the_promotion_it_performs(wf_client, db):
    """
    Concluding an experiment INSERTS AND PROMOTES a deployment. That is a
    production change and belongs in the same log as every other promotion,
    rather than being implied by an experiment status flipping to 'concluded'.
    """
    winner = _deployment("w-dep", ORG_A, 2, "ep-a", "promoted")
    with patch.object(workflow_management, "get_latest_promoted_deployment",
                      AsyncMock(return_value=_deployment(DEP_A, ORG_A, 1, "ep-a", "promoted"))), \
         patch.object(workflow_management, "get_promoted_deployment_by_version",
                      AsyncMock(return_value=winner)):
        r = wf_client.put(f"/api/experiments/{EXP_A}/conclude", json={"winner_version": 2})
    assert r.status_code == 200, r.text

    row = _only(db, audit.DEPLOYMENT_PROMOTED)
    assert row["org_id"] == ORG_A
    # The manual path DOES have a human actor and names them.
    assert row["actor_id"] == USER_A
    assert row["metadata"]["new_status"] == "promoted"
    assert row["metadata"]["outcome"] == "success"


def test_concluding_a_foreign_experiment_records_nothing(wf_client, db):
    r = wf_client.put(f"/api/experiments/{UNKNOWN}/conclude", json={"winner_version": 2})
    assert r.status_code == 404
    assert _audit_rows(db) == []


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT — provider credentials
# ═══════════════════════════════════════════════════════════════════════════


def test_provider_credential_create_writes_exactly_one_row_without_the_key(key_client, db):
    r = key_client.post(
        "/api/api-keys",
        json={"provider": "openai", "api_key": REAL_KEY, "org_id": ORG_A,
              "name": REAL_PROMPT},
    )
    assert r.status_code == 200, r.text

    row = _only(db, audit.PROVIDER_CREDENTIAL_CREATED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["resource_type"] == audit.RESOURCE_PROVIDER_CREDENTIAL
    assert row["metadata"]["provider"] == "openai"
    assert row["metadata"]["outcome"] == "success"
    # The credential and the free-text field are both absent from the WHOLE row.
    _assert_no_content_anywhere(db)


def test_provider_credential_is_stored_against_the_verified_org_not_the_body(key_client, db):
    r = key_client.post(
        "/api/api-keys",
        json={"provider": "openai", "api_key": REAL_KEY, "org_id": ORG_B},
    )
    assert r.status_code == 200, r.text
    assert r.json()["org_id"] == ORG_A
    assert not any(k["org_id"] == ORG_B and k["provider"] == "openai"
                   for k in db.rows["api_keys"])
    assert _only(db, audit.PROVIDER_CREDENTIAL_CREATED)["org_id"] == ORG_A


def test_provider_credential_delete_writes_exactly_one_row(key_client, db):
    r = key_client.delete(f"/api/api-keys/{ORG_A}/{PKEY_A}")
    assert r.status_code == 200, r.text
    assert _row(db, "api_keys", PKEY_A) is None

    row = _only(db, audit.PROVIDER_CREDENTIAL_DELETED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, PKEY_A)
    assert row["metadata"]["outcome"] == "success"


def test_provider_credential_delete_of_a_foreign_key_is_refused_and_filed_under_the_victim(key_client, db):
    r = key_client.delete(f"/api/api-keys/{ORG_A}/{PKEY_B}")
    assert r.status_code == 404
    assert _row(db, "api_keys", PKEY_B) is not None

    row = _only(db, audit.PROVIDER_CREDENTIAL_DELETE_REFUSED)
    # Under the VICTIM's org: "did anyone reach for my credential?" is a
    # question only the owning tenant can be expected to ask.
    assert row["org_id"] == ORG_B
    assert row["actor_id"] == USER_A
    assert row["resource_id"] == PKEY_B
    assert row["metadata"]["outcome"] == "refused"
    assert row["metadata"]["reason_code"] == audit.REASON_CROSS_TENANT
    _assert_no_content_anywhere(db)


def test_provider_credential_delete_of_an_unknown_key_is_refused_under_the_callers_org(key_client, db):
    r = key_client.delete(f"/api/api-keys/{ORG_A}/{UNKNOWN}")
    assert r.status_code == 404
    row = _only(db, audit.PROVIDER_CREDENTIAL_DELETE_REFUSED)
    assert row["org_id"] == ORG_A
    assert row["metadata"]["reason_code"] == audit.REASON_NOT_FOUND


def test_provider_credential_foreign_and_unknown_deletes_are_byte_identical(key_client):
    foreign = key_client.delete(f"/api/api-keys/{ORG_A}/{PKEY_B}")
    unknown = key_client.delete(f"/api/api-keys/{ORG_A}/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT — server API keys (the credential for the public execution surface)
# ═══════════════════════════════════════════════════════════════════════════


def test_server_key_create_writes_exactly_one_row_without_the_key(key_client, db):
    r = key_client.post("/api/service-api-keys",
                        json={"api_key": REAL_KEY, "org_id": ORG_A})
    assert r.status_code == 200, r.text

    row = _only(db, audit.SERVER_KEY_CREATED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["resource_type"] == audit.RESOURCE_SERVER_API_KEY
    assert row["metadata"]["outcome"] == "success"
    _assert_no_content_anywhere(db)


def test_server_key_update_records_an_update(key_client, db):
    r = key_client.put(f"/api/service-api-keys/{ORG_A}/{SKEY_A}",
                       json={"rate_limit_per_minute": 5})
    assert r.status_code == 200, r.text

    row = _only(db, audit.SERVER_KEY_UPDATED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, SKEY_A)
    assert _audit_rows(db, audit.SERVER_KEY_REVOKED) == []


def test_server_key_update_to_revoked_is_recorded_as_a_revoke(key_client, db):
    """
    THE QUIET PATH. `status="revoked"` here takes a production credential out of
    service exactly as DELETE does, but reads as an ordinary settings edit. It
    must land in the log as a revoke, or the revoke count is wrong.
    """
    r = key_client.put(f"/api/service-api-keys/{ORG_A}/{SKEY_A}",
                       json={"status": "revoked"})
    assert r.status_code == 200, r.text
    assert _row(db, "service_api_keys", SKEY_A)["status"] == "revoked"

    row = _only(db, audit.SERVER_KEY_REVOKED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, SKEY_A)
    assert row["metadata"]["new_status"] == "revoked"
    assert _audit_rows(db, audit.SERVER_KEY_UPDATED) == []


def test_server_key_update_of_a_foreign_key_is_refused_and_filed_under_the_victim(key_client, db):
    before = dict(_row(db, "service_api_keys", SKEY_B))
    r = key_client.put(f"/api/service-api-keys/{ORG_A}/{SKEY_B}",
                       json={"status": "revoked"})
    assert r.status_code == 404
    assert _row(db, "service_api_keys", SKEY_B) == before

    row = _only(db, audit.SERVER_KEY_REVOKE_REFUSED)
    assert row["org_id"] == ORG_B          # the tenant whose key was reached for
    assert row["actor_id"] == USER_A
    assert row["resource_id"] == SKEY_B
    assert row["metadata"]["reason_code"] == audit.REASON_CROSS_TENANT
    assert row["metadata"]["outcome"] == "refused"
    _assert_no_content_anywhere(db)


def test_a_refused_non_revoke_edit_is_not_counted_as_a_refused_revoke(key_client, db):
    r = key_client.put(f"/api/service-api-keys/{ORG_A}/{SKEY_B}",
                       json={"rate_limit_per_minute": 1})
    assert r.status_code == 404
    row = _only(db, audit.SERVER_KEY_UPDATE_REFUSED)
    assert row["org_id"] == ORG_B
    assert row["metadata"]["outcome"] == "refused"
    assert _audit_rows(db, audit.SERVER_KEY_REVOKE_REFUSED) == []


def test_server_key_update_of_an_unknown_key_is_refused_under_the_callers_org(key_client, db):
    """The other refusal branch: no such key anywhere, so there is no victim to
    file under and the caller's own org is the only honest answer."""
    r = key_client.put(f"/api/service-api-keys/{ORG_A}/{UNKNOWN}",
                       json={"status": "revoked"})
    assert r.status_code == 404
    row = _only(db, audit.SERVER_KEY_REVOKE_REFUSED)
    assert row["org_id"] == ORG_A
    assert row["actor_id"] == USER_A
    assert row["metadata"]["reason_code"] == audit.REASON_NOT_FOUND


def test_server_key_delete_writes_exactly_one_revoked_row(key_client, db):
    r = key_client.delete(f"/api/service-api-keys/{ORG_A}/{SKEY_A}")
    assert r.status_code == 200, r.text
    assert _row(db, "service_api_keys", SKEY_A) is None

    row = _only(db, audit.SERVER_KEY_REVOKED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, SKEY_A)
    assert row["metadata"]["outcome"] == "success"


def test_server_key_delete_of_a_foreign_key_is_refused_and_filed_under_the_victim(key_client, db):
    """
    THE INCIDENT QUESTION, on the route that has an org in its path. The other
    route to the same operation (`main.delete_service_api_key`) already files
    this under the owning org; a resource class must not have two different
    answers depending on which route was used.
    """
    before = dict(_row(db, "service_api_keys", SKEY_B))
    r = key_client.delete(f"/api/service-api-keys/{ORG_A}/{SKEY_B}")
    assert r.status_code == 404
    assert _row(db, "service_api_keys", SKEY_B) == before

    row = _only(db, audit.SERVER_KEY_REVOKE_REFUSED)
    assert row["org_id"] == ORG_B
    assert row["actor_id"] == USER_A
    assert row["resource_id"] == SKEY_B
    assert row["metadata"]["reason_code"] == audit.REASON_CROSS_TENANT


def test_server_key_foreign_and_unknown_deletes_are_byte_identical(key_client):
    foreign = key_client.delete(f"/api/service-api-keys/{ORG_A}/{SKEY_B}")
    unknown = key_client.delete(f"/api/service-api-keys/{ORG_A}/{UNKNOWN}")
    assert foreign.status_code == unknown.status_code
    assert foreign.content == unknown.content


def test_every_refusal_is_marked_refused_two_independent_ways(key_client, db):
    key_client.delete(f"/api/service-api-keys/{ORG_A}/{SKEY_B}")
    key_client.delete(f"/api/api-keys/{ORG_A}/{PKEY_B}")
    rows = _audit_rows(db)
    assert len(rows) == 2
    for row in rows:
        assert row["action"].endswith(".refused")
        assert row["metadata"]["outcome"] == "refused"


def test_an_audit_failure_never_breaks_the_operation(key_client, db):
    """
    Best effort, always. An audit trail that can take the product down gets
    turned off, and then there is no trail at all.
    """
    with patch.object(audit, "_write", side_effect=RuntimeError("audit is down")):
        r = key_client.delete(f"/api/service-api-keys/{ORG_A}/{SKEY_A}")
    assert r.status_code == 200, r.text
    assert _row(db, "service_api_keys", SKEY_A) is None
