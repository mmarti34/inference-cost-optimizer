"""
The security audit trail: that it is written, and that it is written correctly.

WHAT THESE TESTS ARE FOR
────────────────────────
`public.audit_log` existed with a well-formed schema and zero rows. Nothing in
the repository referenced it. The consequence was that "did anyone exploit the
endpoint that let any authenticated user revoke any tenant's production API
key?" had no answer. These tests are the standing proof that the answer now
exists, and that producing it did not cost anything it should not have.

Four properties, each tested against the real handler rather than against the
writer in isolation — a writer that works but is never called is the bug this
whole change exists to fix:

  1. EVERY instrumented action writes EXACTLY ONE row, with the expected
     action, org, actor and resource id.
  2. An exception inside the audit writer does not break the operation. The
     mutation still commits and the caller still gets its success response.
  3. No secret and no customer content reaches a row — asserted against a
     payload carrying a realistic provider key AND a prompt, checked over the
     ENTIRE serialised row rather than field by field.
  4. `org_id` on the row is the org the guard VERIFIED, never the one in the
     request body.

Property 4 was mutation-tested: replacing `principal=_user` in
`cursor_tokens.create_cursor_token` with an org taken from `body.org_id` makes
`test_org_id_is_the_verified_org_not_the_body` fail with the body's org.

THE FAKE DATABASE really applies the filters a query declares, in the same
spirit as `test_tenant_ownership._FakeDB`. That matters here for a specific
reason: several refusal branches are reached precisely BECAUSE an org-scoped
update or delete matched nothing. A recorder that returned whatever the handler
asked for could not reach them at all.
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

import json  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import audit  # noqa: E402
import cursor_deploy  # noqa: E402
import cursor_tokens  # noqa: E402
import main  # noqa: E402
import org_access_control  # noqa: E402
import organization_management  # noqa: E402
import secrets_management  # noqa: E402
from auth_dependency import AuthenticatedUser  # noqa: E402
from routers import optimization_router  # noqa: E402

ORG_A = "11111111-1111-1111-1111-111111111111"   # the caller's verified org
ORG_B = "22222222-2222-2222-2222-222222222222"   # a different tenant
USER_A = "cccccccc-cccc-cccc-cccc-cccccccccccc"
USER_OTHER = "dddddddd-dddd-dddd-dddd-dddddddddddd"

TOKEN_A = "0a0a0a0a-0000-0000-0000-000000000001"   # cursor token, USER_A, ORG_A
TOKEN_OTHER = "0a0a0a0a-0000-0000-0000-000000000002"  # same org, another member
SECRET_A = "0b0b0b0b-0000-0000-0000-000000000001"
SECRET_B = "0b0b0b0b-0000-0000-0000-000000000002"  # belongs to ORG_B
KEY_B = "0c0c0c0c-0000-0000-0000-000000000002"     # ORG_B's production key
MEMBER_A = "0d0d0d0d-0000-0000-0000-000000000001"
REC_A = "0e0e0e0e-0000-0000-0000-000000000001"
DEP_A = "0f0f0f0f-0000-0000-0000-000000000001"
WF_A = "0f0f0f0f-0000-0000-0000-0000000000aa"

#: A realistic provider credential and a realistic prompt. Neither may ever
#: appear anywhere in an audit row, in any field, in any form.
REAL_KEY = "sk-proj-Ab3dEfGh1JkLmN0pQrStUvWxYz0123456789AbCdEfGhIjKl"
REAL_PROMPT = "Summarise the attached patient intake note and list red flags."


# ─── A Postgres double that actually applies the declared filters ───────────


class _Query:
    def __init__(self, db, table):
        self._db = db
        self._table = table
        self._op = "select"
        self._payload = None
        self._eq = []
        self._in = None
        self._is_null = []
        self._single = False
        self._limit = None

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

    def neq(self, *a, **k):
        return self

    def in_(self, col, vals):
        self._in = (col, [str(v) for v in vals])
        return self

    def is_(self, col, _val):
        self._is_null.append(col)
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

    def maybe_single(self):
        self._single = True
        return self

    def _matches(self, row):
        for col, val in self._eq:
            if str(row.get(col)) != str(val):
                return False
        if self._in is not None:
            col, vals = self._in
            if str(row.get(col)) not in vals:
                return False
        for col in self._is_null:
            if row.get(col) is not None:
                return False
        return True

    def execute(self):
        rows = self._db.rows.setdefault(self._table, [])
        if self._op == "insert":
            payload = self._payload if isinstance(self._payload, list) else [self._payload]
            created = []
            for item in payload:
                row = dict(item)
                row.setdefault("id", self._db.next_uuid())
                row.setdefault("created_at", "2026-01-01T00:00:00Z")
                rows.append(row)
                created.append(dict(row))
            return MagicMock(data=created)

        matched = [r for r in rows if self._matches(r)]

        if self._op == "update":
            for r in matched:
                r.update(self._payload)
            return MagicMock(data=[dict(r) for r in matched])

        if self._op == "delete":
            for r in matched:
                rows.remove(r)
            return MagicMock(data=[dict(r) for r in matched])

        out = [dict(r) for r in matched]
        if self._limit is not None:
            out = out[: self._limit]
        if self._single:
            return MagicMock(data=(out[0] if out else None))
        return MagicMock(data=out)


class _FakeDB:
    def __init__(self, rows):
        self.rows = {k: [dict(r) for r in v] for k, v in rows.items()}
        self._n = 0

    def next_uuid(self):
        self._n += 1
        return "99999999-9999-9999-9999-%012d" % self._n

    def table(self, name):
        return _Query(self, name)

    # ── what the tests assert on ────────────────────────────────────────
    @property
    def audit_rows(self):
        return self.rows.setdefault(audit.AUDIT_TABLE, [])

    def audit_actions(self):
        return [r["action"] for r in self.audit_rows]

    def only_audit_row(self, action):
        """The single row for `action`. Fails loudly on zero or on duplicates."""
        rows = [r for r in self.audit_rows if r["action"] == action]
        assert len(rows) == 1, (
            "expected exactly one %r row, got %d (all actions: %r)"
            % (action, len(rows), self.audit_actions())
        )
        return rows[0]


def _seed():
    return _FakeDB({
        "organizations": [
            {"id": ORG_A, "slug": "org-a", "name": "A", "plan": "enterprise", "type": "Organization"},
            {"id": ORG_B, "slug": "org-b", "name": "B", "plan": "enterprise", "type": "Organization"},
        ],
        "organization_members": [
            {"id": MEMBER_A, "org_id": ORG_A, "user_id": USER_A, "role": "admin", "status": "active"},
            {"id": "m-other", "org_id": ORG_A, "user_id": USER_OTHER, "role": "member", "status": "active"},
        ],
        "cursor_tokens": [
            {"id": TOKEN_A, "org_id": ORG_A, "user_id": USER_A, "name": "Cursor", "status": "active"},
            {"id": TOKEN_OTHER, "org_id": ORG_A, "user_id": USER_OTHER, "name": "Theirs", "status": "active"},
        ],
        "org_secrets": [
            {"id": SECRET_A, "org_id": ORG_A, "name": "STRIPE_KEY", "encrypted_value": "enc", "description": ""},
            {"id": SECRET_B, "org_id": ORG_B, "name": "VICTIM_KEY", "encrypted_value": "enc", "description": ""},
        ],
        "service_api_keys": [
            {"id": KEY_B, "org_id": ORG_B, "hashed_key": "h", "name": "Production", "status": "active"},
        ],
        "api_keys": [],
        "projects": [{"id": "proj-a", "org_id": ORG_A, "name": "Default"}],
        "workflows": [],
        "workflow_deployments": [],
        "invite_tokens": [],
        "join_requests": [],
        "user_profiles": [{"user_id": USER_A, "subscription_tier": "team"}],
        audit.AUDIT_TABLE: [],
    })


def _member_of_org_a():
    user = AuthenticatedUser(user_id=USER_A, email="a@b.c", email_verified=True)
    user._verified_org_id = ORG_A
    user._org_role = "admin"
    return user


def _plain_authenticated_user():
    """`require_auth` only: authenticated, but with NO org verified."""
    return AuthenticatedUser(user_id=USER_A, email="a@b.c", email_verified=True)


@pytest.fixture
def db():
    return _seed()


def _router_client(module, router, db, prefix="", overrides=None, extra_patches=()):
    """A one-router app wired so both the handler and `audit` see the same DB."""
    app = FastAPI()
    app.include_router(router, prefix=prefix)
    for dep, override in (overrides or {}).items():
        app.dependency_overrides[dep] = override
    stack = [patch.object(module, "supabase", db), patch.object(audit, "supabase", db)]
    stack.extend(extra_patches)
    for ctx in stack:
        ctx.start()
    try:
        yield TestClient(app)
    finally:
        for ctx in reversed(stack):
            ctx.stop()
        app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Cursor tokens — a live, org-scoped credential
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cursor_client(db):
    yield from _router_client(
        cursor_tokens, cursor_tokens.router, db,
        overrides={cursor_tokens.require_org_member: _member_of_org_a},
    )


def test_cursor_token_create_writes_exactly_one_row(cursor_client, db):
    r = cursor_client.post("/api/cursor-tokens", json={"org_id": ORG_A, "name": "Cursor"})
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.CURSOR_TOKEN_CREATED)
    assert row["org_id"] == ORG_A
    assert row["actor_id"] == USER_A
    assert row["resource_type"] == audit.RESOURCE_CURSOR_TOKEN
    assert row["metadata"]["outcome"] == "success"
    # The row identifies the credential without containing it.
    assert row["resource_id"] in [t["id"] for t in db.rows["cursor_tokens"]]
    assert r.json()["cursor_token"] not in json.dumps(row)


def test_cursor_token_revoke_writes_exactly_one_row(cursor_client, db):
    r = cursor_client.delete(f"/api/cursor-tokens/{ORG_A}/{TOKEN_A}")
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.CURSOR_TOKEN_REVOKED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, TOKEN_A)
    assert row["metadata"]["outcome"] == "success"


def test_cursor_token_revoke_of_another_members_token_is_recorded_as_refused(cursor_client, db):
    """
    The revoke is scoped by user_id as well as org_id, so this matches nothing
    and 404s. THAT is the event worth keeping: a member reaching for a
    credential that is not theirs leaves no other trace.
    """
    r = cursor_client.delete(f"/api/cursor-tokens/{ORG_A}/{TOKEN_OTHER}")
    assert r.status_code == 404

    row = db.only_audit_row(audit.CURSOR_TOKEN_REVOKE_REFUSED)
    assert row["org_id"] == ORG_A
    assert row["actor_id"] == USER_A
    assert row["resource_id"] == TOKEN_OTHER
    assert row["metadata"]["outcome"] == "refused"
    assert row["metadata"]["reason_code"] == audit.REASON_NOT_FOUND
    # The refusal did not become a mutation.
    assert db.rows["cursor_tokens"][1]["status"] == "active"


def test_audit_action_names_mark_refusals_two_independent_ways(cursor_client, db):
    cursor_client.delete(f"/api/cursor-tokens/{ORG_A}/{TOKEN_OTHER}")
    row = db.audit_rows[0]
    assert row["action"].endswith(".refused")
    assert row["metadata"]["outcome"] == "refused"


# ═══════════════════════════════════════════════════════════════════════════
# Org secrets
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def secrets_client(db):
    yield from _router_client(
        secrets_management, secrets_management.router, db,
        overrides={secrets_management.require_org_member: _member_of_org_a},
        extra_patches=[patch.object(secrets_management, "encrypt_api_key", lambda v: "enc:" + v)],
    )


def test_secret_create_writes_exactly_one_row(secrets_client, db):
    r = secrets_client.post(
        "/secrets",
        json={"org_id": ORG_A, "name": "NEW_SECRET", "value": REAL_KEY, "description": "d"},
    )
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.ORG_SECRET_CREATED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["resource_type"] == audit.RESOURCE_ORG_SECRET
    assert row["resource_id"] == r.json()["id"]


def test_secret_delete_writes_exactly_one_row(secrets_client, db):
    r = secrets_client.delete(f"/secrets/{ORG_A}/{SECRET_A}")
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.ORG_SECRET_DELETED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, SECRET_A)


def test_secret_delete_across_tenants_is_recorded_as_refused(secrets_client, db):
    """SECRET_B belongs to ORG_B. The org-scoped delete matches nothing."""
    r = secrets_client.delete(f"/secrets/{ORG_A}/{SECRET_B}")
    assert r.status_code in (404, 500)  # handler wraps its own 404; either way, refused

    row = db.only_audit_row(audit.ORG_SECRET_DELETE_REFUSED)
    assert row["org_id"] == ORG_A          # filed under the CALLER's org
    assert row["resource_id"] == SECRET_B
    assert row["metadata"]["outcome"] == "refused"
    # And the victim's secret is untouched.
    assert any(s["id"] == SECRET_B for s in db.rows["org_secrets"])


def test_secret_update_writes_exactly_one_row(secrets_client, db):
    r = secrets_client.put(f"/secrets/{ORG_A}/{SECRET_A}", json={"value": REAL_KEY})
    assert r.status_code == 200, r.text
    row = db.only_audit_row(audit.ORG_SECRET_UPDATED)
    assert (row["org_id"], row["resource_id"]) == (ORG_A, SECRET_A)


# ═══════════════════════════════════════════════════════════════════════════
# main.py — server API keys and provider credentials
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def main_client(db):
    main.app.dependency_overrides[main.require_org_member] = _member_of_org_a
    main.app.dependency_overrides[main.require_auth] = _plain_authenticated_user
    with patch.object(main, "supabase", db), \
         patch.object(audit, "supabase", db), \
         patch.object(main, "encrypt_api_key", lambda v: "enc:" + v):
        yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_server_key_create_writes_exactly_one_row(main_client, db):
    r = main_client.post(f"/generate-service-api-key/{ORG_A}", json={"name": "Production"})
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.SERVER_KEY_CREATED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["resource_type"] == audit.RESOURCE_SERVER_API_KEY
    # The plaintext key was returned to the caller exactly once and is nowhere
    # in the audit row.
    assert r.json()["api_key"] not in json.dumps(row)


def test_cross_tenant_server_key_revoke_is_recorded_under_the_victims_org(main_client, db):
    """
    THE INCIDENT QUESTION, in executable form.

    KEY_B is ORG_B's production key. USER_A is authenticated but is not a member
    of ORG_B, so the revoke is refused with the same opaque 404 as a nonexistent
    key. Before this change that attempt left no trace anywhere, which is
    precisely why "did anyone try?" could not be answered.

    The row is filed under ORG_B — the org that OWNS the key — so the victim's
    own audit trail shows the attempt. That org id is read off the
    `service_api_keys` row the server fetched; the caller supplied only a key id.
    """
    r = main_client.delete(f"/delete-service-api-key/{KEY_B}")
    assert r.status_code == 404

    row = db.only_audit_row(audit.SERVER_KEY_REVOKE_REFUSED)
    assert row["org_id"] == ORG_B
    assert row["actor_id"] == USER_A
    assert row["resource_id"] == KEY_B
    assert row["metadata"]["outcome"] == "refused"
    assert row["metadata"]["reason_code"] == audit.REASON_CROSS_TENANT
    # The refusal is still a refusal: the victim's key survives.
    assert any(k["id"] == KEY_B for k in db.rows["service_api_keys"])


def test_server_key_revoke_by_a_member_writes_exactly_one_row(main_client, db):
    db.rows["service_api_keys"].append(
        {"id": "0c0c0c0c-0000-0000-0000-00000000000a", "org_id": ORG_A, "hashed_key": "h", "status": "active"}
    )
    r = main_client.delete("/delete-service-api-key/0c0c0c0c-0000-0000-0000-00000000000a")
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.SERVER_KEY_REVOKED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert not [k for k in db.rows["service_api_keys"] if k["org_id"] == ORG_A]


def test_provider_credential_create_then_overwrite_are_distinct_rows(main_client, db):
    body = {"org_id": ORG_A, "provider": "openai", "api_key": REAL_KEY}
    assert main_client.post("/store-key", json=body).status_code == 200
    assert main_client.post("/store-key", json=body).status_code == 200

    created = db.only_audit_row(audit.PROVIDER_CREDENTIAL_CREATED)
    overwritten = db.only_audit_row(audit.PROVIDER_CREDENTIAL_OVERWRITTEN)
    assert created["metadata"]["provider"] == "openai"
    assert overwritten["metadata"]["provider"] == "openai"
    assert created["org_id"] == overwritten["org_id"] == ORG_A


def test_provider_credential_delete_writes_exactly_one_row(main_client, db):
    main_client.post("/store-key", json={"org_id": ORG_A, "provider": "openai", "api_key": REAL_KEY})
    r = main_client.request("DELETE", "/delete-key", json={"org_id": ORG_A, "provider": "openai"})
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.PROVIDER_CREDENTIAL_DELETED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["metadata"]["provider"] == "openai"


# ═══════════════════════════════════════════════════════════════════════════
# Property 3 — no secret, no customer content, anywhere in the row
# ═══════════════════════════════════════════════════════════════════════════


def test_no_credential_or_prompt_reaches_any_audit_row(main_client, db):
    """
    Asserted over the WHOLE serialised row, not field by field: a future call
    site that tucks a key into metadata under a plausible name has to fail this,
    not merely fail the field it was expected to be in.
    """
    main_client.post("/store-key", json={
        "org_id": ORG_A,
        "provider": "openai",
        "api_key": REAL_KEY,
        "name": REAL_PROMPT,
    })
    assert db.audit_rows, "the endpoint must have written an audit row at all"

    serialised = json.dumps(db.audit_rows)
    assert REAL_KEY not in serialised
    assert REAL_PROMPT not in serialised
    # Not even a prefix of the credential: dropped, never truncated.
    assert REAL_KEY[:12] not in serialised
    assert "sk-proj" not in serialised


def test_writer_drops_disallowed_metadata_keys_without_inspecting_them(db):
    """
    The allow-list is the mechanism, not a redactor. A key called `api_key` is
    dropped because it is not on the list — the same as a key called anything
    else that is not on the list.
    """
    with patch.object(audit, "supabase", db):
        audit.record(
            audit.ORG_SECRET_CREATED,
            principal=_member_of_org_a(),
            resource_type=audit.RESOURCE_ORG_SECRET,
            resource_id=SECRET_A,
            metadata={
                "api_key": REAL_KEY,
                "prompt": REAL_PROMPT,
                "request_body": {"messages": [{"content": REAL_PROMPT}]},
                "provider": "openai",          # on the allow-list, survives
            },
        )
    row = db.only_audit_row(audit.ORG_SECRET_CREATED)
    assert row["metadata"]["provider"] == "openai"
    assert row["metadata"]["_metadata_filtered"] is True
    assert REAL_KEY not in json.dumps(row)
    assert REAL_PROMPT not in json.dumps(row)


def test_writer_drops_oversized_and_non_scalar_values(db):
    with patch.object(audit, "supabase", db):
        audit.record(
            audit.ORG_SECRET_CREATED,
            principal=_member_of_org_a(),
            metadata={"provider": "x" * 500, "role": ["admin"], "prior_status": "verified"},
        )
    row = db.only_audit_row(audit.ORG_SECRET_CREATED)
    assert "provider" not in row["metadata"]
    assert "role" not in row["metadata"]
    assert row["metadata"]["prior_status"] == "verified"
    assert row["metadata"]["_metadata_filtered"] is True


def test_writer_refuses_an_action_name_that_is_not_in_the_vocabulary(db):
    with patch.object(audit, "supabase", db):
        audit.record("org_secret.deleeted", principal=_member_of_org_a())
    assert db.audit_rows == []


# ═══════════════════════════════════════════════════════════════════════════
# Property 4 — the org is the VERIFIED org, never the body's
# ═══════════════════════════════════════════════════════════════════════════


def test_org_id_is_the_verified_org_not_the_body(cursor_client, db):
    """
    MUTATION-TESTED. The guard here is overridden to verify ORG_A while the body
    names ORG_B, which is the shape the confused-deputy fix in
    `require_org_member` exists to prevent reaching a handler at all. If
    `cursor_tokens.create_cursor_token` is changed to file its audit row under
    `body.org_id`, this test fails with ORG_B.

    An audit row filed under an attacker-chosen tenant is worse than no row: it
    is a false alibi for the org that did nothing.
    """
    r = cursor_client.post("/api/cursor-tokens", json={"org_id": ORG_B, "name": "Cursor"})
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.CURSOR_TOKEN_CREATED)
    assert row["org_id"] == ORG_A
    assert row["org_id"] != ORG_B


def test_record_refuses_anything_that_is_not_a_guard_produced_identity(db):
    """
    The same invariant `resource_access._require_verified_identity` enforces —
    but it must not take the request down, so it logs and writes nothing rather
    than raising into the handler.
    """
    with patch.object(audit, "supabase", db):
        audit.record(audit.ORG_SECRET_DELETED, principal={"org_id": ORG_B})   # a dict, not a principal
        audit.record(audit.ORG_SECRET_DELETED, principal=ORG_B)               # a bare string
    assert db.audit_rows == []


def test_a_principal_with_no_verified_org_writes_nothing(db):
    """`org_id` is NOT NULL with an FK. A row we cannot file honestly is not forged."""
    with patch.object(audit, "supabase", db):
        audit.record(audit.ORG_SECRET_DELETED, principal=_plain_authenticated_user())
    assert db.audit_rows == []


def test_server_derived_requires_its_provenance_argument(db):
    with patch.object(audit, "supabase", db):
        audit.record_server_derived(
            audit.SERVER_KEY_REVOKED, org_id=ORG_A, derived_from="", actor_id=USER_A
        )
    assert db.audit_rows == []


# ═══════════════════════════════════════════════════════════════════════════
# Property 2 — a broken audit writer never breaks the operation
# ═══════════════════════════════════════════════════════════════════════════


class _ExplodingClient:
    """Every path into the database raises. The operation must still complete."""

    def table(self, name):
        raise RuntimeError("audit_log is unreachable")


def test_audit_failure_does_not_break_a_successful_revocation(cursor_client, db):
    with patch.object(audit, "supabase", _ExplodingClient()):
        r = cursor_client.delete(f"/api/cursor-tokens/{ORG_A}/{TOKEN_A}")

    assert r.status_code == 200, r.text
    assert r.json() == {"status": "revoked", "id": TOKEN_A}
    # The operation COMMITTED — this is the half that matters.
    assert db.rows["cursor_tokens"][0]["status"] == "revoked"
    assert db.audit_rows == []


def test_audit_failure_does_not_break_a_successful_secret_delete(secrets_client, db):
    with patch.object(audit, "supabase", _ExplodingClient()):
        r = secrets_client.delete(f"/secrets/{ORG_A}/{SECRET_A}")

    assert r.status_code == 200, r.text
    assert not [s for s in db.rows["org_secrets"] if s["id"] == SECRET_A]


def test_an_exception_raised_inside_the_writer_is_swallowed(cursor_client, db):
    """
    Not just a database failure: ANY exception from the writer, including a
    programming error in `_write` itself.
    """
    def _boom(**_kwargs):
        raise ValueError("bug in the audit writer")

    with patch.object(audit, "_write", _boom):
        r = cursor_client.delete(f"/api/cursor-tokens/{ORG_A}/{TOKEN_A}")

    assert r.status_code == 200, r.text
    assert db.rows["cursor_tokens"][0]["status"] == "revoked"


def test_audit_failure_does_not_break_the_cross_tenant_refusal_path(main_client, db):
    """A broken writer must not convert an opaque 404 into a 500 — that would
    turn the audit trail into the existence oracle the 404 exists to prevent."""
    with patch.object(audit, "supabase", _ExplodingClient()):
        r = main_client.delete(f"/delete-service-api-key/{KEY_B}")
    assert r.status_code == 404
    assert r.json() == {"detail": "Service API key not found."}


# ═══════════════════════════════════════════════════════════════════════════
# Membership and organization settings
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def org_client(db):
    yield from _router_client(
        org_access_control, org_access_control.router, db,
        overrides={
            org_access_control.require_org_member: _member_of_org_a,
            org_access_control.require_org_admin: _member_of_org_a,
            org_access_control.require_auth: _plain_authenticated_user,
        },
        extra_patches=[patch.object(org_access_control, "send_invite_email", lambda **k: True)],
    )


def test_member_invite_writes_exactly_one_row_without_the_invitees_email(org_client, db):
    r = org_client.post(
        "/api/organizations/invite",
        json={"org_id": ORG_A, "email": "Newcomer@Example.com"},
    )
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.MEMBER_INVITED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["resource_type"] == audit.RESOURCE_ORG_MEMBER
    assert row["metadata"]["role"] == "member"
    # PII stays in organization_members, which resource_id resolves.
    assert "newcomer@example.com" not in json.dumps(row).lower()
    assert row["resource_id"] in [m["id"] for m in db.rows["organization_members"]]


def test_member_removal_writes_exactly_one_row(org_client, db):
    r = org_client.delete(f"/api/organizations/members/{ORG_A}/{USER_OTHER}")
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.MEMBER_REMOVED)
    assert (row["org_id"], row["actor_id"]) == (ORG_A, USER_A)
    assert row["metadata"]["target_user_id"] == USER_OTHER


def test_member_removal_by_a_non_admin_is_recorded_as_refused(org_client, db):
    db.rows["organization_members"][0]["role"] = "member"   # demote the caller
    r = org_client.delete(f"/api/organizations/members/{ORG_A}/{USER_OTHER}")
    assert r.status_code == 403

    row = db.only_audit_row(audit.MEMBER_REMOVE_REFUSED)
    assert row["metadata"]["reason_code"] == audit.REASON_NOT_ADMIN
    assert row["metadata"]["outcome"] == "refused"
    assert row["metadata"]["target_user_id"] == USER_OTHER
    # Still a member.
    assert any(m["user_id"] == USER_OTHER for m in db.rows["organization_members"])


def test_invite_revocation_writes_exactly_one_row(org_client, db):
    db.rows["organization_members"].append(
        {"id": "m-invited", "org_id": ORG_A, "invited_email": "x@y.z", "status": "invited", "role": "member"}
    )
    r = org_client.post(
        "/api/organizations/revoke-invite",
        json={"org_id": ORG_A, "member_id": "m-invited"},
    )
    assert r.status_code == 200, r.text
    row = db.only_audit_row(audit.MEMBER_INVITE_REVOKED)
    assert (row["org_id"], row["resource_id"]) == (ORG_A, "m-invited")


@pytest.fixture
def org_mgmt_client(db):
    yield from _router_client(
        organization_management, organization_management.router, db,
        overrides={
            organization_management.require_org_admin: _member_of_org_a,
            organization_management.require_org_member: _member_of_org_a,
            organization_management.require_auth: _plain_authenticated_user,
        },
    )


def test_organization_settings_update_writes_exactly_one_row(org_mgmt_client, db):
    r = org_mgmt_client.patch(f"/organizations/{ORG_A}", json={"name": "Renamed"})
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.ORGANIZATION_UPDATED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, ORG_A)
    # The new name is customer free text and is deliberately not in the row.
    assert "Renamed" not in json.dumps(row)


# ═══════════════════════════════════════════════════════════════════════════
# Deployment promotion (cursor deploy) and optimization decisions
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def deploy_client(db):
    yield from _router_client(
        cursor_deploy, cursor_deploy.router, db,
        overrides={cursor_deploy.require_org_member: _member_of_org_a},
        extra_patches=[
            patch.object(cursor_deploy, "check_workflow_limit", MagicMock()),
            patch.object(cursor_deploy, "check_server_key_limit", MagicMock()),
        ],
    )


def test_cursor_deploy_records_both_the_promotion_and_the_minted_key(deploy_client, db):
    r = deploy_client.post("/api/cursor/deploy-from-parsed", json={
        "org_id": ORG_A,
        "endpoint_slug": "my-endpoint",
        "parsed": {"provider": "openai", "model": "gpt-4o-mini", "api_type": "chat"},
    })
    assert r.status_code == 200, r.text

    promoted = db.only_audit_row(audit.DEPLOYMENT_PROMOTED)
    assert (promoted["org_id"], promoted["actor_id"]) == (ORG_A, USER_A)
    assert promoted["metadata"]["version"] == 1

    minted = db.only_audit_row(audit.SERVER_KEY_CREATED)
    assert minted["org_id"] == ORG_A
    # A route that deploys AND mints a credential leaves two distinguishable
    # rows, not one event that hides the other.
    assert r.json()["server_key"] not in json.dumps(db.audit_rows)


@pytest.fixture
def optimization_client(db):
    yield from _router_client(
        optimization_router, optimization_router.router, db, prefix="/api",
        overrides={optimization_router.require_org_member: _member_of_org_a},
    )


def _rec_row(status):
    return {
        "id": REC_A, "org_id": ORG_A, "status": status,
        "workload_id": None, "objective": "cost", "policy_id": None,
        "candidate_strategy_id": None, "candidate_cost": None,
        "candidate_quality": None, "candidate_latency_p95_ms": None,
        "confidence": None,
    }


def test_recommendation_reject_writes_exactly_one_row(optimization_client, db):
    with patch.object(optimization_router.service, "transition", lambda *a, **k: _rec_row("rejected")), \
         patch.object(optimization_router.service, "recommendation_row_to_response", lambda row, **k: {"ok": True}):
        r = optimization_client.post(
            f"/api/optimization/{ORG_A}/recommendations/{REC_A}/reject",
            json={"reason": "the candidate regressed a metric we care about"},
        )
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.RECOMMENDATION_REJECTED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, REC_A)
    # The operator's free-text reason lives on the recommendation, not here.
    assert "regressed" not in json.dumps(row)


def test_recommendation_accept_writes_exactly_one_row(optimization_client, db):
    verified = optimization_router.domain.STATUS_VERIFIED
    with patch.object(optimization_router.service, "get_recommendation", lambda *a, **k: _rec_row(verified)), \
         patch.object(optimization_router.service, "require_evidence", lambda *a, **k: [{"id": "ev"}]), \
         patch.object(optimization_router.service, "transition", lambda *a, **k: _rec_row("canary")), \
         patch.object(optimization_router.service, "recommendation_row_to_response", lambda row, **k: {"ok": True}), \
         patch.object(optimization_router, "_create_candidate_deployment", lambda org, rec: {"id": DEP_A, "version": 2, "endpoint_slug": "e"}), \
         patch.object(optimization_router.allocation, "record_decision", MagicMock()):
        r = optimization_client.post(
            f"/api/optimization/{ORG_A}/recommendations/{REC_A}/accept"
        )
    assert r.status_code == 200, r.text

    row = db.only_audit_row(audit.RECOMMENDATION_ACCEPTED)
    assert (row["org_id"], row["actor_id"], row["resource_id"]) == (ORG_A, USER_A, REC_A)
    assert row["metadata"]["prior_status"] == verified
    assert row["metadata"]["new_status"] == optimization_router.domain.STATUS_CANARY
    assert row["metadata"]["deployment_id"] == DEP_A


# ═══════════════════════════════════════════════════════════════════════════
# The vocabulary itself
# ═══════════════════════════════════════════════════════════════════════════


def test_every_refusal_action_is_in_the_vocabulary_and_marked():
    assert audit.REFUSAL_ACTIONS <= audit.ACTIONS
    for action in audit.REFUSAL_ACTIONS:
        assert action.endswith(".refused"), action


def test_no_action_constant_is_duplicated():
    constants = {
        name: value for name, value in vars(audit).items()
        if name.isupper() and isinstance(value, str) and "." in value
        and value in audit.ACTIONS
    }
    assert len(set(constants.values())) == len(constants), constants
    assert set(constants.values()) == set(audit.ACTIONS)
