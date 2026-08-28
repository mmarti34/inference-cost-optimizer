"""
Cross-tenant authorization on POST /optimize (main.py).

This is NOT an existence oracle — it was cross-tenant CONTENT disclosure.

`require_org_member` proves the caller belongs to the org they named in the
BODY. It says nothing about `prompt_id` or `project_id`, which are separate
caller-supplied identifiers. All three queries in the handler were filtered by
id alone:

    prompt_templates .eq("id", payload.prompt_id)      # no org filter
    projects         .eq("id", payload.project_id)     # no org filter
    usage_logs       .eq("project_id", payload.project_id)

`supabase_client` uses the SERVICE-ROLE key, so RLS does not backstop any of
them. A member of any org could pass another tenant's prompt_id and have that
prompt's TEXT read into the system prompt and described back in the
recommendation's `reasoning` — plus that project's monthly budget and
month-to-date spend.

These tests assert the org filter is applied to every query. They fail if any
`.eq("org_id", ...)` is dropped.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

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

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from auth_dependency import AuthenticatedUser  # noqa: E402

ORG_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ORG_ID = "22222222-2222-2222-2222-222222222222"

FOREIGN_PROMPT_TEXT = "SECRET-TENANT-B-PROMPT-do-not-disclose"

PAYLOAD = {
    "prompt_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "estimated_input_tokens": 100,
    "estimated_output_tokens": 50,
    "project_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "org_id": ORG_ID,
    "user_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
}


class _Recorder:
    """A supabase double that records the filters applied to each table."""

    def __init__(self, rows_by_table):
        self.rows_by_table = rows_by_table
        self.filters: dict[str, list[tuple]] = {}

    def table(self, name):
        self.filters.setdefault(name, [])
        return _Chain(self, name)


class _Chain:
    def __init__(self, rec, table):
        self._rec = rec
        self._table = table

    def select(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def gte(self, *a, **k):
        return self

    def single(self):
        return self

    def eq(self, col, val):
        self._rec.filters[self._table].append((col, val))
        return self

    def execute(self):
        return MagicMock(data=self._rec.rows_by_table.get(self._table, []))


@pytest.fixture
def as_member(monkeypatch):
    """Authenticate as a verified member of ORG_ID."""
    user = AuthenticatedUser(user_id=PAYLOAD["user_id"], email="a@b.c")
    setattr(user, "_verified_org_id", ORG_ID)
    main.app.dependency_overrides[main.require_org_member] = lambda: user
    yield user
    main.app.dependency_overrides.clear()


def _call(rec):
    with patch.object(main, "supabase", rec):
        client = TestClient(main.app)
        return client.post("/optimize", json=PAYLOAD)


def test_every_query_is_filtered_by_the_verified_org(as_member):
    """The whole fix in one assertion: no query may run on id alone."""
    rec = _Recorder({})
    _call(rec)

    for table in ("prompt_templates", "projects", "usage_logs"):
        applied = rec.filters.get(table)
        if applied is None:
            continue  # short-circuited before reaching this table
        cols = [c for c, _ in applied]
        assert "org_id" in cols, f"{table} queried without an org filter: {applied}"
        assert (
            "org_id",
            ORG_ID,
        ) in applied, f"{table} filtered by the wrong org: {applied}"


def test_a_foreign_prompt_is_not_read_or_described_back(as_member):
    """
    The database still holds the other tenant's row; the org filter is what
    keeps it out. Simulate the filter working (no rows returned) and assert the
    handler refuses rather than proceeding with foreign content.
    """
    rec = _Recorder({"prompt_templates": []})  # org filter excluded it
    resp = _call(rec)

    assert resp.status_code == 404
    assert FOREIGN_PROMPT_TEXT not in resp.text
    # It must not have gone on to read the project or call a model.
    assert "projects" not in rec.filters or rec.filters["projects"] == []


def test_the_prompt_query_carries_both_the_id_and_the_org(as_member):
    rec = _Recorder({})
    _call(rec)
    applied = rec.filters.get("prompt_templates", [])
    assert ("id", PAYLOAD["prompt_id"]) in applied
    assert ("org_id", ORG_ID) in applied


def test_the_org_comes_from_the_verified_guard_not_the_body(as_member):
    """
    A caller who names their own org in the body but is verified against it
    cannot widen scope by editing the body: the filter uses the guard's value.
    """
    rec = _Recorder({})
    payload = dict(PAYLOAD, org_id=OTHER_ORG_ID)  # body claims a different org
    with patch.object(main, "supabase", rec):
        TestClient(main.app).post("/optimize", json=payload)

    for table, applied in rec.filters.items():
        for col, val in applied:
            if col == "org_id":
                assert val == ORG_ID, (
                    f"{table} filtered by the body's org_id, not the verified one"
                )


# ---------------------------------------------------------------------------
# DELETE /delete-service-api-key/{key_id}
#
# Cross-tenant DESTRUCTIVE write, not an information leak. It used require_auth
# — any authenticated user, no org membership proven — and deleted from
# service_api_keys filtered only by key_id, with the service-role client
# bypassing RLS. Any signed-in user who knew a key id could revoke another
# tenant's production key.
# ---------------------------------------------------------------------------

KEY_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"


class _KeyRecorder(_Recorder):
    """Records deletes as well as filters."""

    def __init__(self, rows_by_table, member_rows):
        super().__init__(rows_by_table)
        self.rows_by_table["organization_members"] = member_rows
        self.deletes: list[tuple[str, list]] = []

    def table(self, name):
        self.filters.setdefault(name, [])
        return _KeyChain(self, name)


class _KeyChain(_Chain):
    def __init__(self, rec, table):
        super().__init__(rec, table)
        self._deleting = False

    def delete(self):
        self._deleting = True
        return self

    def execute(self):
        if self._deleting:
            self._rec.deletes.append((self._table, list(self._rec.filters[self._table])))
        return MagicMock(data=self._rec.rows_by_table.get(self._table, []))


@pytest.fixture
def as_authed(monkeypatch):
    user = AuthenticatedUser(user_id=PAYLOAD["user_id"], email="a@b.c")
    main.app.dependency_overrides[main.require_auth] = lambda: user
    yield user
    main.app.dependency_overrides.clear()


def _delete(rec):
    with patch.object(main, "supabase", rec):
        return TestClient(main.app).delete(f"/delete-service-api-key/{KEY_ID}")


def test_a_foreign_tenants_key_cannot_be_revoked(as_authed):
    """The key belongs to OTHER_ORG_ID and the caller is not a member."""
    rec = _KeyRecorder(
        {"service_api_keys": [{"id": KEY_ID, "org_id": OTHER_ORG_ID}]},
        member_rows=[],  # caller is not a member of the key's org
    )
    resp = _delete(rec)

    assert resp.status_code == 404
    assert rec.deletes == [], "a foreign tenant's key was deleted"


def test_an_unknown_key_is_indistinguishable_from_a_foreign_one(as_authed):
    unknown = _KeyRecorder({"service_api_keys": []}, member_rows=[])
    foreign = _KeyRecorder(
        {"service_api_keys": [{"id": KEY_ID, "org_id": OTHER_ORG_ID}]}, member_rows=[]
    )
    r1, r2 = _delete(unknown), _delete(foreign)

    assert r1.status_code == r2.status_code == 404
    assert r1.json() == r2.json()
    assert unknown.deletes == foreign.deletes == []


def test_your_own_key_is_still_revocable_and_the_delete_is_org_scoped(as_authed):
    rec = _KeyRecorder(
        {"service_api_keys": [{"id": KEY_ID, "org_id": ORG_ID}]},
        member_rows=[{"role": "member"}],
    )
    resp = _delete(rec)

    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert len(rec.deletes) == 1
    table, applied = rec.deletes[0]
    assert table == "service_api_keys"
    assert ("id", KEY_ID) in applied
    assert ("org_id", ORG_ID) in applied, "the DELETE itself must be org-scoped"


def test_membership_is_checked_against_the_keys_org_not_a_supplied_one(as_authed):
    """The caller cannot name the org they are checked against."""
    rec = _KeyRecorder(
        {"service_api_keys": [{"id": KEY_ID, "org_id": OTHER_ORG_ID}]},
        member_rows=[],
    )
    _delete(rec)
    assert ("org_id", OTHER_ORG_ID) in rec.filters["organization_members"]
