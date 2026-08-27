"""
Tests for POST /v1/outcomes — OptiML's customer-facing outcome ingestion.

This endpoint is called by a customer's BACKEND holding the same **server API
key** it uses for POST /v1/chat/completions. It previously demanded a Supabase
user session, which a backend does not have, so the single capability the
product exists to provide — attaching a real business outcome to an attempt —
was not callable by the customer it was designed for.

What must hold, and is asserted here rather than assumed:

  AUTH.        The server API key, validated by the SAME `validate_api_key`
               primitive production inference uses. Revoked and malformed keys
               fail closed with 401.
  ORG.         Derived exclusively from the validated key. An `org_id` asserted
               in the body, a header or the query string is never trusted, and
               a conflicting one is REJECTED rather than silently overridden or
               silently ignored.
  ISOLATION.   Another org's request id reads as the SAME 404 as an id that
               does not exist anywhere, so the endpoint cannot be used to probe
               which request ids exist in another tenant.
  SEMANTICS.   Idempotency, delayed arrival, revision chains, provenance and
               provenance_rank all survive the auth change unchanged.

Style follows test_public_execution.py and test_openai_compat.py, including the
Crypto stub so the router imports without pycryptodome.

The Supabase fake below is a real (if small) in-memory table rather than a
MagicMock: idempotency and "creates no second row" are claims about STORAGE,
and a mock that returns whatever it is told would assert nothing about either.
"""
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Stub Crypto so the router imports without pycryptodome.
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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from optimization import domain, outcomes as outcomes_mod
from routers.optimization_router import public_router
from utils.encryption import hash_service_api_key

app = FastAPI()
app.include_router(public_router)

ORG_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ORG_ID = "22222222-2222-2222-2222-222222222222"

LIVE_KEY = "sk_live_customer_backend_key"
REVOKED_KEY = "sk_live_revoked_key"
AUTH = {"Authorization": f"Bearer {LIVE_KEY}"}

# A direct-inference attempt is addressed by the X-OptiML-Request-Id the
# /v1/chat/completions response returned.
MY_REQUEST_ID = "chatcmpl-aaaaaaaaaaaaaaaaaaaaaaaa"
THEIR_REQUEST_ID = "chatcmpl-bbbbbbbbbbbbbbbbbbbbbbbb"
NOBODYS_REQUEST_ID = "chatcmpl-cccccccccccccccccccccccc"


# ═════════════════════════════════════════════════════════════════════════════
# A small real table, not a mock
# ═════════════════════════════════════════════════════════════════════════════
class _Query:
    """Supports exactly the chain shapes optimization/outcomes.py builds."""

    def __init__(self, rows):
        self.rows = list(rows)

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self.rows = [r for r in self.rows if _cmp(r.get(col), val)]
        return self

    def is_(self, col, _val):
        self.rows = [r for r in self.rows if r.get(col) is None]
        return self

    def gte(self, col, val):
        self.rows = [r for r in self.rows if str(r.get(col) or "") >= str(val)]
        return self

    def order(self, col, desc=False):
        self.rows.sort(key=lambda r: str(r.get(col) or ""), reverse=desc)
        return self

    def limit(self, n):
        self.rows = self.rows[: max(0, int(n))]
        return self

    def execute(self):
        return SimpleNamespace(data=[dict(r) for r in self.rows])


def _cmp(actual, expected):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(actual) == bool(expected)
    return str(actual) == str(expected)


class _Update:
    def __init__(self, rows, patch_values):
        self.rows = rows
        self.patch = patch_values
        self.filters = []

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def execute(self):
        hit = [r for r in self.rows if all(_cmp(r.get(c), v) for c, v in self.filters)]
        for r in hit:
            r.update(self.patch)
        return SimpleNamespace(data=[dict(r) for r in hit])


class _Table:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_a, **_k):
        return _Query(self.rows)

    def insert(self, row):
        new = dict(row)
        new.setdefault("id", str(uuid.uuid4()))
        # provenance_rank is a GENERATED column in production
        # (public.outcome_provenance_rank). Emulate it so tests can assert it.
        new["provenance_rank"] = domain.provenance_rank(new.get("provenance"))
        new.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.rows.append(new)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[dict(new)]))

    def update(self, patch_values):
        return _Update(self.rows, patch_values)


class _FakeSupabase:
    def __init__(self):
        self.tables: dict[str, list[dict]] = {}

    def table(self, name):
        return _Table(self.tables.setdefault(name, []))

    def rows(self, name):
        return self.tables.setdefault(name, [])


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    """Backing store for the `outcomes` table."""
    return _FakeSupabase()


@pytest.fixture
def keys():
    """
    service_api_keys as production actually stores them: hashed, with a status.

    Both keys are real rows. The difference between them is `status`, which is
    the only thing that should decide whether the request is served.
    """
    fake = _FakeSupabase()
    fake.rows("service_api_keys").extend([
        {
            "id": "key-live",
            "org_id": ORG_ID,
            "hashed_key": hash_service_api_key(LIVE_KEY),
            "api_key": None,
            "key_type": "live",
            "rate_limit_per_minute": 60,
            "status": "active",
        },
        {
            "id": "key-revoked",
            "org_id": ORG_ID,
            "hashed_key": hash_service_api_key(REVOKED_KEY),
            "api_key": None,
            "key_type": "live",
            "rate_limit_per_minute": 60,
            "status": "revoked",
        },
    ])
    return fake


@pytest.fixture
def attempts():
    """
    Attempt ownership, org-scoped. MY_REQUEST_ID belongs to ORG_ID and
    THEIR_REQUEST_ID belongs to OTHER_ORG_ID; NOBODYS_REQUEST_ID exists nowhere.
    """
    owned = {
        (ORG_ID, MY_REQUEST_ID): SimpleNamespace(workload_id=None),
        (OTHER_ORG_ID, THEIR_REQUEST_ID): SimpleNamespace(workload_id=None),
    }

    def get_attempt(org_id, attempt_ref, *, attempt_source="workflow_run"):
        return owned.get((str(org_id), str(attempt_ref)))

    return get_attempt


@pytest.fixture
def wired(db, keys, attempts):
    """
    Real validate_api_key against a real (fake-backed) key table, real
    record_outcome against a real (fake-backed) outcomes table. Only the
    attempt lookup is substituted, and it enforces org scoping itself.
    """
    with patch("api_key_validation.supabase", keys), patch(
        "optimization.outcomes.supabase", db
    ), patch("optimization.attempts.get_attempt", attempts):
        yield db


def _payload(**over):
    body = {
        "outcome_type": "ticket_resolved",
        "idempotency_key": "idem-" + uuid.uuid4().hex[:12],
        "request_id": MY_REQUEST_ID,
        "success": True,
        "provenance": "business_outcome",
    }
    body.update(over)
    return body


# ═════════════════════════════════════════════════════════════════════════════
# 1. The server API key works — the whole point of the change
# ═════════════════════════════════════════════════════════════════════════════
def test_server_api_key_with_own_request_id_succeeds(client, wired):
    """
    The credential a customer's backend already holds for /v1/chat/completions
    is accepted here. This is the case that returned 401 in production.
    """
    resp = client.post("/v1/outcomes", json=_payload(), headers=AUTH)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["created"] is True
    assert body["idempotent_replay"] is False

    outcome = body["outcome"]
    assert outcome["outcome_type"] == "ticket_resolved"
    assert outcome["attempt_ref"] == MY_REQUEST_ID
    # Org comes from the KEY, and is what actually got written.
    assert outcome["org_id"] == ORG_ID
    assert wired.rows("outcomes")[0]["org_id"] == ORG_ID
    # Provenance survives, and the generated rank with it.
    assert outcome["provenance"] == "business_outcome"
    assert outcome["provenance_rank"] == domain.provenance_rank("business_outcome")


def test_org_id_is_never_taken_from_the_request(client, wired):
    """
    Tenant isolation: the written row is stamped with the key's org even when
    the caller says nothing about org at all.
    """
    client.post("/v1/outcomes", json=_payload(), headers=AUTH)
    rows = wired.rows("outcomes")
    assert len(rows) == 1
    assert rows[0]["org_id"] == ORG_ID
    assert rows[0]["org_id"] != OTHER_ORG_ID


# ═════════════════════════════════════════════════════════════════════════════
# 2. Another org's request id is INDISTINGUISHABLE from one that does not exist
# ═════════════════════════════════════════════════════════════════════════════
def test_another_orgs_request_id_is_indistinguishable_from_not_found(client, wired):
    """
    A request id is obscurity, not an access control. If "belongs to someone
    else" and "does not exist" differed in status, body, or anything else, this
    endpoint would be an oracle for another tenant's request ids.
    """
    theirs = client.post(
        "/v1/outcomes", json=_payload(request_id=THEIR_REQUEST_ID), headers=AUTH
    )
    nobodys = client.post(
        "/v1/outcomes", json=_payload(request_id=NOBODYS_REQUEST_ID), headers=AUTH
    )

    assert theirs.status_code == 404
    assert nobodys.status_code == 404
    # Byte-for-byte identical, not merely "both 404".
    assert theirs.json() == nobodys.json()
    # And the response must not echo the id back as confirmation it was seen.
    assert THEIR_REQUEST_ID not in theirs.text

    # Nothing was written for either.
    assert wired.rows("outcomes") == []


def test_another_orgs_request_id_writes_nothing_into_either_org(client, wired):
    """The refusal is a refusal, not a write to the caller's own org."""
    client.post("/v1/outcomes", json=_payload(request_id=THEIR_REQUEST_ID), headers=AUTH)
    assert wired.rows("outcomes") == []


# ═════════════════════════════════════════════════════════════════════════════
# 3 & 4. Keys that must fail closed
# ═════════════════════════════════════════════════════════════════════════════
def test_revoked_server_key_is_401(client, wired):
    """
    A revoked key is a real row with a real matching hash. Only `status` says
    no — and that must be enough. Fail closed.
    """
    resp = client.post(
        "/v1/outcomes",
        json=_payload(),
        headers={"Authorization": f"Bearer {REVOKED_KEY}"},
    )
    assert resp.status_code == 401
    assert wired.rows("outcomes") == []


def test_malformed_key_is_401(client, wired):
    """A token that matches no key at all is rejected."""
    resp = client.post(
        "/v1/outcomes",
        json=_payload(),
        headers={"Authorization": "Bearer not-a-real-key"},
    )
    assert resp.status_code == 401
    assert wired.rows("outcomes") == []


def test_missing_authorization_header_is_401(client, wired):
    resp = client.post("/v1/outcomes", json=_payload())
    assert resp.status_code == 401
    assert wired.rows("outcomes") == []


def test_non_bearer_authorization_is_401(client, wired):
    """A user-style credential is not a server API key and is not accepted."""
    resp = client.post(
        "/v1/outcomes", json=_payload(), headers={"Authorization": "Basic abc123"}
    )
    assert resp.status_code == 401
    assert wired.rows("outcomes") == []


# ═════════════════════════════════════════════════════════════════════════════
# 5. A conflicting org_id is REJECTED, not silently overridden or ignored
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "where",
    ["body", "header", "query"],
)
def test_conflicting_org_id_is_rejected(client, wired, where):
    """
    Silently overriding would let a customer believe they had written to an org
    they had not. Silently ignoring hides the same mistake. Both are refusals
    disguised as successes, so this is an explicit 403.
    """
    url, json_body, headers = "/v1/outcomes", _payload(), dict(AUTH)
    if where == "body":
        json_body["org_id"] = OTHER_ORG_ID
    elif where == "header":
        headers["X-Org-Id"] = OTHER_ORG_ID
    else:
        url = f"/v1/outcomes?org_id={OTHER_ORG_ID}"

    resp = client.post(url, json=json_body, headers=headers)

    assert resp.status_code == 403, resp.text
    assert "org_id" in resp.json()["detail"]
    # Critically: it was NOT written under the key's org instead.
    assert wired.rows("outcomes") == []


def test_matching_org_id_in_body_is_allowed(client, wired):
    """Agreeing with the key is not a conflict — only disagreement is."""
    resp = client.post("/v1/outcomes", json=_payload(org_id=ORG_ID), headers=AUTH)
    assert resp.status_code == 200, resp.text
    assert resp.json()["outcome"]["org_id"] == ORG_ID


def test_conflicting_org_id_is_rejected_before_any_attempt_lookup(client, wired):
    """
    The org conflict is refused on its own terms, not merged into the 404. A
    caller who fat-fingers org_id gets told that, rather than being told their
    perfectly valid request id does not exist.
    """
    resp = client.post(
        "/v1/outcomes",
        json=_payload(org_id=OTHER_ORG_ID, request_id=NOBODYS_REQUEST_ID),
        headers=AUTH,
    )
    assert resp.status_code == 403


# ═════════════════════════════════════════════════════════════════════════════
# 6. Idempotency — outcome feeds are webhooks, and webhooks retry
# ═════════════════════════════════════════════════════════════════════════════
def test_duplicate_idempotency_key_returns_existing_and_creates_no_second_row(
    client, wired
):
    body = _payload(idempotency_key="webhook-delivery-42", value=1.0)

    first = client.post("/v1/outcomes", json=body, headers=AUTH)
    second = client.post("/v1/outcomes", json=body, headers=AUTH)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    assert first.json()["created"] is True
    assert second.json()["created"] is False
    assert second.json()["idempotent_replay"] is True

    # The SAME row, returned unchanged — not a duplicate, not a mutation.
    assert second.json()["outcome"]["id"] == first.json()["outcome"]["id"]
    assert len(wired.rows("outcomes")) == 1


def test_idempotent_replay_does_not_mutate_the_original(client, wired):
    """A retry carrying a different value must not overwrite what was recorded."""
    first = client.post(
        "/v1/outcomes",
        json=_payload(idempotency_key="retry-me", value=1.0),
        headers=AUTH,
    )
    second = client.post(
        "/v1/outcomes",
        json=_payload(idempotency_key="retry-me", value=999.0),
        headers=AUTH,
    )
    assert first.json()["outcome"]["value"] == 1.0
    assert second.json()["outcome"]["value"] == 1.0
    assert len(wired.rows("outcomes")) == 1


def test_idempotency_keys_do_not_leak_across_orgs(client, wired):
    """
    Idempotency is scoped per org. A key another tenant already used must not
    cause this tenant's outcome to be swallowed as a replay.
    """
    wired.rows("outcomes").append({
        "id": str(uuid.uuid4()),
        "org_id": OTHER_ORG_ID,
        "idempotency_key": "shared-key",
        "outcome_type": "ticket_resolved",
        "provenance": "business_outcome",
        "is_current": True,
    })
    resp = client.post(
        "/v1/outcomes", json=_payload(idempotency_key="shared-key"), headers=AUTH
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["created"] is True
    assert resp.json()["outcome"]["org_id"] == ORG_ID
    assert len(wired.rows("outcomes")) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 7. Delayed arrival — occurred_at may long precede recorded_at
# ═════════════════════════════════════════════════════════════════════════════
def test_delayed_outcome_stores_both_timestamps(client, wired):
    """
    A request at 10:00 may be resolved by a customer at 13:00 and reported at
    18:00. `occurred_at` is when it happened in the world; `recorded_at` is when
    OptiML learned. The gap is the point, so neither may be collapsed into now.
    """
    happened = datetime.now(timezone.utc) - timedelta(hours=6)
    resp = client.post(
        "/v1/outcomes",
        json=_payload(occurred_at=happened.isoformat(), outcome_type="refund_issued"),
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    outcome = resp.json()["outcome"]

    occurred = datetime.fromisoformat(outcome["occurred_at"].replace("Z", "+00:00"))
    recorded = datetime.fromisoformat(outcome["recorded_at"].replace("Z", "+00:00"))

    # The caller's timestamp was honoured, not replaced with server time.
    assert abs((occurred - happened).total_seconds()) < 2
    # And it is genuinely earlier than when we learned of it.
    assert occurred < recorded
    assert (recorded - occurred) > timedelta(hours=5)


def test_outcome_without_occurred_at_defaults_to_now(client, wired):
    resp = client.post("/v1/outcomes", json=_payload(), headers=AUTH)
    outcome = resp.json()["outcome"]
    occurred = datetime.fromisoformat(outcome["occurred_at"].replace("Z", "+00:00"))
    assert abs((datetime.now(timezone.utc) - occurred).total_seconds()) < 10


# ═════════════════════════════════════════════════════════════════════════════
# 8. Plural and named — many outcomes attach to ONE attempt
# ═════════════════════════════════════════════════════════════════════════════
def test_multiple_named_outcome_types_on_one_request_id_are_all_stored(client, wired):
    """
    One support response accumulates several distinct, differently-timed
    signals. They are deliberately not collapsed into a single quality score:
    the distinct names are what let a policy declare which one decides success.
    """
    named = ["thumbs_up", "ticket_resolved", "escalation", "reopened_7d"]
    for i, outcome_type in enumerate(named):
        resp = client.post(
            "/v1/outcomes",
            json=_payload(
                outcome_type=outcome_type,
                idempotency_key=f"signal-{i}",
                success=(outcome_type != "escalation"),
            ),
            headers=AUTH,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["created"] is True

    rows = wired.rows("outcomes")
    assert len(rows) == len(named)
    assert sorted(r["outcome_type"] for r in rows) == sorted(named)
    # All attached to the same attempt, all in the caller's org.
    assert {r["external_attempt_ref"] for r in rows} == {MY_REQUEST_ID}
    assert {r["org_id"] for r in rows} == {ORG_ID}


def test_distinct_provenances_keep_their_own_rank(client, wired):
    """
    provenance_rank is what keeps incompatible signals from being averaged
    together. It must follow provenance, per row.
    """
    for i, prov in enumerate(["business_outcome", "human", "heuristic"]):
        resp = client.post(
            "/v1/outcomes",
            json=_payload(provenance=prov, idempotency_key=f"prov-{i}"),
            headers=AUTH,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["outcome"]["provenance_rank"] == domain.provenance_rank(prov)

    ranks = {r["provenance"]: r["provenance_rank"] for r in wired.rows("outcomes")}
    assert ranks["business_outcome"] > ranks["heuristic"]


# ═════════════════════════════════════════════════════════════════════════════
# Revision / correction chains still behave after the auth change
# ═════════════════════════════════════════════════════════════════════════════
def test_correction_supersedes_rather_than_overwrites(client, wired):
    """
    Business data gets revised. A correction is a NEW row; the original is
    retained and marked, never overwritten, because savings math may already
    have consumed the old value and an audit must see what it consumed.
    """
    created = client.post(
        "/v1/outcomes",
        json=_payload(idempotency_key="orig", value=100.0, outcome_type="refund_issued"),
        headers=AUTH,
    )
    assert created.status_code == 200, created.text
    original_id = created.json()["outcome"]["id"]

    revised, was_created = outcomes_mod.correct_outcome(
        ORG_ID,
        original_id,
        idempotency_key="orig-correction-1",
        correction_reason="refund was partially reversed",
        value=40.0,
    )
    assert was_created is True
    assert revised["revision"] == 2
    assert str(revised["supersedes_outcome_id"]) == original_id
    assert revised["outcome_value"] == 40.0

    rows = {str(r["id"]): r for r in wired.rows("outcomes")}
    # Original retained, with its ORIGINAL value intact.
    assert rows[original_id]["outcome_value"] == 100.0
    assert rows[original_id]["is_current"] is False
    assert str(rows[original_id]["superseded_by_outcome_id"]) == str(revised["id"])
    assert rows[original_id]["superseded_at"]

    chain = outcomes_mod.revision_chain(ORG_ID, str(revised["id"]))
    assert [c["revision"] for c in chain] == [1, 2]
    assert [c["outcome_value"] for c in chain] == [100.0, 40.0]


def test_revision_chain_is_org_scoped(client, wired):
    """Another org cannot read the chain, even holding the outcome id."""
    created = client.post(
        "/v1/outcomes", json=_payload(idempotency_key="chain-scope"), headers=AUTH
    )
    outcome_id = created.json()["outcome"]["id"]
    assert outcomes_mod.revision_chain(ORG_ID, outcome_id) != []
    assert outcomes_mod.revision_chain(OTHER_ORG_ID, outcome_id) == []


# ═════════════════════════════════════════════════════════════════════════════
# Payload validation still reports as 400, distinct from the isolation 404
# ═════════════════════════════════════════════════════════════════════════════
def test_bad_provenance_is_400_not_404(client, wired):
    """
    An invalid payload is the caller's own bug and should say so. Collapsing it
    into the isolation 404 would make integration undebuggable.
    """
    resp = client.post("/v1/outcomes", json=_payload(provenance="vibes"), headers=AUTH)
    assert resp.status_code == 400
    assert "provenance" in resp.json()["detail"]


def test_missing_idempotency_key_is_422(client, wired):
    """idempotency_key is REQUIRED: outcome feeds are webhooks and webhooks retry."""
    body = _payload()
    body.pop("idempotency_key")
    resp = client.post("/v1/outcomes", json=body, headers=AUTH)
    assert resp.status_code == 422


def test_no_attempt_and_no_workload_is_400(client, wired):
    body = _payload()
    body.pop("request_id")
    resp = client.post("/v1/outcomes", json=body, headers=AUTH)
    assert resp.status_code == 400


def test_another_orgs_workload_id_is_indistinguishable_from_not_found(client, wired):
    """
    A caller-supplied workload_id is verified, not trusted, and refused with the
    same customer-safe 404 as the attempt path.
    """
    with patch(
        "routers.optimization_router.workloads_mod.get_workload", return_value=None
    ):
        resp = client.post(
            "/v1/outcomes",
            json=_payload(workload_id="99999999-9999-9999-9999-999999999999"),
            headers=AUTH,
        )
    assert resp.status_code == 404
    assert wired.rows("outcomes") == []


# ═════════════════════════════════════════════════════════════════════════════
# The dashboard endpoint is a DIFFERENT endpoint and keeps user auth
# ═════════════════════════════════════════════════════════════════════════════
def test_dashboard_outcomes_endpoint_still_requires_user_auth():
    """
    POST /api/optimization/{org_id}/outcomes is internal and unchanged. The two
    coexist: a server API key is not a substitute for a dashboard session.
    """
    from routers.optimization_router import create_outcome, router as dashboard_router

    deps = create_outcome.__defaults__ or ()
    assert any(
        getattr(d, "dependency", None) is not None
        and getattr(d.dependency, "__name__", "") == "require_org_member"
        for d in deps
    ), "dashboard create_outcome must still depend on require_org_member"

    paths = {r.path for r in dashboard_router.routes}
    assert "/optimization/{org_id}/outcomes" in paths


def test_public_outcomes_endpoint_takes_no_user_dependency():
    """
    The customer endpoint must not have picked up a user-auth dependency: a
    customer backend has no Supabase session, and that is the whole bug.
    """
    from routers.optimization_router import create_outcome_public

    deps = create_outcome_public.__defaults__ or ()
    assert not any(
        getattr(getattr(d, "dependency", None), "__name__", "") == "require_org_member"
        for d in deps
    )
