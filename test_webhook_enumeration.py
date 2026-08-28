"""
The incoming-webhook receiver: pre-authentication uniformity, and that nothing
after authentication was flattened along with it.

WHAT WAS WRONG
──────────────
``POST /api/webhooks/trigger/{endpoint_path}`` takes an endpoint path and a
request body and nothing else — no auth dependency, by design, because the
sender is an external service and the shared secret is the credential. The
lookup it performs is necessarily global (``endpoint_path`` is the only locator
an external sender has), and it ran BEFORE signature verification with its
result reaching the caller:

    unknown path                      -> 404 "Webhook not found or inactive"
    known path, bad/absent signature  -> 401 "Missing webhook signature..."
    known path, row has no secret     -> 503 "This webhook has no signing..."

Contrast any two and a fully unauthenticated caller learns whether an
``endpoint_path`` exists and is active — across every tenant. Paths are often
descriptive (``stripe-customer-acme-foo``), so what leaks is customer names and
vendor relationships, not merely row existence.

WHAT THESE TESTS PIN
────────────────────
  1. UNIFORMITY. Unknown path, known+invalid signature and known+absent
     signature are ONE response: same status, same body bytes, same headers.
     Compared as whole response objects, not field by field, so a future edit
     that adds a distinguishing header is caught too.
  2. THE 503 IS FOLDED IN. A row with no secret can never authenticate anyone,
     so collapsing it into the same 401 costs no legitimate sender anything and
     removes the last pre-auth signal. Same for an inactive webhook.
  3. NOT FLATTENED AFTER AUTH. A valid signature still executes; a handler
     failure after a valid signature is still a 5xx, because a sender that
     cannot tell "your handler broke, retry" from "rejected, stop" is worse off
     than it was before the fix.
  4. THE LOOKUP STILL HAPPENS. The fix must be "the result cannot reach the
     caller", never "stop looking it up" — the receiver cannot verify a
     signature without the row's secret.
  5. NO CREDENTIAL IN THE AUDIT TRAIL. Webhook create/update/rotate/delete are
     recorded; the signing secret, the signature and the request body are not.

MUTATION-TESTED. Restoring the old ``404 "Webhook not found or inactive"`` on
the unknown-path branch makes ``test_unknown_and_bad_signature_are_one_response``
fail. Restoring the old 503 on the no-secret branch makes
``test_webhook_without_a_secret_is_the_same_failure`` fail.
"""
import hashlib
import hmac
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub Crypto so the workflow imports load without pycryptodome.
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

import audit  # noqa: E402
import resource_access  # noqa: E402
import webhook_management  # noqa: E402
from auth_dependency import AuthenticatedUser, require_org_member  # noqa: E402

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
USER_A = "cccccccc-cccc-cccc-cccc-cccccccccccc"
WF_A = "0f0f0f0f-0000-0000-0000-0000000000aa"
WEBHOOK_A = "0a0a0a0a-0000-0000-0000-000000000001"
WEBHOOK_NOSECRET = "0a0a0a0a-0000-0000-0000-000000000002"
WEBHOOK_INACTIVE = "0a0a0a0a-0000-0000-0000-000000000003"
WEBHOOK_B = "0a0a0a0a-0000-0000-0000-0000000000bb"

#: Deliberately descriptive, which is the point: the leak was not structural.
PATH_A = "stripe-customer-acme-foo"
PATH_NOSECRET = "legacy-inbound-no-secret"
PATH_INACTIVE = "retired-zendesk-hook"
PATH_UNKNOWN = "no-such-endpoint-anywhere"

#: A realistic signing secret. It must never appear in an audit row.
REAL_SECRET = "whsec_9f2b7c1d4e6a8b0c2d4e6f8a0b1c3d5e7f9a1b3c5d7e9f0a2b4c6d8e0f1a3b5c"
BODY = b'{"event":"invoice.paid","customer":"acme","amount":42000}'


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ─── A Postgres double that applies the filters a query declares ─────────────


class _Query:
    def __init__(self, db, table):
        self._db, self._table = db, table
        self._op, self._payload = "select", None
        self._eq, self._limit, self._single = [], None, False

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

    def order(self, *a, **k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def single(self):
        self._single = True
        return self

    maybe_single = single

    def _matches(self, row):
        return all(str(row.get(c)) == str(v) for c, v in self._eq)

    def execute(self):
        self._db.queries.append((self._table, self._op, list(self._eq)))
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
        return MagicMock(data=(out[0] if out else None) if self._single else out)


class _FakeDB:
    def __init__(self, rows):
        self.rows = {k: [dict(r) for r in v] for k, v in rows.items()}
        self.queries = []
        self._n = 0

    def next_uuid(self):
        self._n += 1
        return "99999999-9999-9999-9999-%012d" % self._n

    def table(self, name):
        return _Query(self, name)

    @property
    def audit_rows(self):
        return self.rows.setdefault(audit.AUDIT_TABLE, [])

    def audit_actions(self):
        return [r["action"] for r in self.audit_rows]

    def only_audit_row(self, action):
        rows = [r for r in self.audit_rows if r["action"] == action]
        assert len(rows) == 1, "expected one %r row, got %d (all: %r)" % (
            action, len(rows), self.audit_actions())
        return rows[0]

    def lookups_of(self, table):
        return [q for q in self.queries if q[0] == table and q[1] == "select"]


def _seed():
    return _FakeDB({
        "webhook_triggers": [
            {"id": WEBHOOK_A, "org_id": ORG_A, "workflow_id": WF_A,
             "name": "Stripe", "endpoint_path": PATH_A, "secret": REAL_SECRET,
             "payload_template": "{{body}}", "is_active": True, "trigger_count": 3},
            {"id": WEBHOOK_NOSECRET, "org_id": ORG_A, "workflow_id": WF_A,
             "name": "Legacy", "endpoint_path": PATH_NOSECRET, "secret": None,
             "payload_template": "{{body}}", "is_active": True, "trigger_count": 0},
            {"id": WEBHOOK_INACTIVE, "org_id": ORG_A, "workflow_id": WF_A,
             "name": "Retired", "endpoint_path": PATH_INACTIVE, "secret": REAL_SECRET,
             "payload_template": "{{body}}", "is_active": False, "trigger_count": 0},
            {"id": WEBHOOK_B, "org_id": ORG_B, "workflow_id": "wf-b",
             "name": "Theirs", "endpoint_path": "other-tenant-hook",
             "secret": REAL_SECRET, "payload_template": "{{body}}",
             "is_active": True, "trigger_count": 0},
        ],
        "workflows": [
            {"id": WF_A, "org_id": ORG_A, "graph_json": {"nodes": []}, "variables": {}},
        ],
        audit.AUDIT_TABLE: [],
    })


@pytest.fixture
def db():
    return _seed()


@pytest.fixture
def client(db):
    """The receiver, wired so the handler and `audit` share one database."""
    app = FastAPI()
    app.include_router(webhook_management.router, prefix="/api")
    app.dependency_overrides[require_org_member] = _member_of_org_a
    with patch.object(webhook_management, "supabase", db), \
            patch.object(resource_access, "supabase", db), \
            patch.object(audit, "supabase", db), \
            patch.object(webhook_management, "check_and_increment_usage"), \
            patch.object(webhook_management, "check_monthly_request_limit"), \
            patch.object(webhook_management, "increment_monthly_usage"):
        yield TestClient(app)
    app.dependency_overrides.clear()


def _member_of_org_a():
    user = AuthenticatedUser(user_id=USER_A, email="a@b.c", email_verified=True)
    user._verified_org_id = ORG_A
    user._org_role = "admin"
    return user


def _deliver(client, path, signature=None, body=BODY):
    headers = {"content-type": "application/json"}
    if signature is not None:
        headers["X-Webhook-Signature"] = signature
    return client.post("/api/webhooks/trigger/" + path, content=body, headers=headers)


_VOLATILE_HEADERS = {"date", "server", "x-request-id", "x-trace-id"}


def _shape(response):
    """Everything a caller can observe: status, body BYTES, and headers."""
    return (
        response.status_code,
        response.content,
        tuple(sorted(
            (k.lower(), v) for k, v in response.headers.items()
            if k.lower() not in _VOLATILE_HEADERS
        )),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pre-authentication uniformity
# ═══════════════════════════════════════════════════════════════════════════


def test_unknown_and_bad_signature_are_one_response(client):
    """THE test. An unknown path and a real path with a wrong signature are
    indistinguishable to an unauthenticated caller."""
    unknown = _deliver(client, PATH_UNKNOWN, signature="0" * 64)
    bad_sig = _deliver(client, PATH_A, signature="0" * 64)

    assert bad_sig.status_code == 401
    assert _shape(unknown) == _shape(bad_sig)


def test_missing_signature_is_the_same_response(client):
    """Omitting the header entirely must not be its own status code either."""
    unknown = _deliver(client, PATH_UNKNOWN)
    missing = _deliver(client, PATH_A)
    bad_sig = _deliver(client, PATH_A, signature="0" * 64)

    assert _shape(unknown) == _shape(missing) == _shape(bad_sig)


def test_all_pre_auth_failures_collapse_to_one_shape(client):
    """Every pre-auth outcome at once, including the two that used to be a 404
    and a 503. One distinct shape across the whole set, or the endpoint is still
    an existence oracle."""
    shapes = {
        "unknown path": _shape(_deliver(client, PATH_UNKNOWN, signature="0" * 64)),
        "unknown path, no signature": _shape(_deliver(client, PATH_UNKNOWN)),
        "known path, bad signature": _shape(_deliver(client, PATH_A, signature="0" * 64)),
        "known path, no signature": _shape(_deliver(client, PATH_A)),
        "known path, sha256= prefixed bad signature":
            _shape(_deliver(client, PATH_A, signature="sha256=" + "0" * 64)),
        "known path, no secret configured": _shape(_deliver(client, PATH_NOSECRET, signature="0" * 64)),
        "known path, inactive": _shape(_deliver(client, PATH_INACTIVE, signature="0" * 64)),
        "another tenant's path, bad signature":
            _shape(_deliver(client, "other-tenant-hook", signature="0" * 64)),
    }
    assert len(set(shapes.values())) == 1, {k: v[0] for k, v in shapes.items()}
    assert next(iter(shapes.values()))[0] == 401


def test_webhook_without_a_secret_is_the_same_failure(client):
    """The folded 503. A row with no signing secret can authenticate nobody, so
    answering it differently only told an anonymous caller that the path exists.
    It refuses exactly as before — it just no longer says so out loud."""
    no_secret = _deliver(client, PATH_NOSECRET, signature="0" * 64)
    unknown = _deliver(client, PATH_UNKNOWN, signature="0" * 64)

    assert no_secret.status_code == 401
    assert _shape(no_secret) == _shape(unknown)


def test_no_response_body_mentions_the_path_or_the_webhook(client):
    """The uniform failure must not leak through its detail string either."""
    for path in (PATH_A, PATH_NOSECRET, PATH_INACTIVE, PATH_UNKNOWN):
        blob = _deliver(client, path, signature="0" * 64).text
        assert path not in blob
        assert "not found" not in blob.lower()
        assert REAL_SECRET not in blob


def test_a_valid_signature_for_a_different_webhook_does_not_transfer(client):
    """A signature that is valid for org B's webhook is not valid on org A's
    path. Same secret in this fixture, so only the path differs — the check must
    not become 'any well-formed signature'."""
    sig = _sign(REAL_SECRET, BODY)
    with patch("workflow_runtime.execute_workflow") as ex:
        ex.return_value = {"final_output": "ok", "total_cost": 0.0, "total_latency": 1}
        # Sanity: this signature IS valid for a real webhook.
        assert _deliver(client, PATH_A, signature=sig).status_code == 200
    # ...and the same signature on a path that does not exist is still refused.
    assert _deliver(client, PATH_UNKNOWN, signature=sig).status_code == 401


# ═══════════════════════════════════════════════════════════════════════════
# 2. The lookup still happens
# ═══════════════════════════════════════════════════════════════════════════


def test_the_lookup_still_runs_for_an_unknown_path(client, db):
    """The fix must be 'the result cannot reach the caller', not 'stop looking'.
    A receiver that skipped the lookup could never verify a signature at all."""
    _deliver(client, PATH_UNKNOWN, signature="0" * 64)

    lookups = db.lookups_of("webhook_triggers")
    assert lookups, "the webhook_triggers lookup was removed"
    filters = dict(lookups[0][2])
    assert filters["endpoint_path"] == PATH_UNKNOWN
    assert str(filters["is_active"]) == "True"


def test_the_lookup_runs_identically_for_a_known_path(client, db):
    """Same query shape either way: the observable difference the fix removes is
    in the RESPONSE, not in what the server does."""
    _deliver(client, PATH_UNKNOWN, signature="0" * 64)
    unknown_filters = [dict(q[2]) for q in db.lookups_of("webhook_triggers")]
    db.queries.clear()

    _deliver(client, PATH_A, signature="0" * 64)
    known_filters = [dict(q[2]) for q in db.lookups_of("webhook_triggers")]

    assert len(unknown_filters) == len(known_filters) == 1
    assert set(unknown_filters[0]) == set(known_filters[0])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Post-authentication semantics are NOT flattened
# ═══════════════════════════════════════════════════════════════════════════


def test_a_valid_signature_executes_the_workflow(client, db):
    sig = _sign(REAL_SECRET, BODY)
    with patch("workflow_runtime.execute_workflow") as execute:
        execute.return_value = {
            "final_output": "processed", "total_cost": 0.01, "total_latency": 120,
        }
        response = _deliver(client, PATH_A, signature=sig)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["output"] == "processed"
    assert execute.call_count == 1
    # The org came from the row the server read, never from the request.
    assert execute.call_args.kwargs["org_id"] == ORG_A


def test_the_sha256_prefix_is_still_accepted(client):
    sig = "sha256=" + _sign(REAL_SECRET, BODY)
    with patch("workflow_runtime.execute_workflow") as execute:
        execute.return_value = {"final_output": "ok", "total_cost": 0, "total_latency": 1}
        assert _deliver(client, PATH_A, signature=sig).status_code == 200


def test_a_handler_error_after_valid_auth_is_still_a_5xx(client):
    """Retry behaviour depends on this. Flattening it would leave a sender
    unable to tell 'your handler broke, retry' from 'rejected, stop'."""
    sig = _sign(REAL_SECRET, BODY)
    with patch("workflow_runtime.execute_workflow", side_effect=RuntimeError("provider down")):
        response = _deliver(client, PATH_A, signature=sig)

    assert response.status_code == 500
    assert response.status_code != webhook_management.WEBHOOK_AUTH_FAILED_STATUS


def test_a_missing_workflow_after_valid_auth_is_not_the_auth_failure(client, db):
    """Post-auth 404s stay 404s — the uniformity rule stops at authentication."""
    db.rows["workflows"] = []
    sig = _sign(REAL_SECRET, BODY)
    response = _deliver(client, PATH_A, signature=sig)

    assert response.status_code == 404
    assert _shape(response) != _shape(_deliver(client, PATH_UNKNOWN, signature=sig))


def test_the_trigger_count_is_updated_only_after_authentication(client, db):
    def _count():
        return [r for r in db.rows["webhook_triggers"] if r["id"] == WEBHOOK_A][0]["trigger_count"]

    _deliver(client, PATH_A, signature="0" * 64)
    assert _count() == 3, "a refused delivery must not touch the row"

    with patch("workflow_runtime.execute_workflow") as execute:
        execute.return_value = {"final_output": "ok", "total_cost": 0, "total_latency": 1}
        _deliver(client, PATH_A, signature=_sign(REAL_SECRET, BODY))
    assert _count() == 4


def test_a_refused_delivery_never_executes_a_workflow(client):
    with patch("workflow_runtime.execute_workflow") as execute:
        for path in (PATH_UNKNOWN, PATH_A, PATH_NOSECRET, PATH_INACTIVE):
            _deliver(client, path, signature="0" * 64)
            _deliver(client, path)
        assert execute.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. The audit trail: written, and carrying nothing it should not
# ═══════════════════════════════════════════════════════════════════════════


def _create(client, name="Stripe prod", workflow_id=WF_A, secret=None):
    payload = {"org_id": ORG_A, "workflow_id": workflow_id, "name": name}
    if secret is not None:
        payload["secret"] = secret
    return client.post("/api/webhooks", json=payload)


def test_creating_a_webhook_is_recorded(client, db):
    response = _create(client)
    assert response.status_code == 200

    row = db.only_audit_row(audit.WEBHOOK_CREATED)
    assert row["org_id"] == ORG_A
    assert row["actor_id"] == USER_A
    assert row["resource_type"] == audit.RESOURCE_WEBHOOK_TRIGGER
    assert row["metadata"]["outcome"] == "success"
    assert row["metadata"]["workflow_id"] == WF_A


def test_pointing_a_webhook_at_another_tenants_workflow_is_refused_and_recorded(client, db):
    response = _create(client, workflow_id="wf-b")
    assert response.status_code == 404

    row = db.only_audit_row(audit.WEBHOOK_CREATE_REFUSED)
    assert row["metadata"]["outcome"] == "refused"
    assert row["metadata"]["reason_code"] == audit.REASON_NOT_FOUND
    assert audit.WEBHOOK_CREATED not in db.audit_actions()


def test_rotating_the_secret_is_its_own_recorded_action(client, db):
    response = client.put(
        "/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_A),
        json={"org_id": ORG_A, "secret": REAL_SECRET},
    )
    assert response.status_code == 200

    db.only_audit_row(audit.WEBHOOK_UPDATED)
    rotated = db.only_audit_row(audit.WEBHOOK_SECRET_ROTATED)
    assert rotated["resource_id"] == WEBHOOK_A


def test_deactivating_and_deleting_are_recorded(client, db):
    client.put(
        "/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_A),
        json={"org_id": ORG_A, "is_active": False},
    )
    assert db.only_audit_row(audit.WEBHOOK_UPDATED)["metadata"]["new_status"] == "inactive"

    assert client.delete("/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_A)).status_code == 200
    assert db.only_audit_row(audit.WEBHOOK_DELETED)["resource_id"] == WEBHOOK_A


def test_touching_another_tenants_webhook_is_refused_and_recorded(client, db):
    """WEBHOOK_B belongs to ORG_B; the caller is verified into ORG_A."""
    assert client.put(
        "/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_B), json={"org_id": ORG_A, "name": "mine now"},
    ).status_code == 404
    assert client.delete("/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_B)).status_code == 404

    for action in (audit.WEBHOOK_UPDATE_REFUSED, audit.WEBHOOK_DELETE_REFUSED):
        row = db.only_audit_row(action)
        assert row["org_id"] == ORG_A
        assert row["metadata"]["outcome"] == "refused"
    # And the foreign row is untouched.
    assert [r for r in db.rows["webhook_triggers"] if r["id"] == WEBHOOK_B][0]["name"] == "Theirs"


def test_no_secret_signature_or_body_reaches_the_audit_log(client, db):
    """Asserted over the ENTIRE serialised table, not field by field, so a
    future metadata key cannot smuggle a credential past a narrower check."""
    _create(client, secret=REAL_SECRET)
    client.put(
        "/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_A),
        json={"org_id": ORG_A, "secret": REAL_SECRET, "payload_template": BODY.decode()},
    )
    client.delete("/api/webhooks/%s/%s" % (ORG_A, WEBHOOK_A))
    _deliver(client, PATH_A, signature=_sign(REAL_SECRET, BODY))
    _deliver(client, PATH_A, signature="deadbeef" * 8)

    assert db.audit_rows, "nothing was recorded at all — the test proves nothing"
    blob = json.dumps(db.audit_rows, default=str)
    for forbidden in (REAL_SECRET, "whsec_", "deadbeef", "invoice.paid", "acme", "42000"):
        assert forbidden not in blob, "audit_log carries %r" % forbidden
    # Nor the endpoint path, which is half of the receiver's credential.
    assert PATH_A not in blob


def test_the_unauthenticated_receiver_writes_no_audit_rows(client, db):
    """The trigger endpoint has no principal and no verified org, and it is
    reachable by anyone. Recording per-delivery rows there would let a stranger
    write into a tenant's audit trail at will."""
    for path in (PATH_UNKNOWN, PATH_A, PATH_NOSECRET):
        _deliver(client, path, signature="0" * 64)
    assert db.audit_rows == []


def test_every_webhook_action_is_in_the_closed_vocabulary():
    for action in (
        audit.WEBHOOK_CREATED, audit.WEBHOOK_UPDATED, audit.WEBHOOK_SECRET_ROTATED,
        audit.WEBHOOK_DELETED, audit.WEBHOOK_CREATE_REFUSED,
        audit.WEBHOOK_UPDATE_REFUSED, audit.WEBHOOK_DELETE_REFUSED,
    ):
        assert action in audit.ACTIONS
    for refusal in (
        audit.WEBHOOK_CREATE_REFUSED, audit.WEBHOOK_UPDATE_REFUSED,
        audit.WEBHOOK_DELETE_REFUSED,
    ):
        assert refusal in audit.REFUSAL_ACTIONS


# ═══════════════════════════════════════════════════════════════════════════
# 5. The synthetic secret
# ═══════════════════════════════════════════════════════════════════════════


def test_the_synthetic_secret_is_unguessable_and_not_a_real_one(client, db):
    """It stands in for a missing secret so the same hmac path runs. If it were
    predictable, an unknown path would become an EXECUTABLE endpoint."""
    synthetic = webhook_management._SYNTHETIC_WEBHOOK_SECRET
    assert len(synthetic) >= 64
    assert synthetic != REAL_SECRET
    assert synthetic not in {r.get("secret") for r in db.rows["webhook_triggers"]}

    # Even signing correctly WITH it cannot make an unknown path execute.
    sig = _sign(synthetic, BODY)
    with patch("workflow_runtime.execute_workflow") as execute:
        assert _deliver(client, PATH_UNKNOWN, signature=sig).status_code == 401
        assert _deliver(client, PATH_NOSECRET, signature=sig).status_code == 401
        assert execute.call_count == 0
