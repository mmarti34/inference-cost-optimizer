"""
The confused-deputy bypass in require_org_member.

The guard resolved org_id by taking the FIRST source that had a value, in the
order path -> query -> body -> header. Handlers read `payload.org_id` — the
BODY. So the two could name different orgs, and the precedence made that
exploitable:

    POST /api/cursor-tokens?org_id=<attacker's own org>
    {"org_id": "<victim org>", ...}

The guard verified the ATTACKER'S org and passed; the handler acted on the
VICTIM'S. 21 handlers read `<payload>.org_id` under this guard, and two of them
mint credentials (cursor_tokens.create_cursor_token, cursor_deploy.
deploy_from_parsed), turning a cross-tenant write into a live credential for
another tenant.

The guard no longer picks a winner between sources. It collects them all and
refuses the request if they disagree, so "the org the guard verified" and "the
org the handler reads" are the same value by construction — whichever source a
given handler happens to read.
"""
from __future__ import annotations

from typing import Optional

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

import auth_dependency
from auth_dependency import AuthenticatedUser

ATTACKER_ORG = "11111111-1111-1111-1111-111111111111"
VICTIM_ORG = "22222222-2222-2222-2222-222222222222"


class Payload(BaseModel):
    """Module level on purpose: FastAPI resolves body models via the function's
    globals, so a class defined inside the fixture is treated as a query param
    and every request 422s before the guard is ever exercised."""

    org_id: str


@pytest.fixture
def app(monkeypatch):
    """
    A handler with the exact shape of the 21 real ones: guarded by
    require_org_member, acting on payload.org_id.
    """
    # Membership: the attacker belongs to ATTACKER_ORG and nothing else.
    class _Result:
        def __init__(self, data):
            self.data = data

    class _Chain:
        def __init__(self, org):
            self._org = org

        def select(self, *a, **k):
            return self

        def eq(self, col, val):
            if col == "org_id":
                self._org = val
            return self

        def limit(self, *a, **k):
            return self

        def execute(self):
            ok = self._org == ATTACKER_ORG
            return _Result([{"role": "admin", "status": "active"}] if ok else [])

    class _DB:
        def table(self, name):
            return _Chain(None)

    monkeypatch.setattr(auth_dependency, "supabase", _DB())
    monkeypatch.setattr(
        auth_dependency,
        "require_auth",
        lambda *a, **k: AuthenticatedUser(user_id="attacker", email="a@b.c"),
    )

    application = FastAPI()

    async def _guard(
        request: Request,
        x_org_id: Optional[str] = Header(None),
    ) -> AuthenticatedUser:
        user = AuthenticatedUser(user_id="attacker", email="a@b.c")
        return await auth_dependency.require_org_member(
            request=request, auth_user=user, x_org_id=x_org_id
        )

    @application.post("/act")
    async def act(payload: Payload, user: AuthenticatedUser = Depends(_guard)):
        # Deliberately the UNCONVERTED shape: acts on the body's org_id.
        return {
            "guard_verified": auth_dependency.verified_org_id(user),
            "handler_acts_on": payload.org_id,
        }

    return application


def test_the_bypass_is_refused(app):
    """query=attacker's own org, body=victim's org. This was the exploit."""
    r = TestClient(app).post(f"/act?org_id={ATTACKER_ORG}", json={"org_id": VICTIM_ORG})
    assert r.status_code == 400
    assert r.json()["detail"] == "Conflicting org_id values in request."


def test_the_header_variant_is_refused_too(app):
    r = TestClient(app).post(
        "/act", json={"org_id": VICTIM_ORG}, headers={"X-Org-Id": ATTACKER_ORG}
    )
    assert r.status_code == 400


def test_a_conflict_never_reaches_the_handler(app):
    """
    The point is not the status code — it is that the handler never runs on an
    org the guard did not verify.
    """
    r = TestClient(app).post(f"/act?org_id={ATTACKER_ORG}", json={"org_id": VICTIM_ORG})
    assert "handler_acts_on" not in r.text
    assert VICTIM_ORG not in r.text


def test_an_ordinary_request_still_works(app):
    """One org_id, in the body. The overwhelmingly common shape."""
    r = TestClient(app).post("/act", json={"org_id": ATTACKER_ORG})
    assert r.status_code == 200
    body = r.json()
    assert body["guard_verified"] == body["handler_acts_on"] == ATTACKER_ORG


def test_the_same_org_in_two_places_is_not_a_conflict(app):
    """Agreeing duplicates are legitimate and must not be broken."""
    r = TestClient(app).post(f"/act?org_id={ATTACKER_ORG}", json={"org_id": ATTACKER_ORG})
    assert r.status_code == 200
    assert r.json()["guard_verified"] == ATTACKER_ORG


def test_membership_is_still_enforced(app):
    """The conflict check must not have replaced the membership check."""
    r = TestClient(app).post("/act", json={"org_id": VICTIM_ORG})
    assert r.status_code == 403


def test_guard_and_handler_can_never_disagree(app):
    """
    The invariant, stated directly: for every request the guard admits, the org
    it verified equals the org the handler reads.
    """
    client = TestClient(app)
    for params, body, header in [
        ("", {"org_id": ATTACKER_ORG}, None),
        (f"?org_id={ATTACKER_ORG}", {"org_id": ATTACKER_ORG}, None),
        ("", {"org_id": ATTACKER_ORG}, ATTACKER_ORG),
    ]:
        headers = {"X-Org-Id": header} if header else {}
        r = client.post(f"/act{params}", json=body, headers=headers)
        if r.status_code == 200:
            payload = r.json()
            assert payload["guard_verified"] == payload["handler_acts_on"]
