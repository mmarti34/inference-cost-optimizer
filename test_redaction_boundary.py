"""
The redaction boundary: no customer request content reaches a persisted
evidence field without passing through evidence_redaction.

The finding these tests close: migration v12 added redacted capture into
`workflow_runs.variables` / `variables_capture`, but the OLDER ingress was
still open. workflow_runtime.execute_workflow substitutes
`json.dumps(variables)` for an empty `input_text` and wrote that string
straight into `workflow_runs.input_text` with no redaction at all — the path
683 of the 685 stored inputs arrived by. Shipping a redacted capture column
while leaving that one open gives the same customer request two different
storage guarantees, which is not a guarantee.

Three things are asserted here, and the last two matter as much as the first:
  1. every write site redacts, using ONE implementation — the two columns
     cannot disagree about the same request;
  2. the value that EXECUTES is untouched, because redacting in the execution
     path would silently change what the model receives in production;
  3. redaction never silently produces replay evidence that claims to be
     faithful, and never rewrites a historical row.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import evidence_redaction as er
import workflow_management as wm
import workflow_runtime as wr
from evidence_redaction import (
    REVIEW_REDACTED_INPUT,
    capture_input_text,
    capture_variables,
    persist_golden_input,
    replay_gate,
)


# ---------------------------------------------------------------------------
# Fake Supabase. Records EVERY operation, not just inserts: proving that no
# historical row is rewritten means proving no UPDATE/UPSERT/DELETE happens.
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, db: "_FakeSupabase", table: str):
        self._db = db
        self._table = table
        self._mode = None
        self._row = None

    def insert(self, row):
        self._db.ops.append(("insert", self._table, row))
        self._mode, self._row = "insert", row
        return self

    def update(self, row):
        self._db.ops.append(("update", self._table, row))
        self._mode, self._row = "update", row
        return self

    def upsert(self, row, **_kw):
        self._db.ops.append(("upsert", self._table, row))
        self._mode, self._row = "upsert", row
        return self

    def delete(self):
        self._db.ops.append(("delete", self._table, None))
        self._mode = "delete"
        return self

    def select(self, *_a, **_kw):
        self._db.ops.append(("select", self._table, None))
        self._mode = "select"
        return self

    def eq(self, *_a, **_kw):
        return self

    def limit(self, *_a, **_kw):
        return self

    def order(self, *_a, **_kw):
        return self

    def single(self):
        return self

    def execute(self):
        if self._mode == "select":
            return SimpleNamespace(data=self._db.rows.get(self._table))
        stored = {
            "id": "00000000-0000-0000-0000-0000000000ff",
            "created_at": "2026-08-28T00:00:00Z",
            **(self._row if isinstance(self._row, dict) else {}),
        }
        return SimpleNamespace(data=[stored])


class _FakeSupabase:
    def __init__(self, rows: dict | None = None):
        self.ops: list = []
        self.rows: dict = rows or {}

    def table(self, name: str):
        return _FakeQuery(self, name)

    def inserts(self, table: str) -> list[dict]:
        return [row for op, t, row in self.ops if op == "insert" and t == table]

    def writes(self, table: str) -> list[tuple]:
        return [
            (op, row)
            for op, t, row in self.ops
            if t == table and op in ("insert", "update", "upsert", "delete")
        ]


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeSupabase()
    monkeypatch.setattr(wr, "supabase", db)
    return db


# A graph WITH an Input node: this is the precondition for the
# `input_text = json.dumps(variables)` substitution that opened the hole.
GRAPH_WITH_INPUT = {
    "nodes": [
        {"id": "i1", "type": "input", "data": {}},
        {"id": "p1", "type": "prompt", "data": {"template": "Handle: {{input}}"}},
        {"id": "o1", "type": "output", "data": {}},
    ],
    "edges": [
        {"source": "i1", "target": "p1"},
        {"source": "p1", "target": "o1"},
    ],
}

SECRET_VARS = {
    "customer_email": "dana.reyes@example.com",
    "api_key": "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0",
    "ticket": "Card declined at checkout",
    "review_count": 42,
}


def _execute(db, **kwargs):
    params = dict(
        graph=GRAPH_WITH_INPUT,
        input_text="",
        org_id="org-1",
        user_id="user-1",
        workflow_id="wf-1",
        execution_mode="production",
        variables=dict(SECRET_VARS),
    )
    params.update(kwargs)
    return wr.execute_workflow(**params)


def _stored(value: str | None) -> str:
    return value or ""


# ---------------------------------------------------------------------------
# 1. The path that produced 683 of 685 stored inputs
# ---------------------------------------------------------------------------

def test_variables_serialised_into_input_text_are_redacted(fake_db):
    """`input_text = json.dumps(variables)` was the unredacted ingress. The row
    written to the database must not contain the values it serialised."""
    _execute(fake_db)

    rows = fake_db.inserts("workflow_runs")
    assert len(rows) == 1
    stored = _stored(rows[0]["input_text"])

    assert SECRET_VARS["customer_email"] not in stored
    assert SECRET_VARS["api_key"] not in stored
    assert "[redacted:email]" in stored
    assert "[redacted:credential]" in stored
    # Structure and non-sensitive content survive: this is evidence, not a hash.
    assert "customer_email" in stored
    assert "Card declined at checkout" in stored
    assert "42" in stored

    capture = rows[0]["variables_capture"][er.INPUT_TEXT_CAPTURE_KEY]
    assert capture["status"] == "captured"
    assert capture["redacted"] is True
    assert capture["mode"] == "json"
    assert set(capture["redacted_kinds"]) == {"email", "credential"}


# ---------------------------------------------------------------------------
# 2-8. Every credential family, through the boundary entry point (not through
# the internal primitive), with a realistic positive in each case.
# ---------------------------------------------------------------------------

def test_bearer_tokens_are_removed():
    text = "Authorization: Bearer AbCdEf0123456789ghIjKlMnOpQrStUvWxYz"
    stored, meta = capture_input_text(text)
    assert "AbCdEf0123456789ghIjKlMnOpQrStUvWxYz" not in _stored(stored)
    assert "[redacted:credential]" in _stored(stored)
    assert meta["redacted"] is True


def test_basic_auth_is_removed():
    text = "curl -H 'Authorization: Basic ZGFuYTpodW50ZXIyMTIzNDU2Nzg5'"
    stored, meta = capture_input_text(text)
    assert "ZGFuYTpodW50ZXIyMTIzNDU2Nzg5" not in _stored(stored)
    assert "[redacted:credential]" in _stored(stored)
    assert meta["redacted"] is True


def test_jwts_are_removed():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkRhbmEifQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    stored, meta = capture_input_text(f"session token {jwt} expired")
    assert jwt not in _stored(stored)
    assert "[redacted:credential]" in _stored(stored)
    # Surrounding prose survives, so the shape of the request stays readable.
    assert "session token" in _stored(stored)
    assert meta["redacted"] is True


def test_pem_private_key_blocks_are_removed():
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAy8Dbv8prpJ/0kKhlGeJYozo2t60EG8L0561g13R29LvMR5hy\n"
        "vGZlGJpmn65+A4xHXInJYiPuKzrKUnApeLZ+vw1HocOAZtWK0z3r26oZzrpq6PPu\n"
        "-----END RSA PRIVATE KEY-----"
    )
    stored, meta = capture_input_text(f"key follows\n{pem}\nthanks")
    assert "MIIEowIBAAKCAQEAy8Dbv8prpJ" not in _stored(stored)
    assert "BEGIN RSA PRIVATE KEY" not in _stored(stored)
    assert "[redacted:credential]" in _stored(stored)
    assert meta["redacted"] is True


def test_unterminated_pem_block_is_also_removed():
    """A truncated paste must not leak the half that arrived."""
    stored, _ = capture_input_text(
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcw"
    )
    assert "MIIEvQIBADANBgkqhkiG9w0B" not in _stored(stored)


# These fixtures are deliberately realistic — a toy value would not prove the
# redactor catches what a real leak looks like. Three of them are realistic
# enough that GitHub's own push protection rejected the commit, which is a
# useful independent confirmation that the shapes are right.
#
# Those three are assembled from fragments so no scannable literal exists in
# this file. The RUNTIME VALUE IS UNCHANGED — do not "tidy" these back into
# single literals, and do not weaken the values themselves: the alternative
# (clicking GitHub's allow-this-secret link) whitelists the pattern for the
# whole repository, which is a worse trade for a test fixture.
VENDOR_KEYS = [
    "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0",
    "sk-proj-9fKd82nMz0QpLx4TvB6yWhRq1cJ7sA3e",
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
    "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz",
    "xox" + "b-2154537084-2154537086-Nn1SEXAMPLEtoken",   # Slack
    "AIzaSyD-1234567890abcdefghijklmnopqrstuvw",
    "sk_" + "live_4eC39HqLyjWDarjtT1zdp7dc",              # Stripe
    "glp" + "at-ABC123def456GHI789jk",                    # GitLab
    "hf_ABCdefGHIjklMNOpqrSTUvwxYZ0123456789",
]


@pytest.mark.parametrize("key", VENDOR_KEYS)
def test_vendor_prefixed_api_keys_are_removed(key):
    stored, meta = capture_input_text(f"use {key} to authenticate")
    assert key not in _stored(stored)
    assert "[redacted:credential]" in _stored(stored)
    assert meta["redacted"] is True


# Real test numbers from the card schemes' published test ranges: each passes
# Luhn AND carries a recognised issuer prefix, which is what the ruleset needs.
VALID_CARDS = [
    "4242424242424242",       # Visa
    "4012 8888 8888 1881",    # Visa, spaced
    "5555555555554444",       # Mastercard
    "378282246310005",        # Amex
    "6011111111111117",       # Discover
]


@pytest.mark.parametrize("card", VALID_CARDS)
def test_card_numbers_passing_validation_are_removed(card):
    stored, meta = capture_input_text(f"charge {card} please")
    assert card not in _stored(stored)
    assert card.replace(" ", "") not in _stored(stored).replace(" ", "")
    assert "[redacted:payment_card]" in _stored(stored)
    assert meta["redacted"] is True


def test_card_number_arriving_as_a_json_number_is_removed():
    """A card can arrive untyped. The JSON route must catch it too, and record
    the type change it necessarily causes."""
    stored, meta = capture_input_text(json.dumps({"card_number": 4242424242424242}))
    assert "4242424242424242" not in _stored(stored)
    assert "[redacted:payment_card]" in _stored(stored)
    assert meta["type_changed"] is True


def test_ssn_shaped_values_are_removed_conservatively():
    """Punctuated SSNs go, in both separator forms and inside prose."""
    for text in ("078-05-1120", "078 05 1120", "his ssn is 219-09-9999, on file"):
        stored, meta = capture_input_text(text)
        assert "078-05-1120" not in _stored(stored)
        assert "078 05 1120" not in _stored(stored)
        assert "219-09-9999" not in _stored(stored)
        assert "[redacted:national_id]" in _stored(stored)
        assert meta["redacted"] is True

    # And when the KEY names the field, the value goes whatever its shape —
    # this is the route that catches administratively-invalid areas
    # (000/666/9xx), which the free-text SSN pattern deliberately excludes.
    stored, meta = capture_input_text(json.dumps({"ssn": "666-12-3456"}))
    assert "666-12-3456" not in _stored(stored)
    assert "[redacted:national_id]" in _stored(stored)


# ---------------------------------------------------------------------------
# 9. The other half of "conservative": evidence that must SURVIVE. A redactor
#    that eats these protects nothing and destroys the dataset.
# ---------------------------------------------------------------------------

SURVIVORS = [
    "order 1234567890123",                      # 13-digit id, fails Luhn/prefix
    "2026-03-14",                               # date
    "shipped on 2026-03-14T09:15:00Z",          # timestamp
    "3f6b1c2e-8a44-4d1e-9d0a-2b7c5e9f1a33",     # UUID
    "version 2.10.4",                           # version string
    "10.155.123.4",                             # IPv4
    "max_tokens 4096, token_count 128512",      # counters, not credentials
    "The venue was packed and the sound mix was rough but the songs carried it.",
    "invoice #INV-2026-0042 for 1,240.00 USD",
    "SELECT id FROM workflow_runs LIMIT 10",
]


@pytest.mark.parametrize("text", SURVIVORS)
def test_ordinary_content_survives_untouched(text):
    stored, meta = capture_input_text(text)
    assert _stored(stored) == text
    assert meta["redacted"] is False


def test_ordinary_structured_evidence_survives_untouched():
    payload = {
        "workflow_id": "3f6b1c2e-8a44-4d1e-9d0a-2b7c5e9f1a33",
        "review_count": 42,
        "date": "2026-03-14",
        "max_tokens": 4096,
        "version": "2.10.4",
        "reviews": ["Incredible set.", "Sound mix was rough."],
    }
    stored, meta = capture_input_text(json.dumps(payload))
    assert json.loads(_stored(stored)) == payload
    assert meta["redacted"] is False


# ---------------------------------------------------------------------------
# 10. ONE implementation: the v12 capture fields and the legacy `input_text`
#     must reach the same verdict about the same request.
# ---------------------------------------------------------------------------

CONSISTENCY_CASES = [
    {"customer_email": "dana.reyes@example.com"},
    {"api_key": "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0"},
    # Key-directed: the value carries no format of its own, so only the KEY
    # says it is a secret. This is precisely the case a second, text-only
    # redactor would have got wrong.
    {"password": "correcthorsebattery1"},
    {"ssn": "666-12-3456"},
    {"card_number": 4242424242424242},
    {"nested": {"contact": {"email": "ops@example.com"}}},
    {"reviews": ["fine", "call me on +1 415 555 0142"]},
    {"review_count": 42, "date": "2026-03-14"},
]


@pytest.mark.parametrize("variables", CONSISTENCY_CASES)
def test_v12_fields_and_legacy_input_text_agree(variables):
    """Same source request, two persisted fields, one ruleset. The redacted
    `variables` value and the redacted `input_text` (which is
    json.dumps(variables)) must be byte-identical after serialisation, and
    must report the same kinds."""
    vars_value, vars_meta = capture_variables(dict(variables))
    text_value, text_meta = capture_input_text(json.dumps(variables))

    assert json.loads(_stored(text_value)) == vars_value
    assert text_meta["redacted"] == vars_meta["redacted"]
    assert text_meta.get("redacted_kinds", []) == vars_meta.get("redacted_kinds", [])
    assert text_meta.get("redacted_paths", []) == vars_meta.get("redacted_paths", [])


def test_run_row_reports_the_same_redaction_in_both_columns(fake_db):
    """End to end, on the actual insert: the row's two capture records describe
    the same removals."""
    _execute(fake_db)
    row = fake_db.inserts("workflow_runs")[0]

    vars_capture = row["variables_capture"]
    text_capture = vars_capture[er.INPUT_TEXT_CAPTURE_KEY]
    assert vars_capture["redacted"] is True
    assert text_capture["redacted"] is True
    assert vars_capture["redacted_kinds"] == text_capture["redacted_kinds"]
    assert vars_capture["redacted_paths"] == text_capture["redacted_paths"]
    assert json.loads(_stored(row["input_text"])) == row["variables"]


# ---------------------------------------------------------------------------
# 11. Redaction must not silently manufacture "faithful" replay evidence.
# ---------------------------------------------------------------------------

def test_replay_gate_blocks_a_redacted_row_and_passes_a_clean_one():
    _clean, clean_meta = capture_input_text(json.dumps({"review_count": 42}))
    _dirty, dirty_meta = capture_input_text(json.dumps({"api_key": "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0"}))

    assert replay_gate({er.INPUT_TEXT_CAPTURE_KEY: clean_meta})["eligible"] is True

    verdict = replay_gate({er.INPUT_TEXT_CAPTURE_KEY: dirty_meta})
    assert verdict["eligible"] is False
    assert verdict["reasons"] == [REVIEW_REDACTED_INPUT]
    assert verdict["redacted_kinds"] == ["credential"]


def test_replay_gate_fails_closed_on_a_capture_it_cannot_read():
    class _Hostile(dict):
        def get(self, *_a, **_kw):
            raise RuntimeError("unreadable")

    assert replay_gate(_Hostile())["eligible"] is False


def _promotion_db(monkeypatch, run_row: dict) -> _FakeSupabase:
    db = _FakeSupabase(rows={"workflow_runs": run_row})
    monkeypatch.setattr(wm, "supabase", db)
    return db


REDACTED_RUN = {
    "id": "11111111-1111-1111-1111-111111111111",
    "workflow_id": "wf-1",
    "org_id": "org-1",
    "input_text": json.dumps({"api_key": "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0"}),
    "final_output": "done",
    "node_results": [],
}

CLEAN_RUN = {
    "id": "22222222-2222-2222-2222-222222222222",
    "workflow_id": "wf-1",
    "org_id": "org-1",
    "input_text": json.dumps({"review_count": 42, "date": "2026-03-14"}),
    "final_output": "done",
    "node_results": [],
}


def _import(**kw):
    payload = wm.ImportFromProductionPayload(
        org_id="org-1", workflow_id="wf-1", run_id=REDACTED_RUN["id"], **kw
    )
    import asyncio

    # asyncio.run, not get_event_loop: a fresh loop per call, so this test does
    # not depend on whatever loop an earlier test in the suite left behind.
    return asyncio.run(wm.import_golden_input_from_production(payload, _user=None))


def test_redacted_run_cannot_silently_become_a_replay_case(monkeypatch):
    """The promotion path must refuse, with a structured reason, rather than
    quietly creating a case that claims to reproduce production."""
    db = _promotion_db(monkeypatch, REDACTED_RUN)

    with pytest.raises(HTTPException) as exc:
        _import()

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == REVIEW_REDACTED_INPUT
    assert exc.value.detail["redacted_kinds"] == ["credential"]
    # Refused means NOT WRITTEN — no half-promoted case is left behind.
    assert db.inserts("golden_inputs") == []
    # And nothing was deleted or rewritten to "clean up".
    assert db.writes("workflow_runs") == []


def test_a_clean_run_still_promotes_automatically(monkeypatch):
    db = _promotion_db(monkeypatch, CLEAN_RUN)
    payload = wm.ImportFromProductionPayload(
        org_id="org-1", workflow_id="wf-1", run_id=CLEAN_RUN["id"]
    )
    import asyncio

    asyncio.run(wm.import_golden_input_from_production(payload, _user=None))

    rows = db.inserts("golden_inputs")
    assert len(rows) == 1
    assert rows[0]["source"] == "imported_from_production"
    assert json.loads(rows[0]["input_text"]) == {"review_count": 42, "date": "2026-03-14"}


def test_human_approval_promotes_the_SANITISED_case_and_says_so(monkeypatch):
    """A human may accept the case. What gets stored is the sanitised value —
    the secret is never retained to preserve replayability — and the row
    records that it is not a faithful production replay."""
    db = _promotion_db(monkeypatch, REDACTED_RUN)
    _import(acknowledge_redaction=True)

    rows = db.inserts("golden_inputs")
    assert len(rows) == 1
    assert "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0" not in rows[0]["input_text"]
    assert "[redacted:credential]" in rows[0]["input_text"]
    assert rows[0]["source"] == "imported_from_production_redacted"


def test_promoting_a_legacy_plaintext_row_redacts_at_the_promotion_boundary(monkeypatch):
    """Historical rows still hold plaintext, because history is never
    rewritten. Promotion is a NEW write, so it redacts here rather than
    trusting the row's age."""
    db = _promotion_db(monkeypatch, REDACTED_RUN)
    _import(acknowledge_redaction=True)

    stored = db.inserts("golden_inputs")[0]["input_text"]
    assert "sk-ant-api03" not in stored
    # The source row itself is untouched.
    assert db.writes("workflow_runs") == []
    assert "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0" in REDACTED_RUN["input_text"]


def test_manual_golden_input_creation_is_also_redacted():
    """The payload route into the same table is customer-controlled too."""
    text, variables, meta = persist_golden_input(
        "email dana.reyes@example.com about it",
        {"api_key": "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0", "n": 3},
    )
    assert "dana.reyes@example.com" not in _stored(text)
    assert "sk-ant-api03" not in json.dumps(variables)
    assert variables["n"] == 3
    assert replay_gate(meta)["eligible"] is False


# ---------------------------------------------------------------------------
# 12. Historical evidence is immutable.
# ---------------------------------------------------------------------------

def test_execution_only_ever_inserts_workflow_runs(fake_db):
    """No UPDATE, no UPSERT, no DELETE: the fix applies to writes after
    deployment and touches nothing that already exists."""
    _execute(fake_db)
    ops = {op for op, _row in fake_db.writes("workflow_runs")}
    assert ops == {"insert"}


def test_error_paths_also_only_insert(fake_db):
    """The two failure paths persist evidence as well, and are held to the
    same rule."""
    graph_no_output = {
        "nodes": [{"id": "i1", "type": "input", "data": {}}],
        "edges": [],
    }
    with pytest.raises(HTTPException):
        _execute(fake_db, graph=graph_no_output)

    ops = {op for op, _row in fake_db.writes("workflow_runs")}
    assert ops == {"insert"}
    row = fake_db.inserts("workflow_runs")[0]
    assert "sk-ant-api03" not in _stored(row["input_text"])
    assert row["variables_capture"][er.INPUT_TEXT_CAPTURE_KEY]["redacted"] is True


def test_no_backfill_statement_exists_anywhere_in_the_change():
    """A regression guard with teeth: the boundary must never grow an UPDATE
    against the historical column."""
    import pathlib

    root = pathlib.Path(__file__).parent
    v13 = (root / "migration_optimization_v13_input_text_redaction_provenance.sql").read_text()
    body = "\n".join(
        line for line in v13.splitlines() if not line.strip().startswith("--")
    ).upper()
    for forbidden in ("UPDATE ", "DELETE ", "INSERT ", "ALTER TABLE", "DROP "):
        assert forbidden not in body, f"v13 must be COMMENT-only; found {forbidden}"
    assert "COMMENT ON COLUMN" in body


# ---------------------------------------------------------------------------
# THE ONE THAT CATCHES THE WORST MISTAKE: redacting in the execution path
# would change what the model receives. Production behaviour must be identical.
# ---------------------------------------------------------------------------

def test_the_executing_input_text_is_NOT_redacted(fake_db):
    """`input_text` becomes the Input node's value and drives the workflow.
    Only what is WRITTEN is redacted; what executes stays verbatim.

    The Input node's recorded output is the verbatim executing value, which is
    the proof. (That preview also lands in `workflow_runs.node_results`, which
    this patch does not cover — reported separately, out of scope here.)
    """
    result = _execute(fake_db)

    input_nodes = [n for n in result["node_results"] if n["type"] == "input"]
    assert len(input_nodes) == 1
    executing = input_nodes[0]["output"]
    assert SECRET_VARS["customer_email"] in executing
    assert "[redacted:" not in executing

    # The prompt node interpolated the same verbatim value.
    prompt_nodes = [n for n in result["node_results"] if n["type"] == "prompt"]
    assert SECRET_VARS["customer_email"] in prompt_nodes[0]["output"]

    # And the persisted copy of the same run is redacted.
    assert SECRET_VARS["customer_email"] not in _stored(
        fake_db.inserts("workflow_runs")[0]["input_text"]
    )


def test_caller_supplied_input_text_is_returned_to_the_caller_verbatim(fake_db):
    """The non-variables case: a plain input_text must reach the output node
    unchanged, so the customer's response is byte-identical to before."""
    plain = "Please refund the card ending 4242424242424242 for dana.reyes@example.com"
    result = _execute(fake_db, input_text=plain, variables=None)

    # The graph's prompt node wraps it ("Handle: {{input}}"), so what matters is
    # that the caller's text flowed through VERBATIM — card and address intact.
    assert plain in result["final_output"]
    row = fake_db.inserts("workflow_runs")[0]
    assert "4242424242424242" not in _stored(row["input_text"])
    assert "dana.reyes@example.com" not in _stored(row["input_text"])
    assert "[redacted:payment_card]" in _stored(row["input_text"])


# ---------------------------------------------------------------------------
# Capture must never become a customer-facing error.
# ---------------------------------------------------------------------------

def test_a_broken_redactor_never_breaks_a_run(fake_db, monkeypatch):
    def _explode(*_a, **_kw):
        raise RuntimeError("redactor is down")

    monkeypatch.setattr(er, "redact_text", _explode)
    monkeypatch.setattr(er, "_walk", _explode)

    result = _execute(fake_db)
    assert result["final_output"]

    row = fake_db.inserts("workflow_runs")[0]
    capture = row["variables_capture"][er.INPUT_TEXT_CAPTURE_KEY]
    # Unavailable with a reason code — never a placeholder, never the raw value.
    assert capture["status"] == "unavailable"
    assert capture["reason"] == "capture_failed"
    assert row["input_text"] is None


def test_absent_empty_and_unavailable_stay_distinct():
    assert capture_input_text(None)[1]["status"] == "absent"
    assert capture_input_text("")[1]["status"] == "empty"
    assert capture_input_text(None)[0] is None
    assert capture_input_text("")[0] is None


def test_redaction_runs_before_truncation():
    """A secret straddling the 5000-character storage cap must not survive as a
    usable prefix."""
    key = "sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0"
    text = ("a" * (er.MAX_VALUE_CHARS - 20)) + " " + key
    stored, meta = capture_input_text(text)
    assert "sk-ant-api03" not in _stored(stored)
    assert meta["redacted"] is True


# ---------------------------------------------------------------------------
# The execution trace: the THIRD copy of the same request, on the SAME ROW.
# ---------------------------------------------------------------------------

def test_node_results_output_previews_are_redacted(fake_db):
    """The Input node records input_text[:200] and the Prompt node records the
    interpolated template. Redacting input_text and not these would have left
    the same secret 200 characters away on the same row."""
    _execute(fake_db)
    row = fake_db.inserts("workflow_runs")[0]

    persisted = json.dumps(row["node_results"])
    assert SECRET_VARS["customer_email"] not in persisted
    assert SECRET_VARS["api_key"] not in persisted
    assert "[redacted:email]" in persisted
    assert "[redacted:credential]" in persisted

    capture = row["variables_capture"][er.NODE_RESULTS_CAPTURE_KEY]
    assert capture["status"] == "captured"
    assert capture["redacted"] is True
    # Paths are indexed by trace entry, so a reader can find the step.
    assert all(p.startswith("[") for p in capture["redacted_paths"])


def test_the_executing_and_streamed_node_results_are_NOT_redacted(fake_db):
    """The same rule as input_text: only the database copy is redacted. The
    trace this call returns — and that the SSE path already streamed — is
    verbatim."""
    result = _execute(fake_db)

    returned = json.dumps(result["node_results"])
    assert SECRET_VARS["customer_email"] in returned
    assert "[redacted:" not in returned

    # And the object handed to the boundary was not mutated in place: the
    # persisted copy is a different structure.
    assert result["node_results"] is not fake_db.inserts("workflow_runs")[0]["node_results"]


def test_trace_structure_survives_redaction(fake_db):
    """Structure is load-bearing: the step count is shown in the UI, `status`
    and `error` produce every error rate in the product, and `node_id` is the
    join key other entries refer to by value."""
    result = _execute(fake_db)
    stored = fake_db.inserts("workflow_runs")[0]["node_results"]

    assert len(stored) == len(result["node_results"])
    assert [n["node_id"] for n in stored] == [n["node_id"] for n in result["node_results"]]
    assert [n["type"] for n in stored] == [n["type"] for n in result["node_results"]]


def test_error_rate_signal_survives_redaction(fake_db):
    """The canonical error parser must reach the same verdict on the persisted
    trace as on the executing one — redaction changes what a step says, never
    whether it failed."""
    from optimization.attempts import node_result_has_error

    graph_no_output = {
        "nodes": [{"id": "i1", "type": "input", "data": {}}],
        "edges": [],
    }
    with pytest.raises(HTTPException):
        _execute(fake_db, graph=graph_no_output)

    stored = fake_db.inserts("workflow_runs")[0]["node_results"]
    assert any(node_result_has_error(nr) for nr in stored)


def test_provider_credentials_echoed_into_error_detail_are_removed():
    """`error_detail` is where a provider's own error text lands, and provider
    errors routinely quote the key that was rejected."""
    trace = [{
        "node_id": "n1",
        "type": "model",
        "status": "error",
        "error": True,
        "error_status": 401,
        "error_detail": "401 unauthorized for key sk-ant-api03-QZ7bLmT4vR9wX2yK6pN8dF1sG3hJ5aC0",
        "model": "claude-sonnet-4-5-20250929",
        "provider": "anthropic",
        "latency_ms": 12,
        "tokens": 0,
        "cost": 0,
    }]
    safe, meta = er.capture_node_results(trace)

    assert "sk-ant-api03" not in json.dumps(safe)
    assert "[redacted:credential]" in safe[0]["error_detail"]
    # Structural fields untouched, including the ones error rates read.
    assert safe[0]["status"] == "error"
    assert safe[0]["error"] is True
    assert safe[0]["error_status"] == 401
    assert safe[0]["model"] == "claude-sonnet-4-5-20250929"
    assert meta["entry_count"] == 1


def test_structural_identifiers_are_never_mistaken_for_secrets():
    """A long mixed-case node_id trips the opaque-secret heuristic on its own.
    Redacting it would break the router entry that refers to it by value."""
    node_id = "aiStep_7fK2mQ9xL4pR8sT1vW3zB6nH"
    trace = [
        {"node_id": node_id, "type": "ai-step", "output": "fine", "cost": 0},
        {"node_id": "r1", "type": "router", "selected": node_id,
         "candidates": [node_id, "n2"], "router_selected_node_id": node_id},
    ]
    safe, meta = er.capture_node_results(trace)

    assert safe[0]["node_id"] == node_id
    assert safe[1]["selected"] == node_id
    assert safe[1]["candidates"] == [node_id, "n2"]
    assert meta["redacted"] is False
    # Proof this is not an accident of the heuristic: the same string under a
    # content key IS redacted.
    assert er.looks_like_opaque_secret(node_id) is True


def test_a_key_outside_the_structural_set_is_redacted_by_default():
    """The exception list is closed. Anything a future node type adds is
    redacted unless someone deliberately adds it."""
    safe, meta = er.capture_node_results(
        [{"node_id": "n1", "type": "custom", "some_future_field": "write to dana.reyes@example.com"}]
    )
    assert "dana.reyes@example.com" not in json.dumps(safe)
    assert meta["redacted"] is True


def test_node_results_capture_feeds_the_promotion_gate(monkeypatch):
    """A run whose only redaction was in the trace must still be gated."""
    run = {
        "id": "33333333-3333-3333-3333-333333333333",
        "workflow_id": "wf-1",
        "org_id": "org-1",
        "input_text": json.dumps({"review_count": 42}),
        "final_output": "done",
        "node_results": [{"node_id": "n1", "type": "input",
                          "output": "reach me at dana.reyes@example.com"}],
    }
    db = _promotion_db(monkeypatch, run)
    payload = wm.ImportFromProductionPayload(
        org_id="org-1", workflow_id="wf-1", run_id=run["id"]
    )
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(wm.import_golden_input_from_production(payload, _user=None))

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == REVIEW_REDACTED_INPUT
    assert exc.value.detail["redacted_kinds"] == ["email"]
    assert db.inserts("golden_inputs") == []
    assert db.writes("workflow_runs") == []


def test_empty_and_absent_traces_stay_distinct():
    assert er.capture_node_results(None)[1]["status"] == "absent"
    assert er.capture_node_results([])[0] == []
    assert er.capture_node_results([])[1]["status"] == "empty"


# ---------------------------------------------------------------------------
# National identifiers: ITINs are real PII and live in the 9xx block.
# ---------------------------------------------------------------------------

# 9XX-XX-XXXX with group digits in 50-65, 70-88, 90-92, 94-99.
VALID_ITINS = [
    "912-70-1234",   # group 70, low end of 70-88
    "950-65-4321",   # group 65, top of 50-65
    "988-88-1111",   # group 88, top of 70-88
    "901-90-2222",   # group 90, low end of 90-92
    "977-94-3333",   # group 94, low end of 94-99
    "999-99-9999",   # group 99, top of 94-99
    "912 70 1234",   # space separator
]


@pytest.mark.parametrize("itin", VALID_ITINS)
def test_itins_are_removed(itin):
    """The 9xx block is where the IRS issues ITINs. Excluding all of 9xx kept
    real taxpayer identifiers in plaintext."""
    stored, meta = capture_input_text(f"taxpayer {itin} on file")
    assert itin not in _stored(stored)
    assert "[redacted:national_id]" in _stored(stored)
    assert meta["redacted"] is True


# 9xx areas whose group digits fall in the gaps the IRS does not issue
# (00-49, 66-69, 89, 93): a formatted internal identifier, not an ITIN.
NON_ITIN_9XX = ["900-45-6789", "912-66-1234", "912-89-1234", "912-93-1234", "912-49-1234"]


@pytest.mark.parametrize("value", NON_ITIN_9XX)
def test_non_itin_9xx_values_are_left_alone(value):
    stored, meta = capture_input_text(f"reference {value} in the ledger")
    assert value in _stored(stored)
    assert meta["redacted"] is False


@pytest.mark.parametrize("value", ["000-12-3456", "666-12-3456"])
def test_structurally_impossible_areas_still_pass_through(value):
    """`000` and `666` are issued by no scheme, so excluding them is precision.
    That half of the lookahead is deliberately unchanged."""
    stored, meta = capture_input_text(f"ticket {value} logged")
    assert value in _stored(stored)
    assert meta["redacted"] is False


@pytest.mark.parametrize("value", ["000-12-3456", "666-12-3456", "900-45-6789"])
def test_the_key_still_catches_what_the_shape_rule_passes_over(value):
    """When the key names the field, the value goes whatever its shape — so the
    free-text pass-throughs above are not a gap wherever a key is present."""
    stored, _ = capture_input_text(json.dumps({"ssn": value}))
    assert value not in _stored(stored)
    assert "[redacted:national_id]" in _stored(stored)


def test_the_ruleset_version_records_the_change():
    """A later ruleset must not silently reinterpret rows written by an
    earlier one."""
    assert er.REDACTION_VERSION == 2
    assert capture_input_text("x")[1]["redaction_version"] == 2
