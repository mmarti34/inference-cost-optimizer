"""
Evidence capture: production input variables reach `workflow_runs`.

The gap these tests close: `workflow_runs` recorded `input_text` and had no
`variables` column at all, so a workflow driven by named variables stored
nothing about its inputs and none of its traffic could ever be promoted into an
evaluation case. Measured before the change: the "Review Summary" workflow takes
five variables and had stored 0 inputs across 358 production runs.

Three things are asserted here, and the second and third matter as much as the
first:
  1. the variables are captured, on ALL THREE insert paths — the two error paths
     included, because a failing input is exactly the edge case an evaluation
     set most needs;
  2. redaction is strict enough to be useful and LOUD enough to be safe — every
     pattern is tested with a realistic positive AND a near-miss negative, and
     every modification is recorded where a downstream reader will find it;
  3. capture can never break a production run.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import evidence_redaction as er
import workflow_runtime as wr
from evidence_redaction import capture_variables, luhn_ok, redact_text


# ---------------------------------------------------------------------------
# Fake Supabase: records every insert instead of performing one.
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, recorder: list, table_name: str):
        self._recorder = recorder
        self._table = table_name
        self._row = None

    def insert(self, row):
        self._row = row
        self._recorder.append((self._table, row))
        return self

    def execute(self):
        return SimpleNamespace(data=[{"id": "00000000-0000-0000-0000-0000000000ff"}])


class _FakeSupabase:
    def __init__(self):
        self.inserts: list = []

    def table(self, name: str):
        return _FakeQuery(self.inserts, name)


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeSupabase()
    monkeypatch.setattr(wr, "supabase", db)
    return db


def _runs(db: _FakeSupabase) -> list[dict]:
    return [row for table, row in db.inserts if table == "workflow_runs"]


# The Review Summary shape: five named variables, driven entirely by variables
# with no input_text, and no Input node in the graph.
REVIEW_VARS = {
    "artist": "Phoebe Bridgers",
    "venue": "The Fillmore",
    "review_count": 42,
    "reviews": ["Incredible set.", "Sound mix was rough but the songs carried it."],
    "date": "2026-03-14",
}

GRAPH_OK = {
    "nodes": [
        {"id": "p1", "type": "prompt", "data": {"template": "Summarise {{artist}} at {{venue}}"}},
        {"id": "o1", "type": "output", "data": {}},
    ],
    "edges": [{"source": "p1", "target": "o1"}],
}

# No output node: execution raises HTTPException(400) and takes the error path.
GRAPH_NO_OUTPUT = {
    "nodes": [{"id": "p1", "type": "prompt", "data": {"template": "Summarise {{artist}}"}}],
    "edges": [],
}


def _execute(**kwargs):
    params = dict(
        graph=GRAPH_OK,
        input_text="",
        org_id="org-1",
        user_id="user-1",
        workflow_id="wf-1",
        execution_mode="production",
        variables=dict(REVIEW_VARS),
    )
    params.update(kwargs)
    return wr.execute_workflow(**params)


# ---------------------------------------------------------------------------
# 1. A variable-driven run captures its variables
# ---------------------------------------------------------------------------

def test_variable_driven_run_captures_its_variables(fake_db):
    """The Review Summary shape: five named variables, no input_text. Before
    this change the run recorded nothing about any of them."""
    _execute()

    rows = _runs(fake_db)
    assert len(rows) == 1
    row = rows[0]

    assert row["variables"] == REVIEW_VARS
    assert row["variables_capture"]["status"] == "captured"
    assert row["variables_capture"]["key_count"] == 5
    assert row["variables_capture"]["redacted"] is False
    assert row["variables_capture"]["truncated"] is False
    # And the row is still valid JSON for a jsonb column.
    json.dumps(row["variables"])
    json.dumps(row["variables_capture"])


def test_capture_row_still_carries_everything_it_carried_before(fake_db):
    """Capture is additive. No existing field changes."""
    _execute(input_text="")
    row = _runs(fake_db)[0]
    for field in (
        "workflow_id", "org_id", "user_id", "input_text", "final_output",
        "node_results", "total_cost", "total_latency_ms", "endpoint_slug",
        "version", "execution_mode",
    ):
        assert field in row, field
    assert row["execution_mode"] == "production"


# ---------------------------------------------------------------------------
# 2. Absent / empty / unavailable are three distinct recorded states
# ---------------------------------------------------------------------------

def test_no_variables_is_null_with_a_reason_not_an_empty_object(fake_db):
    _execute(variables=None, input_text="just some text")
    row = _runs(fake_db)[0]
    assert row["variables"] is None
    assert row["variables_capture"]["status"] == "absent"
    assert row["variables_capture"]["reason"] == "no_variables_supplied"


def test_empty_mapping_is_distinguishable_from_no_mapping(fake_db):
    """`{}` is a real observation about the caller and must not be recorded the
    same way as "the caller sent nothing"."""
    _execute(variables={}, input_text="just some text")
    row = _runs(fake_db)[0]
    assert row["variables"] is None
    assert row["variables_capture"]["status"] == "empty"
    assert row["variables_capture"]["reason"] == "caller_supplied_empty_mapping"
    # Never `{}` masquerading as "no variables".
    assert row["variables"] != {}


def test_statuses_are_disjoint():
    assert capture_variables(None)[1]["status"] == "absent"
    assert capture_variables({})[1]["status"] == "empty"
    assert capture_variables({"a": 1})[1]["status"] == "captured"
    assert capture_variables("not a mapping")[1]["status"] == "unavailable"
    assert capture_variables("not a mapping")[1]["reason"] == "variables_not_a_mapping"
    assert capture_variables("not a mapping")[0] is None


# ---------------------------------------------------------------------------
# 3. Redaction patterns: realistic positives AND near-miss negatives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,kind",
    [
        ("write to dana.k+tickets@example.co.uk about it", er.KIND_EMAIL),
        ("reach me on +44 20 7946 0958 tomorrow", er.KIND_PHONE),
        ("call (415) 555-2671 after six", er.KIND_PHONE),
        ("call 415.555.2671 after six", er.KIND_PHONE),
        ("charged 4111 1111 1111 1111 for the tickets", er.KIND_CARD),
        ("amex 3782 822463 10005 on file", er.KIND_CARD),
        ("ssn 123-45-6789 on the form", er.KIND_NATIONAL_ID),
        ("nino AB 12 34 56 C recorded", er.KIND_NATIONAL_ID),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwx", er.KIND_CREDENTIAL),
        ("key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA", er.KIND_CREDENTIAL),
        ("aws AKIAIOSFODNN7EXAMPLE in the config", er.KIND_CREDENTIAL),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N", er.KIND_CREDENTIAL),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END RSA PRIVATE KEY-----", er.KIND_CREDENTIAL),
    ],
)
def test_redacts_realistic_positive(value, kind):
    out, kinds = redact_text(value)
    assert kind in kinds, f"{value!r} was not detected as {kind}"
    assert er.marker(kind) in out
    # The marker names what was removed — redaction is never anonymous.
    assert out != value


@pytest.mark.parametrize(
    "value,why",
    [
        ("order 1234567812345678 shipped", "16 digits but fails the Luhn checksum"),
        ("order 4571234567890111 shipped", "starts with 4 but fails Luhn"),
        ("reference 1700000000000 logged", "13-digit millisecond timestamp, no separators"),
        ("upgrade to v2.10.4 today", "version string, not a phone number"),
        ("release 10.155.123.4567 build", "dotted build number, not a phone number"),
        ("host 192.168.100.4567 unreachable", "address-shaped, not a phone number"),
        ("due 2024-01-15 at the latest", "ISO date, not an SSN"),
        ("case 000-45-6789 filed", "administratively invalid SSN area 000"),
        ("account 9876543210 credited", "bare ten digits with no separators and no key evidence"),
        ("the-fillmore-san-francisco-2024-tour", "lowercase slug, not a secret"),
        ("550e8400-e29b-41d4-a716-446655440000", "UUID, not a secret"),
        ("Great show, 5 stars, would go again", "ordinary prose"),
    ],
)
def test_near_miss_is_left_alone(value, why):
    out, kinds = redact_text(value)
    assert kinds == [], f"false positive ({why}): {value!r} -> {out!r}"
    assert out == value


def test_luhn_is_what_separates_a_card_from_an_order_number():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("1234567812345678")
    assert er.looks_like_payment_card("4111111111111111")
    # Passes no issuer prefix even though it is 16 digits.
    assert not er.looks_like_payment_card("1234567812345678")
    # Passes Luhn but belongs to no scheme -> not treated as a card.
    assert luhn_ok("1234567812345670")
    assert not er.looks_like_payment_card("1234567812345670")


def test_token_shaped_key_names_are_not_treated_as_secrets():
    """`max_tokens` and `token_count` are counters. Redacting them would corrupt
    replay while protecting nothing."""
    safe, meta = capture_variables({"max_tokens": 512, "token_count": "51234567", "output_tokens": 900})
    assert safe == {"max_tokens": 512, "token_count": "51234567", "output_tokens": 900}
    assert meta["redacted"] is False


def test_key_name_directs_redaction_when_the_value_has_no_shape_of_its_own():
    safe, meta = capture_variables({
        "customer_phone": "5551234567",
        "api_key": "abcdefghijklmnop",
        "password": "hunter2hunter2",
    })
    assert safe["customer_phone"] == er.marker(er.KIND_PHONE)
    assert safe["api_key"] == er.marker(er.KIND_CREDENTIAL)
    assert safe["password"] == er.marker(er.KIND_CREDENTIAL)


def test_numeric_card_is_redacted_and_the_type_change_is_recorded():
    safe, meta = capture_variables({"payment": 4111111111111111})
    assert safe["payment"] == er.marker(er.KIND_CARD)
    assert meta["type_changed"] is True
    assert "payment" in meta["redacted_paths"]


def test_booleans_and_ordinary_numbers_are_untouched():
    safe, _ = capture_variables({"flag": True, "count": 42, "ratio": 0.5, "nothing": None})
    assert safe == {"flag": True, "count": 42, "ratio": 0.5, "nothing": None}


# ---------------------------------------------------------------------------
# 4. Redaction is recorded, so a downstream reader can tell a case was modified
# ---------------------------------------------------------------------------

def test_redaction_is_recorded_where_curation_will_look(fake_db):
    _execute(variables={
        "artist": "Phoebe Bridgers",
        "contact": "dana@example.com",
        "reviews": ["loved it", "call me on (415) 555-2671"],
    })
    row = _runs(fake_db)[0]
    cap = row["variables_capture"]

    assert cap["status"] == "captured"
    assert cap["redacted"] is True
    assert sorted(cap["redacted_kinds"]) == ["email", "phone"]
    # Exact locations, so the reader knows which fields differ from what
    # production actually ran — not merely that something differs.
    assert cap["redacted_paths"] == ["contact", "reviews[1]"]
    assert {"path": "contact", "kinds": ["email"]} in cap["redactions"]
    assert cap["redaction_version"] == er.REDACTION_VERSION
    # The unredacted field is untouched, so the case is still partly faithful.
    assert row["variables"]["artist"] == "Phoebe Bridgers"


def test_a_clean_run_is_marked_clean():
    _, meta = capture_variables(dict(REVIEW_VARS))
    assert meta["redacted"] is False
    assert "redactions" not in meta
    assert "redacted_paths" not in meta


# ---------------------------------------------------------------------------
# 5. Structure and keys survive redaction
# ---------------------------------------------------------------------------

def test_structure_keys_and_types_survive_redaction():
    original = {
        "artist": "Phoebe Bridgers",
        "review_count": 42,
        "customer": {"name": "Dana", "email": "dana@example.com", "vip": True},
        "reviews": [
            {"text": "loved it", "rating": 5},
            {"text": "email me at x@y.io", "rating": 3},
        ],
    }
    safe, meta = capture_variables(original)

    assert list(safe.keys()) == list(original.keys())
    assert list(safe["customer"].keys()) == ["name", "email", "vip"]
    assert isinstance(safe["reviews"], list) and len(safe["reviews"]) == 2
    assert isinstance(safe["reviews"][1], dict)
    assert safe["review_count"] == 42 and isinstance(safe["review_count"], int)
    assert safe["customer"]["vip"] is True
    assert safe["reviews"][0] == {"text": "loved it", "rating": 5}
    assert safe["reviews"][1]["rating"] == 3
    # Only the values changed.
    assert safe["customer"]["email"] == er.marker(er.KIND_EMAIL)
    assert safe["reviews"][1]["text"] == f"email me at {er.marker(er.KIND_EMAIL)}"
    assert meta["redacted_paths"] == ["customer.email", "reviews[1].text"]


def test_surrounding_text_is_preserved_around_a_redacted_span():
    out, _ = redact_text("Please refund dana@example.com for order 77")
    assert out == "Please refund [redacted:email] for order 77"


# ---------------------------------------------------------------------------
# 6. Over-long values are truncated AND marked
# ---------------------------------------------------------------------------

def test_long_value_is_truncated_and_the_truncation_is_marked(fake_db):
    long_review = "x" * (er.MAX_VALUE_CHARS + 2500)
    _execute(variables={"artist": "Phoebe Bridgers", "reviews": long_review})
    row = _runs(fake_db)[0]

    stored = row["variables"]["reviews"]
    assert stored.startswith("x" * 100)
    assert stored.endswith(er.TRUNCATION_MARKER.format(original=len(long_review)))
    assert len(stored) < len(long_review)

    cap = row["variables_capture"]
    assert cap["truncated"] is True
    assert cap["truncated_paths"] == ["reviews"]
    entry = cap["truncations"][0]
    assert entry["reason"] == "value_length"
    assert entry["original"] == len(long_review)
    assert entry["kept"] == er.MAX_VALUE_CHARS
    # Short values are not touched.
    assert row["variables"]["artist"] == "Phoebe Bridgers"


def test_oversize_blob_is_null_with_a_reason_not_a_silent_partial():
    huge = {f"k{i}": "y" * 4000 for i in range(200)}
    safe, meta = capture_variables(huge)
    assert safe is None
    assert meta["status"] == "unavailable"
    assert meta["reason"] == "oversize"
    assert meta["serialized_chars"] > er.MAX_SERIALIZED_CHARS
    assert meta["key_count"] == 200


def test_long_list_is_capped_and_the_cap_is_recorded():
    safe, meta = capture_variables({"reviews": ["ok"] * (er.MAX_LIST_ITEMS + 30)})
    assert len(safe["reviews"]) == er.MAX_LIST_ITEMS + 1
    assert safe["reviews"][-1] == er.LIST_MARKER.format(omitted=30)
    assert meta["truncated"] is True
    assert meta["truncations"][0]["reason"] == "list_items"


def test_deep_nesting_is_capped_and_recorded():
    node: dict = {"leaf": "bottom"}
    for _ in range(20):
        node = {"child": node}
    safe, meta = capture_variables(node)
    assert meta["truncated"] is True
    assert any(t["reason"] == "depth" for t in meta["truncations"])
    assert er.DEPTH_MARKER in json.dumps(safe)


# ---------------------------------------------------------------------------
# 7. Capture can never break a production run
# ---------------------------------------------------------------------------

def test_exception_inside_capture_does_not_break_the_run(fake_db, monkeypatch):
    """The whole point of best-effort: a redaction bug must cost evidence, never
    a customer's request."""
    def _explode(_variables):
        raise RuntimeError("redaction blew up")

    monkeypatch.setattr(wr, "capture_variables", _explode)

    result = _execute()

    # The run succeeded ...
    assert result["final_output"]
    # ... the row was still inserted ...
    rows = _runs(fake_db)
    assert len(rows) == 1
    # ... and the variables are recorded as unavailable, with the reason.
    assert rows[0]["variables"] is None
    cap = rows[0]["variables_capture"]
    assert cap["status"] == "unavailable"
    assert cap["reason"] == "capture_failed"
    assert cap["error_type"] == "RuntimeError"


def test_unserialisable_value_is_recorded_as_unavailable_not_raised(fake_db):
    """A value that explodes when rendered must produce NULL-with-a-reason, not
    an exception escaping capture.

    Note what this test also documents: such a value breaks the run itself, in
    `_apply_variables`, for reasons that have nothing to do with capture and
    that predate it. Capture's obligation is to add no failure of its own and
    to still record what it can — which on that path means the generic-error
    insert happens and carries an honest `unavailable` status.
    """
    class Exploding:
        def __repr__(self):
            raise ValueError("nope")
        __str__ = __repr__

    # Capture alone: never raises, records the reason.
    value, meta = capture_variables({"artist": "Phoebe Bridgers", "bad": Exploding()})
    assert value is None
    assert meta["status"] == "unavailable"
    assert meta["reason"] == "capture_failed"

    # And through the runtime: the run fails on its own merits, but the row is
    # still inserted and still carries the capture record.
    with pytest.raises(HTTPException):
        _execute(variables={"artist": "Phoebe Bridgers", "bad": Exploding()})
    rows = _runs(fake_db)
    assert len(rows) == 1
    assert rows[0]["variables"] is None
    assert rows[0]["variables_capture"]["status"] == "unavailable"


def test_capture_itself_never_raises_on_hostile_input():
    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("hostile")

    for bad in (Hostile(), object(), b"\xff\xfe", {"k": Hostile()}, [1, 2, 3]):
        value, meta = capture_variables(bad)
        assert isinstance(meta, dict) and "status" in meta
        assert meta["status"] in ("captured", "absent", "empty", "unavailable")
        if meta["status"] != "captured":
            assert value is None


def test_capture_is_computed_at_most_once_per_run(fake_db, monkeypatch):
    calls = {"n": 0}
    real = er.capture_variables

    def _counting(v):
        calls["n"] += 1
        return real(v)

    monkeypatch.setattr(wr, "capture_variables", _counting)
    _execute()
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 8. All three insert paths capture — the error paths included
# ---------------------------------------------------------------------------

def test_success_path_captures(fake_db):
    _execute()
    rows = _runs(fake_db)
    assert len(rows) == 1
    assert rows[0]["final_output"] is not None
    assert rows[0]["variables"] == REVIEW_VARS
    assert rows[0]["variables_capture"]["status"] == "captured"


def test_http_error_path_captures(fake_db):
    """A failing input is exactly the edge case an evaluation set most needs."""
    with pytest.raises(HTTPException) as exc:
        _execute(graph=GRAPH_NO_OUTPUT)
    assert exc.value.status_code == 400

    rows = _runs(fake_db)
    assert len(rows) == 1
    assert rows[0]["final_output"] is None
    assert rows[0]["node_results"][-1]["status"] == "error"
    assert rows[0]["variables"] == REVIEW_VARS
    assert rows[0]["variables_capture"]["status"] == "captured"


def test_generic_error_path_captures(fake_db, monkeypatch):
    def _boom(context, from_node_id):
        raise TypeError("not an HTTPException")

    monkeypatch.setattr(wr, "_get_previous_output", _boom)

    with pytest.raises(HTTPException) as exc:
        _execute()
    assert exc.value.status_code == 500

    rows = _runs(fake_db)
    assert len(rows) == 1
    assert rows[0]["final_output"] is None
    assert rows[0]["node_results"][-1]["status"] == "error"
    assert rows[0]["variables"] == REVIEW_VARS
    assert rows[0]["variables_capture"]["status"] == "captured"


def test_error_paths_redact_exactly_like_the_success_path(fake_db):
    dirty = {"contact": "dana@example.com", "artist": "Phoebe Bridgers"}

    with pytest.raises(HTTPException):
        _execute(graph=GRAPH_NO_OUTPUT, variables=dict(dirty))
    err_row = _runs(fake_db)[0]

    fake_db.inserts.clear()
    _execute(variables=dict(dirty))
    ok_row = _runs(fake_db)[0]

    assert err_row["variables"] == ok_row["variables"]
    assert err_row["variables_capture"]["redacted_paths"] == \
        ok_row["variables_capture"]["redacted_paths"]
    assert err_row["variables"]["contact"] == er.marker(er.KIND_EMAIL)


# ---------------------------------------------------------------------------
# 9. The streaming production path captures too
# ---------------------------------------------------------------------------

def test_streaming_linear_insert_captures(monkeypatch):
    import workflow_streaming as ws

    db = _FakeSupabase()
    monkeypatch.setattr(ws, "supabase", db)

    ws._insert_workflow_run_linear(
        org_id="org-1", workflow_id="wf-1", endpoint_slug="slug", version=1,
        input_text="", final_output="out", node_results=[], total_cost=0.0,
        total_latency_ms=5, variables={"contact": "dana@example.com"},
        variables_supplied=True,
    )
    row = _runs(db)[0]
    assert row["variables"] == {"contact": er.marker(er.KIND_EMAIL)}
    assert row["variables_capture"]["status"] == "captured"


def test_streaming_unwired_caller_is_not_recorded_as_having_no_variables(monkeypatch):
    """An un-wired call site must not be indistinguishable from a run that
    genuinely had no variables."""
    import workflow_streaming as ws

    db = _FakeSupabase()
    monkeypatch.setattr(ws, "supabase", db)

    ws._insert_workflow_run_linear(
        org_id="org-1", workflow_id="wf-1", endpoint_slug="slug", version=1,
        input_text="x", final_output="out", node_results=[], total_cost=0.0,
        total_latency_ms=5,
    )
    cap = _runs(db)[0]["variables_capture"]
    assert cap["status"] == "unavailable"
    assert cap["reason"] == "not_captured_at_this_call_site"
