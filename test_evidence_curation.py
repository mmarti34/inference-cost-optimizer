"""
Evidence curation: production traffic -> deduplicated candidates -> a human
decision -> replay evidence.

What these tests are really asserting is one claim, in several shapes: THAT AN
UNREVIEWED CANDIDATE CANNOT BECOME A BENCHMARK CASE. Everything else here —
dedup, buckets, counters, readiness, tenant isolation — exists to make that
claim survivable in a real product, and each is tested against the failure it
was built to prevent rather than against its own happy path.

Only two edges are stubbed: the database (an in-memory table store) and the
authenticated principal. The redaction boundary, the replay gate, the
fingerprint, the bucketing and the whole review lifecycle are the real code.
"""
import copy
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# Same import shim the other router tests use, so these run under a bare venv.
if "Crypto" not in sys.modules:  # pragma: no cover - import shim only
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for _sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        sys.modules["Crypto." + _sub] = types.ModuleType("Crypto." + _sub)
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from fastapi.testclient import TestClient  # noqa: E402

import audit  # noqa: E402
import evidence_redaction as redaction  # noqa: E402
import main  # noqa: E402
import resource_access  # noqa: E402
from auth_dependency import AuthenticatedUser  # noqa: E402
from optimization import benchmark as benchmark_mod  # noqa: E402
from optimization import curation  # noqa: E402
from optimization import policies as policies_mod  # noqa: E402
from optimization import workloads as workloads_mod  # noqa: E402
from routers import optimization_router  # noqa: E402

ORG_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ORG_ID = "99999999-9999-9999-9999-999999999999"
WORKLOAD_ID = "22222222-2222-2222-2222-222222222222"
OTHER_WORKLOAD_ID = "22222222-2222-2222-2222-222222222299"
WORKFLOW_ID = "33333333-3333-3333-3333-333333333333"
OTHER_WORKFLOW_ID = "33333333-3333-3333-3333-333333333399"
USER_ID = "44444444-4444-4444-4444-444444444444"
UNKNOWN_CANDIDATE_ID = "55555555-5555-5555-5555-555555555555"

FOREIGN_SECRET = "TENANT-B-CUSTOMER-INPUT-do-not-disclose"


# ---------------------------------------------------------------------------
# In-memory stand-in for the supabase client
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, table, log):
        self._store = store
        self._table = table
        self._log = log
        self._op = "select"
        self._payload = None
        self._filters = []
        self._order = None
        self._limit = None

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, row):
        self._op = "insert"
        self._payload = row
        return self

    def update(self, patch):
        self._op = "update"
        self._payload = patch
        return self

    def delete(self):
        self._op = "delete"
        return self

    def single(self):
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def gte(self, col, val):
        self._filters.append(("gte", col, val))
        return self

    def in_(self, col, vals):
        self._filters.append(("in", col, list(vals)))
        return self

    def is_(self, col, val):
        self._filters.append(("is", col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _matches(self, row):
        for op, col, val in self._filters:
            actual = row.get(col)
            if op == "eq":
                if val is None:
                    if actual is not None:
                        return False
                elif str(actual) != str(val):
                    return False
            elif op == "gte":
                if actual is None or str(actual) < str(val):
                    return False
            elif op == "in":
                if str(actual) not in {str(v) for v in val}:
                    return False
            elif op == "is":
                if val is None and actual is not None:
                    return False
        return True

    def execute(self):
        rows = self._store.setdefault(self._table, [])
        applied = [(c, v) for _op, c, v in self._filters]

        if self._op == "insert":
            batch = self._payload if isinstance(self._payload, list) else [self._payload]
            written = []
            for item in batch:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("created_at", _iso(_now()))
                # THE DEDUP INDEX, in the double. Without it the fake would
                # accept duplicates the real schema refuses, and the dedup
                # tests would pass against a database that does not exist.
                if self._table == "evidence_candidates":
                    key = (row.get("org_id"), row.get("workload_id"), row.get("fingerprint"))
                    if any(
                        (r.get("org_id"), r.get("workload_id"), r.get("fingerprint")) == key
                        for r in rows
                    ):
                        raise RuntimeError("duplicate key value violates unique constraint")
                rows.append(row)
                written.append(dict(row))
            self._log.append(("insert", self._table, applied))
            return _Result(written)

        matched = [r for r in rows if self._matches(r)]

        if self._op == "update":
            for r in matched:
                r.update(self._payload)
            self._log.append(("update", self._table, applied))
            return _Result([dict(r) for r in matched])

        if self._op == "delete":
            for r in matched:
                rows.remove(r)
            self._log.append(("delete", self._table, applied))
            return _Result([dict(r) for r in matched])

        self._log.append(("select", self._table, applied))
        if self._order:
            col, desc = self._order
            matched = sorted(
                matched,
                key=lambda r: (r.get(col) is None, r.get(col)),
                reverse=bool(desc),
            )
        if self._limit is not None:
            matched = matched[: self._limit]
        return _Result([dict(r) for r in matched])


class FakeSupabase:
    def __init__(self):
        self.store = {}
        self.log = []

    def table(self, name):
        return _Query(self.store, name, self.log)

    def rows(self, name):
        return [dict(r) for r in self.store.get(name, [])]

    def seed(self, name, *rows):
        self.store.setdefault(name, []).extend(dict(r) for r in rows)

    def filters_for(self, op, table):
        return [f for o, t, f in self.log if o == op and t == table]


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


_PATCH_TARGETS = (
    curation,
    workloads_mod,
    benchmark_mod,
    policies_mod,
    resource_access,
    audit,
)


@pytest.fixture
def db():
    fake = FakeSupabase()
    patches = [patch.object(mod, "supabase", fake) for mod in _PATCH_TARGETS]
    for p in patches:
        p.start()
    try:
        yield fake
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def client(db):
    user = AuthenticatedUser(user_id=USER_ID, email="a@b.c")
    setattr(user, "_verified_org_id", ORG_ID)
    main.app.dependency_overrides[optimization_router.require_org_member] = lambda: user
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _workload(org_id=ORG_ID, workload_id=WORKLOAD_ID, workflow_id=WORKFLOW_ID):
    return {
        "id": workload_id,
        "org_id": org_id,
        "name": "Review Summary",
        "surface": "runtime",
        "identity_kind": "workflow",
        "identity_ref": workflow_id,
        "default_objective": "cost",
    }


def _run(
    *,
    org_id=ORG_ID,
    workflow_id=WORKFLOW_ID,
    input_text="summarise ticket 1",
    variables=None,
    output='{"summary": "ok", "sentiment": "positive"}',
    created_at=None,
    cost=0.001,
    latency=900,
    node_results=None,
    variables_capture=None,
    run_id=None,
):
    return {
        "id": run_id or str(uuid.uuid4()),
        "org_id": org_id,
        "workflow_id": workflow_id,
        "endpoint_slug": None,
        "execution_mode": "production",
        "input_text": input_text,
        "variables": variables,
        "variables_capture": variables_capture,
        "final_output": output,
        "node_results": node_results if node_results is not None else [
            {"node_id": "n1", "type": "ai-step", "status": "success", "cost": cost,
             "latency_ms": latency, "model": "gpt-4o", "provider": "openai"}
        ],
        "total_cost": cost,
        "total_latency_ms": latency,
        "created_at": created_at or _iso(_now() - timedelta(days=1)),
    }


def _seed_workload(db, *, quality_check=True, workload=None):
    db.seed("workloads", workload or _workload())
    if quality_check:
        db.seed("eval_suites", {
            "id": str(uuid.uuid4()), "org_id": ORG_ID, "workflow_id": WORKFLOW_ID,
            "name": "suite", "enabled": True,
            "checks": [{"type": "deterministic", "name": "exact", "enabled": True}],
        })


def _derive(db, workload=None):
    return curation.derive_candidates(ORG_ID, workload or _workload(), lookback_days=30)


def _candidates(db, org_id=ORG_ID, workload_id=WORKLOAD_ID):
    return [
        r for r in db.rows("evidence_candidates")
        if r["org_id"] == org_id and r["workload_id"] == workload_id
    ]


# ===========================================================================
# 1. DEDUP — the 742:39 ratio is the whole reason this feature exists
# ===========================================================================

def test_fingerprint_is_deterministic_across_calls():
    a = curation.fingerprint_of("Summarise  the\nticket", {"k": "V"})
    b = curation.fingerprint_of("Summarise the ticket", {"k": "V"})
    assert a == b
    # Nothing about it depends on dict ordering or on `hash()`.
    assert curation.fingerprint_of("x", {"a": 1, "b": 2}) == curation.fingerprint_of("x", {"b": 2, "a": 1})


def test_case_is_folded_on_free_text_but_never_on_variable_values():
    """The one asymmetry in the normalisation, asserted rather than described.

    Free text differing only in case is not a distinct benchmark case. A
    variable value differing in case routinely IS one — it is an id, a slug, an
    enum or a path — and collapsing those would silently delete a case.
    """
    assert curation.fingerprint_of("Summarise It", None) == curation.fingerprint_of("summarise it", None)
    assert curation.fingerprint_of("x", {"env": "PROD"}) != curation.fingerprint_of("x", {"env": "prod"})


def test_variable_key_identity_survives_normalisation():
    """Same value under a different key is a different input, not a duplicate."""
    assert curation.fingerprint_of("", {"a": "x"}) != curation.fingerprint_of("", {"b": "x"})


def test_742_runs_collapse_to_39_candidates(db):
    """The measured production ratio, reproduced end to end.

    39 distinct inputs across 742 runs is the largest real workload in the
    database. If dedup ever regresses, this is the number that moves, and a
    counter reading "742 cases captured" is exactly the lie the feature exists
    to prevent.
    """
    _seed_workload(db)
    runs = []
    for i in range(742):
        distinct = i % 39
        # Whitespace and case noise of the kind real traffic actually carries.
        # The noise cycles on the REPEAT index, not on `distinct`: keying it off
        # `i` directly would correlate with `i % 39` (39 is divisible by 3) and
        # give every distinct input exactly one spelling, so the normalisation
        # would never actually be exercised. That is a real trap and it caught
        # this test once already.
        noise = (i // 39) % 3
        if noise == 0:
            text = f"summarise ticket {distinct}"
        elif noise == 1:
            text = f"  Summarise\tticket {distinct}  "
        else:
            text = f"SUMMARISE\n ticket  {distinct}"
        runs.append(_run(
            input_text=text,
            created_at=_iso(_now() - timedelta(days=2, seconds=i)),
        ))
    db.seed("workflow_runs", *runs)

    result = _derive(db)

    assert result["runs_observed"] == 742
    assert result["distinct"] == 39
    assert result["inserted"] == 39
    assert len(_candidates(db)) == 39
    assert result["coverage"]["dedup_ratio"] == pytest.approx(19.03, abs=0.01)
    # Every collapsed run is accounted for — 703 rows did not vanish.
    assert sum(c["occurrences"] for c in _candidates(db)) == 742


def test_derivation_is_idempotent(db):
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"case {i}") for i in range(10)])

    first = _derive(db)
    second = _derive(db)

    assert first["inserted"] == 10
    assert second["inserted"] == 0
    assert len(_candidates(db)) == 10


def test_the_exemplar_run_is_the_earliest_and_does_not_depend_on_scan_order(db):
    """Determinism: the same runs must always produce the same candidate row."""
    _seed_workload(db)
    early = _run(input_text="same", created_at=_iso(_now() - timedelta(days=20)), run_id="a" * 8)
    late = _run(input_text="SAME", created_at=_iso(_now() - timedelta(days=2)), run_id="b" * 8)
    db.seed("workflow_runs", late, early)

    _derive(db)
    rows = _candidates(db)

    assert len(rows) == 1
    assert rows[0]["source_run_id"] == "a" * 8
    assert rows[0]["captured_at"] == early["created_at"]
    assert rows[0]["last_seen_at"] == late["created_at"]


# ===========================================================================
# 2. THE PROPOSED LABEL
# ===========================================================================

def test_a_derived_candidate_has_a_proposal_and_no_expected_output(db):
    """The central semantic: production output is a PROPOSAL, not an answer."""
    _seed_workload(db)
    db.seed("workflow_runs", _run(output="the production answer"))

    _derive(db)
    row = _candidates(db)[0]

    assert row["production_output"] == "the production answer"
    assert row["proposed_expected_output"] == "the production answer"
    assert row["expected_output"] is None
    assert row["state"] == curation.STATE_CAPTURED
    assert row["golden_input_id"] is None


def test_deriving_candidates_never_writes_golden_inputs(db):
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(30)])

    _derive(db)

    assert db.rows("golden_inputs") == []


def test_an_unapproved_candidate_never_appears_in_load_golden_inputs(db):
    """
    THE NON-NEGOTIABLE. `benchmark._load_golden_inputs` is THE case set, and it
    has no idea this feature exists — which is the point of the separate table.
    A `status` column on `golden_inputs` would have made this test depend on
    ten readers each remembering to filter.
    """
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(25)])
    _derive(db)
    assert len(_candidates(db)) == 25

    assert benchmark_mod._load_golden_inputs(ORG_ID, WORKFLOW_ID) == []

    # And it stays true after every non-approving decision.
    for row in _candidates(db)[:5]:
        curation.review(ORG_ID, row, decision=curation.DECISION_REJECT, reviewer_id=USER_ID)
    for row in _candidates(db)[5:10]:
        curation.review(ORG_ID, row, decision=curation.DECISION_NOT_USEFUL, reviewer_id=USER_ID)

    assert benchmark_mod._load_golden_inputs(ORG_ID, WORKFLOW_ID) == []

    # Approval — and only approval — puts a case in front of the benchmark.
    curation.review(ORG_ID, _candidates(db)[10], decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)
    cases = benchmark_mod._load_golden_inputs(ORG_ID, WORKFLOW_ID)
    assert len(cases) == 1


def test_approval_copies_the_humans_answer_not_the_production_output(db):
    """An edit must not silently benchmark against what production emitted."""
    _seed_workload(db)
    db.seed("workflow_runs", _run(output="production was wrong"))
    _derive(db)
    row = _candidates(db)[0]

    curation.review(
        ORG_ID, row, decision=curation.DECISION_EDIT,
        expected_output="the correct answer", reviewer_id=USER_ID,
    )

    golden = db.rows("golden_inputs")
    assert len(golden) == 1
    assert golden[0]["expected_output"] == "the correct answer"
    assert "production was wrong" not in str(golden[0])
    # The observation itself is preserved, unedited.
    assert _candidates(db)[0]["production_output"] == "production was wrong"
    assert _candidates(db)[0]["state"] == curation.STATE_EDITED


def test_edit_without_an_expected_output_is_refused(db):
    _seed_workload(db)
    db.seed("workflow_runs", _run())
    _derive(db)

    with pytest.raises(curation.CurationRefused) as exc:
        curation.review(
            ORG_ID, _candidates(db)[0], decision=curation.DECISION_EDIT,
            expected_output="   ", reviewer_id=USER_ID,
        )
    assert exc.value.detail["code"] == curation.REFUSE_EXPECTED_OUTPUT_REQUIRED
    assert db.rows("golden_inputs") == []


# ===========================================================================
# 3. APPROVAL: exactly one case, and idempotent
# ===========================================================================

def test_approval_creates_exactly_one_golden_input_and_is_idempotent(db):
    _seed_workload(db)
    db.seed("workflow_runs", _run(output="answer"))
    _derive(db)
    row = _candidates(db)[0]

    first = curation.review(ORG_ID, row, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)
    assert first["golden_input_id"] is not None
    assert first["decision_recorded"] is True
    assert len(db.rows("golden_inputs")) == 1

    # The same decision again, from a stale copy of the row — a double-clicked
    # Approve, or a retried request.
    second = curation.review(ORG_ID, row, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)
    assert len(db.rows("golden_inputs")) == 1, "a second case was created"
    assert second["decision_recorded"] is False

    # And from the fresh row, which is what a well-behaved client would send.
    third = curation.review(
        ORG_ID, _candidates(db)[0], decision=curation.DECISION_APPROVE, reviewer_id=USER_ID
    )
    assert len(db.rows("golden_inputs")) == 1
    assert third["golden_input_id"] == first["golden_input_id"]


def test_a_different_decision_on_an_already_reviewed_candidate_is_refused(db):
    """A recorded human decision is not silently overwritten by the next one."""
    _seed_workload(db)
    db.seed("workflow_runs", _run())
    _derive(db)
    row = _candidates(db)[0]
    curation.review(ORG_ID, row, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)

    with pytest.raises(curation.CurationRefused) as exc:
        curation.review(
            ORG_ID, _candidates(db)[0], decision=curation.DECISION_REJECT, reviewer_id=USER_ID
        )
    assert exc.value.detail["code"] == curation.REFUSE_ALREADY_REVIEWED
    assert len(db.rows("golden_inputs")) == 1


def test_the_golden_input_name_carries_no_customer_content(db):
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="patient MRN 55 has a rash"))
    _derive(db)

    curation.review(
        ORG_ID, _candidates(db)[0], decision=curation.DECISION_APPROVE, reviewer_id=USER_ID
    )

    assert "patient" not in db.rows("golden_inputs")[0]["name"]


# ===========================================================================
# 4. REDACTED / PROVENANCE-UNKNOWN: visible, never silently replayable
# ===========================================================================

def test_a_redacted_candidate_is_visible_in_the_queue_but_gated(db):
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="contact me at alice@example.com about ticket 9"))
    _derive(db)

    q = curation.queue(ORG_ID, WORKLOAD_ID)
    assert len(q["candidates"]) == 1, "a redacted candidate must not be hidden"
    entry = q["candidates"][0]
    assert entry["capture"]["replay_eligible"] is False
    assert redaction.REVIEW_REDACTED_INPUT in entry["capture"]["reason_codes"]
    assert entry["capture"]["redacted"] is True
    assert redaction.KIND_EMAIL in entry["capture"]["redacted_kinds"]
    # The value itself is gone, and the marker says what was removed.
    assert "alice@example.com" not in str(entry)
    assert redaction.marker(redaction.KIND_EMAIL) in entry["input_text"]


def test_a_redacted_candidate_cannot_be_approved_without_acknowledgement(db):
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="reach bob@example.com now"))
    _derive(db)
    row = _candidates(db)[0]

    with pytest.raises(curation.CurationRefused) as exc:
        curation.review(ORG_ID, row, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)

    assert exc.value.detail["code"] == redaction.REVIEW_REDACTED_INPUT
    assert exc.value.detail["candidate_preserved"] is True
    assert db.rows("golden_inputs") == []
    # Nothing about the row changed: it is preserved for inspection, not moved.
    assert _candidates(db)[0]["state"] == curation.STATE_CAPTURED


def test_provenance_unknown_is_gated_too_because_unknown_is_not_clean(db):
    """`redacted: false` on a row nobody could inspect is not evidence of absence."""
    _seed_workload(db)
    db.seed("workflow_runs", _run(
        input_text="ordinary text",
        variables_capture={
            "status": "unavailable", "reason": "capture_failed",
            "redacted": False, "truncated": False, "redaction_version": 2,
        },
    ))
    _derive(db)
    row = _candidates(db)[0]

    assert row["replay_eligible"] is False
    assert redaction.REVIEW_UNKNOWN_PROVENANCE in row["replay_reason_codes"]
    assert row["redacted"] is False, "unknown provenance is not the same as redacted"

    with pytest.raises(curation.CurationRefused) as exc:
        curation.review(ORG_ID, row, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)
    assert exc.value.detail["code"] == redaction.REVIEW_REDACTED_INPUT
    assert redaction.REVIEW_UNKNOWN_PROVENANCE in exc.value.detail["reasons"]


def test_an_acknowledged_redacted_candidate_is_approvable_and_says_so(db):
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="reach carol@example.com now"))
    _derive(db)

    out = curation.review(
        ORG_ID, _candidates(db)[0], decision=curation.DECISION_APPROVE,
        acknowledge_redaction=True, reviewer_id=USER_ID,
    )

    assert out["golden_input_id"] is not None
    assert _candidates(db)[0]["review_acknowledged_redaction"] is True
    # The case is marked so nobody later mistakes it for a faithful replay.
    assert db.rows("golden_inputs")[0]["source"] == curation.GOLDEN_SOURCE_CURATED_REDACTED


def test_rejecting_a_redacted_candidate_needs_no_acknowledgement(db):
    """The gate protects replay evidence. Deciding something is NOT evidence
    creates no evidence, so there is nothing to gate."""
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="reach dave@example.com now"))
    _derive(db)

    out = curation.review(
        ORG_ID, _candidates(db)[0], decision=curation.DECISION_REJECT, reviewer_id=USER_ID
    )
    assert out["decision_recorded"] is True
    assert db.rows("golden_inputs") == []


# ===========================================================================
# 5. BUCKETS and INFERRED CHECKS
# ===========================================================================

def test_buckets_partition_the_population_and_are_named_by_code(db):
    _seed_workload(db)
    runs = [_run(input_text=f"ordinary request number {i}") for i in range(20)]
    runs.append(_run(
        input_text="errored request",
        node_results=[{"node_id": "n1", "type": "ai-step", "status": "error",
                       "error": "upstream 500"}],
    ))
    runs.append(_run(input_text="long " + ("padding " * 400)))
    runs.append(_run(input_text="odd shape", output="just prose, not json"))
    db.seed("workflow_runs", *runs)

    _derive(db)
    rows = _candidates(db)
    counts = {b["code"]: b["count"] for b in curation.bucket_counts(rows)}

    assert sum(counts.values()) == len(rows), "buckets must partition the queue"
    assert counts.get(curation.BUCKET_FAILURE) == 1
    assert counts.get(curation.BUCKET_UNUSUAL_OUTPUT) == 1
    assert counts.get(curation.BUCKET_LONG_INPUT) == 1
    assert counts.get(curation.BUCKET_RANDOM, 0) > 0
    assert set(counts) <= set(curation.BUCKET_PRIORITY)


def test_an_unreadable_trace_is_not_bucketed_as_a_failure(db):
    """An unknown is not a failure. `parse_step_results` says so, and the
    bucketing must not upgrade "we could not read it" into "it errored"."""
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="x", node_results="corrupt-not-a-list"))
    _derive(db)

    row = _candidates(db)[0]
    assert row["bucket"] != curation.BUCKET_FAILURE
    assert row["bucket_signals"]["error_known"] is False


def test_inferred_checks_omit_what_does_not_apply():
    """A check that could not be measured is ABSENT, never `passed: False`."""
    prose = curation.infer_checks(production_output="just prose", node_results=[])
    codes = {c["code"] for c in prose}
    assert curation.CHECK_OUTPUT_VALID_JSON not in codes, "prose is not a failed JSON case"
    assert curation.CHECK_OUTPUT_PRESENT in codes

    broken = curation.infer_checks(production_output='{"a": 1', node_results=[])
    assert {"code": curation.CHECK_OUTPUT_VALID_JSON, "passed": False} in broken

    unreadable = curation.infer_checks(production_output="x", node_results="not-a-list")
    assert curation.CHECK_NO_EXECUTION_ERROR not in {c["code"] for c in unreadable}


def test_field_checks_need_a_reference_and_are_omitted_without_one():
    single = curation.infer_checks(
        production_output='{"a": 1}', node_results=[], modal_json_fields=None
    )
    assert curation.CHECK_OUTPUT_FIELDS_PRESENT not in {c["code"] for c in single}

    reffed = curation.infer_checks(
        production_output='{"a": 1}', node_results=[],
        modal_json_fields={"a": "number", "b": "string"},
    )
    assert {"code": curation.CHECK_OUTPUT_FIELDS_PRESENT, "passed": False} in reffed


def test_a_missing_json_field_is_flagged_from_the_workloads_own_outputs(db):
    _seed_workload(db)
    runs = [_run(input_text=f"c{i}", output='{"summary": "s", "sentiment": "positive"}')
            for i in range(10)]
    runs.append(_run(input_text="truncated one", output='{"summary": "s"}'))
    db.seed("workflow_runs", *runs)

    _derive(db)
    odd = [r for r in _candidates(db) if r["production_output"] == '{"summary": "s"}'][0]

    assert {"code": curation.CHECK_OUTPUT_FIELDS_PRESENT, "passed": False} in odd["checks"]


# ===========================================================================
# 6. COUNTERS and READINESS
# ===========================================================================

def test_counters_keep_all_four_numbers_apart(db):
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(25)])
    _derive(db)
    rows = _candidates(db)

    for r in rows[:6]:
        curation.review(ORG_ID, r, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)
    for r in rows[6:10]:
        curation.review(ORG_ID, r, decision=curation.DECISION_REJECT, reviewer_id=USER_ID)

    counts = curation.counters(_candidates(db), runs_observed=25, required=20)

    assert counts == {
        "runs_observed": 25,
        "distinct_captured": 25,
        "reviewed": 10,
        "approved": 6,
        "required": 20,
    }
    # The lie this guards against: captured is NOT progress toward the floor.
    assert counts["approved"] != counts["distinct_captured"]


def test_runs_observed_is_null_when_the_scan_failed_never_zero(db):
    _seed_workload(db)

    class _Boom(FakeSupabase):
        def table(self, name):
            if name == "workflow_runs":
                raise RuntimeError("connection reset")
            return super().table(name)

    boom = _Boom()
    boom.store = db.store
    with patch.object(curation, "supabase", boom):
        result = curation.derive_candidates(ORG_ID, _workload())

    assert result["runs_observed"] is None
    assert result["coverage"]["error"] == "query_failed"


def test_readiness_needs_approved_cases_AND_a_quality_signal(db):
    """Both conditions, independently. Neither alone makes a workload ready."""
    _seed_workload(db, quality_check=False)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(40)])
    _derive(db)

    # (a) nothing approved, no quality check.
    overview = curation.evidence_overview(ORG_ID, _workload(), derive=False)
    assert overview["readiness"]["ready"] is False
    assert "sample_size_below_threshold" in overview["readiness"]["reason_codes"]
    assert "evidence_awaiting_review" in overview["readiness"]["reason_codes"]
    assert "evidence_quality_check_absent" in overview["readiness"]["reason_codes"]

    # (b) enough approved, STILL no quality check -> still not ready.
    for r in _candidates(db)[:25]:
        curation.review(ORG_ID, r, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)
    overview = curation.evidence_overview(ORG_ID, _workload(), derive=False)
    assert overview["counters"]["approved"] == 25
    assert overview["readiness"]["ready"] is False
    assert overview["readiness"]["reason_codes"] == ["evidence_quality_check_absent"]

    # (c) add the quality check -> ready flips.
    db.seed("eval_suites", {
        "id": str(uuid.uuid4()), "org_id": ORG_ID, "workflow_id": WORKFLOW_ID,
        "checks": [{"type": "deterministic", "enabled": True}], "enabled": True,
    })
    overview = curation.evidence_overview(ORG_ID, _workload(), derive=False)
    assert overview["readiness"]["ready"] is True
    assert overview["readiness"]["reason_codes"] == []


def test_an_llm_judge_is_not_a_quality_signal(db):
    """`model_graded` must never satisfy readiness — benchmark._run_quality_checks
    excludes it from the measured verdict, so promising one here would promise a
    number the benchmark cannot produce."""
    _seed_workload(db, quality_check=False)
    db.seed("eval_suites", {
        "id": str(uuid.uuid4()), "org_id": ORG_ID, "workflow_id": WORKFLOW_ID,
        "checks": [{"type": "model_graded", "enabled": True}], "enabled": True,
    })

    signal = curation.quality_signal(ORG_ID, WORKFLOW_ID)

    assert signal["usable"] is False
    assert signal["reason_codes"] == ["evidence_quality_check_absent"]
    assert signal["provenance"] is None


def test_a_quality_signal_that_could_not_be_read_is_null_not_false(db):
    class _Boom(FakeSupabase):
        def table(self, name):
            if name == "eval_suites":
                raise RuntimeError("connection reset")
            return super().table(name)

    with patch.object(curation, "supabase", _Boom()):
        signal = curation.quality_signal(ORG_ID, WORKFLOW_ID)

    assert signal["usable"] is None, "'could not tell' must not read as 'absent'"
    assert signal["reason_codes"] == ["quality_not_measured"]


def test_readiness_never_starts_a_benchmark(db):
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(30)])
    _derive(db)
    for r in _candidates(db)[:25]:
        curation.review(ORG_ID, r, decision=curation.DECISION_APPROVE, reviewer_id=USER_ID)

    overview = curation.evidence_overview(ORG_ID, _workload(), derive=False)

    assert overview["readiness"]["ready"] is True
    assert db.rows("optimization_benchmarks") == []
    assert db.rows("optimization_jobs") == []


# ===========================================================================
# 7. THE API
# ===========================================================================

def test_the_evidence_endpoint_returns_the_contracted_shape(db, client):
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(30)])

    resp = client.get(f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/evidence")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body["counters"]) == {
        "runs_observed", "distinct_captured", "reviewed", "approved", "required"
    }
    assert body["readiness"]["ready"] is False
    assert isinstance(body["readiness"]["reason_codes"], list)
    assert all(isinstance(c, str) for c in body["readiness"]["reason_codes"])
    assert all(set(b) == {"code", "count"} for b in body["buckets"])
    assert body["counters"]["distinct_captured"] == 30


def test_the_queue_endpoint_returns_the_contracted_shape(db, client):
    _seed_workload(db)
    db.seed("workflow_runs", *[_run(input_text=f"c{i}") for i in range(30)])
    _derive(db)

    resp = client.get(
        f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/evidence/queue?limit=5"
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["candidates"]) == 5
    assert body["remaining"] == 25
    entry = body["candidates"][0]
    for key in ("id", "state", "bucket", "input_text", "variables", "production_output",
                "proposed_expected_output", "checks", "capture", "source_run_id",
                "captured_at"):
        assert key in entry, key
    assert set(entry["capture"]) >= {
        "redacted", "redacted_kinds", "replay_eligible", "reason_codes"
    }
    assert all(set(c) == {"code", "passed"} for c in entry["checks"])


def test_the_review_endpoint_returns_candidate_golden_id_and_counters(db, client):
    _seed_workload(db)
    db.seed("workflow_runs", _run())
    _derive(db)
    cid = _candidates(db)[0]["id"]

    resp = client.post(
        f"/api/optimization/{ORG_ID}/evidence/{cid}/review",
        json={"decision": "approve"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate"]["state"] == curation.STATE_APPROVED
    assert body["golden_input_id"] is not None
    assert body["counters"]["approved"] == 1
    assert body["counters"]["distinct_captured"] == 1


def test_the_review_endpoint_409s_on_unacknowledged_redaction(db, client):
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="mail erin@example.com"))
    _derive(db)
    cid = _candidates(db)[0]["id"]

    resp = client.post(
        f"/api/optimization/{ORG_ID}/evidence/{cid}/review", json={"decision": "approve"}
    )

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == redaction.REVIEW_REDACTED_INPUT
    assert db.rows("golden_inputs") == []


def test_an_unknown_decision_is_refused(db, client):
    _seed_workload(db)
    db.seed("workflow_runs", _run())
    _derive(db)
    cid = _candidates(db)[0]["id"]

    resp = client.post(
        f"/api/optimization/{ORG_ID}/evidence/{cid}/review", json={"decision": "looks_fine"}
    )

    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == curation.REFUSE_UNKNOWN_DECISION
    assert db.rows("golden_inputs") == []


# ===========================================================================
# 8. AUDIT
# ===========================================================================

def test_every_review_decision_is_audited_without_customer_content(db, client):
    _seed_workload(db)
    db.seed("workflow_runs",
            _run(input_text="SENSITIVE-INPUT-TEXT", output="SENSITIVE-OUTPUT-TEXT"),
            _run(input_text="another one"))
    _derive(db)
    rows = _candidates(db)

    client.post(f"/api/optimization/{ORG_ID}/evidence/{rows[0]['id']}/review",
                json={"decision": "approve"})
    client.post(f"/api/optimization/{ORG_ID}/evidence/{rows[1]['id']}/review",
                json={"decision": "reject"})

    entries = db.rows("audit_log")
    actions = [e["action"] for e in entries]
    assert audit.EVIDENCE_APPROVED in actions
    assert audit.EVIDENCE_REJECTED in actions
    for e in entries:
        assert e["org_id"] == ORG_ID
        assert e["actor_id"] == USER_ID
        assert e["resource_type"] == audit.RESOURCE_EVIDENCE_CANDIDATE
        blob = str(e)
        assert "SENSITIVE-INPUT-TEXT" not in blob
        assert "SENSITIVE-OUTPUT-TEXT" not in blob


def test_a_refused_approval_is_audited_as_a_refusal(db, client):
    _seed_workload(db)
    db.seed("workflow_runs", _run(input_text="mail frank@example.com"))
    _derive(db)
    cid = _candidates(db)[0]["id"]

    client.post(f"/api/optimization/{ORG_ID}/evidence/{cid}/review",
                json={"decision": "approve"})

    entries = [e for e in db.rows("audit_log") if e["action"] == audit.EVIDENCE_APPROVE_REFUSED]
    assert len(entries) == 1
    assert entries[0]["metadata"]["outcome"] == "refused"
    assert entries[0]["metadata"]["replay_eligible"] is False


def test_every_new_audit_action_is_in_the_closed_vocabulary():
    for action in (audit.EVIDENCE_APPROVED, audit.EVIDENCE_EDITED, audit.EVIDENCE_REJECTED,
                   audit.EVIDENCE_NOT_USEFUL, audit.EVIDENCE_APPROVE_REFUSED):
        assert action in audit.ACTIONS
    assert audit.EVIDENCE_APPROVE_REFUSED in audit.REFUSAL_ACTIONS


# ===========================================================================
# 9. TENANT ISOLATION — the standing shape
#    own -> success | foreign -> opaque | unknown -> identical | foreign
#    mutation -> zero side effects
# ===========================================================================

def _seed_two_tenants(db):
    """One candidate in each org, with the foreign one carrying a marker."""
    db.seed("workloads", _workload(), _workload(
        org_id=OTHER_ORG_ID, workload_id=OTHER_WORKLOAD_ID, workflow_id=OTHER_WORKFLOW_ID
    ))
    mine = {
        "id": str(uuid.uuid4()), "org_id": ORG_ID, "workload_id": WORKLOAD_ID,
        "workflow_id": WORKFLOW_ID, "fingerprint": "fp-mine", "fingerprint_version": 1,
        "input_text": "my own input", "variables": None,
        "production_output": "mine", "proposed_expected_output": "mine",
        "expected_output": None, "capture": {}, "replay_eligible": True,
        "replay_reason_codes": [], "redacted": False, "redacted_kinds": [],
        "bucket": curation.BUCKET_COMMON, "bucket_signals": {}, "checks": [],
        "state": curation.STATE_CAPTURED, "occurrences": 1,
        "captured_at": _iso(_now()), "source_run_id": None, "golden_input_id": None,
        "review_acknowledged_redaction": False,
    }
    theirs = dict(
        mine,
        id=str(uuid.uuid4()), org_id=OTHER_ORG_ID, workload_id=OTHER_WORKLOAD_ID,
        workflow_id=OTHER_WORKFLOW_ID, fingerprint="fp-theirs",
        input_text=FOREIGN_SECRET, production_output=FOREIGN_SECRET,
        proposed_expected_output=FOREIGN_SECRET,
    )
    db.seed("evidence_candidates", mine, theirs)
    return mine, theirs


def test_own_candidate_review_succeeds(db, client):
    mine, _ = _seed_two_tenants(db)

    resp = client.post(f"/api/optimization/{ORG_ID}/evidence/{mine['id']}/review",
                       json={"decision": "approve"})

    assert resp.status_code == 200
    assert len(db.rows("golden_inputs")) == 1


def test_a_foreign_candidate_is_opaque_and_indistinguishable_from_an_unknown_one(db, client):
    _mine, theirs = _seed_two_tenants(db)

    foreign = client.post(f"/api/optimization/{ORG_ID}/evidence/{theirs['id']}/review",
                          json={"decision": "approve"})
    unknown = client.post(f"/api/optimization/{ORG_ID}/evidence/{UNKNOWN_CANDIDATE_ID}/review",
                          json={"decision": "approve"})

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json(), "the endpoint is an existence oracle"
    assert foreign.json()["detail"] == resource_access.EVIDENCE_CANDIDATE_NOT_FOUND
    assert FOREIGN_SECRET not in foreign.text


def test_a_foreign_mutation_has_zero_side_effects(db, client):
    _mine, theirs = _seed_two_tenants(db)
    before = copy.deepcopy(db.store)

    for decision in ("approve", "edit", "reject", "not_useful"):
        resp = client.post(
            f"/api/optimization/{ORG_ID}/evidence/{theirs['id']}/review",
            json={"decision": decision, "expected_output": "hijacked",
                  "acknowledge_redaction": True},
        )
        assert resp.status_code == 404

    assert db.store == before, "a foreign candidate was modified"
    assert db.rows("golden_inputs") == []
    assert db.rows("audit_log") == []


def test_the_ownership_lookup_carries_both_the_id_and_the_org(db, client):
    """The whole fix in one assertion: the row is never read on id alone."""
    _mine, theirs = _seed_two_tenants(db)
    db.log.clear()

    client.post(f"/api/optimization/{ORG_ID}/evidence/{theirs['id']}/review",
                json={"decision": "approve"})

    lookups = db.filters_for("select", "evidence_candidates")
    assert lookups, "no ownership lookup was performed"
    for applied in lookups:
        cols = [c for c, _ in applied]
        assert "org_id" in cols, f"candidate read without an org filter: {applied}"
        assert ("org_id", ORG_ID) in applied


def test_every_review_write_is_org_scoped(db, client):
    """Mutations are DOUBLY scoped: the check and the write are one statement,
    so there is no check-then-act window."""
    mine, _theirs = _seed_two_tenants(db)
    db.log.clear()

    client.post(f"/api/optimization/{ORG_ID}/evidence/{mine['id']}/review",
                json={"decision": "approve"})

    updates = db.filters_for("update", "evidence_candidates")
    assert updates
    for applied in updates:
        assert ("org_id", ORG_ID) in applied, f"unscoped update: {applied}"


def test_a_foreign_workloads_evidence_is_opaque(db, client):
    _seed_two_tenants(db)

    foreign = client.get(
        f"/api/optimization/{ORG_ID}/workloads/{OTHER_WORKLOAD_ID}/evidence"
    )
    unknown = client.get(
        f"/api/optimization/{ORG_ID}/workloads/{UNKNOWN_CANDIDATE_ID}/evidence"
    )

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()
    assert FOREIGN_SECRET not in foreign.text


def test_a_foreign_workloads_queue_is_opaque(db, client):
    _seed_two_tenants(db)

    foreign = client.get(
        f"/api/optimization/{ORG_ID}/workloads/{OTHER_WORKLOAD_ID}/evidence/queue"
    )
    unknown = client.get(
        f"/api/optimization/{ORG_ID}/workloads/{UNKNOWN_CANDIDATE_ID}/evidence/queue"
    )

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()
    assert FOREIGN_SECRET not in foreign.text


def test_the_queue_never_returns_another_tenants_candidate(db, client):
    _seed_two_tenants(db)
    # The foreign candidate is filed under the caller's OWN workload id, which
    # is the shape a query filtered on workload_id alone would leak.
    db.seed("evidence_candidates", dict(
        db.rows("evidence_candidates")[1], id=str(uuid.uuid4()),
        workload_id=WORKLOAD_ID, fingerprint="fp-planted",
    ))

    resp = client.get(f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/evidence/queue")

    assert resp.status_code == 200
    assert FOREIGN_SECRET not in resp.text
    assert len(resp.json()["candidates"]) == 1


def test_derivation_never_reads_another_tenants_runs(db):
    _seed_workload(db)
    db.seed("workflow_runs",
            _run(input_text="mine"),
            _run(org_id=OTHER_ORG_ID, workflow_id=WORKFLOW_ID, input_text=FOREIGN_SECRET))

    _derive(db)
    rows = _candidates(db)

    assert len(rows) == 1
    assert FOREIGN_SECRET not in str(rows)


def test_the_resource_helper_refuses_a_caller_supplied_org():
    """The executable invariant: there is no parameter to pass an org into."""
    with pytest.raises(TypeError):
        resource_access.get_evidence_candidate_for_org("some-id", ORG_ID)  # type: ignore[arg-type]
