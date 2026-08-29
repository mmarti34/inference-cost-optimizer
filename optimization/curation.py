"""
CURATION — the missing bridge between captured traffic and replay evidence.

WHAT WAS MISSING
----------------
OptiML could observe, capture, prove and optimize. It could not CURATE. Every
benchmark path reads `golden_inputs` as its case set, and `golden_inputs` was
populated by a customer arriving with a golden dataset. Essentially nobody
does.

Measured across the whole database when this module was written:

    169 distinct inputs, all orgs, all time      (140 of them our own)
    largest single real workload: 39 distinct inputs across 742 runs

Both numbers shape everything below.

  * THE UNIT OF WORK IS TENS, NOT THOUSANDS. There are no embeddings here and
    no clustering. At 39 examples an embedding model is an expensive way to
    reproduce what an exact normalised match already gets right, and it buys a
    similarity threshold nobody could defend to a customer asking why two of
    their cases were merged. Every signal in this module is cheap, exact and
    explainable in one sentence.
  * DEDUP, NOT VOLUME, IS THE CONSTRAINT. 742 runs -> 39 candidates is 19:1. A
    reviewer shown 742 rows reviews nothing, and a counter reading "742 cases
    captured" is false. The fingerprint is a UNIQUE index in the schema, not a
    convention in this file, so a duplicate cannot be inserted even by a buggy
    caller.

THE ONE RULE EVERYTHING ELSE SERVES
-----------------------------------
A PRODUCTION OUTPUT IS A PROPOSED LABEL, NEVER AN AUTOMATIC GOLDEN ANSWER.

A production INPUT is excellent benchmark material — real, representative, and
exactly the traffic the customer cares about getting right. A production OUTPUT
is what one model produced on one day under one prompt. If the workload is
being optimized *because* its output is mediocre, treating that output as the
expected answer scores every candidate on its ability to reproduce the
mediocrity, and the benchmark concludes something other than what it says.

So candidates live in their own table (`evidence_candidates`) and only the
approval path in this module writes `golden_inputs`. That makes "everything in
`golden_inputs` is human-approved" true BY CONSTRUCTION rather than by
convention: the ten existing readers of `golden_inputs` are untouched and
cannot be forgotten. See migration_optimization_v14_evidence_candidates.sql for
why a `status` column on `golden_inputs` was rejected.

Nothing in this module approves anything. `infer_checks` is structural, and it
exists to make a human's review faster, never to replace it.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not call a provider, it does not start a benchmark, and it does not
make a workload optimization-ready by itself. `readiness()` reports codes and
facts; the spend-triggering action stays an explicit, separate request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import evidence_redaction as redaction
from supabase_client import supabase

from optimization import attempts as attempts_mod
from optimization import domain
from optimization import policies as policies_mod
from optimization import workloads as workloads_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

STATE_CAPTURED = "captured"
STATE_PROPOSED = "proposed_for_review"
STATE_APPROVED = "human_approved"
STATE_EDITED = "human_edited"
STATE_REJECTED = "rejected"
STATE_NOT_USEFUL = "not_useful"

#: States a review decision may still act on. Everything else is a recorded
#: human decision and is terminal.
REVIEWABLE_STATES = (STATE_CAPTURED, STATE_PROPOSED)
#: States that count toward benchmarkability, and the ONLY ones that may carry
#: a golden_input_id. Kept apart rather than collapsed into one "approved":
#: a workload whose cases are mostly `human_edited` is one whose production
#: output is routinely wrong, and that is a finding about the workload.
APPROVED_STATES = (STATE_APPROVED, STATE_EDITED)
REVIEWED_STATES = (STATE_APPROVED, STATE_EDITED, STATE_REJECTED, STATE_NOT_USEFUL)

DECISION_APPROVE = "approve"
DECISION_EDIT = "edit"
DECISION_REJECT = "reject"
DECISION_NOT_USEFUL = "not_useful"
DECISIONS = (DECISION_APPROVE, DECISION_EDIT, DECISION_REJECT, DECISION_NOT_USEFUL)

_DECISION_TO_STATE = {
    DECISION_APPROVE: STATE_APPROVED,
    DECISION_EDIT: STATE_EDITED,
    DECISION_REJECT: STATE_REJECTED,
    DECISION_NOT_USEFUL: STATE_NOT_USEFUL,
}

# ── Diversity buckets. CODES ONLY; the frontend owns every word. ────────────
#
# Assignment is by STRICT PRIORITY, so a candidate has exactly one bucket and
# the bucket counts partition the queue — "12 common · 6 long-input · 4 failure
# · 3 unusual output shape · 5 random coverage" must sum to the total, or the
# reviewer cannot trust either number.
BUCKET_FAILURE = "failure"
BUCKET_UNUSUAL_OUTPUT = "unusual_output_shape"
BUCKET_UNUSUAL_VARIABLES = "unusual_variable_shape"
BUCKET_LONG_INPUT = "long_input"
BUCKET_OUTLIER = "outlier_cost_latency"
BUCKET_RANDOM = "random_coverage"
BUCKET_COMMON = "common"

#: Highest-signal first. A run that ERRORED is the case a hand-built dataset
#: is most likely to be missing, so it outranks everything; `common` is the
#: residue, which is why it is last.
BUCKET_PRIORITY = (
    BUCKET_FAILURE,
    BUCKET_UNUSUAL_OUTPUT,
    BUCKET_UNUSUAL_VARIABLES,
    BUCKET_LONG_INPUT,
    BUCKET_OUTLIER,
    BUCKET_RANDOM,
    BUCKET_COMMON,
)

# ── Inferred structural check codes ─────────────────────────────────────────
CHECK_NO_EXECUTION_ERROR = "no_execution_error"
CHECK_OUTPUT_PRESENT = "output_present"
CHECK_OUTPUT_VALID_JSON = "output_valid_json"
CHECK_OUTPUT_FIELDS_PRESENT = "output_json_fields_present"
CHECK_OUTPUT_FIELD_TYPES = "output_json_field_types_stable"

# ── Refusal codes returned to the API, never prose ──────────────────────────
REFUSE_REDACTION_UNACKNOWLEDGED = redaction.REVIEW_REDACTED_INPUT
REFUSE_ALREADY_REVIEWED = "evidence_candidate_already_reviewed"
REFUSE_EXPECTED_OUTPUT_REQUIRED = "expected_output_required"
REFUSE_NO_WORKFLOW_FOR_REPLAY = "workload_has_no_replayable_workflow"
REFUSE_UNKNOWN_DECISION = "unknown_decision"
REFUSE_STORAGE_UNAVAILABLE = "evidence_candidate_storage_unavailable"

#: `source` written onto the `golden_inputs` row this module creates. Two
#: values, so a reader can tell a faithful curated case from one a human
#: knowingly approved despite redaction — the same distinction
#: workflow_management.import_golden_input_from_production already draws.
GOLDEN_SOURCE_CURATED = "curated_from_production"
GOLDEN_SOURCE_CURATED_REDACTED = "curated_from_production_redacted"

CANDIDATE_COLS = (
    "id, org_id, workload_id, workflow_id, source_run_id, captured_at, last_seen_at, "
    "occurrences, input_text, variables, production_output, proposed_expected_output, "
    "expected_output, capture, replay_eligible, replay_reason_codes, redacted, "
    "redacted_kinds, fingerprint, fingerprint_version, bucket, bucket_signals, checks, "
    "state, reviewed_by, reviewed_at, review_acknowledged_redaction, golden_input_id, "
    "created_at, updated_at"
)

_RUN_COLS = (
    "id, org_id, workflow_id, endpoint_slug, execution_mode, input_text, variables, "
    "variables_capture, final_output, node_results, total_cost, total_latency_ms, created_at"
)
#: Pre-v12 databases have neither `variables` nor `variables_capture`. The scan
#: retries without them rather than failing, exactly as the existing promotion
#: path does — and a row read this way carries no capture provenance, which the
#: replay gate correctly reports as `capture_provenance_unavailable`.
_RUN_COLS_LEGACY = (
    "id, org_id, workflow_id, endpoint_slug, execution_mode, input_text, "
    "final_output, node_results, total_cost, total_latency_ms, created_at"
)

_MAX_SCAN_ROWS = 2000
#: Hard cap on candidates held per workload. At 39 distinct inputs on the
#: largest real workload this never binds; it exists so a pathological workload
#: (every request unique) degrades into a bounded queue instead of an unbounded
#: table. When it binds it is REPORTED, never silent.
MAX_CANDIDATES_PER_WORKLOAD = 500

DEFAULT_LOOKBACK_DAYS = 30

#: How many of the residual `common` candidates get relabelled
#: `random_coverage`. Not a statistical sample — a deliberate, deterministic
#: chronological spread so the queue is not entirely one afternoon's traffic.
RANDOM_COVERAGE_TARGET = 5

#: Distinct buckets an approved case set should span. Capped at what the
#: candidate population actually CONTAINS: a workload with no failures and one
#: output shape genuinely has one bucket, and demanding diversity that does not
#: exist would make it permanently un-ready for a reason the customer cannot
#: act on.
MIN_APPROVED_BUCKETS = 3

#: Fallback floor on approved cases. The real value comes from the workload's
#: policy (`min_sample_size`) or `benchmark.DEFAULT_MIN_SAMPLE_SIZE`; this
#: constant is only used when neither can be read.
FALLBACK_MIN_APPROVED_CASES = 20


class CurationRefused(Exception):
    """A review decision was refused. Carries a structured detail, never prose."""

    def __init__(self, detail: dict):
        super().__init__(detail.get("code", "refused"))
        self.detail = detail


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 1. DEDUP — deterministic, dull, and documented
# ---------------------------------------------------------------------------
#
# The whole point of the fingerprint is that 742 runs become 39 candidates. It
# has to be defensible to a customer who asks why two of their inputs were
# treated as the same case, so every rule below is one sentence long and none
# of them is semantic.
#
# INPUT_TEXT is normalised as:
#   1. Unicode NFC. Canonical composition only — "e" + combining acute becomes
#      "é" and nothing else changes. NFKC was rejected: it is a COMPATIBILITY
#      mapping and would rewrite "½" to "1/2" and "ﬁ" to "fi", which is a
#      content change, not a normalisation.
#   2. Whitespace runs collapse to one space, then strip. This is exactly
#      `benchmark._normalize_output` / `workflow_management._normalize_output`,
#      the convention this codebase ALREADY uses to decide whether two eval
#      outputs are the same. Reusing it means the dedup rule and the quality
#      rule cannot drift apart.
#   3. Case folded.
#
# VARIABLE VALUES get 1 and 2 but NOT 3. This is the only asymmetry and it is
# deliberate: a named variable routinely holds an identifier, slug, enum value,
# file path, model name or ticket reference, where case is load-bearing and a
# case difference is a real difference. Free-text input that differs only in
# case is not a distinct benchmark case. Where the two errors are balanced,
# under-collapsing a variable costs one duplicate in the queue, while
# over-collapsing one silently deletes a case.
#
# WHAT THIS COSTS, STATED PLAINLY: a workload whose behaviour depends on the
# CASING of its free-text input has two genuinely distinct cases collapsed into
# one. `occurrences` records that the collapse happened. If such a workload ever
# turns up, the fix is to bump `FINGERPRINT_VERSION` — not to add a heuristic
# that guesses when case matters.
#
# REDACTION INTERACTS WITH DEDUP, ON PURPOSE. The fingerprint is computed over
# the values AFTER the redaction boundary, so two runs that differed only in a
# redacted email collapse into one candidate. That is correct: neither can
# replay faithfully, both are already gated, and keeping them apart would
# inflate the counter with rows a reviewer cannot tell apart.

FINGERPRINT_VERSION = 1

_WHITESPACE_RE = re.compile(r"\s+")


def _collapse(value: str) -> str:
    """NFC, whitespace runs to one space, stripped. Rule 1 + 2, nothing else."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def normalize_input_text(value: Any) -> str:
    """Free-text input, normalised for dedup: collapsed AND case folded."""
    if value is None:
        return ""
    return _collapse(str(value)).casefold()


def normalize_variables(value: Any) -> Any:
    """Variables, normalised for dedup: collapsed, CASE PRESERVED.

    Structure is preserved so that `{"a": "x"}` and `{"b": "x"}` stay distinct;
    mapping keys are sorted at serialisation time by `json.dumps(sort_keys=True)`
    so key ORDER never changes a fingerprint.
    """
    if isinstance(value, dict):
        return {str(k): normalize_variables(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_variables(v) for v in value]
    if isinstance(value, str):
        return _collapse(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    # Anything else is stringified rather than dropped: dropping it would make
    # two different inputs fingerprint identically, which is the one failure
    # mode this function must not have.
    return _collapse(str(value))


def fingerprint_of(input_text: Any, variables: Any) -> str:
    """The dedup key: sha256 over canonical JSON of the normalised input.

    Deterministic across processes and across runs — no `hash()`, no dict
    ordering, no locale. The version is part of the hashed payload so that a
    future normalisation change cannot silently collide with an old row.
    """
    payload = {
        "v": FINGERPRINT_VERSION,
        "text": normalize_input_text(input_text),
        "vars": normalize_variables(variables) if isinstance(variables, dict) else None,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. CHEAP SIGNALS — the only things this module knows how to see
# ---------------------------------------------------------------------------

SHAPE_JSON_OBJECT = "json_object"
SHAPE_JSON_ARRAY = "json_array"
SHAPE_EMPTY = "empty"
SHAPE_TEXT = "text"


def output_shape(output: Any) -> str:
    """Four values, decided by parsing — never by guessing.

    A string that *starts* like JSON but does not parse is `text`, not a broken
    object: the shape describes what the output IS, and whether it should have
    been JSON is what the `output_valid_json` check answers separately.
    """
    if output is None:
        return SHAPE_EMPTY
    text = str(output).strip()
    if not text:
        return SHAPE_EMPTY
    if text[:1] in ("{", "["):
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return SHAPE_TEXT
        if isinstance(parsed, dict):
            return SHAPE_JSON_OBJECT
        if isinstance(parsed, list):
            return SHAPE_JSON_ARRAY
    return SHAPE_TEXT


def variable_key_shape(variables: Any) -> list[str]:
    """The sorted TOP-LEVEL key set. Keys only — never a value."""
    if not isinstance(variables, dict):
        return []
    return sorted(str(k) for k in variables.keys())


def _parsed_json_object(output: Any) -> Optional[dict]:
    if output is None:
        return None
    text = str(output).strip()
    if text[:1] != "{":
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


# ---------------------------------------------------------------------------
# 3. INFERRED CHECKS — structural, never semantic
# ---------------------------------------------------------------------------

def infer_checks(
    *,
    production_output: Any,
    node_results: Any,
    modal_json_fields: Optional[dict[str, str]] = None,
) -> list[dict]:
    """What OptiML can assert about a production output WITHOUT judgement.

    Returns `[{"code": ..., "passed": bool}]`.

    A CHECK THAT DOES NOT APPLY IS OMITTED — never recorded as passed, never
    recorded as failed. "We could not measure this" is not the same as "this
    failed", and a queue that renders the two identically teaches the reviewer
    to ignore the column. This is the same discipline as
    `benchmark._run_quality_checks`, which skips rather than scoring a check it
    has nothing to compare against.

    Nothing here is semantic and nothing here approves anything. These exist so
    a human reviewing 39 cases can skip the ones that are structurally fine and
    spend their attention on the ones that are not.
    """
    checks: list[dict] = []

    # Applies whenever the trace is readable at all. `unparseable` means we
    # learned nothing, so the check is omitted rather than passed.
    facts = attempts_mod.parse_step_results(node_results)
    if not facts.unparseable:
        checks.append({"code": CHECK_NO_EXECUTION_ERROR, "passed": not facts.has_error})

    # Always applicable: either there is an output or there is not.
    text = "" if production_output is None else str(production_output).strip()
    checks.append({"code": CHECK_OUTPUT_PRESENT, "passed": bool(text)})

    # Applicable only when the output is EVIDENTLY meant to be JSON — it opens
    # with a brace or bracket. Prose is not a failed JSON case, it is not a
    # JSON case; but a truncated `{"a": 1` is exactly the failure this catches.
    if text[:1] in ("{", "["):
        try:
            json.loads(text)
            valid = True
        except (ValueError, TypeError):
            valid = False
        checks.append({"code": CHECK_OUTPUT_VALID_JSON, "passed": valid})

    # The two field-level checks need a REFERENCE — the field set the rest of
    # this workload's JSON outputs agree on. With no reference there is nothing
    # to compare against, so both are omitted.
    obj = _parsed_json_object(production_output)
    if obj is not None and modal_json_fields:
        missing = [k for k in modal_json_fields if k not in obj]
        checks.append({"code": CHECK_OUTPUT_FIELDS_PRESENT, "passed": not missing})
        mismatched = [
            k for k, t in modal_json_fields.items()
            if k in obj and _json_type(obj[k]) != t
        ]
        checks.append({"code": CHECK_OUTPUT_FIELD_TYPES, "passed": not mismatched})

    return checks


def _modal_json_fields(outputs: Iterable[Any]) -> dict[str, str]:
    """Fields that at least 80% of this workload's JSON-object outputs carry.

    The reference the two field-level checks compare against. Derived from the
    workload's OWN observed outputs, never from a schema someone declared, so
    it cannot assert a requirement the workload has never actually met.

    Returns {} when fewer than three JSON objects were observed: two agreeing
    outputs are a coincidence, not a contract.
    """
    objects = [o for o in (_parsed_json_object(x) for x in outputs) if o is not None]
    if len(objects) < 3:
        return {}
    total = len(objects)
    counts: dict[str, int] = {}
    types: dict[str, dict[str, int]] = {}
    for obj in objects:
        for k, v in obj.items():
            key = str(k)
            counts[key] = counts.get(key, 0) + 1
            types.setdefault(key, {})
            t = _json_type(v)
            types[key][t] = types[key].get(t, 0) + 1
    fields: dict[str, str] = {}
    for key, n in sorted(counts.items()):
        if n / total < 0.8:
            continue
        # The type this field MOSTLY has. Ties break alphabetically so the
        # reference is deterministic.
        fields[key] = sorted(types[key].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return fields


# ---------------------------------------------------------------------------
# 4. DERIVATION
# ---------------------------------------------------------------------------

def _scan_runs(
    org_id: str,
    *,
    workflow_id: Optional[str],
    endpoint_slug: Optional[str],
    lookback_days: int,
) -> tuple[list[dict], dict]:
    """Production runs for one workload over a window, newest first.

    Returns (rows, coverage). `coverage` states the window, the row cap and
    whether the scan was truncated, so a partial sample is never presented as
    complete — the same contract `evidence.observed_production_traffic` uses.
    """
    since = _utc_now() - timedelta(days=max(1, lookback_days))
    coverage: dict[str, Any] = {
        "window_days": lookback_days,
        "since": _iso(since),
        "row_cap": _MAX_SCAN_ROWS,
        "truncated": False,
        "source": "workflow_runs(execution_mode=production)",
    }

    def _run(cols: str):
        q = (
            supabase.table("workflow_runs")
            .select(cols)
            .eq("org_id", org_id)
            .eq("execution_mode", "production")
            .gte("created_at", _iso(since))
        )
        if workflow_id:
            q = q.eq("workflow_id", workflow_id)
        if endpoint_slug:
            q = q.eq("endpoint_slug", endpoint_slug)
        return q.order("created_at", desc=True).limit(_MAX_SCAN_ROWS).execute()

    try:
        resp = _run(_RUN_COLS)
    except Exception:
        try:
            resp = _run(_RUN_COLS_LEGACY)
            coverage["capture_columns"] = "absent_pre_v12"
        except Exception as exc:  # pragma: no cover - network/db
            logger.warning("curation._scan_runs failed: %s", type(exc).__name__)
            return [], {**coverage, "error": "query_failed"}

    rows = getattr(resp, "data", None) or []
    if len(rows) >= _MAX_SCAN_ROWS:
        coverage["truncated"] = True
    return list(rows), coverage


def _redact_for_storage(run: dict) -> dict:
    """Put one run's content through the SINGLE persist-time boundary.

    `evidence_candidates` holds customer request and response content exactly
    as `golden_inputs` does, so it goes through `evidence_redaction`, at the
    write, and nowhere else. A run persisted BEFORE that boundary shipped still
    carries plaintext (history is immutable and is never rewritten), so the
    ruleset is re-run here rather than trusted for the row's age. Re-redacting
    an already-redacted value is a no-op: the markers match no pattern.

    The replay verdict comes from `evidence_redaction.replay_gate`, which is
    reused and NOT reimplemented — it already encodes "unknown is not clean".
    """
    stored_text, stored_vars, capture = redaction.persist_golden_input(
        run.get("input_text"), run.get("variables")
    )
    stored_output, output_capture = redaction.capture_input_text(run.get("final_output"))
    # The trace is the third persisted copy of the same request. It is not
    # stored on the candidate (there is no column and no replay use for it) but
    # it IS part of the same run, so it is judged by the same gate — exactly as
    # workflow_management.import_golden_input_from_production does.
    _safe_trace, trace_capture = redaction.capture_node_results(run.get("node_results"))

    gate = redaction.replay_gate(
        run.get("variables_capture"),
        capture,
        output_capture,
        {"node_results_capture": trace_capture},
    )

    full_capture = dict(capture)
    full_capture["output_capture"] = output_capture
    full_capture[redaction.NODE_RESULTS_CAPTURE_KEY] = trace_capture

    return {
        "input_text": stored_text,
        "variables": stored_vars,
        "production_output": stored_output,
        "capture": full_capture,
        "replay_eligible": bool(gate["eligible"]),
        "replay_reason_codes": list(gate["reasons"]),
        "redacted": redaction.REVIEW_REDACTED_INPUT in gate["reasons"],
        "redacted_kinds": list(gate["redacted_kinds"]),
    }


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    return domain.percentile(sorted(values), pct)


def _assign_buckets(entries: list[dict]) -> None:
    """Assign exactly one bucket per entry, in place.

    Bucket assignment is a property of the POPULATION, not of a row on its own:
    "long input" only means anything relative to this workload's other inputs.
    So the thresholds are measured here, from these entries, and stored in
    `bucket_signals` alongside the value that was compared — a bucket
    assignment must be auditable without re-scanning the runs.

    A bucket NEVER affects whether a candidate may be approved. It orders a
    queue and labels a count; that is all.
    """
    if not entries:
        return

    lengths = [float(e["signals"]["input_length"]) for e in entries]
    median_len = domain.percentile(sorted(lengths), 50)
    p90_len = domain.percentile(sorted(lengths), 90)

    costs = [float(e["signals"]["cost_usd"]) for e in entries if e["signals"].get("cost_usd") is not None]
    latencies = [
        float(e["signals"]["latency_ms"]) for e in entries if e["signals"].get("latency_ms") is not None
    ]
    p95_cost = _percentile(costs, 95)
    p95_latency = _percentile(latencies, 95)

    shape_counts: dict[str, int] = {}
    keyset_counts: dict[str, int] = {}
    for e in entries:
        s = e["signals"]
        shape_counts[s["output_shape"]] = shape_counts.get(s["output_shape"], 0) + 1
        key = json.dumps(s["variable_keys"], sort_keys=True)
        keyset_counts[key] = keyset_counts.get(key, 0) + 1

    def _modal(counts: dict[str, int]) -> Optional[str]:
        # Ties break alphabetically, so "which shape is normal" is deterministic
        # and the same population always produces the same buckets.
        if not counts:
            return None
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    modal_shape = _modal(shape_counts)
    modal_keyset = _modal(keyset_counts)

    thresholds = {
        "median_input_length": median_len,
        "p90_input_length": p90_len,
        "p95_cost_usd": p95_cost,
        "p95_latency_ms": p95_latency,
        "modal_output_shape": modal_shape,
        "population": len(entries),
    }

    for e in entries:
        s = e["signals"]
        s["thresholds"] = thresholds

        if s.get("had_error"):
            e["bucket"] = BUCKET_FAILURE
        elif modal_shape is not None and s["output_shape"] != modal_shape:
            e["bucket"] = BUCKET_UNUSUAL_OUTPUT
        elif (
            modal_keyset is not None
            and json.dumps(s["variable_keys"], sort_keys=True) != modal_keyset
        ):
            e["bucket"] = BUCKET_UNUSUAL_VARIABLES
        elif (
            p90_len is not None
            and median_len is not None
            and s["input_length"] >= p90_len
            # AND at least twice the median. p90 alone would label the top
            # decile "long" on a workload whose inputs are all the same size,
            # which says nothing. Both conditions means genuinely long.
            and s["input_length"] >= 2 * max(median_len, 1)
        ):
            e["bucket"] = BUCKET_LONG_INPUT
        elif (
            (p95_cost is not None and s.get("cost_usd") is not None and s["cost_usd"] > p95_cost)
            or (
                p95_latency is not None
                and s.get("latency_ms") is not None
                and s["latency_ms"] > p95_latency
            )
        ):
            e["bucket"] = BUCKET_OUTLIER
        else:
            e["bucket"] = BUCKET_COMMON

    # Carve a deterministic chronological spread out of the residue, so a queue
    # of otherwise-identical `common` cases is not all from one afternoon.
    commons = sorted(
        [e for e in entries if e["bucket"] == BUCKET_COMMON],
        key=lambda e: (e["captured_at"], e["fingerprint"]),
    )
    # Never relabel more than half: `random_coverage` is a spread THROUGH the
    # common cases, not a rename of them.
    take = min(RANDOM_COVERAGE_TARGET, len(commons) // 2)
    if take > 0:
        step = len(commons) / take
        for i in range(take):
            commons[int(i * step)]["bucket"] = BUCKET_RANDOM


def derive_candidates(
    org_id: str,
    workload: dict,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict:
    """Turn one workload's production runs into deduplicated candidate rows.

    IDEMPOTENT. Re-running over the same window inserts nothing new: the
    fingerprint is UNIQUE per (org_id, workload_id) in the schema, and this
    function only ever inserts a fingerprint it did not find. It NEVER changes
    a candidate's `state`, never touches `expected_output`, and never writes
    `golden_inputs`. Buckets and inferred checks are refreshed on UNREVIEWED
    rows only — a reviewed row keeps the bucket and the checks it was reviewed
    under, because those were part of what the human saw.

    Returns measured facts: {runs_observed, distinct, inserted, refreshed,
    coverage}. `runs_observed` is None when the scan failed, never 0 — "we did
    not look" and "there was nothing" are different answers.
    """
    workload_id = str(workload.get("id"))
    workflow_id = workloads_mod.resolve_workflow_id(org_id, workload)
    endpoint_slug = (
        str(workload.get("identity_ref"))
        if workload.get("identity_kind") == "endpoint" and workload.get("identity_ref")
        else None
    )

    runs, coverage = _scan_runs(
        org_id,
        workflow_id=workflow_id,
        endpoint_slug=endpoint_slug,
        lookback_days=lookback_days,
    )
    if coverage.get("error"):
        return {
            "runs_observed": None,
            "distinct": None,
            "inserted": 0,
            "refreshed": 0,
            "coverage": coverage,
        }

    # ── Group by fingerprint. THE EXEMPLAR IS DETERMINISTIC: the earliest run
    # by (created_at, id), so the same set of runs always yields the same
    # candidate rows regardless of the order the scan returned them in.
    groups: dict[str, dict] = {}
    for run in runs:
        stored = _redact_for_storage(run)
        fp = fingerprint_of(stored["input_text"], stored["variables"])
        created = str(run.get("created_at") or "")
        rid = str(run.get("id") or "")
        g = groups.get(fp)
        if g is None:
            groups[fp] = {
                "fingerprint": fp,
                "stored": stored,
                "run": run,
                "occurrences": 1,
                "first": (created, rid),
                "last": created,
                "outputs": [stored["production_output"]],
            }
            continue
        g["occurrences"] += 1
        g["outputs"].append(stored["production_output"])
        if created > g["last"]:
            g["last"] = created
        if (created, rid) < g["first"]:
            g["first"] = (created, rid)
            g["stored"] = stored
            g["run"] = run

    modal_fields = _modal_json_fields(
        out for g in groups.values() for out in g["outputs"]
    )

    entries: list[dict] = []
    for fp, g in groups.items():
        run = g["run"]
        stored = g["stored"]
        facts = attempts_mod.parse_step_results(run.get("node_results"))
        signals = {
            "input_length": len(normalize_input_text(stored["input_text"])),
            "output_shape": output_shape(stored["production_output"]),
            "variable_keys": variable_key_shape(stored["variables"]),
            # `unparseable` means we could not read the trace, so "did it
            # error" is unknown. An unknown is not a failure and must not be
            # bucketed as one.
            "had_error": (False if facts.unparseable else facts.has_error),
            "error_known": not facts.unparseable,
            "cost_usd": (float(run["total_cost"]) if run.get("total_cost") is not None else None),
            "latency_ms": (
                float(run["total_latency_ms"]) if run.get("total_latency_ms") is not None else None
            ),
        }
        entries.append({
            "fingerprint": fp,
            "stored": stored,
            "run": run,
            "occurrences": g["occurrences"],
            "captured_at": g["first"][0],
            "last_seen_at": g["last"],
            "source_run_id": g["first"][1],
            "signals": signals,
            "checks": infer_checks(
                production_output=stored["production_output"],
                node_results=run.get("node_results"),
                modal_json_fields=modal_fields,
            ),
        })

    _assign_buckets(entries)

    existing = _existing_by_fingerprint(org_id, workload_id)
    if existing is None:
        return {
            "runs_observed": len(runs),
            "distinct": len(entries),
            "inserted": 0,
            "refreshed": 0,
            "coverage": {**coverage, "error": "candidate_storage_unavailable"},
        }

    inserted = refreshed = 0
    capped = False
    # Deterministic write order: highest-signal bucket first, then oldest. When
    # MAX_CANDIDATES_PER_WORKLOAD binds, what survives is the diverse tail, not
    # whichever rows the scan happened to return first.
    entries.sort(key=lambda e: (BUCKET_PRIORITY.index(e["bucket"]), e["captured_at"], e["fingerprint"]))

    for e in entries:
        row = existing.get(e["fingerprint"])
        if row is None:
            if len(existing) + inserted >= MAX_CANDIDATES_PER_WORKLOAD:
                capped = True
                continue
            if _insert_candidate(org_id, workload_id, workflow_id, e):
                inserted += 1
            continue
        if str(row.get("state")) in REVIEWABLE_STATES and _refresh_candidate(org_id, row, e):
            refreshed += 1

    return {
        "runs_observed": len(runs),
        "distinct": len(entries),
        "inserted": inserted,
        "refreshed": refreshed,
        "coverage": {
            **coverage,
            "dedup_ratio": (round(len(runs) / len(entries), 2) if entries else None),
            "candidate_cap": MAX_CANDIDATES_PER_WORKLOAD,
            "candidate_cap_reached": capped,
            "fingerprint_version": FINGERPRINT_VERSION,
        },
    }


def _existing_by_fingerprint(org_id: str, workload_id: str) -> Optional[dict[str, dict]]:
    """Fingerprint -> row for this workload, or None when the table is unreadable.

    None and {} are different answers: {} means the workload has no candidates,
    None means we could not tell, and the caller must not insert on None.
    """
    rows = _list_candidates(org_id, workload_id)
    if rows is None:
        return None
    return {str(r.get("fingerprint")): r for r in rows}


def _list_candidates(org_id: str, workload_id: str) -> Optional[list[dict]]:
    try:
        resp = (
            supabase.table("evidence_candidates")
            .select(CANDIDATE_COLS)
            .eq("org_id", org_id)
            .eq("workload_id", workload_id)
            .limit(MAX_CANDIDATES_PER_WORKLOAD)
            .execute()
        )
        return list(getattr(resp, "data", None) or [])
    except Exception as exc:  # pragma: no cover - network/db
        logger.warning("curation._list_candidates failed: %s", type(exc).__name__)
        return None


def _insert_candidate(
    org_id: str, workload_id: str, workflow_id: Optional[str], entry: dict
) -> bool:
    stored = entry["stored"]
    payload = {
        "org_id": org_id,
        "workload_id": workload_id,
        "workflow_id": workflow_id,
        "source_run_id": entry["source_run_id"] or None,
        "captured_at": entry["captured_at"] or None,
        "last_seen_at": entry["last_seen_at"] or None,
        "occurrences": entry["occurrences"],
        "input_text": stored["input_text"],
        "variables": stored["variables"],
        "production_output": stored["production_output"],
        # THE PROPOSAL, and the reason this column is not called
        # `expected_output`. Written explicitly as a pair so the distinction is
        # visible AT THE INSERT and not only in the schema: the proposal is
        # seeded from what production emitted, and the expected output — the
        # only value that ever reaches `golden_inputs` — stays NULL until a
        # human decides. Relying on the column default would make the most
        # important line in this file the one that is not there.
        "proposed_expected_output": stored["production_output"],
        "expected_output": None,
        "capture": stored["capture"],
        "replay_eligible": stored["replay_eligible"],
        "replay_reason_codes": stored["replay_reason_codes"],
        "redacted": stored["redacted"],
        "redacted_kinds": stored["redacted_kinds"],
        "fingerprint": entry["fingerprint"],
        "fingerprint_version": FINGERPRINT_VERSION,
        "bucket": entry["bucket"],
        "bucket_signals": entry["signals"],
        "checks": entry["checks"],
        "state": STATE_CAPTURED,
        # The four columns that together say "no human has touched this",
        # written explicitly rather than left to the column defaults. Derivation
        # creates nothing approved, and the code that says so should be
        # readable at the insert instead of inferred from its absence.
        "reviewed_by": None,
        "reviewed_at": None,
        "review_acknowledged_redaction": False,
        "golden_input_id": None,
    }
    try:
        supabase.table("evidence_candidates").insert(payload).execute()
        return True
    except Exception as exc:
        # A UNIQUE violation here is the dedup index doing its job under a
        # concurrent derivation, not an error worth surfacing.
        logger.info("curation._insert_candidate skipped: %s", type(exc).__name__)
        return False


def _refresh_candidate(org_id: str, row: dict, entry: dict) -> bool:
    """Update the population-derived fields on an UNREVIEWED candidate.

    Deliberately narrow. `state`, `expected_output`, `reviewed_by`,
    `reviewed_at` and `golden_input_id` are never written here — this is a
    read-model refresh, and it must be impossible for a derivation pass to
    advance a lifecycle. The org filter is on the UPDATE itself, not merely on
    a prior read, so the check and the write are one statement.
    """
    patch = {
        "occurrences": entry["occurrences"],
        "last_seen_at": entry["last_seen_at"] or None,
        "bucket": entry["bucket"],
        "bucket_signals": entry["signals"],
        "checks": entry["checks"],
        "updated_at": _iso(_utc_now()),
    }
    try:
        (
            supabase.table("evidence_candidates")
            .update(patch)
            .eq("id", str(row["id"]))
            .eq("org_id", org_id)
            .execute()
        )
        return True
    except Exception as exc:  # pragma: no cover - network/db
        logger.warning("curation._refresh_candidate failed: %s", type(exc).__name__)
        return False


# ---------------------------------------------------------------------------
# 5. COUNTERS — four numbers that must never be collapsed into one
# ---------------------------------------------------------------------------

def counters(rows: list[dict], *, runs_observed: Optional[int], required: int) -> dict:
    """captured / reviewed / approved / required, kept strictly apart.

    "39/30 cases" is a LIE when captured is not the same as trustworthy. A
    captured candidate carries a production output that nobody has looked at;
    presenting it as progress toward a benchmark is exactly the mistake this
    whole feature exists to prevent. Four numbers, four meanings:

      runs_observed      production runs scanned in the window. None when the
                         scan failed — never 0, which would read as "no traffic".
      distinct_captured  candidates after dedup. THE HONEST DENOMINATOR.
      reviewed           a human reached a decision, any decision.
      approved           reviewed AND usable as a replay case. The only one of
                         the four that counts toward benchmarkability.
      required           the floor `approved` must reach.
    """
    states = [str(r.get("state") or "") for r in rows]
    return {
        "runs_observed": runs_observed,
        "distinct_captured": len(rows),
        "reviewed": sum(1 for s in states if s in REVIEWED_STATES),
        "approved": sum(1 for s in states if s in APPROVED_STATES),
        "required": required,
    }


def bucket_counts(rows: list[dict]) -> list[dict]:
    """[{code, count}] over every bucket the population contains, richest first.

    Codes only. The frontend turns `{"code": "long_input", "count": 6}` into
    "6 long-input"; changing that phrasing is never an API change.
    """
    counts: dict[str, int] = {}
    for r in rows:
        code = str(r.get("bucket") or BUCKET_COMMON)
        counts[code] = counts.get(code, 0) + 1
    return [
        {"code": code, "count": counts[code]}
        for code in sorted(counts, key=lambda c: (-counts[c], BUCKET_PRIORITY.index(c)
                                                  if c in BUCKET_PRIORITY else 99, c))
    ]


def required_cases(org_id: str, workload_id: str) -> int:
    """The approved-case floor, taken from the same place a benchmark takes it.

    Resolved from the workload's effective policy (`min_sample_size`), falling
    back to `benchmark.DEFAULT_MIN_SAMPLE_SIZE`. Read from the benchmark module
    rather than duplicated, so a readiness verdict here and a benchmark's own
    sample-size refusal can never disagree about the number.
    """
    try:
        policy = policies_mod.get_effective_policy(org_id, workload_id)
        configured = policies_mod.constraints_of(policy).get("min_sample_size")
        if configured:
            return int(configured)
    except Exception as exc:  # pragma: no cover
        logger.warning("curation.required_cases policy read failed: %s", type(exc).__name__)
    try:
        from optimization import benchmark as benchmark_mod

        return int(benchmark_mod.DEFAULT_MIN_SAMPLE_SIZE)
    except Exception:  # pragma: no cover
        return FALLBACK_MIN_APPROVED_CASES


# ---------------------------------------------------------------------------
# 6. QUALITY SIGNAL and READINESS
# ---------------------------------------------------------------------------

#: The check types that produce a MEASURABLE quality verdict. Mirrors
#: `benchmark._QUALITY_CHECK_PROVENANCE` exactly — if these two ever disagree,
#: readiness would promise a quality number the benchmark cannot produce.
#: `model_graded` is absent on purpose and must stay absent: an LLM judge is
#: not a quality signal here, and `benchmark._run_quality_checks` already
#: excludes it from the measured verdict.
_USABLE_CHECK_TYPES = {"deterministic": "deterministic", "structural": "schema", "format": "schema"}


def quality_signal(org_id: str, workflow_id: Optional[str]) -> dict:
    """Can a replay of this workload produce a quality VERDICT at all?

    Reads the workload's own `eval_suites` row — the same rows the eval UI
    writes and the same rows `benchmark._load_eval_checks` reads.

    Returns measured facts or NULL with a reason. `usable` is None (not False)
    when the suite could not be read: "this workload has no quality check" and
    "we could not tell" are different findings, and reporting the second as the
    first would tell a customer to fix something that may already be fine.
    """
    out: dict[str, Any] = {
        "usable": None,
        "provenance": None,
        "check_types": [],
        "reason_codes": [],
    }
    if not workflow_id:
        # A direct-inference workload has no workflow and therefore no eval
        # suite. That is a real observation, not a read failure.
        out["usable"] = False
        out["reason_codes"] = ["evidence_quality_check_absent"]
        return out
    try:
        resp = (
            supabase.table("eval_suites")
            .select("id, org_id, workflow_id, checks, enabled")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover - network/db
        logger.warning("curation.quality_signal failed: %s", type(exc).__name__)
        out["reason_codes"] = ["quality_not_measured"]
        return out

    raw = (rows[0].get("checks") or []) if rows else []
    types = sorted({
        str(c.get("type") or "deterministic").lower()
        for c in raw
        if isinstance(c, dict) and c.get("enabled", True)
    })
    usable_types = [t for t in types if t in _USABLE_CHECK_TYPES]
    out["check_types"] = types
    out["usable"] = bool(usable_types)
    if usable_types:
        out["provenance"] = max(
            (_USABLE_CHECK_TYPES[t] for t in usable_types),
            key=domain.provenance_rank,
        )
    else:
        out["reason_codes"] = ["evidence_quality_check_absent"]
    return out


def readiness(
    *,
    rows: list[dict],
    counts: dict,
    signal: dict,
) -> dict:
    """Is this workload optimization-ready?

    THREE conditions, and all three must hold. They are not blended into a
    score, because a score cannot be argued with and these three can:

      1. enough APPROVED cases (never captured cases — see `counters`);
      2. those approved cases span enough distinct buckets, capped at what the
         candidate population actually CONTAINS. A workload with one output
         shape and no failures genuinely has one bucket, and demanding
         diversity that does not exist would make it permanently un-ready for
         a reason nobody can act on;
      3. a usable quality check exists — deterministic, structural or format.
         NEVER an LLM judge.

    Returns `{ready, reason_codes, reasons}`. `reason_codes` is the contracted
    list of bare codes; `reasons` carries the same codes WITH the facts behind
    them (observed, required, unit), because a frontend that has to render "12
    of 20" should not have to guess the numbers. All wording is the frontend's.

    Ready is NEVER a trigger. Nothing here starts a benchmark or spends money.
    """
    reasons: list[dict] = []

    approved = counts["approved"]
    required = counts["required"]
    if approved < required:
        reasons.append(domain.reason(
            "sample_size_below_threshold",
            observed=approved, required=required,
            unit="cases", dataset="evidence_candidates",
        ))
        unreviewed = counts["distinct_captured"] - counts["reviewed"]
        if unreviewed > 0:
            reasons.append(domain.reason(
                "evidence_awaiting_review",
                observed=unreviewed, unit="candidates",
            ))
    if approved == 0:
        reasons.append(domain.reason(
            "no_replay_cases", observed=0, unit="cases", dataset="evidence_candidates",
        ))

    available_buckets = {str(r.get("bucket") or BUCKET_COMMON) for r in rows}
    approved_buckets = {
        str(r.get("bucket") or BUCKET_COMMON)
        for r in rows
        if str(r.get("state") or "") in APPROVED_STATES
    }
    bucket_floor = min(MIN_APPROVED_BUCKETS, len(available_buckets)) if available_buckets else 0
    if len(approved_buckets) < bucket_floor:
        reasons.append(domain.reason(
            "coverage_gap",
            observed=len(approved_buckets), required=bucket_floor,
            unit="buckets", dataset="evidence_candidates",
        ))

    if signal.get("usable") is not True:
        for code in (signal.get("reason_codes") or ["quality_not_measured"]):
            reasons.append(domain.reason(code))

    codes: list[str] = []
    for r in reasons:
        if r["code"] not in codes:
            codes.append(r["code"])

    return {"ready": not reasons, "reason_codes": codes, "reasons": reasons}


# ---------------------------------------------------------------------------
# 7. PRESENTATION
# ---------------------------------------------------------------------------

def candidate_row_to_response(row: dict) -> dict:
    """One queue entry, exactly as the API contract defines it.

    `production_output` and `proposed_expected_output` are BOTH returned and
    both are the same value until a human edits one. That redundancy is the
    contract: the reviewer has to see that the thing they are being asked to
    approve is a proposal derived from an observation, not a fact.
    """
    capture = row.get("capture") if isinstance(row.get("capture"), dict) else {}
    return {
        "id": str(row["id"]),
        "state": row.get("state") or STATE_CAPTURED,
        "bucket": row.get("bucket") or BUCKET_COMMON,
        "input_text": row.get("input_text"),
        "variables": row.get("variables"),
        "production_output": row.get("production_output"),
        "proposed_expected_output": row.get("proposed_expected_output"),
        # NULL until a human decided. Never defaulted to the proposal.
        "expected_output": row.get("expected_output"),
        "checks": [
            {"code": c.get("code"), "passed": bool(c.get("passed"))}
            for c in (row.get("checks") or [])
            if isinstance(c, dict)
        ],
        "capture": {
            "redacted": row.get("redacted"),
            "redacted_kinds": list(row.get("redacted_kinds") or []),
            "replay_eligible": row.get("replay_eligible"),
            "reason_codes": list(row.get("replay_reason_codes") or []),
            "redaction_version": capture.get("redaction_version"),
        },
        "occurrences": row.get("occurrences"),
        "source_run_id": (str(row["source_run_id"]) if row.get("source_run_id") else None),
        "captured_at": row.get("captured_at"),
        "reviewed_at": row.get("reviewed_at"),
        "acknowledged_redaction": bool(row.get("review_acknowledged_redaction")),
        "golden_input_id": (str(row["golden_input_id"]) if row.get("golden_input_id") else None),
    }


def _queue_sort_key(row: dict):
    bucket = str(row.get("bucket") or BUCKET_COMMON)
    rank = BUCKET_PRIORITY.index(bucket) if bucket in BUCKET_PRIORITY else len(BUCKET_PRIORITY)
    return (rank, str(row.get("captured_at") or ""), str(row.get("id")))


def queue(org_id: str, workload_id: str, *, limit: int = 25) -> dict:
    """Unreviewed candidates, highest-signal bucket first, oldest first within.

    Ordered rather than random so a reviewer who stops after ten has spent
    those ten on the cases most likely to be missing from a hand-built set —
    failures first, then unusual shapes. Deterministic, so re-opening the queue
    shows the same thing in the same order.

    Reviewed candidates are excluded: this is a work list, not a record.
    """
    rows = _list_candidates(org_id, workload_id)
    if rows is None:
        return {"candidates": [], "remaining": None, "unavailable": REFUSE_STORAGE_UNAVAILABLE}
    pending = [r for r in rows if str(r.get("state") or "") in REVIEWABLE_STATES]
    pending.sort(key=_queue_sort_key)
    n = max(1, min(int(limit or 25), 200))
    page = pending[:n]
    return {
        "candidates": [candidate_row_to_response(r) for r in page],
        "remaining": max(0, len(pending) - len(page)),
    }


def counters_for(org_id: str, workload_id: str) -> Optional[dict]:
    """The four counters for one workload, WITHOUT rescanning production.

    A review is not the moment to re-derive: the caller has just made a
    decision and needs to see what it did to the numbers, not a fresh traffic
    scan. `runs_observed` is therefore None here — the honest value for a
    question this call did not ask.
    """
    rows = _list_candidates(org_id, workload_id)
    if rows is None:
        return None
    return counters(
        rows, runs_observed=None, required=required_cases(org_id, workload_id)
    )


def evidence_overview(
    org_id: str,
    workload: dict,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    derive: bool = True,
) -> dict:
    """The GET /evidence payload: counters, readiness, quality signal, buckets.

    WHY THIS REFRESHES. Derivation runs here rather than behind a separate
    action the frontend would have to remember to call. It is a MATERIALISED
    READ, not a mutation of anything a user can see: it is a pure function of
    `workflow_runs`, it is idempotent under the fingerprint UNIQUE index, it
    cannot change a candidate's state, and it never writes `golden_inputs`. The
    one thing in this whole module that spends money or changes a conclusion —
    approval — is a POST, and stays one.
    """
    workload_id = str(workload.get("id"))
    workflow_id = workloads_mod.resolve_workflow_id(org_id, workload)

    derivation = (
        derive_candidates(org_id, workload, lookback_days=lookback_days)
        if derive
        else {"runs_observed": None, "coverage": {"derived": False}}
    )

    rows = _list_candidates(org_id, workload_id)
    if rows is None:
        return {
            "counters": {
                "runs_observed": derivation.get("runs_observed"),
                "distinct_captured": None,
                "reviewed": None,
                "approved": None,
                "required": required_cases(org_id, workload_id),
            },
            "readiness": {
                "ready": False,
                "reason_codes": [REFUSE_STORAGE_UNAVAILABLE],
                "reasons": [],
            },
            "quality_signal": quality_signal(org_id, workflow_id),
            "buckets": [],
            "coverage": derivation.get("coverage", {}),
        }

    counts = counters(
        rows,
        runs_observed=derivation.get("runs_observed"),
        required=required_cases(org_id, workload_id),
    )
    signal = quality_signal(org_id, workflow_id)
    return {
        "counters": counts,
        "readiness": readiness(rows=rows, counts=counts, signal=signal),
        "quality_signal": signal,
        "buckets": bucket_counts(rows),
        "coverage": derivation.get("coverage", {}),
    }


# ---------------------------------------------------------------------------
# 8. REVIEW — the only path from captured traffic to replay evidence
# ---------------------------------------------------------------------------

def review(
    org_id: str,
    candidate: dict,
    *,
    decision: str,
    expected_output: Optional[str] = None,
    acknowledge_redaction: bool = False,
    reviewer_id: Optional[str] = None,
) -> dict:
    """Record one human decision, and — only on approval — create the case.

    `candidate` MUST be a row already proven to belong to `org_id` by
    `resource_access.get_evidence_candidate_for_org`. Every write below carries
    the org filter as well, so the check and the write are one statement rather
    than a check-then-act window.

    THE INVARIANT: a `golden_inputs` row is created HERE and only here, from
    `expected_output` — the human's answer — and never from `production_output`.
    There is no other code path in this module that inserts into
    `golden_inputs`, and no argument to this function that skips the decision.

    IDEMPOTENT BY CONSTRUCTION, in two layers. The lifecycle claim is a single
    conditional UPDATE filtered on `state IN (captured, proposed_for_review)`;
    a second identical request matches zero rows and creates nothing. Beneath
    that, `uq_evidence_candidates_golden_input` refuses a second golden input
    for the same candidate even if the claim were somehow bypassed.

    Returns {candidate, golden_input_id, decision_recorded}. Raises
    CurationRefused with a structured detail for every refusal.
    """
    if decision not in DECISIONS:
        raise CurationRefused({"code": REFUSE_UNKNOWN_DECISION, "accepted": list(DECISIONS)})

    candidate_id = str(candidate["id"])
    current_state = str(candidate.get("state") or STATE_CAPTURED)
    target_state = _DECISION_TO_STATE[decision]

    # ── Already reviewed? ──────────────────────────────────────────────────
    if current_state not in REVIEWABLE_STATES:
        if current_state == target_state:
            # The same decision, again. Idempotent: nothing is written, the
            # existing outcome is returned, and no second golden input exists
            # because none was created.
            return {
                "candidate": candidate,
                "golden_input_id": (
                    str(candidate["golden_input_id"]) if candidate.get("golden_input_id") else None
                ),
                "decision_recorded": False,
            }
        # A DIFFERENT decision on a candidate a human already decided. Refused
        # rather than silently applied: the first decision is a record, and
        # overwriting it would erase who concluded what.
        raise CurationRefused({
            "code": REFUSE_ALREADY_REVIEWED,
            "state": current_state,
            "requested": decision,
        })

    # ── The redaction gate. Reused, not reimplemented. ─────────────────────
    replay_eligible = candidate.get("replay_eligible")
    gate_codes = list(candidate.get("replay_reason_codes") or [])
    needs_ack = decision in (DECISION_APPROVE, DECISION_EDIT) and replay_eligible is not True
    if needs_ack and not acknowledge_redaction:
        # Visible, but never SILENTLY replayable. The candidate is not hidden,
        # not deleted and not altered; a human may still approve it, but must
        # say so, and the saying-so is recorded on the row.
        raise CurationRefused({
            "code": REFUSE_REDACTION_UNACKNOWLEDGED,
            "reasons": gate_codes or [redaction.REVIEW_UNKNOWN_PROVENANCE],
            "redacted_kinds": list(candidate.get("redacted_kinds") or []),
            "candidate_preserved": True,
            "resolution": "resubmit_with_acknowledge_redaction_or_edit_expected_output",
        })

    # ── What the expected output will be ───────────────────────────────────
    new_expected: Optional[str] = None
    if decision == DECISION_APPROVE:
        # Approval means "the production output IS the right answer" — an
        # affirmative human claim about this specific case, which is the whole
        # difference between this and reading `golden_inputs` off the traffic.
        new_expected = candidate.get("proposed_expected_output")
        if new_expected is None or str(new_expected).strip() == "":
            # There is nothing to approve. Refused rather than storing an empty
            # expectation, which would make every candidate trivially pass a
            # deterministic check.
            raise CurationRefused({
                "code": REFUSE_EXPECTED_OUTPUT_REQUIRED,
                "reason": "no_production_output_to_approve",
            })
    elif decision == DECISION_EDIT:
        if expected_output is None or str(expected_output).strip() == "":
            raise CurationRefused({"code": REFUSE_EXPECTED_OUTPUT_REQUIRED})
        # A reviewer-supplied expected output is customer content arriving over
        # the API and destined for two persisted evidence fields, so it crosses
        # the SAME write boundary as everything else here.
        new_expected, _meta = redaction.capture_input_text(expected_output)
        if new_expected is None:
            raise CurationRefused({
                "code": REFUSE_EXPECTED_OUTPUT_REQUIRED,
                "reason": "expected_output_not_storable",
            })

    if decision in (DECISION_APPROVE, DECISION_EDIT) and not candidate.get("workflow_id"):
        # `golden_inputs.workflow_id` is required, and a direct-inference
        # workload has no workflow. Refused with a code rather than inventing a
        # workflow id or writing a case nothing can replay.
        raise CurationRefused({
            "code": REFUSE_NO_WORKFLOW_FOR_REPLAY,
            "workload_id": str(candidate.get("workload_id")),
        })

    # ── Claim the lifecycle. ONE conditional, org-scoped UPDATE. ───────────
    now = _iso(_utc_now())
    patch: dict[str, Any] = {
        "state": target_state,
        "reviewed_by": reviewer_id,
        "reviewed_at": now,
        "updated_at": now,
        "review_acknowledged_redaction": bool(needs_ack and acknowledge_redaction),
    }
    if new_expected is not None:
        patch["expected_output"] = new_expected

    try:
        claim = (
            supabase.table("evidence_candidates")
            .update(patch)
            .eq("id", candidate_id)
            .eq("org_id", org_id)
            .in_("state", list(REVIEWABLE_STATES))
            .execute()
        )
    except Exception as exc:
        logger.warning("curation.review claim failed: %s", type(exc).__name__)
        raise CurationRefused({"code": REFUSE_STORAGE_UNAVAILABLE}) from exc

    claimed = getattr(claim, "data", None) or []
    if not claimed:
        # Somebody else reviewed it between the read and the write. Re-read and
        # return their outcome rather than racing them.
        fresh = _get_candidate(org_id, candidate_id) or candidate
        return {
            "candidate": fresh,
            "golden_input_id": (
                str(fresh["golden_input_id"]) if fresh.get("golden_input_id") else None
            ),
            "decision_recorded": False,
        }

    row = claimed[0] if isinstance(claimed[0], dict) else candidate

    if decision not in (DECISION_APPROVE, DECISION_EDIT):
        # rejected / not_useful are terminal and produce NO replay case. That
        # is the point of having them: a decision that something is not
        # evidence has to be recordable.
        return {"candidate": row, "golden_input_id": None, "decision_recorded": True}

    golden_id = _create_golden_input(
        org_id,
        candidate=candidate,
        expected_output=new_expected,
        replay_eligible=(replay_eligible is True),
    )
    if golden_id:
        try:
            linked = (
                supabase.table("evidence_candidates")
                .update({"golden_input_id": golden_id, "updated_at": _iso(_utc_now())})
                .eq("id", candidate_id)
                .eq("org_id", org_id)
                .execute()
            )
            data = getattr(linked, "data", None) or []
            if data and isinstance(data[0], dict):
                row = data[0]
        except Exception as exc:  # pragma: no cover - network/db
            # The case exists and the decision is recorded; only the link is
            # missing. Reported as a measured fact (golden_input_id is returned
            # from the insert), never as a silent success.
            logger.warning("curation.review link failed: %s", type(exc).__name__)

    return {"candidate": row, "golden_input_id": golden_id, "decision_recorded": True}


def _create_golden_input(
    org_id: str,
    *,
    candidate: dict,
    expected_output: Optional[str],
    replay_eligible: bool,
) -> Optional[str]:
    """The ONE bridge into `golden_inputs`. Called only from `review`.

    Writes the existing `golden_inputs` shape unchanged, so all ten readers of
    that table keep working and none of them needs to know this feature exists.
    """
    # Re-run the boundary on the values being WRITTEN. They were redacted on
    # the way into `evidence_candidates`, and re-redacting a marker is a no-op,
    # but the boundary is defined as "at the persist", and this is a persist.
    stored_text, stored_vars, _capture = redaction.persist_golden_input(
        candidate.get("input_text"), candidate.get("variables")
    )
    payload = {
        "org_id": org_id,
        "workflow_id": str(candidate["workflow_id"]),
        # NO CUSTOMER CONTENT IN THE NAME. The fingerprint prefix identifies
        # the case without quoting the customer's input into a label that ends
        # up in dropdowns, logs and screenshots.
        "name": f"Curated {str(candidate.get('fingerprint') or '')[:8]}",
        "input_text": stored_text,
        "variables": stored_vars,
        # The human's answer. NEVER candidate["production_output"].
        "expected_output": expected_output,
        "source": GOLDEN_SOURCE_CURATED if replay_eligible else GOLDEN_SOURCE_CURATED_REDACTED,
        "source_run_id": candidate.get("source_run_id"),
    }
    try:
        result = supabase.table("golden_inputs").insert(payload).execute()
    except Exception as exc:
        logger.warning("curation._create_golden_input failed: %s", type(exc).__name__)
        raise CurationRefused({"code": REFUSE_STORAGE_UNAVAILABLE}) from exc

    rows = getattr(result, "data", None) or []
    if not rows:
        raise CurationRefused({"code": REFUSE_STORAGE_UNAVAILABLE})
    return str(rows[0]["id"])


def _get_candidate(org_id: str, candidate_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("evidence_candidates")
            .select(CANDIDATE_COLS)
            .eq("id", candidate_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover - network/db
        logger.warning("curation._get_candidate failed: %s", type(exc).__name__)
        return None
