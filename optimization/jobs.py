"""
Benchmark execution as a PERSISTED JOB.

WHY THIS MODULE EXISTS
----------------------
`POST /workloads/{id}/optimize` used to run the whole benchmark inside the
request and await the verdict. The first real production run — 140 replay cases
across 7 arms — took ~28 minutes. Railway's edge timeout is 300 seconds. The
client got a connection error at 5 minutes; the server kept working for another
23 and wrote its evidence to a benchmark row nobody was told the id of. The
benchmark succeeded and the API told nobody. A 20-minute HTTP connection is not
a contract; it is a bet that nothing between the two processes has an opinion
about idle sockets.

THE JOB IS THE BENCHMARK ROW. There is no separate `optimization_jobs` table.
`optimization_benchmarks` already carries `status`, `started_at` and
`completed_at`, and the id the caller needs to poll is the benchmark id it will
eventually cite as evidence. A second table would mean two ids, two lifecycles
and a join that can disagree with itself.

TWO STATE AXES, DELIBERATELY SEPARATE
-------------------------------------
`status`          the LIFECYCLE axis. Coarse, CHECK-constrained, already exists,
                  already indexed, already read by everything else in the
                  system: queued -> running -> completed | failed.
`progress_state`  the PHASE axis. Fine-grained, new, exists so a frontend can
                  render where a 28-minute run actually is.

Collapsing them would have meant widening the `status` CHECK with eight phase
names and teaching every existing reader that `stage_2` means "still running".
Separating them means every existing query — `.eq("status", "completed")`,
the org/status index, the RLS policies — keeps working unchanged.

NOTHING HERE MEASURES ANYTHING. This module starts, tracks, and reaps runs. It
never touches an arm, a metric, a conclusion or an evidence-maturity value. Every number
it writes is a count of work already done (cases run, arms finished), never an
estimate of work remaining and never a prediction of a verdict.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The lifecycle axis: `optimization_benchmarks.status`
# ---------------------------------------------------------------------------
#
# `pending` predates this module. It is what _insert_benchmark wrote between the
# row's creation and the run's start, and historical rows still carry it. It is
# retained as an ACTIVE status rather than migrated away, because rewriting the
# status of rows that already exist — including the two preserved historical
# benchmarks — to make a new vocabulary look tidy is exactly the kind of history
# rewrite this codebase refuses elsewhere.

STATUS_PENDING = "pending"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

#: A job in one of these has not reached a verdict. It is either being worked on
#: or it is orphaned; there is no third possibility, which is what makes the
#: heartbeat lease below decidable.
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_QUEUED, STATUS_RUNNING)
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED)


# ---------------------------------------------------------------------------
# The phase axis: `optimization_benchmarks.progress_state`
# ---------------------------------------------------------------------------
#
# STAGE NAMES ARE DATA, NOT FREE TEXT. `stage_2` is not a string a handler
# invents; it is produced by stage_state(k) from the stage index the staged
# evaluation plan actually resolved, validated by is_valid_progress_state(), and
# constrained in the database by a CHECK regex. An unknown phase name cannot be
# written from Python and cannot be inserted by SQL.
#
# ── WHY THIS LIST DIFFERS FROM THE ONE ORIGINALLY SPECIFIED ────────────────
#
# The brief proposed: queued, candidate_screening, stage_1, stage_2,
# verification, concluding, completed, failed. Three changes, each forced by
# what the loop in optimization/benchmark.py actually does:
#
# 1. `stage_1`/`stage_2` are PARAMETERISED, not a fixed pair. The number of
#    stages is policy data — `evaluation_stage_sizes`, default [30, 60, 133],
#    per-workload overridable — so a plan may have one stage or five. Hard-coding
#    two would misreport a three-stage run, and would have no name at all for the
#    single unstaged pass used when staged_evaluation_enabled is false. The band
#    is therefore `stage_{k}` for k >= 1, and `stages_planned` in progress_detail
#    says how many there are, so a determinate progress bar is still possible.
#
# 2. `preparing` added. Resolving the workflow, loading the promoted deployment
#    graph, and loading the golden inputs happen before any candidate exists.
#    Reporting that window as `queued` would say the job had not started when it
#    had, and every refusal that concludes `insufficient_evidence` — no workflow,
#    no baseline graph, sample size below floor — is decided inside it.
#
# 3. `baseline_measurement` added. The baseline arm runs to completion over the
#    FULL case set before any candidate starts; in the 140-case incident that
#    was one seventh of the total work and several minutes of wall clock. Naming
#    it `candidate_screening` or `stage_1` would attribute the baseline's cost
#    and duration to a candidate that had not been touched yet.
#
# `verification` is kept and means the policy comparison: constraint evaluation,
# quality non-inferiority, the frontier and the consideration funnel. It runs no
# model calls.

PROGRESS_QUEUED = "queued"
PROGRESS_PREPARING = "preparing"
PROGRESS_CANDIDATE_SCREENING = "candidate_screening"
PROGRESS_BASELINE_MEASUREMENT = "baseline_measurement"
PROGRESS_VERIFICATION = "verification"
PROGRESS_CONCLUDING = "concluding"
PROGRESS_COMPLETED = "completed"
PROGRESS_FAILED = "failed"

#: Every phase name that is not a stage. Ordered.
FIXED_PROGRESS_STATES: tuple[str, ...] = (
    PROGRESS_QUEUED,
    PROGRESS_PREPARING,
    PROGRESS_CANDIDATE_SCREENING,
    PROGRESS_BASELINE_MEASUREMENT,
    PROGRESS_VERIFICATION,
    PROGRESS_CONCLUDING,
    PROGRESS_COMPLETED,
    PROGRESS_FAILED,
)

_STAGE_PATTERN = re.compile(r"^stage_([1-9][0-9]*)$")

#: What each phase means. Descriptions, like domain.REASON_CODES, are the
#: documented vocabulary — the API returns the CODE, never this text.
PROGRESS_STATES: dict[str, str] = {
    PROGRESS_QUEUED: (
        "Persisted and accepted. No work has begun and nothing has been spent."
    ),
    PROGRESS_PREPARING: (
        "Resolving the workload's workflow, its promoted deployment graph and "
        "its golden inputs. A refusal for missing baseline or sample size is "
        "decided here."
    ),
    PROGRESS_CANDIDATE_SCREENING: (
        "Generating candidate strategies and screening them for eligibility. No "
        "arm has been executed."
    ),
    PROGRESS_BASELINE_MEASUREMENT: (
        "Executing the baseline arm over the full case set. This is the "
        "reference every candidate is paired against."
    ),
    "stage_{k}": (
        "Executing candidate arms over stage k of the resolved staged-evaluation "
        "plan. Not monotone on its own: arms are evaluated one after another and "
        "each walks the plan from stage 1, so this band cycles. The monotone "
        "quantity is (arms_completed, stage_index) in progress_detail."
    ),
    PROGRESS_VERIFICATION: (
        "Comparing measured arms against the policy: constraints, quality "
        "non-inferiority, frontier, consideration funnel. Runs no model calls."
    ),
    PROGRESS_CONCLUDING: (
        "Writing the immutable conclusion and, only on safe_improvement_found, "
        "the recommendation that cites it."
    ),
    PROGRESS_COMPLETED: "The run reached a conclusion. The conclusion may be a refusal.",
    PROGRESS_FAILED: (
        "The run did not reach a conclusion. progress_detail.failure carries the "
        "code."
    ),
}

#: Monotone ordering. A phase never moves to a lower band. Used by the frontend
#: to render progress and by the tests to assert advancement without asserting
#: an exact transition list, which would break whenever a stage plan changes.
_BANDS: dict[str, int] = {
    PROGRESS_QUEUED: 0,
    PROGRESS_PREPARING: 1,
    PROGRESS_CANDIDATE_SCREENING: 2,
    PROGRESS_BASELINE_MEASUREMENT: 3,
    # stage_* == 4
    PROGRESS_VERIFICATION: 5,
    PROGRESS_CONCLUDING: 6,
    PROGRESS_COMPLETED: 7,
    PROGRESS_FAILED: 7,
}
_STAGE_BAND = 4


def stage_state(stage_index: int) -> str:
    """The phase name for stage `stage_index` of the resolved plan. 1-based."""
    k = int(stage_index)
    if k < 1:
        raise ValueError(f"stage_index must be >= 1, got {stage_index!r}")
    return f"stage_{k}"


def is_valid_progress_state(state: Any) -> bool:
    if not isinstance(state, str):
        return False
    return state in FIXED_PROGRESS_STATES or bool(_STAGE_PATTERN.match(state))


def progress_band(state: str) -> int:
    """The monotone band of a phase. Raises on an unknown phase."""
    if state in _BANDS:
        return _BANDS[state]
    if _STAGE_PATTERN.match(state or ""):
        return _STAGE_BAND
    raise ValueError(f"Unknown progress state {state!r}.")


def progress_plan(stages_planned: Optional[int]) -> list[str]:
    """
    The full ordered phase list for a run with `stages_planned` stages.

    Returned to the caller on the job envelope so a frontend can render a
    determinate progress indicator instead of guessing the vocabulary. When the
    stage plan is not yet resolved the stage band is absent rather than
    fabricated with a guessed count.
    """
    stages = (
        [stage_state(k) for k in range(1, int(stages_planned) + 1)]
        if stages_planned and int(stages_planned) > 0
        else []
    )
    return [
        PROGRESS_QUEUED,
        PROGRESS_PREPARING,
        PROGRESS_CANDIDATE_SCREENING,
        PROGRESS_BASELINE_MEASUREMENT,
        *stages,
        PROGRESS_VERIFICATION,
        PROGRESS_CONCLUDING,
        PROGRESS_COMPLETED,
    ]


# ---------------------------------------------------------------------------
# Job kinds — WHICH INTENT was expressed, kept as data on the row
# ---------------------------------------------------------------------------

JOB_KIND_OPTIMIZE = "optimize"
JOB_KIND_EXPLORE = "explore"
JOB_KIND_RECOMMENDATION_BENCHMARK = "recommendation_benchmark"

JOB_KINDS: dict[str, str] = {
    JOB_KIND_OPTIMIZE: (
        "The full loop. A safe_improvement_found conclusion creates a "
        "recommendation that cites this benchmark."
    ),
    JOB_KIND_EXPLORE: (
        "Exploratory. Measures and concludes; never creates a recommendation, "
        "whatever the conclusion."
    ),
    JOB_KIND_RECOMMENDATION_BENCHMARK: (
        "Evidence for an EXISTING recommendation. The conclusion drives that "
        "recommendation's lifecycle transition."
    ),
}


# ---------------------------------------------------------------------------
# Why a job did not reach a conclusion — codes, never prose
# ---------------------------------------------------------------------------
#
# Kept here rather than in domain.REASON_CODES because these are OPERATIONAL
# facts about a process, not evidential reasons about a workload. A job that
# died with the worker has no verdict about anything, and putting `worker_lost`
# next to `quality_below_threshold` would invite it being rendered as one.

JOB_FAILURE_CODES: dict[str, str] = {
    "worker_lost": (
        "The process running this job stopped reporting before it reached a "
        "conclusion. Arms already measured are retained in "
        "benchmark_candidate_results; no conclusion was written, because none "
        "was reached."
    ),
    "superseded_by_existing_job": (
        "A concurrent request for the same idempotency key created this row a "
        "moment after an equivalent job already existed. This row was abandoned "
        "before any provider call was made and spent nothing."
    ),
    "start_failed": (
        "The job was persisted but could not be handed to a worker. Nothing was "
        "executed."
    ),
}


# ---------------------------------------------------------------------------
# Worker identity, heartbeat and lease
# ---------------------------------------------------------------------------

DEFAULT_HEARTBEAT_SECONDS = 30
DEFAULT_LEASE_SECONDS = 300
MIN_HEARTBEAT_SECONDS = 5
MIN_LEASE_SECONDS = 30

#: Identifies THIS process. Host and pid are for a human reading the row; the
#: random suffix distinguishes two runs of the same pid after a fast restart,
#: which is otherwise indistinguishable and would let a new process mistake an
#: orphan for one of its own.
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def heartbeat_seconds() -> int:
    return _env_int(
        "OPTIML_BENCHMARK_HEARTBEAT_SECONDS", DEFAULT_HEARTBEAT_SECONDS, MIN_HEARTBEAT_SECONDS
    )


def lease_seconds() -> int:
    """
    How long a job may go without a heartbeat before it is declared orphaned.

    Must be comfortably larger than the heartbeat interval: the cost of being
    wrong in one direction is a benchmark killed while it is alive and its spend
    wasted, and in the other a stale row lingering for a few more minutes. The
    default is 10x the default heartbeat.
    """
    return _env_int("OPTIML_BENCHMARK_LEASE_SECONDS", DEFAULT_LEASE_SECONDS, MIN_LEASE_SECONDS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
#
# WHAT MAKES TWO REQUESTS "THE SAME REQUEST", AND WHY THIS KEY
#
# The failure being defended against is not a malicious duplicate; it is the one
# the incident produced. A caller posts /optimize, the edge times out at 300s,
# the caller sees a connection error and — reasonably, because as far as it
# knows nothing happened — posts again. Under a client-supplied-only key the
# retrying client generates a fresh uuid and the second POST launches a second
# 28-minute, 140-case, 7-arm run against real providers. The guarantee would be
# in the hands of the client that had just been given no way to know the truth.
#
# So the DEFAULT key is DERIVED, from the tuple that decides what a run will
# actually do:
#
#     (org_id, workload_id, method, objective, min_sample_size,
#      create_recommendation, recommendation_id)
#
# Read it as a SCOPE, not a request id: at most one IN-FLIGHT benchmark per org,
# workload, objective, method and recommendation-intent. `create_recommendation`
# is in the tuple because /optimize and the exploratory /benchmark differ in
# exactly that boolean and must not be collapsed into each other — an
# exploratory run must never satisfy a request that was supposed to be able to
# produce a proposal. `recommendation_id` is in it because evidence gathered FOR
# a specific recommendation is not interchangeable with evidence gathered for
# no recommendation at all.
#
# ENFORCED ONLY AGAINST ACTIVE JOBS. Once a job is completed or failed the key
# is free again, because re-running a benchmark after it has finished is a
# legitimate and intended act: evidence ages, golden inputs are added, the
# policy version changes. Idempotency here means "do not run two of these at
# once", not "never run this twice".
#
# A CLIENT MAY OVERRIDE with an Idempotency-Key header, namespaced separately so
# a client-chosen string can never collide with a derived one. That is the
# escape hatch for a caller that genuinely wants two concurrent runs with
# identical parameters, and the explicit path for a client that does retry
# properly.
#
# THE TRADE-OFF, STATED: a user cannot launch two concurrent benchmarks with
# identical parameters on the same workload without supplying distinct
# Idempotency-Key headers. That is intended. Two such runs replay the same cases
# through the same providers for the same verdict; the second one is spend with
# no information in it.
#
# THE DATABASE IS THE BACKSTOP. A partial unique index on
# (org_id, idempotency_key) WHERE status IN ('pending','queued','running') makes
# the guarantee hold across processes, where a read-then-insert check cannot.
# create_job() does both: it checks first (so the ordinary duplicate never
# creates a row at all) and resolves a lost race afterwards (so a duplicate that
# slipped through is abandoned before it spends anything).

_CLIENT_KEY_PREFIX = "client:"
_DERIVED_KEY_PREFIX = "auto:"


def _digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def idempotency_key(
    *,
    org_id: str,
    workload_id: str,
    job_kind: str,
    objective: Optional[str] = None,
    min_sample_size: Optional[int] = None,
    create_recommendation: bool = False,
    recommendation_id: Optional[str] = None,
    client_key: Optional[str] = None,
) -> str:
    """
    The key two requests must share to be treated as the same in-flight job.

    Always scoped by org_id, including in the client-supplied case: a header
    value is a string a tenant controls, and two tenants choosing "retry-1" must
    not be able to observe each other's jobs.
    """
    if client_key and client_key.strip():
        return _CLIENT_KEY_PREFIX + _digest({
            "org_id": str(org_id),
            "workload_id": str(workload_id),
            "client_key": client_key.strip()[:200],
        })
    return _DERIVED_KEY_PREFIX + _digest({
        "org_id": str(org_id),
        "workload_id": str(workload_id),
        "job_kind": job_kind,
        "objective": objective,
        "min_sample_size": min_sample_size,
        "create_recommendation": bool(create_recommendation),
        "recommendation_id": (str(recommendation_id) if recommendation_id else None),
    })


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
#
# The job columns are added by migration_optimization_v9_async_jobs.sql, which
# is deliberately NOT applied by this code. Between shipping this and applying
# that migration the columns do not exist, and a write naming them would fail
# the whole insert — which would take /optimize down entirely rather than
# degrading it. So every job write goes through _attempt(), which retries once
# without the job columns and latches into a degraded mode that is loud in the
# logs and visible on the status endpoint. Degraded mode still runs benchmarks;
# it just cannot report their phase or reap their orphans, which is exactly the
# behaviour that exists today.

JOB_COLUMNS = (
    "progress_state",
    "progress_detail",
    "heartbeat_at",
    "worker_id",
    "idempotency_key",
    "job_kind",
    "requested_by",
)

JOB_COLS_SELECT = (
    "id, org_id, workload_id, status, method, objective, error, "
    "started_at, completed_at, created_at, conclusion, confidence, "
    "progress_state, progress_detail, heartbeat_at, worker_id, "
    "idempotency_key, job_kind, requested_by"
)

_FALLBACK_COLS_SELECT = (
    "id, org_id, workload_id, status, method, objective, error, "
    "started_at, completed_at, created_at, conclusion, confidence"
)

_state = {"degraded": False, "degraded_reason": None}


def schema_status() -> dict:
    """Whether the v9 job columns are usable in this deployment."""
    return {
        "job_columns_available": not _state["degraded"],
        "degraded_reason": _state["degraded_reason"],
        "worker_id": WORKER_ID,
        "heartbeat_seconds": heartbeat_seconds(),
        "lease_seconds": lease_seconds(),
    }


def _looks_like_missing_column(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "column",
            "does not exist",
            "42703",
            "pgrst204",
            "could not find",
            "schema cache",
        )
    )


def _degrade(exc: BaseException) -> None:
    if _state["degraded"]:
        return
    _state["degraded"] = True
    _state["degraded_reason"] = str(exc)[:300]
    logger.critical(
        "optimization_benchmarks is missing the async-job columns; falling back to "
        "status-only tracking. Benchmarks still run, but progress reporting and "
        "orphan reaping are DISABLED until "
        "migration_optimization_v9_async_jobs.sql is applied. Cause: %s",
        _state["degraded_reason"],
    )


def strip_job_columns(patch: dict) -> dict:
    return {k: v for k, v in patch.items() if k not in JOB_COLUMNS}


def _attempt(with_columns: Callable[[], Any], without_columns: Callable[[], Any]) -> Any:
    """Run the full write; on a missing-column failure latch degraded and retry."""
    if _state["degraded"]:
        return without_columns()
    try:
        return with_columns()
    except Exception as exc:
        if not _looks_like_missing_column(exc):
            raise
        _degrade(exc)
        return without_columns()


def select_cols() -> str:
    return _FALLBACK_COLS_SELECT if _state["degraded"] else JOB_COLS_SELECT


def get_job(org_id: str, benchmark_id: str) -> Optional[dict]:
    """One job row, ALWAYS re-filtered by org. Returns None for another tenant's id."""
    def _q(cols: str):
        resp = (
            supabase.table("optimization_benchmarks")
            .select(cols)
            .eq("id", benchmark_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None

    try:
        return _attempt(lambda: _q(JOB_COLS_SELECT), lambda: _q(_FALLBACK_COLS_SELECT))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("get_job failed: %s", type(exc).__name__)
        return None


def list_active_jobs(
    org_id: Optional[str] = None, *, workload_id: Optional[str] = None, limit: int = 200
) -> list[dict]:
    """
    Every job that has not reached a verdict.

    org_id=None is the REAPER path only and is never reachable from an HTTP
    route: every /api route in this package takes org_id in the path and
    re-filters by the verified org. The reaper is a process-local maintenance
    task with no caller and no response.
    """
    def _q(cols: str):
        q = supabase.table("optimization_benchmarks").select(cols)
        if org_id:
            q = q.eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        q = q.in_("status", list(ACTIVE_STATUSES))
        resp = q.order("created_at", desc=True).limit(max(1, min(limit, 500))).execute()
        return getattr(resp, "data", None) or []

    try:
        return _attempt(lambda: _q(JOB_COLS_SELECT), lambda: _q(_FALLBACK_COLS_SELECT))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("list_active_jobs failed: %s", type(exc).__name__)
        return []


def find_active_job(org_id: str, key: str) -> Optional[dict]:
    """The oldest in-flight job for this org and idempotency key, if any."""
    if _state["degraded"] or not key:
        return None
    try:
        resp = (
            supabase.table("optimization_benchmarks")
            .select(JOB_COLS_SELECT)
            .eq("org_id", org_id)
            .eq("idempotency_key", key)
            .in_("status", list(ACTIVE_STATUSES))
            .order("created_at", desc=False)
            .limit(5)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:
        if _looks_like_missing_column(exc):
            _degrade(exc)
            return None
        logger.warning("find_active_job failed: %s", type(exc).__name__)
        return None


def patch_job(
    org_id: str,
    benchmark_id: str,
    patch: dict,
    *,
    only_if_active: bool = False,
) -> list[dict]:
    """
    Update a job row, org-filtered.

    `only_if_active` narrows the update to rows that have not reached a verdict.
    That is what makes the reaper safe: a worker finishing a run at the same
    instant the reaper decides it is orphaned wins, because the reaper's update
    then matches zero rows and it can see that it matched zero.
    """
    def _q(payload: dict):
        q = supabase.table("optimization_benchmarks").update(payload).eq(
            "id", benchmark_id
        ).eq("org_id", org_id)
        if only_if_active:
            q = q.in_("status", list(ACTIVE_STATUSES))
        return getattr(q.execute(), "data", None) or []

    try:
        return _attempt(lambda: _q(dict(patch)), lambda: _q(strip_job_columns(patch)))
    except Exception as exc:
        logger.warning("patch_job failed: %s", type(exc).__name__)
        return []


def job_insert_columns(
    *,
    idem_key: Optional[str],
    job_kind: str,
    requested_by: Optional[str],
    stages_planned: Optional[int] = None,
) -> dict:
    """The job columns a freshly created benchmark row carries."""
    now = _iso(_utc_now())
    return {
        "status": STATUS_QUEUED,
        "progress_state": PROGRESS_QUEUED,
        "progress_detail": {
            "phase": PROGRESS_QUEUED,
            "plan": progress_plan(stages_planned),
            "stages_planned": stages_planned,
            "updated_at": now,
        },
        "heartbeat_at": now,
        "worker_id": None,
        "idempotency_key": idem_key,
        "job_kind": job_kind,
        "requested_by": requested_by,
    }


def fail_job(
    org_id: str,
    benchmark_id: str,
    *,
    code: str,
    detail: Optional[str] = None,
    existing_detail: Optional[dict] = None,
) -> bool:
    """
    Mark an ACTIVE job failed with a structured code. Returns True if it changed.

    Never writes a conclusion. A job that died has no verdict about the
    workload, and manufacturing a `benchmark_failed` conclusion from outside the
    run — with no policy, no materiality and no measured arms in hand — would be
    inventing evidence about a customer's system. The absence of a conclusion is
    the honest record, and every arm that was measured before the process died
    is still in benchmark_candidate_results.
    """
    if code not in JOB_FAILURE_CODES:
        raise ValueError(f"Unknown job failure code '{code}'.")
    now = _iso(_utc_now())
    progress_detail = dict(existing_detail or {})
    progress_detail.update({
        "phase": PROGRESS_FAILED,
        "failure": {"code": code, "detail": (detail[:300] if detail else None)},
        "updated_at": now,
    })
    changed = patch_job(
        org_id,
        benchmark_id,
        {
            "status": STATUS_FAILED,
            "progress_state": PROGRESS_FAILED,
            "progress_detail": progress_detail,
            "completed_at": now,
            "error": f"{code}: {detail}"[:500] if detail else code,
        },
        only_if_active=True,
    )
    return bool(changed)


# ---------------------------------------------------------------------------
# Creating a job
# ---------------------------------------------------------------------------

def create_job(
    org_id: str,
    *,
    workload_id: str,
    job_kind: str,
    insert_row: Callable[[dict], Optional[dict]],
    objective: Optional[str] = None,
    min_sample_size: Optional[int] = None,
    create_recommendation: bool = False,
    recommendation_id: Optional[str] = None,
    client_key: Optional[str] = None,
    requested_by: Optional[str] = None,
) -> tuple[Optional[dict], bool, str]:
    """
    Claim the idempotency key and persist a queued job.

    Returns (row, created, key). `created` is False when an equivalent job was
    already in flight — the caller MUST NOT start execution in that case, which
    is the whole double-spend guarantee.

    `insert_row` is injected rather than imported so this module never has to
    know how a benchmark row is built. optimization/benchmark.py owns that.
    """
    if job_kind not in JOB_KINDS:
        raise ValueError(f"Unknown job kind '{job_kind}'.")

    key = idempotency_key(
        org_id=org_id,
        workload_id=workload_id,
        job_kind=job_kind,
        objective=objective,
        min_sample_size=min_sample_size,
        create_recommendation=create_recommendation,
        recommendation_id=recommendation_id,
        client_key=client_key,
    )

    # 1. The ordinary duplicate: an equivalent job is already in flight. No row
    #    is created, so there is nothing to clean up and nothing was spent.
    existing = find_active_job(org_id, key)
    if existing is not None:
        return existing, False, key

    # 2. Create. In production a partial unique index may reject this because a
    #    concurrent request won the race; that is a duplicate, not an error.
    columns = job_insert_columns(
        idem_key=key, job_kind=job_kind, requested_by=requested_by
    )
    try:
        row = insert_row(columns)
    except Exception as exc:
        if _looks_like_missing_column(exc):
            _degrade(exc)
            row = insert_row(strip_job_columns(columns))
        else:
            winner = find_active_job(org_id, key)
            if winner is not None:
                return winner, False, key
            raise
    if row is None:
        winner = find_active_job(org_id, key)
        if winner is not None:
            return winner, False, key
        return None, False, key

    # 3. Resolve a race the pre-check could not see. Without the DB constraint
    #    (and in any store that lacks one) two requests can both pass step 1.
    #    Deterministic tie-break: oldest row wins, everyone else abandons BEFORE
    #    executing anything, so the loser costs a row and zero provider calls.
    if not _state["degraded"]:
        winner = find_active_job(org_id, key)
        if winner is not None and str(winner.get("id")) != str(row.get("id")):
            fail_job(
                org_id,
                str(row["id"]),
                code="superseded_by_existing_job",
                detail=f"benchmark {winner.get('id')} already held this key",
            )
            return winner, False, key

    return row, True, key


# ---------------------------------------------------------------------------
# Progress reporting, called from inside the run
# ---------------------------------------------------------------------------

class ProgressReporter:
    """
    Writes phase transitions and heartbeats onto the benchmark row.

    Called from the worker THREAD that executes the benchmark, so every method
    is synchronous and every failure is swallowed. A progress write that fails
    must never fail a benchmark: the run is the product, the progress bar is
    not, and losing 28 minutes of provider spend because a JSONB update timed
    out would be a self-inflicted version of the incident this replaces.
    """

    def __init__(self, org_id: str, benchmark_id: str, *, worker_id: str = WORKER_ID):
        self.org_id = str(org_id)
        self.benchmark_id = str(benchmark_id)
        self.worker_id = worker_id
        self.detail: dict = {}
        self.state: str = PROGRESS_QUEUED
        self.observed: list[str] = []

    def __call__(self, state: str, **facts: Any) -> None:
        if not is_valid_progress_state(state):
            # A phase name is data. An invented one is a bug, and it must be
            # loud rather than persisted into the vocabulary.
            logger.error("Refusing to persist unknown progress state %r.", state)
            return
        now = _iso(_utc_now())
        self.state = state
        self.detail.update({k: v for k, v in facts.items() if v is not None})
        self.detail["phase"] = state
        self.detail["updated_at"] = now
        if "stages_planned" in self.detail:
            self.detail["plan"] = progress_plan(self.detail.get("stages_planned"))
        self.observed.append(state)
        patch: dict = {
            "progress_state": state,
            "progress_detail": dict(self.detail),
            "heartbeat_at": now,
            "worker_id": self.worker_id,
        }
        if state != PROGRESS_QUEUED and not self.detail.get("_started"):
            self.detail["_started"] = True
            patch["status"] = STATUS_RUNNING
            patch["started_at"] = now
        try:
            matched = patch_job(
                self.org_id, self.benchmark_id, patch, only_if_active=True
            )
            if matched:
                return
            # The active guard exists so a job the reaper has already declared
            # worker_lost cannot be dragged back to `running` by a worker that
            # turned out to be alive. It also catches a benign case: the run
            # writes `status='completed'` onto the row before its final
            # `concluding` phase, so the last phase of a healthy run arrives
            # after the row is terminal. Record the phase in that case — but
            # never the status, and never over a FAILED row, whose
            # progress_detail carries the failure code that explains it.
            row = get_job(self.org_id, self.benchmark_id) or {}
            if row.get("status") == STATUS_COMPLETED:
                patch_job(
                    self.org_id,
                    self.benchmark_id,
                    {
                        "progress_state": state,
                        "progress_detail": dict(self.detail),
                        "heartbeat_at": now,
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Progress write failed for benchmark %s (%s); the run continues.",
                self.benchmark_id, type(exc).__name__,
            )

    def heartbeat(self) -> None:
        """Liveness only. Writes no phase, so it can never move the state back."""
        try:
            patch_job(
                self.org_id,
                self.benchmark_id,
                {"heartbeat_at": _iso(_utc_now()), "worker_id": self.worker_id},
                only_if_active=True,
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("Heartbeat write failed for benchmark %s.", self.benchmark_id)

    def finalize(self, status: Optional[str]) -> None:
        """Record the terminal phase that the run itself already wrote as status."""
        terminal = (
            PROGRESS_COMPLETED if status == STATUS_COMPLETED else PROGRESS_FAILED
        )
        now = _iso(_utc_now())
        self.state = terminal
        self.detail["phase"] = terminal
        self.detail["updated_at"] = now
        self.observed.append(terminal)
        try:
            patch_job(
                self.org_id,
                self.benchmark_id,
                {
                    "progress_state": terminal,
                    "progress_detail": dict(self.detail),
                    "heartbeat_at": now,
                },
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("Finalize write failed for benchmark %s.", self.benchmark_id)


# ---------------------------------------------------------------------------
# Orphan detection: how a job stops being "running forever"
# ---------------------------------------------------------------------------
#
# THE FAILURE MODE. A worker picks up a job, writes status='running', and the
# process dies — deploy, OOM, Railway moving the container. Nothing in the
# database distinguishes that row from one whose 28-minute run is going fine.
# Poll it a year later and it still says `running`. That is the state the brief
# calls unknowable, and it is what a naive implementation produces, because the
# only process that knew the truth is the one that died.
#
# THE MECHANISM. Liveness is a claim that must be RENEWED. `heartbeat_at` is
# rewritten at every phase transition and, because a single arm over 140 cases
# can run for minutes without a transition, by a ticker every
# OPTIML_BENCHMARK_HEARTBEAT_SECONDS (default 30). A job is orphaned when it is
# in an ACTIVE status and its last heartbeat is older than
# OPTIML_BENCHMARK_LEASE_SECONDS (default 300). A dead process renews nothing,
# so the passage of time itself resolves the ambiguity.
#
# WHEN IT RUNS. Two triggers, deliberately:
#   * every control-loop tick (60s), so an orphan is resolved within about one
#     lease period even while the deployment stays up;
#   * once at STARTUP, so the ordinary case — the process that died IS the one
#     coming back, on a single-worker deployment where nobody else is looking —
#     resolves in seconds rather than waiting for the first tick.
#
# WHAT HAPPENS TO IT: FAILED, NOT RESUMED. Resuming would need the candidate
# ordering, the per-arm case cursor and the in-memory per-case results, none of
# which are persisted, so a "resumed" run would silently re-execute arms and
# double-spend. Every arm that finished before the crash is already durable in
# benchmark_candidate_results, so failing the job loses no measurement and no
# money — only the incomplete run. The row gets status='failed',
# progress_state='failed' and progress_detail.failure.code='worker_lost'. It
# gets no conclusion, because none was reached.
#
# A job that has never heartbeat at all (degraded schema, or a crash between
# insert and first write) falls back to started_at, then created_at, so a row
# with no heartbeat column still cannot outlive its lease.


def is_stale(row: dict, *, now: Optional[datetime] = None, lease: Optional[int] = None) -> bool:
    if (row.get("status") or "") not in ACTIVE_STATUSES:
        return False
    now = now or _utc_now()
    cutoff = now - timedelta(seconds=(lease if lease is not None else lease_seconds()))
    last = (
        _parse_iso(row.get("heartbeat_at"))
        or _parse_iso(row.get("started_at"))
        or _parse_iso(row.get("created_at"))
    )
    if last is None:
        # No timestamp at all. Unknowable is the one state not allowed, and a
        # row that cannot prove it is alive is not.
        return True
    return last < cutoff


def reap_stale_jobs(
    org_id: Optional[str] = None,
    *,
    now: Optional[datetime] = None,
    lease: Optional[int] = None,
) -> dict:
    """
    Fail every job whose lease has expired. Never raises.

    Returns counts and the ids acted on — facts, no prose.
    """
    now = now or _utc_now()
    lease = lease if lease is not None else lease_seconds()
    result: dict = {
        "checked": 0, "stale": 0, "failed": 0, "lease_seconds": lease,
        "benchmark_ids": [], "degraded": _state["degraded"],
    }
    try:
        rows = list_active_jobs(org_id)
    except Exception as exc:  # pragma: no cover - list_active_jobs already guards
        logger.warning("reap_stale_jobs could not list jobs: %s", type(exc).__name__)
        result["error"] = type(exc).__name__
        return result

    result["checked"] = len(rows)
    for row in rows:
        try:
            if not is_stale(row, now=now, lease=lease):
                continue
            result["stale"] += 1
            row_org = str(row.get("org_id") or "")
            bid = str(row.get("id"))
            if not row_org or not bid:
                continue
            last_seen = (
                row.get("heartbeat_at") or row.get("started_at") or row.get("created_at")
            )
            if fail_job(
                row_org,
                bid,
                code="worker_lost",
                detail=(
                    f"no heartbeat since {last_seen}; lease {lease}s expired "
                    f"(last worker {row.get('worker_id') or 'unknown'})"
                ),
                existing_detail=(row.get("progress_detail") or {}),
            ):
                result["failed"] += 1
                result["benchmark_ids"].append(bid)
                logger.warning(
                    "Benchmark %s (org %s) declared orphaned: no heartbeat since %s. "
                    "Marked failed with worker_lost; measured arms are retained.",
                    bid, row_org, last_seen,
                )
        except Exception:
            logger.exception("reap_stale_jobs failed on one row; continuing.")
    return result


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------
#
# WHY THE TASK IS HELD IN A MODULE-LEVEL SET. asyncio keeps only a WEAK
# reference to a running task. `asyncio.create_task(...)` whose result nobody
# stores can be garbage-collected mid-flight, and the observable symptom is a
# benchmark that stops for no reason and a row that says `running` forever —
# the precise failure this module exists to make impossible. The set is the
# strong reference; the done-callback removes it.
#
# WHY CLIENT DISCONNECT CANNOT CANCEL IT. The work lives on a task created on
# the event loop and executed in a thread-pool worker. It is not awaited by, and
# holds no reference to, the request coroutine. Starlette cancelling the request
# on disconnect cancels the handler; the job's task is not in that cancellation
# tree.

_TASKS: set = set()


def active_task_count() -> int:
    return len([t for t in _TASKS if not t.done()])


def _on_job_done(benchmark_id: str, org_id: str) -> Callable[[Any], None]:
    def _cb(task) -> None:
        _TASKS.discard(task)
        if task.cancelled():
            logger.critical(
                "Benchmark job %s was CANCELLED. Its row is left ACTIVE on purpose: "
                "the reaper will declare it worker_lost once its lease expires "
                "rather than this process guessing what it managed to finish.",
                benchmark_id,
            )
            return
        exc = task.exception()
        if exc is not None:
            logger.critical(
                "Benchmark job %s raised outside the run; marking it failed.",
                benchmark_id, exc_info=exc,
            )
            try:
                fail_job(
                    org_id, benchmark_id, code="start_failed", detail=str(exc)[:300]
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Could not mark benchmark %s failed.", benchmark_id)

    return _cb


async def _heartbeat_ticker(reporter: ProgressReporter, stop: asyncio.Event) -> None:
    interval = heartbeat_seconds()
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        await asyncio.to_thread(reporter.heartbeat)


async def _run_job(
    org_id: str, benchmark_id: str, run: Callable[[ProgressReporter], dict]
) -> Optional[dict]:
    reporter = ProgressReporter(org_id, benchmark_id)
    stop = asyncio.Event()
    ticker = asyncio.create_task(_heartbeat_ticker(reporter, stop))
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: run(reporter))
    finally:
        stop.set()
        ticker.cancel()
        try:
            await ticker
        except (asyncio.CancelledError, Exception):
            pass
    status = (result or {}).get("status")
    if status is None:
        row = get_job(org_id, benchmark_id) or {}
        status = row.get("status")
    # finalize() deliberately has NO active guard. If a mis-tuned lease caused
    # the reaper to declare this job worker_lost while it was in fact alive, and
    # it then reached a real conclusion, the real conclusion wins: the run
    # actually measured the arms, and `worker_lost` was a guess about a process.
    # A measurement always outranks an inference about a process.
    reporter.finalize(status)
    return result


def start_job(
    org_id: str, benchmark_id: str, run: Callable[[ProgressReporter], dict]
) -> Any:
    """
    Hand a persisted job to a worker and return immediately.

    `run` receives the ProgressReporter and is executed in a thread, because
    run_benchmark is synchronous and spends most of its life in provider I/O.

    The task is created on the RUNNING loop and parented to nothing. It is not
    awaited by the request handler and holds no reference to it, which is what
    makes a client disconnect a non-event: Starlette cancels the handler, and
    the job's task is not in that cancellation tree.
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(_run_job(org_id, benchmark_id, run))
    _TASKS.add(task)
    task.add_done_callback(_on_job_done(str(benchmark_id), str(org_id)))
    return task


# ---------------------------------------------------------------------------
# The response envelope
# ---------------------------------------------------------------------------

def job_response(row: dict, *, created: Optional[bool] = None) -> dict:
    """
    What every job-returning route hands back. Codes and facts only.

    `progress_state` may be absent on a row written before the v9 migration or
    while the schema is degraded; it is reported as null rather than backfilled
    with a guess, because inventing a phase for a run whose phase was never
    recorded is a fabrication like any other.
    """
    detail = row.get("progress_detail") or {}
    out = {
        "benchmark_id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "workload_id": (str(row["workload_id"]) if row.get("workload_id") else None),
        "job_kind": row.get("job_kind"),
        "status": row.get("status"),
        "progress_state": row.get("progress_state"),
        "progress": {
            "phase": detail.get("phase") or row.get("progress_state"),
            "plan": detail.get("plan") or progress_plan(detail.get("stages_planned")),
            "stages_planned": detail.get("stages_planned"),
            "stage_index": detail.get("stage_index"),
            "arms_total": detail.get("arms_total"),
            "arms_completed": detail.get("arms_completed"),
            "cases_planned": detail.get("cases_planned"),
            "cases_completed": detail.get("cases_completed"),
            "updated_at": detail.get("updated_at"),
        },
        "failure": detail.get("failure"),
        "objective": row.get("objective"),
        "method": row.get("method"),
        "conclusion": row.get("conclusion"),
        "error": row.get("error"),
        "heartbeat_at": row.get("heartbeat_at"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "created_at": row.get("created_at"),
        "terminal": (row.get("status") in TERMINAL_STATUSES),
    }
    if created is not None:
        # False means "this request did not launch anything; here is the job
        # that was already running". The distinction matters to a client
        # deciding whether it is safe to retry.
        out["created"] = bool(created)
    return out
