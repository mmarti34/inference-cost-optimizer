"""
Benchmark execution as an asynchronous job.

WHAT THESE TESTS DEFEND. A real production run — 140 replay cases across 7 arms
— took about 28 minutes against a 300-second edge timeout. The caller got a
connection error, the benchmark finished, and the API told nobody. Everything
here is about the failure modes that replacing one long HTTP request with a
persisted job either fixes or introduces:

  * the caller going away must not cancel or corrupt the run
  * a retry must not launch a second run, and must not double-spend
  * a job, its progress and its result belong to exactly one tenant
  * progress must be real, persisted, and monotone
  * a run whose worker dies must NEVER stay `running` forever

The harness — the in-memory table store, the priced replay runtime, the seeded
workload — is imported from test_optimization_loop rather than copied, so these
tests drive the same real loop against the same edges.
"""
import asyncio
import contextlib
import threading
import time

import pytest

from optimization import benchmark as benchmark_mod  # noqa: E402
from optimization import jobs as jobs_mod  # noqa: E402
from test_optimization_loop import (  # noqa: E402
    ORG_ID,
    OTHER_ORG_ID,
    WORKLOAD_ID,
    FakeRuntime,
    FakeSupabase,
    _patched,
    _seed,
    wait_for_job,
)

OPTIMIZE = f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/optimize"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_job_module_state():
    """
    The degraded-schema latch and the task registry are module globals.

    They are process-wide by design — a deployment either has the v9 columns or
    it does not — which makes them leak between tests unless reset. A test that
    passes only because an earlier test left a flag set is not a test.
    """
    jobs_mod._state["degraded"] = False
    jobs_mod._state["degraded_reason"] = None
    jobs_mod._TASKS.clear()
    yield
    jobs_mod._TASKS.clear()


@pytest.fixture
def db():
    return RecordingSupabase()


class RecordingSupabase(FakeSupabase):
    """
    FakeSupabase that records every phase write AND WHAT THE STORE HELD AFTER IT.

    Recording the intent alone would prove the code called a function. Recording
    the row the update actually returned proves the phase was PERSISTED, which
    is the claim: a benchmark's progress has to survive the process that
    reported it.
    """

    def __init__(self):
        super().__init__()
        self.progress_writes: list[dict] = []

    def table(self, name):
        query = super().table(name)
        if name != "optimization_benchmarks":
            return query
        recorder = self.progress_writes
        original_update = query.update

        def update(patch):
            result = original_update(patch)
            if "progress_state" in patch:
                original_execute = query.execute

                def execute():
                    res = original_execute()
                    rows = getattr(res, "data", None) or []
                    recorder.append({
                        "intended": patch["progress_state"],
                        "matched": len(rows),
                        "stored": (rows[0].get("progress_state") if rows else None),
                        "detail": (rows[0].get("progress_detail") if rows else None),
                        "status": (rows[0].get("status") if rows else None),
                    })
                    return res

                query.execute = execute
            return result

        query.update = update
        return query


class GatedRuntime:
    """
    The priced replay runtime, with a brake.

    A benchmark that finishes in three milliseconds cannot be observed
    mid-flight, and every question here — is it still running when the duplicate
    arrives, does it survive the caller vanishing, what phase is it in — is a
    question about a run in progress. `started` fires on the first provider
    call; `release` holds every call until the test lets go.
    """

    def __init__(self, inner: FakeRuntime):
        self.inner = inner
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    @property
    def calls(self):
        return self.inner.calls

    def hold(self):
        self.release.clear()

    def __call__(self, *args, **kwargs):
        self.started.set()
        if not self.release.wait(timeout=30):
            raise AssertionError("GatedRuntime was never released.")
        return self.inner(*args, **kwargs)


def _client(org_id=ORG_ID):
    """A TestClient whose auth dependency has verified membership of `org_id`."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth_dependency import AuthenticatedUser, require_org_member
    from routers.optimization_router import router as opt_router

    app = FastAPI()
    app.include_router(opt_router, prefix="/api")
    user = AuthenticatedUser(user_id=f"user-of-{org_id}", email="a@b.c", email_verified=True)
    user._verified_org_id = org_id
    app.dependency_overrides[require_org_member] = lambda: user
    # A context manager keeps ONE event loop alive across requests, which is
    # what lets a job started by request A still be running during request B —
    # the situation this whole feature exists to create.
    return TestClient(app)


@contextlib.contextmanager
def _running(db, runtime):
    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


def _wait_until(predicate, timeout=20.0, interval=0.005):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _bench_rows(db):
    return db.rows("optimization_benchmarks")


# ---------------------------------------------------------------------------
# 1. The caller going away does not cancel or corrupt the run
# ---------------------------------------------------------------------------

def test_a_job_survives_the_caller_disconnecting_mid_run(db):
    """
    The exact production failure, inverted.

    Previously the run lived inside the request coroutine, so it was reachable
    from the request's cancellation tree — and a client that walked away at the
    300-second edge timeout was, structurally, a client that could kill work it
    had already paid for. Here the caller is cancelled WHILE the benchmark is
    executing, standing in for Starlette tearing down a handler on disconnect,
    and the run must reach a conclusion anyway.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))

    async def scenario():
        holder = {}

        async def caller():
            # Everything the route does, then the part a real handler cannot
            # do: keep holding the connection open.
            prepared = benchmark_mod.prepare_benchmark(ORG_ID, workload_id=WORKLOAD_ID)
            row, created, _key = jobs_mod.create_job(
                ORG_ID,
                workload_id=WORKLOAD_ID,
                job_kind=jobs_mod.JOB_KIND_OPTIMIZE,
                insert_row=lambda cols: benchmark_mod.create_benchmark_row(
                    ORG_ID, prepared, extra_columns=cols
                ),
                objective=prepared["objective"],
                create_recommendation=True,
                requested_by="user-1",
            )
            assert created is True
            benchmark_id = str(row["id"])
            holder["benchmark_id"] = benchmark_id
            holder["job"] = jobs_mod.start_job(
                ORG_ID,
                benchmark_id,
                lambda reporter: benchmark_mod.run_benchmark(
                    ORG_ID,
                    workload_id=WORKLOAD_ID,
                    prepared=prepared,
                    benchmark_id=benchmark_id,
                    create_recommendation=True,
                    progress=reporter,
                ),
            )
            await asyncio.sleep(3600)  # still on the line

        runtime.hold()
        caller_task = asyncio.create_task(caller())
        assert await asyncio.to_thread(runtime.started.wait, 20), "run never started"

        # THE CLIENT DISCONNECTS.
        caller_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await caller_task
        assert caller_task.cancelled()

        runtime.release.set()
        await asyncio.wait_for(holder["job"], timeout=60)
        return holder["benchmark_id"]

    with _running(db, runtime):
        benchmark_id = asyncio.run(scenario())

    rows = _bench_rows(db)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["progress_state"] == "completed"
    # Not merely "not cancelled" — the evidence the run exists to produce is
    # there, which is what "not corrupted" has to mean.
    assert row.get("conclusion")
    assert row.get("completed_at")
    arms = [
        r for r in db.rows("benchmark_candidate_results")
        if str(r.get("benchmark_id")) == benchmark_id
    ]
    assert any(a.get("arm") == "baseline" for a in arms)
    assert any(a.get("arm") == "candidate" for a in arms)


# ---------------------------------------------------------------------------
# 2. A duplicate request launches nothing and spends nothing
# ---------------------------------------------------------------------------

def test_duplicate_optimize_returns_the_running_job_and_does_not_double_spend(db):
    """
    The retry the incident produced: connection error, so the caller posts again.

    Two guarantees, and the second is the expensive one. Only one benchmark row
    may exist — and no golden case may be replayed twice through the same arm,
    because that is what a second 140-case run against real providers actually
    costs.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))

    with _running(db, runtime):
        with _client() as client:
            runtime.hold()
            first = client.post(OPTIMIZE, json={})
            assert first.status_code == 202, first.text
            first_body = first.json()
            assert first_body["created"] is True

            assert _wait_until(runtime.started.is_set), "first run never started"

            # The duplicate, arriving while the first is mid-flight.
            second = client.post(OPTIMIZE, json={})
            assert second.status_code == 202, second.text
            second_body = second.json()
            assert second_body["created"] is False
            assert second_body["benchmark_id"] == first_body["benchmark_id"]
            assert second_body["status"] in ("queued", "running")

            runtime.release.set()
            wait_for_job(client, ORG_ID, first_body["benchmark_id"])

    assert len(_bench_rows(db)) == 1

    # NO DOUBLE SPEND. Every (arm, case) pair was executed exactly once. This is
    # asserted over the calls themselves rather than against an expected count,
    # so it keeps meaning the same thing if the candidate generator changes.
    executed = [(c["model"], c["input_text"]) for c in runtime.calls]
    assert len(executed) == len(set(executed)), "a case was replayed twice"


def test_a_terminal_job_does_not_block_a_later_run_of_the_same_workload(db):
    """
    Idempotency means "not two at once", not "never twice".

    Evidence ages, golden inputs get added, the policy version moves. Once a job
    is terminal its key is free, and a second POST is a genuinely new benchmark
    rather than a replay of an old verdict.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))

    with _running(db, runtime):
        with _client() as client:
            first = client.post(OPTIMIZE, json={}).json()
            wait_for_job(client, ORG_ID, first["benchmark_id"])

            second = client.post(OPTIMIZE, json={})
            assert second.status_code == 202
            second_body = second.json()
            assert second_body["created"] is True
            assert second_body["benchmark_id"] != first["benchmark_id"]
            wait_for_job(client, ORG_ID, second_body["benchmark_id"])

    assert len(_bench_rows(db)) == 2


def test_explore_and_optimize_are_not_the_same_job(db):
    """
    /benchmark may never create a recommendation; /optimize may.

    They differ in one boolean, so that boolean is part of the idempotency key.
    Were it not, an exploratory run already in flight would be handed to a
    caller that asked for the full loop, and that caller would poll to
    completion and never get the proposal it requested.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))
    explore = f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/benchmark"

    with _running(db, runtime):
        with _client() as client:
            runtime.hold()
            a = client.post(explore, json={}).json()
            assert _wait_until(runtime.started.is_set)
            b = client.post(OPTIMIZE, json={}).json()

            assert a["job_kind"] == jobs_mod.JOB_KIND_EXPLORE
            assert a["creates_recommendation"] is False
            assert b["job_kind"] == jobs_mod.JOB_KIND_OPTIMIZE
            assert b["creates_recommendation"] is True
            assert b["created"] is True
            assert a["benchmark_id"] != b["benchmark_id"]

            runtime.release.set()
            wait_for_job(client, ORG_ID, a["benchmark_id"])
            wait_for_job(client, ORG_ID, b["benchmark_id"])


def test_an_explicit_idempotency_key_separates_two_otherwise_identical_requests(db):
    """The documented escape hatch, and proof it is namespaced per org."""
    key_a = jobs_mod.idempotency_key(
        org_id=ORG_ID, workload_id=WORKLOAD_ID,
        job_kind=jobs_mod.JOB_KIND_OPTIMIZE, client_key="retry-1",
    )
    key_b = jobs_mod.idempotency_key(
        org_id=OTHER_ORG_ID, workload_id=WORKLOAD_ID,
        job_kind=jobs_mod.JOB_KIND_OPTIMIZE, client_key="retry-1",
    )
    # Same string, different tenants: two tenants both choosing "retry-1" must
    # not be able to observe or block each other's jobs.
    assert key_a != key_b
    assert key_a.startswith("client:")

    derived = jobs_mod.idempotency_key(
        org_id=ORG_ID, workload_id=WORKLOAD_ID, job_kind=jobs_mod.JOB_KIND_OPTIMIZE,
    )
    assert derived.startswith("auto:")
    assert derived != key_a

    # create_recommendation is part of the derived key: exploratory and
    # full-loop runs are never confused for one another.
    assert jobs_mod.idempotency_key(
        org_id=ORG_ID, workload_id=WORKLOAD_ID,
        job_kind=jobs_mod.JOB_KIND_OPTIMIZE, create_recommendation=True,
    ) != jobs_mod.idempotency_key(
        org_id=ORG_ID, workload_id=WORKLOAD_ID,
        job_kind=jobs_mod.JOB_KIND_OPTIMIZE, create_recommendation=False,
    )


def test_a_lost_idempotency_race_abandons_the_loser_before_it_spends_anything(db):
    """
    The window a check-then-insert cannot close, closed after the fact.

    In production the partial unique index rejects the second insert. Where
    there is no such index both inserts land, so create_job re-reads, keeps the
    oldest and abandons its own row — and it does so BEFORE handing anything to
    a worker, which is why the loser costs a row and zero provider calls.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = FakeRuntime(n_cases=20)

    with _running(db, runtime):
        prepared = benchmark_mod.prepare_benchmark(ORG_ID, workload_id=WORKLOAD_ID)

        real_find = jobs_mod.find_active_job
        calls = {"n": 0}

        def blind_first_check(org_id, key):
            # The first caller's pre-check sees nothing; by the time it
            # re-reads, a competing row exists. That is the race.
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return real_find(org_id, key)

        winner = benchmark_mod.create_benchmark_row(
            ORG_ID,
            prepared,
            extra_columns=jobs_mod.job_insert_columns(
                idem_key=jobs_mod.idempotency_key(
                    org_id=ORG_ID, workload_id=WORKLOAD_ID,
                    job_kind=jobs_mod.JOB_KIND_OPTIMIZE, objective=prepared["objective"],
                    create_recommendation=True,
                ),
                job_kind=jobs_mod.JOB_KIND_OPTIMIZE,
                requested_by="first-caller",
            ),
        )
        time.sleep(0.01)  # so created_at ordering is unambiguous

        jobs_mod.find_active_job = blind_first_check
        try:
            row, created, _key = jobs_mod.create_job(
                ORG_ID,
                workload_id=WORKLOAD_ID,
                job_kind=jobs_mod.JOB_KIND_OPTIMIZE,
                insert_row=lambda cols: benchmark_mod.create_benchmark_row(
                    ORG_ID, prepared, extra_columns=cols
                ),
                objective=prepared["objective"],
                create_recommendation=True,
                requested_by="second-caller",
            )
        finally:
            jobs_mod.find_active_job = real_find

    assert created is False
    assert str(row["id"]) == str(winner["id"])
    assert runtime.calls == [], "the loser must not have executed anything"

    loser = [
        r for r in _bench_rows(db) if str(r["id"]) != str(winner["id"])
    ]
    assert len(loser) == 1
    assert loser[0]["status"] == "failed"
    assert loser[0]["progress_detail"]["failure"]["code"] == "superseded_by_existing_job"


# ---------------------------------------------------------------------------
# 3. Tenant isolation
# ---------------------------------------------------------------------------

def test_another_org_cannot_read_or_affect_a_job(db):
    """
    A job, its progress and its result are org property.

    The status endpoint is the new attack surface: it takes an id and returns
    what a benchmark is doing. It must answer 404 for another tenant's id —
    identically to how it answers for an id that does not exist, so it cannot be
    used to discover that a benchmark id is real.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))

    with _running(db, runtime):
        with _client(ORG_ID) as owner, _client(OTHER_ORG_ID) as intruder:
            runtime.hold()
            job = owner.post(OPTIMIZE, json={}).json()
            benchmark_id = job["benchmark_id"]
            assert _wait_until(runtime.started.is_set)

            # Reading someone else's job.
            stolen = intruder.get(
                f"/api/optimization/{OTHER_ORG_ID}/benchmarks/{benchmark_id}/status"
            )
            assert stolen.status_code == 404
            unknown = intruder.get(
                f"/api/optimization/{OTHER_ORG_ID}/benchmarks/"
                "00000000-0000-0000-0000-0000000000ff/status"
            )
            # Indistinguishable from "not yours": no oracle for another tenant's ids.
            assert unknown.status_code == 404
            assert stolen.json() == unknown.json()

            # Listing. The intruder's org has no jobs; the owner's does.
            assert intruder.get(f"/api/optimization/{OTHER_ORG_ID}/jobs").json()["jobs"] == []
            mine = owner.get(f"/api/optimization/{ORG_ID}/jobs").json()["jobs"]
            assert [j["benchmark_id"] for j in mine] == [benchmark_id]

            # Reaching across the path org into the owner's org is refused by
            # the handler's re-assertion of the VERIFIED org, not by the path.
            assert intruder.get(
                f"/api/optimization/{ORG_ID}/benchmarks/{benchmark_id}/status"
            ).status_code == 403
            assert intruder.post(OPTIMIZE, json={}).status_code == 403

            # The intruder cannot start a job on the owner's workload even
            # under its own org: the workload is not theirs.
            assert intruder.post(
                f"/api/optimization/{OTHER_ORG_ID}/workloads/{WORKLOAD_ID}/optimize",
                json={},
            ).status_code == 404

            # An identical Idempotency-Key from another tenant neither joins nor
            # blocks the owner's job.
            assert intruder.post(
                f"/api/optimization/{OTHER_ORG_ID}/workloads/{WORKLOAD_ID}/optimize",
                json={}, headers={"Idempotency-Key": "shared"},
            ).status_code == 404

            runtime.release.set()
            wait_for_job(owner, ORG_ID, benchmark_id)

    # And nothing the intruder did created, failed or altered a row.
    rows = _bench_rows(db)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert str(rows[0]["org_id"]) == ORG_ID


def test_the_reaper_only_ever_fails_a_job_within_its_own_org(db):
    """
    The reaper is the one code path with no org in scope, so it must carry the
    org from each ROW it acts on and never from an ambient value.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    old = "2020-01-01T00:00:00Z"
    db.seed("optimization_benchmarks", {
        "id": "aaaaaaaa-0000-0000-0000-000000000001", "org_id": ORG_ID,
        "workload_id": WORKLOAD_ID, "status": "running",
        "progress_state": "stage_1", "progress_detail": {}, "heartbeat_at": old,
    })
    db.seed("optimization_benchmarks", {
        "id": "bbbbbbbb-0000-0000-0000-000000000002", "org_id": OTHER_ORG_ID,
        "workload_id": WORKLOAD_ID, "status": "running",
        "progress_state": "stage_1", "progress_detail": {}, "heartbeat_at": old,
    })

    with _running(db, FakeRuntime()):
        scoped = jobs_mod.reap_stale_jobs(ORG_ID)

    assert scoped["failed"] == 1
    assert scoped["benchmark_ids"] == ["aaaaaaaa-0000-0000-0000-000000000001"]
    by_id = {str(r["id"]): r for r in _bench_rows(db)}
    assert by_id["aaaaaaaa-0000-0000-0000-000000000001"]["status"] == "failed"
    assert by_id["bbbbbbbb-0000-0000-0000-000000000002"]["status"] == "running"


# ---------------------------------------------------------------------------
# 4. Progress states advance and are persisted
# ---------------------------------------------------------------------------

def test_progress_states_advance_monotonically_and_are_persisted(db):
    """
    Every phase written was also STORED, and the sequence never goes backwards.

    Asserting an exact transition list would be asserting the stage plan, which
    is policy data and legitimately changes. What must hold is the band
    ordering: a run never reports an earlier phase than one it has already
    reported, and it passes through preparation, screening, the baseline, at
    least one stage, verification and concluding on its way to completed.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = FakeRuntime(n_cases=20)

    with _running(db, runtime):
        with _client() as client:
            job = client.post(OPTIMIZE, json={}).json()
            benchmark_id = job["benchmark_id"]
            assert job["progress_state"] in ("queued", "preparing")
            final = wait_for_job(client, ORG_ID, benchmark_id)

    # Only writes that actually matched a row count as progress.
    persisted = [w for w in db.progress_writes if w["matched"] > 0]
    assert persisted, "no progress was persisted at all"
    for write in persisted:
        # The value read back out of the store is the value that was intended.
        assert write["stored"] == write["intended"]

    states = [w["stored"] for w in persisted]
    assert all(jobs_mod.is_valid_progress_state(s) for s in states)

    bands = [jobs_mod.progress_band(s) for s in states]
    assert bands == sorted(bands), f"progress went backwards: {states}"

    for expected in (
        jobs_mod.PROGRESS_PREPARING,
        jobs_mod.PROGRESS_CANDIDATE_SCREENING,
        jobs_mod.PROGRESS_BASELINE_MEASUREMENT,
        jobs_mod.PROGRESS_VERIFICATION,
        jobs_mod.PROGRESS_CONCLUDING,
    ):
        assert expected in states, f"{expected} was never reported"
    assert any(s.startswith("stage_") for s in states), "no stage was reported"
    assert states[-1] == jobs_mod.PROGRESS_COMPLETED

    # The status the client last saw agrees with the row.
    row = _bench_rows(db)[0]
    assert row["progress_state"] == "completed"
    assert row["status"] == "completed"
    assert final["progress_state"] == "completed"
    assert final["terminal"] is True

    # The phase PLAN is published, so a client renders progress from data rather
    # than from a vocabulary it hardcoded.
    detail = row["progress_detail"]
    assert detail["stages_planned"] >= 1
    assert detail["plan"] == jobs_mod.progress_plan(detail["stages_planned"])
    assert detail["cases_planned"] == 20
    assert detail["arms_total"] >= 2


def test_stage_names_are_data_and_an_invented_phase_is_never_persisted(db):
    """
    Stage names are generated and validated, not typed.

    `stage_2` exists because a stage plan resolved two stages, and the vocabulary
    is closed: a reporter handed a name outside it writes nothing rather than
    quietly widening the contract the database CHECK and the frontend both rely
    on.
    """
    assert jobs_mod.stage_state(1) == "stage_1"
    assert jobs_mod.stage_state(12) == "stage_12"
    with pytest.raises(ValueError):
        jobs_mod.stage_state(0)

    assert jobs_mod.is_valid_progress_state("stage_3")
    assert not jobs_mod.is_valid_progress_state("stage_0")
    assert not jobs_mod.is_valid_progress_state("stage_two")
    assert not jobs_mod.is_valid_progress_state("almost_done")
    assert not jobs_mod.is_valid_progress_state(None)

    # A plan with three stages names three stages, in order, once.
    plan = jobs_mod.progress_plan(3)
    assert plan.index("stage_1") < plan.index("stage_2") < plan.index("stage_3")
    assert plan[0] == "queued" and plan[-1] == "completed"

    db.seed("optimization_benchmarks", {
        "id": "cccccccc-0000-0000-0000-000000000003", "org_id": ORG_ID,
        "status": "running", "progress_state": "preparing", "progress_detail": {},
    })
    with _running(db, FakeRuntime()):
        reporter = jobs_mod.ProgressReporter(ORG_ID, "cccccccc-0000-0000-0000-000000000003")
        reporter("nearly_finished")

    assert _bench_rows(db)[0]["progress_state"] == "preparing"


def test_a_refusal_still_reports_a_phase_and_completes(db):
    """
    An insufficient_evidence refusal is a successful job, not a failed one.

    It is decided during preparation, so the run goes straight to concluding —
    and it must still land as `completed`, because the system reached a verdict
    about what it knows. A refusal marked `failed` would be indistinguishable
    from a crashed worker.
    """
    _seed(db, golden_inputs=3, production_runs=120)  # below the sample floor
    runtime = FakeRuntime(n_cases=3)

    with _running(db, runtime):
        with _client() as client:
            job = client.post(OPTIMIZE, json={}).json()
            final = wait_for_job(client, ORG_ID, job["benchmark_id"])

    assert final["status"] == "completed"
    assert final["progress_state"] == "completed"
    assert final["result"]["conclusion"] == "insufficient_evidence"
    assert final["failure"] is None
    assert runtime.calls == [], "a refusal must not have spent anything"


# ---------------------------------------------------------------------------
# 5. An interrupted job never stays `running`
# ---------------------------------------------------------------------------

def test_a_job_interrupted_mid_flight_does_not_remain_permanently_running(db):
    """
    THE failure this design exists to prevent.

    A worker sets status='running' and its process dies: deploy, OOM, the
    platform moving the container. Nothing in the row distinguishes that from a
    28-minute run going fine, and the only process that knew is gone — so poll
    it a year later and it still says `running`. That state is unresolvable by
    construction, and it is what a naive implementation produces.

    HOW THE DEATH IS STAGED, AND WHY. Python cannot kill a thread that is
    mid-benchmark, and spawning a real subprocess would test the harness rather
    than the design. So this drives a REAL job to a REAL mid-flight state
    through the HTTP surface, snapshots the row exactly as the database holds it
    at that instant, and then restores that snapshot — which is, precisely and
    by definition, what the database would contain had the process died there.
    The reaper then has to resolve it, and only the passage of time (an unrenewed
    lease) is available to tell it apart from a healthy long run.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))

    with _running(db, runtime):
        with _client() as client:
            runtime.hold()
            job = client.post(OPTIMIZE, json={}).json()
            benchmark_id = job["benchmark_id"]
            assert _wait_until(runtime.started.is_set)

            # A genuine mid-flight row: claimed, phased, heartbeating.
            assert _wait_until(lambda: _bench_rows(db)[0]["status"] == "running")
            mid_flight = dict(_bench_rows(db)[0])
            assert mid_flight["status"] == "running"
            assert jobs_mod.is_valid_progress_state(mid_flight["progress_state"])
            assert mid_flight["heartbeat_at"]
            assert mid_flight["worker_id"]
            assert not mid_flight.get("conclusion")

            runtime.release.set()
            wait_for_job(client, ORG_ID, benchmark_id)

        # THE PROCESS DIES HERE. Restore the row to the instant above: no
        # further write ever arrives, because there is no longer anything to
        # write it.
        db.store["optimization_benchmarks"] = [dict(mid_flight)]

        # Inside its lease it is presumed alive. Declaring a slow run dead would
        # kill it and throw away the provider spend it has already made.
        assert jobs_mod.reap_stale_jobs(lease=3600)["failed"] == 0
        assert _bench_rows(db)[0]["status"] == "running"

        # The lease expires with no heartbeat. Nothing else could have told
        # these two cases apart.
        result = jobs_mod.reap_stale_jobs(lease=0)

    assert result["failed"] == 1
    assert result["benchmark_ids"] == [benchmark_id]

    row = _bench_rows(db)[0]
    assert row["status"] == "failed"
    assert row["progress_state"] == "failed"
    assert row["progress_detail"]["failure"]["code"] == "worker_lost"
    assert row["completed_at"]
    # No verdict was invented for a run that never reached one. A benchmark that
    # died has no conclusion about the workload, and manufacturing one from
    # outside the run would be fabricating evidence.
    assert not row.get("conclusion")
    # The arms it finished before dying are retained. Failing the job loses the
    # incomplete run, never the measurements already paid for.
    assert any(
        str(r.get("benchmark_id")) == benchmark_id
        for r in db.rows("benchmark_candidate_results")
    )


def test_cancelling_the_awaiting_task_does_not_abort_the_run(db):
    """
    Documents what cancellation actually does, because it is not obvious.

    The benchmark executes in a thread-pool worker; cancelling the coroutine
    that awaits it stops the awaiting, not the work. That is the RIGHT
    behaviour — a shutdown or a disconnect should not throw away a run that is
    twenty minutes into real provider spend — and it means the row still reaches
    a real verdict. The row is deliberately not marked failed on cancellation
    either: a process that is going away must not guess how much it finished.
    Only an unrenewed lease decides that.
    """
    _seed(db, golden_inputs=20, production_runs=120)
    runtime = GatedRuntime(FakeRuntime(n_cases=20))

    async def scenario():
        prepared = benchmark_mod.prepare_benchmark(ORG_ID, workload_id=WORKLOAD_ID)
        row, created, _key = jobs_mod.create_job(
            ORG_ID,
            workload_id=WORKLOAD_ID,
            job_kind=jobs_mod.JOB_KIND_OPTIMIZE,
            insert_row=lambda cols: benchmark_mod.create_benchmark_row(
                ORG_ID, prepared, extra_columns=cols
            ),
            objective=prepared["objective"],
        )
        assert created
        benchmark_id = str(row["id"])
        runtime.hold()
        task = jobs_mod.start_job(
            ORG_ID,
            benchmark_id,
            lambda reporter: benchmark_mod.run_benchmark(
                ORG_ID, workload_id=WORKLOAD_ID, prepared=prepared,
                benchmark_id=benchmark_id, progress=reporter,
            ),
        )
        assert await asyncio.to_thread(runtime.started.wait, 20)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        runtime.release.set()
        # Let the abandoned worker thread finish rather than leaking it.
        assert await asyncio.to_thread(
            _wait_until, lambda: _bench_rows(db)[0].get("conclusion") is not None, 60
        )
        return benchmark_id

    with _running(db, runtime):
        benchmark_id = asyncio.run(scenario())

    row = _bench_rows(db)[0]
    assert row["status"] == "completed"
    assert row["conclusion"]


def test_a_stale_job_is_reaped_and_a_live_one_is_not(db):
    """The lease decision itself, on rows with known heartbeats."""
    fresh = jobs_mod._iso(jobs_mod._utc_now())
    stale = "2020-01-01T00:00:00Z"
    db.seed("optimization_benchmarks",
        {"id": "d1", "org_id": ORG_ID, "status": "running",
         "progress_detail": {}, "heartbeat_at": stale, "worker_id": "host:1:aaaa"},
        {"id": "d2", "org_id": ORG_ID, "status": "running",
         "progress_detail": {}, "heartbeat_at": fresh, "worker_id": "host:2:bbbb"},
        {"id": "d3", "org_id": ORG_ID, "status": "queued",
         "progress_detail": {}, "created_at": stale},
        {"id": "d4", "org_id": ORG_ID, "status": "completed",
         "progress_detail": {}, "heartbeat_at": stale, "conclusion": "safe_improvement_found"},
    )

    with _running(db, FakeRuntime()):
        result = jobs_mod.reap_stale_jobs()

    by_id = {str(r["id"]): r for r in _bench_rows(db)}
    assert result["failed"] == 2
    assert by_id["d1"]["status"] == "failed"
    assert by_id["d1"]["progress_detail"]["failure"]["code"] == "worker_lost"
    assert by_id["d2"]["status"] == "running", "a heartbeating job must be left alone"
    # Queued with no heartbeat falls back to created_at, so a job that died
    # between insert and first write cannot outlive its lease either.
    assert by_id["d3"]["status"] == "failed"
    # A terminal row is never touched, whatever its heartbeat says. This is what
    # protects a preserved historical benchmark from a maintenance task.
    assert by_id["d4"]["status"] == "completed"
    assert by_id["d4"]["conclusion"] == "safe_improvement_found"


def test_the_reaper_never_overwrites_a_verdict_that_landed_first(db):
    """
    The race between a worker finishing and the reaper deciding it is dead.

    fail_job narrows its update to ACTIVE rows, so a run that completed a
    millisecond earlier wins and the reaper can SEE that it changed nothing.
    Without that filter, a slow-but-alive benchmark could have its real
    conclusion overwritten by 'worker_lost'.
    """
    db.seed("optimization_benchmarks", {
        "id": "e1", "org_id": ORG_ID, "status": "completed",
        "conclusion": "safe_improvement_found", "progress_state": "completed",
        "progress_detail": {}, "heartbeat_at": "2020-01-01T00:00:00Z",
    })
    with _running(db, FakeRuntime()):
        assert jobs_mod.fail_job(ORG_ID, "e1", code="worker_lost") is False

    row = _bench_rows(db)[0]
    assert row["status"] == "completed"
    assert row["conclusion"] == "safe_improvement_found"


def test_is_stale_treats_a_row_with_no_timestamps_as_dead(db):
    """
    Unknowable is the one state not permitted.

    A row that cannot prove it is alive is not alive. The alternative — leaving
    it active because there is no evidence either way — is exactly how a job
    becomes permanently `running`.
    """
    assert jobs_mod.is_stale({"status": "running"}) is True
    assert jobs_mod.is_stale({"status": "completed"}) is False
    assert jobs_mod.is_stale(
        {"status": "running", "heartbeat_at": jobs_mod._iso(jobs_mod._utc_now())}
    ) is False


def test_the_control_loop_reaps_orphans_on_every_cycle(db):
    """
    The reaper is wired into the EXISTING scheduler, not a parallel one.

    background_jobs.py already runs an in-process loop with an overlap guard and
    a CRITICAL done-callback. A second timer would have been a second thing to
    keep alive.
    """
    import background_jobs

    db.seed("optimization_benchmarks", {
        "id": "f1", "org_id": ORG_ID, "status": "running",
        "progress_detail": {}, "heartbeat_at": "2020-01-01T00:00:00Z",
    })
    with _running(db, FakeRuntime()):
        result = asyncio.run(background_jobs.run_benchmark_job_reaper())

    assert result["failed"] == 1
    assert _bench_rows(db)[0]["status"] == "failed"
