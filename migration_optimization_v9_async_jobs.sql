-- ============================================================================
-- Migration: OptiML optimization layer v9 — benchmark execution as an
--            ASYNCHRONOUS, RECOVERABLE, IDEMPOTENT job.
--
-- RUN AFTER: migration_optimization_v8_staged_evaluation.sql
--
-- *** NOT APPLIED, AND NOT RUN BY THE AGENT THAT WROTE IT. ***
-- The agent executed no SQL of any kind. Apply this by hand, and apply it
-- BEFORE deploying the code that accompanies it (see "ORDERING" below).
--
-- IDEMPOTENT BY CONSTRUCTION. Every statement is ADD COLUMN IF NOT EXISTS,
-- CREATE INDEX IF NOT EXISTS, DROP CONSTRAINT IF EXISTS followed by ADD
-- CONSTRAINT, COMMENT ON, or an UPDATE guarded by `WHERE ... IS NULL`.
-- Re-running it is a no-op.
--
-- ADDITIVE ONLY. No column is dropped, no column is retyped, no row is deleted,
-- and no existing value is overwritten. The single UPDATE fills a column that
-- did not exist a moment earlier, only where it is still NULL.
--
-- HISTORY IS PRESERVED. Benchmark 88813bfb-5581-45a0-abf6-884732a0b19b with
-- recommendation 41d79006-bc14-4180-9d3d-97e71d6a2809 (policy v1), and
-- benchmark 4d5ca24d-... with recommendation 25eabede-... (policy v2, the first
-- genuinely verified win), are untouched. Their `status`, `conclusion`,
-- `conclusion_detail`, `confidence`, their rows in benchmark_conclusions and
-- benchmark_candidate_results, and every recommendation citing them are not
-- read and not written by this migration. The backfill in section 6 sets only
-- the NEW `progress_state` column, and for a completed benchmark it sets it to
-- 'completed' — a restatement of the status those rows already carry, not a
-- reinterpretation of what was concluded.
--
-- NOTHING HERE WEAKENS AUTH, RLS OR ORG ISOLATION. No policy is created,
-- altered or dropped. Every new column lives on optimization_benchmarks, which
-- already has RLS enabled and an org-scoped SELECT policy; the new columns
-- inherit it. No column is exposed through a new view, function or grant.
--
--
-- ── WHY ─────────────────────────────────────────────────────────────────────
--
-- POST /api/optimization/{org}/workloads/{id}/optimize ran the entire benchmark
-- inside the HTTP request and awaited the verdict. The first real production
-- run — 140 replay cases across 7 arms — took about 28 minutes. Railway's edge
-- timeout is 300 seconds. At five minutes the client got a connection error;
-- the server worked for another 23 minutes, completed the benchmark, and wrote
-- its evidence to a row whose id it had told nobody. The measurement existed
-- and was unreachable.
--
-- Execution is now a persisted job. The job IS the benchmark row: this table
-- already carries status, started_at and completed_at, and the id the caller
-- polls is the id it will later cite as evidence. A separate optimization_jobs
-- table would have created a second id, a second lifecycle, and a join that can
-- disagree with itself about whether a run finished.
--
--
-- ── ORDERING (IMPORTANT) ────────────────────────────────────────────────────
--
-- Apply this migration BEFORE deploying the accompanying code. The code
-- degrades safely if you do not — optimization/jobs.py detects the missing
-- columns on first write, latches a degraded mode, logs CRITICAL, and keeps
-- running benchmarks with status-only tracking — but in that mode progress
-- reporting and orphan reaping are DISABLED, which is precisely the behaviour
-- this migration exists to end. GET /api/control-loop/status reports
-- `benchmark_jobs.job_columns_available: false` while degraded.
--
--
-- ── WHAT THIS FILE ADDS ─────────────────────────────────────────────────────
--
--   1. status gains 'queued'                     — accepted, not yet started
--   2. progress_state + progress_detail          — the phase axis
--   3. heartbeat_at + worker_id                  — the lease that makes an
--                                                  orphaned run detectable
--   4. idempotency_key + job_kind + requested_by — one in-flight job per scope
--   5. indexes                                   — the reaper's scan, and the
--                                                  partial unique index that
--                                                  enforces idempotency
--   6. backfill of progress_state for existing rows
-- ============================================================================


-- ============================================================================
-- 1. status: add 'queued'
--
-- TWO STATE AXES, DELIBERATELY SEPARATE.
--
--   status          the LIFECYCLE axis. Coarse, already indexed
--                   (idx_optbench_org_status), already read by the rest of the
--                   system, already CHECK-constrained.
--   progress_state  the PHASE axis, added in section 2.
--
-- The eight-name phase vocabulary is NOT folded into `status`. Doing so would
-- have meant teaching every existing reader that 'stage_2' means "still
-- running", and every existing `.eq("status", "completed")` would have kept
-- working only by luck. Instead `status` gains exactly one new value.
--
-- 'pending' is RETAINED. It is what rows written before this migration carry
-- between insert and start, including historical rows. Migrating them to
-- 'queued' to make the new vocabulary look tidy would be a history rewrite for
-- cosmetic reasons; the application treats 'pending' and 'queued' identically
-- as active states.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    DROP CONSTRAINT IF EXISTS optimization_benchmarks_status_check;

ALTER TABLE public.optimization_benchmarks
    ADD CONSTRAINT optimization_benchmarks_status_check
    CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed'));

COMMENT ON COLUMN public.optimization_benchmarks.status IS
  'LIFECYCLE axis of a benchmark job. pending|queued = accepted, no arm executed, nothing spent (''pending'' is the pre-v9 spelling and is still written by synchronous callers; the two are treated identically). running = a worker holds it and is renewing heartbeat_at. completed = a conclusion was reached, which may be a refusal such as insufficient_evidence. failed = no conclusion was reached; see error and progress_detail.failure.code. A job in pending|queued|running whose heartbeat_at is older than the lease is ORPHANED and is marked failed with progress_detail.failure.code = ''worker_lost''. The FINE-GRAINED phase lives in progress_state, not here.';


-- ============================================================================
-- 2. progress_state + progress_detail — the phase axis
--
-- STAGE NAMES ARE DATA, NOT FREE TEXT. The CHECK below is the database half of
-- that guarantee; optimization/jobs.py holds the other half
-- (is_valid_progress_state / stage_state(k), and a ProgressReporter that
-- refuses to persist a name it does not recognise). A phase name cannot be
-- invented by a handler, by a caller, or by an UPDATE.
--
-- WHY THE STAGE BAND IS A PATTERN AND NOT A FIXED LIST. The number of
-- evaluation stages is POLICY DATA: optimization_policies.constraints
-- .evaluation_stage_sizes, default [30, 60, 133], overridable per workload, and
-- absent entirely when staged_evaluation_enabled is false (one unstaged pass).
-- Enumerating 'stage_1','stage_2' would misreport every three-stage run and
-- would have no name at all for the unstaged one. progress_detail.stages_planned
-- carries the count, so a determinate progress bar is still possible.
--
-- WHY 'preparing' AND 'baseline_measurement' EXIST. Both are real, and both are
-- slow. Resolving the workflow, loading the promoted deployment graph and
-- loading the golden inputs happens before any candidate exists, and every
-- refusal that concludes insufficient_evidence — no workflow, no baseline
-- graph, sample size below floor — is decided inside it. The baseline arm then
-- runs to completion over the FULL case set before any candidate starts; in the
-- 140-case incident that was one seventh of the total work. Reporting either as
-- 'queued' or as 'candidate_screening' would attribute time and spend to the
-- wrong thing.
--
-- 'stage_k' IS NOT MONOTONE ON ITS OWN. Arms are evaluated one after another
-- and each walks the stage plan from stage 1, so this band cycles 1,2,3,1,2,3.
-- The monotone quantity is the pair (progress_detail.arms_completed,
-- progress_detail.stage_index). A frontend must order on that, not on the
-- phase name.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    ADD COLUMN IF NOT EXISTS progress_state  TEXT,
    ADD COLUMN IF NOT EXISTS progress_detail JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.optimization_benchmarks
    DROP CONSTRAINT IF EXISTS optimization_benchmarks_progress_state_check;

ALTER TABLE public.optimization_benchmarks
    ADD CONSTRAINT optimization_benchmarks_progress_state_check
    CHECK (
        progress_state IS NULL
        OR progress_state IN (
            'queued',
            'preparing',
            'candidate_screening',
            'baseline_measurement',
            'verification',
            'concluding',
            'completed',
            'failed'
        )
        OR progress_state ~ '^stage_[1-9][0-9]*$'
    );

COMMENT ON COLUMN public.optimization_benchmarks.progress_state IS
  'PHASE axis of a benchmark job — where a long run actually is. Vocabulary, in order: queued (persisted, nothing spent) -> preparing (resolving workflow, promoted graph and golden inputs; every insufficient_evidence refusal for a missing baseline or a short sample is decided here) -> candidate_screening (generating and screening candidate strategies; no arm executed) -> baseline_measurement (the baseline arm over the FULL case set, the reference every candidate is paired against) -> stage_1..stage_N (candidate arms over stage k of the resolved staged-evaluation plan; N = progress_detail.stages_planned, which is policy data and is NOT always 2) -> verification (policy comparison, quality non-inferiority, frontier, consideration funnel; runs no model calls) -> concluding (writing the immutable conclusion and, only on safe_improvement_found, the recommendation that cites it) -> completed | failed. NULL means the phase was never recorded — a pre-v9 row, or a run made while the schema was degraded. It is left NULL rather than guessed. The stage band CYCLES across arms: order on (progress_detail.arms_completed, progress_detail.stage_index), never on the phase name alone.';

COMMENT ON COLUMN public.optimization_benchmarks.progress_detail IS
  'Facts behind the phase, all COUNTS OF WORK ALREADY DONE — never an estimate of work remaining and never a provisional verdict. Keys: phase (mirrors progress_state), plan (the ordered phase list for this run, so a client need not hardcode the vocabulary), stages_planned, stage_index, arms_total, arms_completed, cases_planned, cases_completed, updated_at, and on a failed job failure:{code, detail} where code is one of worker_lost | superseded_by_existing_job | start_failed. Anything unmeasured is absent or null; nothing here is inferred.';


-- ============================================================================
-- 3. heartbeat_at + worker_id — how an orphaned run stops being "running
--    forever"
--
-- THE FAILURE MODE THIS FIXES. A worker sets status='running' and the process
-- dies: deploy, OOM, the platform moving the container. Nothing in the row
-- distinguishes that from a 28-minute run that is going fine. Poll it a year
-- later and it still says 'running'. The only process that knew the truth is
-- the one that died, so no amount of reading the row can recover it.
--
-- THE MECHANISM: LIVENESS IS A CLAIM THAT MUST BE RENEWED. heartbeat_at is
-- rewritten at every phase transition and, because one arm over 140 cases can
-- run for minutes without a transition, by a ticker every
-- OPTIML_BENCHMARK_HEARTBEAT_SECONDS (default 30). A job is declared ORPHANED
-- when it is in pending|queued|running and heartbeat_at is older than
-- OPTIML_BENCHMARK_LEASE_SECONDS (default 300 — ten heartbeats). A dead process
-- renews nothing, so the passage of time alone decides it.
--
-- WHEN THE CHECK RUNS: on every control-loop tick (60s, the existing scheduler
-- in background_jobs.py), and once at process STARTUP — because on a
-- single-worker deployment the process that lost the job is the one coming
-- back, and nothing else is watching.
--
-- WHAT HAPPENS TO AN ORPHAN: IT IS FAILED, NOT RESUMED. Resuming would need the
-- candidate ordering, the per-arm case cursor and the in-memory per-case
-- results, none of which are persisted; a "resumed" run would silently
-- re-execute arms and double-spend. Every arm that finished before the crash is
-- already durable in benchmark_candidate_results, so failing the job loses no
-- measurement and no money — only the incomplete run. It gets status='failed',
-- progress_state='failed', progress_detail.failure.code='worker_lost', and NO
-- conclusion: a job that died has no verdict about the workload, and writing
-- one from outside the run — with no policy, no materiality and no measured
-- arms in hand — would be inventing evidence.
--
-- worker_id is host:pid:random. The random suffix matters: without it, a
-- process that restarts fast enough to reuse a pid is indistinguishable from
-- the one that died, and a new worker could mistake an orphan for its own.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS worker_id    TEXT;

COMMENT ON COLUMN public.optimization_benchmarks.heartbeat_at IS
  'LEASE. The last moment the worker running this job proved it was alive: rewritten at every phase transition and by a ticker every OPTIML_BENCHMARK_HEARTBEAT_SECONDS (default 30). A row in pending|queued|running whose heartbeat_at is older than OPTIML_BENCHMARK_LEASE_SECONDS (default 300) is ORPHANED — its worker is gone — and is marked failed with failure code worker_lost by the reaper in background_jobs.py, which runs every control-loop tick and once at process startup. This column is the ONLY thing standing between a crashed worker and a row that says ''running'' forever. NULL falls back to started_at, then created_at, so a row that never heartbeat still cannot outlive its lease.';

COMMENT ON COLUMN public.optimization_benchmarks.worker_id IS
  'Which process last held this job: host:pid:random. Diagnostic only — never an authorization input and never a lock. The random suffix distinguishes two runs of the same pid after a fast restart, which is otherwise indistinguishable and would let a new process mistake an orphan for one of its own.';


-- ============================================================================
-- 4. idempotency_key + job_kind + requested_by
--
-- THE FAILURE BEING DEFENDED AGAINST is not a malicious duplicate; it is the
-- one the incident produced. The caller posts /optimize, the edge times out at
-- 300s, the caller sees a connection error and — reasonably, because as far as
-- it knows nothing happened — posts again. Without a key that second POST
-- launches a second 28-minute, 140-case, 7-arm run against real providers.
--
-- THE KEY IS DERIVED, NOT CLIENT-SUPPLIED, BY DEFAULT. sha256 over:
--
--     (org_id, workload_id, method, objective, min_sample_size,
--      create_recommendation, recommendation_id)
--
-- Read it as a SCOPE, not a request id: at most one IN-FLIGHT benchmark per
-- org, workload, objective, method and recommendation-intent. A
-- client-supplied-only key would have put the guarantee in the hands of the
-- client that had just been given no way to know the truth — the retrying
-- client generates a fresh uuid and spends the money anyway.
--
-- create_recommendation is IN the tuple because /optimize and the exploratory
-- /benchmark differ in exactly that boolean: an exploratory run must never be
-- handed back to a caller whose request was supposed to be able to produce a
-- proposal. recommendation_id is in it because evidence gathered FOR a specific
-- recommendation is not interchangeable with evidence gathered for none.
--
-- A client MAY override with an Idempotency-Key header. That value is hashed
-- WITH org_id and under a separate 'client:' namespace, so one tenant choosing
-- "retry-1" can never collide with another tenant's, nor with a derived key.
--
-- ENFORCED ONLY WHILE ACTIVE — hence the PARTIAL unique index in section 5.
-- Once a job is completed or failed the key is free again, because re-running a
-- benchmark after it has finished is legitimate and intended: evidence ages,
-- golden inputs are added, the policy version changes. Idempotency here means
-- "do not run two of these at once", not "never run this twice".
--
-- THE TRADE-OFF, STATED: a user cannot launch two concurrent benchmarks with
-- identical parameters on the same workload without supplying distinct
-- Idempotency-Key headers. That is intended. Two such runs replay the same
-- cases through the same providers for the same verdict; the second is spend
-- with no information in it.
-- ============================================================================
ALTER TABLE public.optimization_benchmarks
    ADD COLUMN IF NOT EXISTS idempotency_key TEXT,
    ADD COLUMN IF NOT EXISTS job_kind        TEXT,
    ADD COLUMN IF NOT EXISTS requested_by    TEXT;

ALTER TABLE public.optimization_benchmarks
    DROP CONSTRAINT IF EXISTS optimization_benchmarks_job_kind_check;

ALTER TABLE public.optimization_benchmarks
    ADD CONSTRAINT optimization_benchmarks_job_kind_check
    CHECK (
        job_kind IS NULL
        OR job_kind IN ('optimize', 'explore', 'recommendation_benchmark')
    );

COMMENT ON COLUMN public.optimization_benchmarks.idempotency_key IS
  'What makes two requests the same IN-FLIGHT job. Default: ''auto:'' + sha256 over (org_id, workload_id, job_kind, objective, min_sample_size, create_recommendation, recommendation_id) — a SCOPE, not a request id, so a caller retrying after an edge timeout is handed the run that is already going instead of starting a second one against real providers. Override: ''client:'' + sha256 over (org_id, workload_id, Idempotency-Key header), a separate namespace so a tenant-chosen string can never collide with a derived key or with another tenant''s. Enforced ONLY against pending|queued|running rows, by idx_optbench_idempotency_active below: once a job is terminal the key is free, because re-running a benchmark later is a legitimate act. NULL on every pre-v9 row and on any benchmark started synchronously.';

COMMENT ON COLUMN public.optimization_benchmarks.job_kind IS
  'WHICH INTENT was expressed, recorded rather than inferred later from which columns happen to be populated. optimize = the full loop; a safe_improvement_found conclusion creates a recommendation citing this benchmark. explore = POST /workloads/{id}/benchmark; measures and concludes but may NEVER create a recommendation, whatever it finds. recommendation_benchmark = evidence for an EXISTING recommendation, whose lifecycle the conclusion then drives. This is part of the idempotency key, so an exploratory run and a full-loop run on the same workload are never confused for one another.';

COMMENT ON COLUMN public.optimization_benchmarks.requested_by IS
  'The authenticated user_id that requested this job, for audit. Never an authorization input: every route is guarded by require_org_member with org_id in the path and re-filters by the verified org, and this column is never read to decide access.';


-- ============================================================================
-- 5. Indexes
--
-- The partial unique index is the REAL idempotency guarantee. The application
-- also checks for an existing active job before inserting, but a check-then-act
-- across two processes has a window; this index closes it. A losing insert
-- raises, and the caller re-reads and returns the winner — having spent
-- nothing, because the decision happens before any provider call.
-- ============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS idx_optbench_idempotency_active
    ON public.optimization_benchmarks (org_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL
      AND status IN ('pending', 'queued', 'running');

-- The reaper's scan: active jobs ordered by staleness.
CREATE INDEX IF NOT EXISTS idx_optbench_active_heartbeat
    ON public.optimization_benchmarks (status, heartbeat_at)
    WHERE status IN ('pending', 'queued', 'running');

-- The dashboard's "what is running for my org right now".
CREATE INDEX IF NOT EXISTS idx_optbench_org_active
    ON public.optimization_benchmarks (org_id, created_at DESC)
    WHERE status IN ('pending', 'queued', 'running');


-- ============================================================================
-- 6. Backfill progress_state for rows that predate it
--
-- Fills the NEW column only, and only where it is still NULL. Reads `status`;
-- writes nothing else. No conclusion, confidence, metric or timestamp is
-- touched, and no row is deleted.
--
-- An in-flight row is left at NULL rather than assigned a phase. It genuinely
-- has no recorded phase, and guessing one would be a fabrication — the same
-- rule that governs every other unmeasurable value in this schema. It is not
-- left unresolvable: it has no heartbeat_at either, so it falls back to
-- started_at/created_at, is already past its lease, and the first reaper pass
-- after this migration will mark it failed with worker_lost. That is the
-- correct outcome — any row that was 'running' when this migration is applied
-- belongs to a process that is definitionally gone.
--
-- The two preserved historical benchmarks are 'completed' and therefore get
-- progress_state='completed': a restatement of the status they already carry.
-- ============================================================================
UPDATE public.optimization_benchmarks
   SET progress_state = 'completed'
 WHERE progress_state IS NULL
   AND status = 'completed';

UPDATE public.optimization_benchmarks
   SET progress_state = 'failed'
 WHERE progress_state IS NULL
   AND status = 'failed';


-- ============================================================================
-- Verification (read-only; run by hand after applying)
-- ============================================================================
-- SELECT status, progress_state, count(*)
--   FROM public.optimization_benchmarks
--  GROUP BY 1, 2 ORDER BY 1, 2;
--
-- -- The two preserved benchmarks, unchanged apart from the new column:
-- SELECT id, status, progress_state, conclusion, confidence, completed_at
--   FROM public.optimization_benchmarks
--  WHERE id IN ('88813bfb-5581-45a0-abf6-884732a0b19b');
--
-- -- Anything still in flight after the migration (expect these to be reaped):
-- SELECT id, org_id, status, heartbeat_at, started_at, created_at
--   FROM public.optimization_benchmarks
--  WHERE status IN ('pending', 'queued', 'running')
--  ORDER BY created_at;
-- ============================================================================
