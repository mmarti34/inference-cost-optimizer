-- ============================================================================
-- Migration: OptiML optimization layer v14 — EVIDENCE CANDIDATES.
--            Production traffic becomes a REVIEW QUEUE, and only a human
--            decision turns a queue entry into replay evidence.
--
-- RUN AFTER: migration_optimization_v13_input_text_redaction_provenance.sql
--
-- *** NOT APPLIED, AND NOT RUN BY THE AGENT THAT WROTE IT. ***
-- The agent executed no SQL of any kind: no DDL, no DML, no SELECT, no
-- migration tool, no Supabase MCP call. Apply this by hand after review.
--
-- IDEMPOTENT BY CONSTRUCTION. One `CREATE TABLE IF NOT EXISTS`, six
-- `CREATE INDEX IF NOT EXISTS`, one `ALTER TABLE ... ENABLE ROW LEVEL
-- SECURITY` (a no-op when already enabled), one `DROP POLICY IF EXISTS` +
-- `CREATE POLICY` pair, two `REVOKE` (no-ops when the privilege is already
-- absent) and a block of `COMMENT ON`. There is no RENAME, no DROP TABLE, no
-- DROP COLUMN, no INSERT, no UPDATE and no DELETE. NO EXISTING ROW IN ANY
-- TABLE IS READ OR WRITTEN. Re-running the file changes nothing.
--
-- NO EXISTING TABLE IS ALTERED. `golden_inputs`, `workflow_runs`, `workloads`
-- and `eval_suites` are referenced by FOREIGN KEY only. Adding an inbound FK
-- takes a SHARE ROW EXCLUSIVE lock on the referenced table for the duration of
-- the validation scan; on tables of this size (workflow_runs is in the low
-- thousands of rows) that is milliseconds, and it does not rewrite them.
--
-- ORDERING RELATIVE TO THE CODE: APPLY THIS MIGRATION FIRST.
-- optimization/curation.py reads and writes `evidence_candidates` on every
-- call to the three new endpoints. Without the table those endpoints return a
-- structured "unavailable" rather than crashing, but they do nothing useful.
-- Nothing else in the product depends on this table, so shipping the code
-- first is degraded, not broken.
--
--
-- ── THE PROBLEM, MEASURED ───────────────────────────────────────────────────
--
-- OptiML can observe, capture, prove and optimize. It could not CURATE. Every
-- benchmark path in this codebase reads `golden_inputs` as its case set, and
-- `golden_inputs` is populated by a customer who arrives with a golden
-- dataset. Essentially nobody does.
--
-- Measured across the whole database at the time of writing:
--
--     169 distinct inputs, all orgs, all time      (140 of them our own)
--     largest single real workload: 39 distinct inputs across 742 runs
--
-- Two things follow from those numbers, and both are load-bearing here.
--
--   1. THE UNIT OF WORK IS TENS, NOT THOUSANDS. Nothing in this design uses
--      embeddings or clustering. At 39 examples an embedding model is an
--      expensive way to reproduce what an exact-match fingerprint already
--      gets right, and it introduces a similarity threshold nobody can defend.
--      The cheap signals below work at 30, 50 and 100 examples, which is the
--      range that actually exists.
--
--   2. DEDUP, NOT VOLUME, IS THE CONSTRAINT. 742 runs collapsing to 39 is an
--      19:1 ratio. A queue that shows the reviewer 742 rows is unusable, and a
--      counter that says "742 cases captured" is a lie. The fingerprint is
--      therefore part of the schema — a UNIQUE index, not an application
--      convention — so a duplicate cannot be inserted even by a buggy caller.
--
--
-- ── WHY A SEPARATE TABLE, AND NOT A `status` COLUMN ON `golden_inputs` ──────
--
-- This is the central design decision of the migration and the reason the file
-- exists at all.
--
-- `golden_inputs` currently means, everywhere, "approved replay evidence".
-- `optimization/benchmark.py:_load_golden_inputs` reads it as THE case set,
-- unfiltered, and there are TEN readers of that table across the codebase
-- (benchmark.py, workloads.py `_golden_input_count`, candidates.py,
-- context_accounting.py, and six handlers in workflow_management.py including
-- the eval replay loop and the workflow-delete cascade).
--
-- Adding `status` to `golden_inputs` and parking unreviewed candidates there
-- would require all ten readers to add `.eq("status", "approved")`. Every one
-- of those filters is a chance to be forgotten, and the failure mode of
-- forgetting is SILENT: an unapproved candidate, whose "expected output" is
-- whatever production happened to emit, is fed into a real benchmark and the
-- benchmark concludes on it. Nobody sees a stack trace. They see a quality
-- number that means something other than what it says.
--
-- A separate table makes "everything in `golden_inputs` is human-approved"
-- true BY CONSTRUCTION rather than by convention. The ten readers are not
-- touched, cannot be forgotten, and stay correct without knowing this table
-- exists. The only bridge between the two tables is one INSERT performed by
-- the approval path.
--
-- This is the same reasoning that made the `resource_access.py` helpers refuse
-- an org parameter outright instead of documenting that callers should pass
-- the right one. A rule you cannot express in the type system, you express in
-- the schema; a rule you express only in a comment is a rule you will lose.
--
--
-- ── A PRODUCTION OUTPUT IS A PROPOSED LABEL, NEVER A GOLDEN ANSWER ─────────
--
-- The single most important semantic in this table, and the reason
-- `production_output` and `expected_output` are two columns instead of one.
--
-- A production INPUT is excellent benchmark material: it is real, it is
-- representative, and it is exactly the traffic the customer cares about
-- getting right. A production OUTPUT is not automatically correct. It is what
-- one model produced on one day under one prompt. If the workload is being
-- optimized precisely because its current output is mediocre, then treating
-- that output as the expected answer bakes the mediocrity into the benchmark
-- and every candidate is scored on its ability to reproduce it.
--
-- So:
--
--   production_output        WHAT HAPPENED. Immutable observation. Written
--                            once at derivation, never edited by review.
--   proposed_expected_output THE PROPOSAL. Seeded from production_output, and
--                            it is a proposal for exactly as long as nobody
--                            has looked at it.
--   expected_output          THE HUMAN'S ANSWER. NULL until a human approves
--                            or edits. This — never production_output — is
--                            what is copied into `golden_inputs`.
--
-- `expected_output IS NULL` and `state IN ('captured','proposed_for_review')`
-- are therefore two spellings of the same fact, and the CHECK constraint
-- `chk_evidence_candidates_approved_has_expected` makes them agree at the
-- database level: a row cannot reach an approved state without a human answer
-- on it.
--
--
-- ── LIFECYCLE ───────────────────────────────────────────────────────────────
--
--   captured             derived from a run; nobody has seen it
--   proposed_for_review   surfaced in the queue
--   human_approved        a human accepted the proposed output as expected
--   human_edited          a human supplied a DIFFERENT expected output
--   rejected              a human judged the production output wrong
--   not_useful            a valid output, but a case not worth benchmarking
--
-- `human_approved` and `human_edited` are kept apart rather than collapsed into
-- one "approved" state because they answer different questions about the
-- dataset. A workload whose cases are 90% `human_edited` is one whose
-- production output is routinely wrong — which is a finding about the
-- workload, not a detail of the review UI. Collapsing them would erase it.
--
-- Likewise `rejected` (the output is wrong) and `not_useful` (the output is
-- fine, the case is uninteresting) are distinct: the first is evidence about
-- quality, the second is evidence about coverage. Both are terminal and
-- neither produces a `golden_input`.
--
-- No state transition is enforced by the schema beyond the two CHECKs below.
-- The lifecycle is enforced in optimization/curation.py, which is also where
-- the audit record is written. A trigger enforcing the graph was considered
-- and rejected: it would put half the rule in SQL and half in Python, and the
-- half in SQL cannot write the audit row.
--
--
-- ── REDACTED AND PROVENANCE-UNKNOWN CANDIDATES ARE VISIBLE, NEVER SILENT ────
--
-- `evidence_redaction.replay_gate()` already encodes the rule and this table
-- stores its verdict rather than re-deriving one:
--
--   replay_eligible = false, reason `redacted_input_requires_review`
--       redaction modified the persisted content. The removed value may be
--       exactly what drove the behaviour the case claims to reproduce.
--   replay_eligible = false, reason `capture_provenance_unavailable`
--       capture was attempted and produced nothing usable. UNKNOWN IS NOT
--       CLEAN: `redacted: false` on a row nobody could inspect is an absence
--       of evidence, not evidence of absence.
--
-- Neither verdict hides the row. Both are surfaced in the queue with their
-- reason codes, and both can be approved — but only by a human who ticks
-- `review_acknowledged_redaction`, which is recorded on the row with the
-- reviewer and the timestamp. That column existing is the point: six months
-- later, "why does this case not replay?" has an answer in the row itself.
--
--
-- ── NOTHING HERE WEAKENS AUTH, RLS OR ORG ISOLATION ────────────────────────
--
-- The table is created WITH RLS ENABLED and one SELECT policy gated on
-- `public.is_org_member(org_id)` — identical to `benchmark_candidate_results`,
-- `benchmark_conclusions` and `recommendation_evidence` from v5. No INSERT,
-- UPDATE or DELETE policy exists, so the `anon` and `authenticated` roles can
-- never write it; all writes go through the backend's service-role client,
-- which is also where the ownership check and the audit record live. `REVOKE
-- ALL ... FROM anon` is defence in depth, so that a future `DISABLE ROW LEVEL
-- SECURITY` on this table does not by itself make it publicly readable.
--
-- No existing policy, function, grant, role or view is created, altered or
-- dropped by this file.
--
-- NOTE ON WHAT THIS TABLE HOLDS. `input_text`, `variables`,
-- `production_output`, `proposed_expected_output` and `expected_output` all
-- carry customer request and response content, redacted at WRITE time by
-- `evidence_redaction.persist_golden_input()` before they reach this table —
-- the same boundary `golden_inputs` uses, applied for the same reason. A value
-- that is never written cannot leak, cannot be exported, and does not depend
-- on the RLS posture staying as this file leaves it.
-- ============================================================================


-- ── 1. The table ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.evidence_candidates (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                   UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    workload_id              UUID NOT NULL REFERENCES public.workloads(id) ON DELETE CASCADE,
    -- NULL for a direct-inference workload, which has no workflow. Resolved
    -- via workloads.resolve_workflow_id, never guessed.
    workflow_id              UUID REFERENCES public.workflows(id) ON DELETE CASCADE,

    -- ── Provenance: which run this came from ────────────────────────────────
    -- ON DELETE SET NULL, deliberately. A candidate that has been reviewed is
    -- a human decision and must outlive the run that suggested it; cascading
    -- would let a log-retention job silently delete approved evidence.
    source_run_id            UUID REFERENCES public.workflow_runs(id) ON DELETE SET NULL,
    captured_at              TIMESTAMPTZ,
    last_seen_at             TIMESTAMPTZ,
    occurrences              INT NOT NULL DEFAULT 1,

    -- ── The captured input ──────────────────────────────────────────────────
    input_text               TEXT,
    variables                JSONB,

    -- ── Output: observation, proposal, answer. Three columns, three meanings.
    production_output        TEXT,
    proposed_expected_output TEXT,
    expected_output          TEXT,

    -- ── Capture / redaction provenance, carried over and re-derived ─────────
    capture                  JSONB,
    replay_eligible          BOOLEAN,
    replay_reason_codes      TEXT[] NOT NULL DEFAULT '{}',
    redacted                 BOOLEAN,
    redacted_kinds           TEXT[] NOT NULL DEFAULT '{}',

    -- ── Dedup ───────────────────────────────────────────────────────────────
    fingerprint              TEXT NOT NULL,
    fingerprint_version      INT NOT NULL DEFAULT 1,

    -- ── Diversity ───────────────────────────────────────────────────────────
    bucket                   TEXT NOT NULL DEFAULT 'common'
                             CHECK (bucket IN ('common', 'long_input', 'failure',
                                               'unusual_output_shape',
                                               'unusual_variable_shape',
                                               'outlier_cost_latency',
                                               'random_coverage')),
    bucket_signals           JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- ── Inferred structural checks ──────────────────────────────────────────
    checks                   JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- ── Lifecycle ───────────────────────────────────────────────────────────
    state                    TEXT NOT NULL DEFAULT 'captured'
                             CHECK (state IN ('captured', 'proposed_for_review',
                                              'human_approved', 'human_edited',
                                              'rejected', 'not_useful')),
    reviewed_by              UUID,
    reviewed_at              TIMESTAMPTZ,
    review_acknowledged_redaction BOOLEAN NOT NULL DEFAULT FALSE,

    -- ── The bridge, and the only one ────────────────────────────────────────
    golden_input_id          UUID REFERENCES public.golden_inputs(id) ON DELETE SET NULL,

    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- An approved row MUST carry a human answer. This is the "a production
    -- output is never automatically golden" rule, expressed where it cannot be
    -- forgotten: no code path can produce an approved candidate whose expected
    -- output was never set by a human, because the database will refuse the row.
    CONSTRAINT chk_evidence_candidates_approved_has_expected
        CHECK (state NOT IN ('human_approved', 'human_edited')
               OR expected_output IS NOT NULL),

    -- Conversely, only an approved row may carry a golden_input_id. A rejected
    -- or unreviewed candidate pointing at a golden input would mean the bridge
    -- was crossed without approval.
    CONSTRAINT chk_evidence_candidates_golden_requires_approval
        CHECK (golden_input_id IS NULL
               OR state IN ('human_approved', 'human_edited'))
);


-- ── 2. Indexes ──────────────────────────────────────────────────────────────

-- THE DEDUP INVARIANT, in the schema rather than in a comment. 742 runs ->
-- 39 candidates is enforced here: a second run with the same normalised input
-- cannot be inserted at all, so the counter cannot be inflated by a retry, a
-- concurrent derivation pass, or a caller that forgot to check first.
-- Scoped to (org, workload) rather than (org): the same input arriving at two
-- different workloads is two different cases.
CREATE UNIQUE INDEX IF NOT EXISTS uq_evidence_candidates_fingerprint
    ON public.evidence_candidates (org_id, workload_id, fingerprint);

-- ONE golden input per candidate, and one candidate per golden input. This is
-- what makes approval IDEMPOTENT at the database level: a double-clicked
-- Approve cannot create a second `golden_input` row, because the second insert
-- has nowhere to record itself.
CREATE UNIQUE INDEX IF NOT EXISTS uq_evidence_candidates_golden_input
    ON public.evidence_candidates (golden_input_id)
    WHERE golden_input_id IS NOT NULL;

-- The queue query: this workload, unreviewed, oldest first.
CREATE INDEX IF NOT EXISTS idx_evidence_candidates_queue
    ON public.evidence_candidates (org_id, workload_id, state, captured_at);

-- The counters query.
CREATE INDEX IF NOT EXISTS idx_evidence_candidates_workload_state
    ON public.evidence_candidates (workload_id, state);

-- "Which cases came from runs we can no longer replay faithfully?"
CREATE INDEX IF NOT EXISTS idx_evidence_candidates_replay_eligible
    ON public.evidence_candidates (org_id, workload_id, replay_eligible);

-- Derivation asks "have I already seen this run?" once per scanned run.
CREATE INDEX IF NOT EXISTS idx_evidence_candidates_source_run
    ON public.evidence_candidates (source_run_id)
    WHERE source_run_id IS NOT NULL;


-- ── 3. RLS. Same posture as the v5 evidence tables. ─────────────────────────

ALTER TABLE public.evidence_candidates ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Org members can view evidence candidates" ON public.evidence_candidates;
CREATE POLICY "Org members can view evidence candidates"
    ON public.evidence_candidates FOR SELECT
    USING (public.is_org_member(org_id));

-- No INSERT/UPDATE/DELETE policy, on purpose. Writes are the review decision,
-- and a review decision must go through the backend so that the ownership
-- check and the audit_log row happen in the same place. The service-role
-- client bypasses RLS, so the backend is unaffected.

-- Defence in depth: with the grants gone, a future `DISABLE ROW LEVEL
-- SECURITY` on this table does not by itself expose customer request content
-- over PostgREST. Two things would have to go wrong instead of one.
REVOKE ALL ON public.evidence_candidates FROM anon;
REVOKE ALL ON public.evidence_candidates FROM authenticated;


-- ── 4. What the columns mean ────────────────────────────────────────────────

COMMENT ON TABLE public.evidence_candidates IS
  'Production traffic proposed as replay evidence, and the human review that decides. A row here is a CANDIDATE: real input captured from a real run, carrying the production output as a PROPOSAL, not as a golden answer. Only the approval path in optimization/curation.py may create a `golden_inputs` row from one, and it copies `expected_output` (the human''s answer) — never `production_output` (what happened). Kept in its own table rather than as a status column on `golden_inputs` because ten readers across the codebase treat `golden_inputs` as the approved case set; a forgotten status filter on any one of them would feed unreviewed cases into a real benchmark silently. Separation makes "everything in golden_inputs is human-approved" true by construction.';

COMMENT ON COLUMN public.evidence_candidates.state IS
  'Lifecycle, six values, all disjoint. captured = derived from a run, nobody has seen it. proposed_for_review = surfaced in the review queue. human_approved = a human accepted the production output as the expected result. human_edited = a human supplied a DIFFERENT expected result. rejected = a human judged the production output wrong. not_useful = the output is fine but the case is not worth benchmarking. human_approved and human_edited are NOT collapsed: a workload whose cases are mostly human_edited is one whose production output is routinely wrong, which is a finding about the workload. rejected and not_useful are NOT collapsed either: the first is evidence about quality, the second about coverage. Only human_approved and human_edited count toward benchmarkability, and only they may carry a golden_input_id.';

COMMENT ON COLUMN public.evidence_candidates.production_output IS
  'WHAT ACTUALLY HAPPENED — `workflow_runs.final_output` for the source run, redacted at write time. An immutable observation. Review never edits it, so the record of what production emitted survives whatever a human decides the answer should have been. NULL means the run recorded no output, which is itself a fact and is reported by the `output_present` inferred check.';

COMMENT ON COLUMN public.evidence_candidates.proposed_expected_output IS
  'THE PROPOSAL. Seeded from production_output at derivation and never changed afterwards. It is what the review UI shows as "is this right?". It is NOT an expected output until a human says so: a production output is what one model produced on one day, and if the workload is being optimized because its output is mediocre, treating that output as the expected answer scores every candidate on its ability to reproduce the mediocrity.';

COMMENT ON COLUMN public.evidence_candidates.expected_output IS
  'THE HUMAN''S ANSWER, and the ONLY value ever copied into `golden_inputs.expected_output`. NULL until a human approves (it becomes the proposal, verbatim) or edits (it becomes what they typed). The CHECK constraint chk_evidence_candidates_approved_has_expected refuses any approved row where this is NULL, so "no golden input without a human answer" is enforced by the database and not by the application.';

COMMENT ON COLUMN public.evidence_candidates.fingerprint IS
  'Deterministic dedup key: sha256 over a canonical JSON of the NORMALISED input. Normalisation is deliberately dull and is documented in optimization/curation.py: Unicode NFC, whitespace runs collapsed to a single space, trimmed — matching the `_normalize_output` convention this codebase already uses for eval comparison — plus case folding on `input_text` ONLY. Case is NOT folded on variable values, because a named variable routinely holds an identifier, slug, enum, path or model name where case is load-bearing, while free-text input that differs only in case is not a distinct benchmark case. Nothing semantic, no embeddings, no similarity threshold: at the 39-distinct-inputs scale this table is built for, an exact normalised match is both sufficient and defensible, and a similarity threshold would be a number nobody could justify. UNIQUE per (org_id, workload_id) so the 742-runs-to-39-candidates collapse is a schema guarantee rather than an application convention.';

COMMENT ON COLUMN public.evidence_candidates.fingerprint_version IS
  'Which normalisation produced this row''s fingerprint. Bump in optimization/curation.py when the rule changes, so that old rows are not silently reinterpreted under a new rule and a mixed-version table is detectable rather than mysterious. Changing the rule does NOT rewrite existing rows: history is immutable here as everywhere else in this schema.';

COMMENT ON COLUMN public.evidence_candidates.occurrences IS
  'How many scanned production runs collapsed into this candidate. This is the number that makes the dedup honest and visible: 742 runs producing 39 candidates shows up here as occurrence counts, not as 703 vanished rows. It is also a crude popularity signal — the input seen 200 times is the one worth getting right — and it is MEASURED over the derivation window only, so it is a count of what was scanned and never a claim about all time.';

COMMENT ON COLUMN public.evidence_candidates.source_run_id IS
  'The EXEMPLAR run: the chronologically earliest run that produced this fingerprint, tie-broken by run id so derivation is deterministic regardless of scan order. ON DELETE SET NULL, not CASCADE: a reviewed candidate is a human decision and must outlive the run that suggested it, or a log-retention job would silently delete approved evidence.';

COMMENT ON COLUMN public.evidence_candidates.capture IS
  'Redaction/capture provenance, in the same shape `workflow_runs.variables_capture` uses and consumed by the same `evidence_redaction.replay_gate()`. Carried over from the source run AND re-derived at this write boundary, because a run persisted before the redaction boundary shipped holds unredacted content and history is never rewritten. Without this column, redaction here would be silent and a curated case that no longer reproduces production would look like a mystery instead of a recorded modification.';

COMMENT ON COLUMN public.evidence_candidates.replay_eligible IS
  'The `evidence_redaction.replay_gate()` verdict, stored rather than re-derived at read time so the queue and the approval path cannot disagree. FALSE does NOT mean hidden and does NOT mean discarded — the candidate stays visible in the queue with its reason codes, and a human may still approve it, which is recorded in review_acknowledged_redaction. NULL means the gate was not run for this row.';

COMMENT ON COLUMN public.evidence_candidates.replay_reason_codes IS
  'Structured codes from replay_gate(), never prose: `redacted_input_requires_review` (redaction modified the persisted content, so the removed value may be exactly what drove the behaviour the case claims to reproduce) and `capture_provenance_unavailable` (capture was attempted and produced nothing usable — UNKNOWN IS NOT CLEAN, and `redacted: false` on a row nobody could inspect is an absence of evidence rather than evidence of absence). The frontend owns all wording.';

COMMENT ON COLUMN public.evidence_candidates.review_acknowledged_redaction IS
  'TRUE when a human approved a candidate that the replay gate had ruled ineligible. The approval path REFUSES such a candidate without it, so a redacted or provenance-unknown case can never become replay evidence silently. Recorded on the row, beside reviewed_by and reviewed_at, so that "why does this case not replay faithfully?" has an answer in the row itself six months later.';

COMMENT ON COLUMN public.evidence_candidates.bucket IS
  'Diversity bucket, one per candidate, assigned by strict priority so the buckets partition the queue and their counts sum to the total. CHEAP SIGNALS ONLY — no embeddings, no clustering: failure (the source run errored) > unusual_output_shape (output shape differs from the workload''s modal shape) > unusual_variable_shape (variable key-set differs from the modal key-set) > long_input (normalised input length at or above the workload''s p90 and at least twice its median) > outlier_cost_latency (cost or latency above the workload''s p95) > random_coverage (a deterministic evenly-spaced chronological sample of the remainder, so the queue is not all from one afternoon) > common. Codes only; the frontend owns the words. Buckets are a property of the POPULATION at derivation time and are recomputed for unreviewed rows on each pass; a reviewed row keeps the bucket it was reviewed under. A bucket never affects whether a candidate may be approved.';

COMMENT ON COLUMN public.evidence_candidates.bucket_signals IS
  'The measured facts that produced the bucket: {input_length, output_shape, variable_keys, had_error, cost_usd, latency_ms, percentiles}. Stored so a bucket assignment can be audited without re-scanning the runs, and so a later change to the bucketing rule can be evaluated against what was actually observed. Measured values or absent — never a placeholder.';

COMMENT ON COLUMN public.evidence_candidates.checks IS
  'STRUCTURAL assertions OptiML can make about the production output without judgement, as [{"code": ..., "passed": bool}]: no_execution_error, output_present, output_valid_json, output_json_fields_present, output_json_field_types_stable. A check that does not APPLY is OMITTED rather than recorded as passed or failed — an absent check is "we could not measure this", which is not the same as a failure. These exist to make review faster by showing what is already known; they are never semantic, never an LLM judgement, and NOTHING here ever approves a candidate automatically.';

COMMENT ON COLUMN public.evidence_candidates.golden_input_id IS
  'The `golden_inputs` row this candidate became, or NULL. THE ONLY BRIDGE between captured traffic and replay evidence, written by exactly one code path — the approval handler — and guarded by two independent mechanisms: chk_evidence_candidates_golden_requires_approval refuses it on an unapproved row, and uq_evidence_candidates_golden_input refuses a second one, which is what makes a double-clicked Approve idempotent rather than duplicating the case.';


-- ── 5. VERIFICATION (read-only; run these by hand, they change nothing) ─────
--
-- 1. The table exists with RLS on and exactly one policy:
--
--    SELECT c.relname, c.relrowsecurity AS rls_enabled,
--           (SELECT count(*) FROM pg_policies p
--             WHERE p.schemaname='public' AND p.tablename='evidence_candidates') AS policies,
--           has_table_privilege('anon', c.oid, 'SELECT')          AS anon_select,
--           has_table_privilege('authenticated', c.oid, 'SELECT') AS authd_select
--      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
--     WHERE n.nspname='public' AND c.relname='evidence_candidates';
--    -- expect rls_enabled = true, policies = 1, anon_select = false
--
-- 2. Both CHECK constraints and both UNIQUE indexes are in force:
--
--    SELECT conname, pg_get_constraintdef(oid)
--      FROM pg_constraint WHERE conrelid='public.evidence_candidates'::regclass
--       AND contype='c';
--    SELECT indexname, indexdef FROM pg_indexes
--     WHERE schemaname='public' AND tablename='evidence_candidates';
--
-- 3. THE INVARIANT THIS WHOLE MIGRATION EXISTS TO PROTECT. Every golden input
--    that came from curation traces back to an approved candidate, and no
--    unapproved candidate has a golden input. Both must return zero rows:
--
--    SELECT count(*) FROM public.evidence_candidates
--     WHERE golden_input_id IS NOT NULL
--       AND state NOT IN ('human_approved','human_edited');
--
--    SELECT count(*) FROM public.evidence_candidates
--     WHERE state IN ('human_approved','human_edited')
--       AND expected_output IS NULL;
--
-- 4. The dedup ratio, once derivation has run — the number this design is
--    built around. Expect roughly 19:1 on the largest real workload:
--
--    SELECT workload_id, count(*) AS candidates, sum(occurrences) AS runs_collapsed,
--           round(sum(occurrences)::numeric / greatest(count(*),1), 1) AS ratio
--      FROM public.evidence_candidates GROUP BY 1 ORDER BY 3 DESC;
--
-- 5. The review funnel, and how much of it cannot replay faithfully:
--
--    SELECT state, bucket, count(*) FROM public.evidence_candidates
--     GROUP BY 1,2 ORDER BY 1,3 DESC;
--
--    SELECT replay_eligible, replay_reason_codes, count(*)
--      FROM public.evidence_candidates GROUP BY 1,2;
--
--
-- ── NOT DONE HERE, AND DELIBERATELY SO ─────────────────────────────────────
--
-- NO COLUMN IS ADDED TO `golden_inputs`, and no existing row in it is touched.
-- A golden input created by approval is written through the existing shape
-- (`source`, `source_run_id`), so every one of the ten readers keeps working
-- unchanged and this migration cannot break a benchmark that ran yesterday.
-- The `source` value distinguishes curated rows for anyone who wants to count
-- them; a dedicated column would be a schema change to `golden_inputs` in
-- service of a report, which is not a trade this migration is willing to make.
--
-- NO EMBEDDING COLUMN, NO VECTOR INDEX, NO pgvector DEPENDENCY. See the
-- measured numbers at the top: the largest real workload has 39 distinct
-- inputs. Semantic clustering at that scale is an expensive way to reproduce
-- what an exact normalised match already gets right, and it would introduce a
-- similarity threshold that no one could defend to a customer asking why two
-- of their cases were merged.
--
-- NO BACKFILL. Derivation is an explicit, idempotent read of `workflow_runs`
-- performed by the API, not by this file. This migration reads no rows at all.
--
-- NO TRIGGER enforces the state graph. Two CHECKs cover the invariants that
-- must never be violated by any writer; the rest of the lifecycle lives in
-- optimization/curation.py, next to the audit record it has to write. Half a
-- rule in SQL and half in Python is worse than all of it in one place.
-- ============================================================================
