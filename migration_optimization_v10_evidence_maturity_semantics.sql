-- ============================================================================
-- Migration: OptiML optimization layer v10 — `confidence` is EVIDENCE MATURITY,
--            it is INTERNAL, and it is not a probability.
--
-- RUN AFTER: migration_optimization_v9_async_jobs.sql
--
-- *** NOT APPLIED, AND NOT RUN BY THE AGENT THAT WROTE IT. ***
-- The agent executed no SQL of any kind. Apply this by hand.
--
-- IDEMPOTENT BY CONSTRUCTION. Every statement is a COMMENT ON. There is no
-- ALTER, no ADD COLUMN, no RENAME, no DROP, no INSERT, no UPDATE and no DELETE.
-- Re-running it is a no-op, and running it changes NO DATA WHATSOEVER.
--
-- ORDERING DOES NOT MATTER. The accompanying code neither reads nor writes any
-- new column, so it is correct before this migration and correct after it.
-- Deploy in either order.
--
-- HISTORY IS PRESERVED, EXACTLY. Not one row is read or written:
--   policy v1  benchmark      88813bfb-5581-45a0-abf6-884732a0b19b
--              recommendation 41d79006-bc14-4180-9d3d-97e71d6a2809
--   policy v2  benchmark      4d5ca24d-93b6-4b8a-a4e7-1c5fcba9fec7
--              recommendation 25eabede-0261-4b42-9882-c01c850d54f1
--              (the first verified win)
--   second win benchmark      fb723f70-b53d-4e5e-be2c-91d67b688eb0
--              (gpt-4o -> gpt-4.1, -20.0% cost, quality 0.9643 -> 0.9714)
--
-- NOTHING HERE WEAKENS AUTH, RLS OR ORG ISOLATION. No policy, view, function,
-- grant or role is created, altered or dropped. A COMMENT is catalog metadata
-- and is not reachable through the data API.
--
--
-- ── WHAT WAS WRONG ──────────────────────────────────────────────────────────
--
-- A production recommendation reported:
--
--     confidence 0.188   band "low"
--
-- immediately beside a quality-safety verdict that was ESTABLISHED: 140 paired
-- evaluations, non-inferiority within 2 percentage points at 95% one-sided,
-- discordant_b = 1, discordant_c = 2.
--
-- Read together, "confidence: low / 0.188" says "about an 18% chance this
-- verdict holds". It says nothing of the kind. `confidence` is a MATURITY
-- INDEX: a multiplicative blend of five unrelated quantities —
--
--     sample size (log-scaled, saturating at 1000)
--   x counterfactual strength of evidence_source
--   x quality-signal provenance rank
--   x (1 - coefficient of variation of observed cost)
--   x historical consistency
--
-- — with a bounded bonus when production confirmed it. No probabilistic
-- statement can be recovered from that product. It answers "how mature is the
-- evidence behind this claim", never "how likely is this claim to be true".
-- 0.188 is low because 140 replay cases is a small, offline sample, not
-- because the non-inferiority result is shaky. The non-inferiority result is
-- the thing that carries a real confidence level, and it says 95%.
--
-- Two numbers on two unrelated axes were printed next to each other, and
-- nothing in the payload said they were unrelated.
--
--
-- ── WHAT CHANGED, AND WHAT DELIBERATELY DID NOT ─────────────────────────────
--
-- CHANGED (in code, this release):
--   * The index is INTERNAL. It is gone from every customer-facing payload:
--       - optimization.service.recommendation_row_to_response no longer returns
--         `evidence.confidence` or `evidence.confidence_band`;
--       - optimization.domain.conclusion_payload no longer returns
--         `confidence` or `confidence_band`, so neither the benchmark response
--         nor the conclusion response carries them;
--       - the /summary coverage query no longer selects the column at all.
--     Each of those payloads instead names the absence with the documented
--     reason code `evidence_maturity_internal_only`, so a client that used to
--     read the field can tell this was a decision and not a regression.
--   * It is NAMED for what it is, everywhere in code:
--       domain.compute_confidence -> domain.compute_evidence_maturity
--       domain.confidence_band    -> domain.evidence_maturity_band
--       domain.CONFIDENCE_BANDS   -> domain.EVIDENCE_MATURITY_BANDS
--     and the local variables and keyword arguments that carry it are
--     `evidence_maturity`.
--   * It keeps doing the internal work it is actually good at: ranking and
--     prioritising candidates, and deciding `more_data_changes_conclusion`.
--
-- NOT CHANGED, ON PURPOSE:
--   * THE COLUMNS ARE NOT RENAMED. `benchmark_conclusions.confidence`,
--     `benchmark_conclusions.confidence_band`,
--     `optimization_recommendations.confidence`,
--     `optimization_benchmarks.confidence` and
--     `allocation_decisions.confidence` keep their physical names.
--
--     Why not rename, when the name is the defect?
--
--       1. A rename is destructive to readers. Every stored row, every cached
--          client, every ad-hoc query and the two preserved historical
--          recommendations resolve through that name. `ALTER TABLE ... RENAME
--          COLUMN` cannot be made idempotent without a catalog probe, and a
--          rename plus a compatibility view would add a second name for one
--          value — the ambiguity this migration exists to remove.
--       2. An additive `evidence_maturity_score` column would be worse. It
--          would create two sources of truth for one number, require a backfill
--          UPDATE that touches the preserved historical rows, and require code
--          that can only run AFTER this migration is applied — which cannot be
--          relied upon, because this file is handed over unapplied. Code that
--          wrote to a column that did not exist yet would fail every
--          conclusion insert.
--       3. The storage was never wrong. The column holds exactly the quantity
--          intended. The defect was the NAME the product read and the PLACE it
--          was printed, and both of those are fixed above, in code, at zero
--          risk to stored data.
--
--     So the fix here is documentation of record: the column comments below are
--     what a future reader — or a `\d+` — sees, and they now say precisely what
--     the value is, what it is not, and that it is internal.
--
--   * THE FORMULA IS UNCHANGED. Byte for byte, the same inputs produce the same
--     number as before this release.
--
--   * THE BAND VOCABULARY IS UNCHANGED: 'low' | 'medium' | 'high'. Renaming the
--     values would either break the existing CHECK constraint or force an
--     UPDATE over the preserved rows. 'low' has always meant "the evidence is
--     EARLY"; it has never meant "this verdict is unlikely".
--
--
-- ── HOW HISTORICAL ROWS ARE HANDLED ─────────────────────────────────────────
--
-- Explicitly, and without reinterpretation.
--
-- Because the formula did not change and the stored values are not touched, a
-- historical row's `confidence` still means exactly what it meant on the day it
-- was written. This is a RENAMING OF THE CONCEPT IN THE PRODUCT, not a change
-- of measurement, so there is no old-semantics / new-semantics split to
-- reconcile and no row needs a version marker.
--
-- What DID change for historical rows is where the number is allowed to appear:
-- reading one of these rows through the API no longer surfaces it. The row is
-- unchanged; the presentation is. That is the whole difference, and it is
-- stated here rather than left to be inferred.
--
-- Conclusions are immutable by design (see v7): a conclusion row is never
-- updated, and re-deciding the same evidence under a new policy inserts a new
-- row. This migration honours that. It rewrites no verdict.
-- ============================================================================


-- ── 1. benchmark_conclusions ────────────────────────────────────────────────

COMMENT ON COLUMN public.benchmark_conclusions.confidence IS
  'EVIDENCE MATURITY, 0..1, from optimization.domain.compute_evidence_maturity. '
  'NOT a probability, NOT a p-value, NOT a confidence level, and NOT the chance '
  'this conclusion is correct. It is a multiplicative blend of sample size, '
  'counterfactual strength of evidence_source, quality-signal provenance rank, '
  'observed cost variance and historical consistency, so that 14 replay '
  'examples cannot look like 180,000 production outcomes. No probabilistic '
  'statement can be recovered from it. INTERNAL: used for candidate ranking and '
  'for more_data_changes_conclusion; deliberately absent from every '
  'customer-facing payload since v10, which reports the safety verdict '
  '(quality_safety), the evidence stage (evidence_source) and the production '
  'status separately and never merges them. Column NOT renamed in v10: the '
  'value and the formula are unchanged, so historical rows are not '
  'reinterpreted and nothing is orphaned. NULL when it cannot be computed — '
  'never a placeholder.';

COMMENT ON COLUMN public.benchmark_conclusions.confidence_band IS
  'Coarse band over the evidence-MATURITY score in `confidence`: low | medium | '
  'high. "low" means the evidence is EARLY (small sample, weak counterfactual, '
  'or a weak quality signal). It has never meant "this verdict is unlikely to '
  'hold" — for that, read quality_safety.established with its confidence_level, '
  'n_pairs and discordant counts. Values are unchanged in v10 because the CHECK '
  'constraint and the preserved historical rows both depend on them. INTERNAL, '
  'like the score. NULL when the score could not be computed.';


-- ── 2. optimization_recommendations ─────────────────────────────────────────

COMMENT ON COLUMN public.optimization_recommendations.confidence IS
  'EVIDENCE MATURITY, 0..1, from optimization.domain.compute_evidence_maturity: '
  'sample size x counterfactual strength of evidence_source x quality-signal '
  'provenance rank x observed variance x historical consistency, plus a bounded '
  'bonus when production confirmed it. NOT a probability and NOT a confidence '
  'level. A recommendation reading 0.188 next to an ESTABLISHED non-inferiority '
  'verdict (140 pairs, 2pp margin, 95% one-sided, discordant_b=1, '
  'discordant_c=2) does not mean an 18% chance the verdict is wrong; it means '
  'the evidence class is early — replay, not production. INTERNAL since v10 and '
  'NOT returned by optimization.service.recommendation_row_to_response, which '
  'exposes instead: quality_safety (safety verdict), evidence_source + '
  'evidence_strength (evidence stage) and status + rollout (production status). '
  'Those three axes are separate and must never be collapsed into one score. '
  'Column NOT renamed in v10 — see the header of '
  'migration_optimization_v10_evidence_maturity_semantics.sql for why an '
  'additive column and a rename were both rejected. NULL when it cannot be '
  'computed — never a placeholder.';


-- ── 3. optimization_benchmarks ──────────────────────────────────────────────

COMMENT ON COLUMN public.optimization_benchmarks.confidence IS
  'EVIDENCE MATURITY for this run''s conclusion, 0..1, from '
  'optimization.domain.compute_evidence_maturity. NOT a probability and NOT a '
  'confidence level — see benchmark_conclusions.confidence. Mirrors the current '
  'conclusion row for convenience; benchmark_conclusions is the immutable, '
  'policy-versioned record. INTERNAL since v10: not returned by '
  'optimization.benchmark.benchmark_row_to_response and not selected by the '
  '/summary coverage query. NULL when it cannot be computed — never a '
  'placeholder.';


-- ── 4. allocation_decisions ───────────────────────────────────────────────
--
-- The decision log is internal by construction (it records what was chosen and
-- what was rejected, for review). The column is commented for the same reason
-- as the others: so nobody reading the catalog mistakes it for a probability.

COMMENT ON COLUMN public.allocation_decisions.confidence IS
  'EVIDENCE MATURITY of the evidence behind this decision, 0..1, from '
  'optimization.domain.compute_evidence_maturity, passed as the '
  '`evidence_maturity` argument to optimization.allocation.record_decision. NOT '
  'a probability and NOT a confidence level. Internal audit trail only. NULL '
  'when it could not be computed.';


-- ── 5. Verification (read-only; run by hand if you want it) ─────────────────
--
-- Nothing below is executed by applying this file. It is here so the change can
-- be checked without writing anything.
--
--   -- The four comments are in place:
--   SELECT c.relname AS table_name,
--          a.attname AS column_name,
--          col_description(c.oid, a.attnum) AS comment
--     FROM pg_class c
--     JOIN pg_attribute a ON a.attrelid = c.oid
--    WHERE c.relname IN ('benchmark_conclusions',
--                        'optimization_recommendations',
--                        'optimization_benchmarks',
--                        'allocation_decisions')
--      AND a.attname IN ('confidence', 'confidence_band')
--      AND a.attnum > 0
--    ORDER BY 1, 2;
--
--   -- The preserved rows are untouched (values identical to pre-v10):
--   SELECT id, conclusion, confidence, confidence_band
--     FROM public.benchmark_conclusions
--    WHERE benchmark_id IN ('88813bfb-5581-45a0-abf6-884732a0b19b',
--                           '4d5ca24d-93b6-4b8a-a4e7-1c5fcba9fec7',
--                           'fb723f70-b53d-4e5e-be2c-91d67b688eb0');
--
--   SELECT id, status, confidence
--     FROM public.optimization_recommendations
--    WHERE id IN ('41d79006-bc14-4180-9d3d-97e71d6a2809',
--                 '25eabede-0261-4b42-9882-c01c850d54f1');
--
-- ============================================================================
