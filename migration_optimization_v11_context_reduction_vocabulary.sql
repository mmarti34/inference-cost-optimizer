-- ============================================================================
-- Migration: OptiML optimization layer v11 — context efficiency joins the
--            documented dimension vocabulary.
--
-- RUN AFTER: migration_optimization_v10_evidence_maturity_semantics.sql
--
-- *** NOT APPLIED, AND NOT RUN BY THE AGENT THAT WROTE IT. ***
-- The agent executed no SQL of any kind. Apply this by hand.
--
-- IDEMPOTENT BY CONSTRUCTION. Every statement is a COMMENT ON. There is no
-- ALTER, no ADD COLUMN, no RENAME, no DROP, no CREATE, no INSERT, no UPDATE and
-- no DELETE. Re-running it is a no-op, and running it changes NO DATA
-- WHATSOEVER.
--
-- NO VIEW IS TOUCHED, so the `DROP VIEW IF EXISTS` dance that earlier
-- migrations needed (CREATE OR REPLACE VIEW cannot retype a column) does not
-- apply here. Nothing is retyped because nothing is typed.
--
-- ORDERING DOES NOT MATTER. The accompanying code neither reads nor writes any
-- new column and adds no new value to any CHECK-constrained column:
-- `optimization_recommendations.dimensions` and
-- `execution_strategies.dimensions` are plain TEXT[] with no constraint, and
-- both `prompt` and `context_length` were already in the documented vocabulary.
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
-- ── WHY THIS EXISTS AT ALL ──────────────────────────────────────────────────
--
-- The column comment written in migration_optimization.sql says:
--
--     "NOTE: only model, provider and prompt are currently APPLICABLE to
--      graph_json by optimization/strategy.py — the rest are vocabulary
--      reserved for future generators."
--
-- That was true when it was written and is now wrong in a way that matters.
-- `context_length` has been applicable on the runtime surface for some time
-- (optimization/strategy.py `_apply_context` writes
-- contextConfig.packaging.maxChars and the per-source budgets, which
-- context_runtime.py enforces), and as of this change a generator actually
-- EMITS it: optimization/candidates.py ContextReductionGenerator.
--
-- A stale comment on a vocabulary column is not cosmetic. It is the thing a
-- reader consults to answer "can this system really change that?", and left as
-- it stands it says no about a dimension that has now produced candidates.
--
--
-- ── WHAT CONTEXT EFFICIENCY IS, AND WHAT IT IS NOT ──────────────────────────
--
-- Internally `context_reduction`; customer-facing wording is owned by the
-- frontend and is "context efficiency". It is deliberately NOT called prompt
-- optimization: that name invites rewriting prompts for style and voice, which
-- is a different and far less defensible thing.
--
-- The claim is narrow, and the narrowness is the value:
--
--     Same model. Same provider. Same sampling parameters. Same tools. Same
--     output contract. ONLY the prompt/context representation changes.
--
-- It introduces NO new dimension, because it needs no new runtime mechanism —
-- it is expressed with `prompt` and `context_length`, both of which the runtime
-- could already apply. What was missing was never the mechanism; it was the
-- evidence. A candidate now carries a MEASURED per-component token breakdown
-- (optimization/context_accounting.py) and then faces the identical machinery
-- every other candidate faces: the same recorded cases, the same deterministic
-- eval checks, the same staged replay with futility stopping, the same real
-- measured provider cost, and the same paired non-inferiority test at the same
-- policy margin.
--
-- The governing rule, recorded here because it is the reason this is a product
-- claim and not a demo: an LLM may PROPOSE a shorter prompt; an LLM may never
-- be the EVIDENCE that the shorter prompt works. Replay is the evidence. A
-- proposal that cannot be measured produces no finding at all.
-- ============================================================================


-- ── The dimension vocabulary, corrected ─────────────────────────────────────

COMMENT ON COLUMN public.optimization_recommendations.dimensions IS
  'Optimization dimensions changed by this candidate. Vocabulary: model, provider, prompt, context_length, reasoning_effort, temperature, max_tokens, top_p, retrieval, reranking, caching, fallback_chain, llm_call_count, workflow_structure, tool_selection, deterministic_code. APPLICABILITY IS A PROPERTY OF THE EXECUTION SURFACE, not a global fact — see optimization/strategy.py SURFACE_APPLICABLE_DIMENSIONS. On the runtime (workflow graph) surface: model, provider, prompt, context_length, fallback_chain. On the direct_inference surface: model, provider, prompt, temperature, max_tokens, top_p. temperature is applicable on direct inference and REFUSED on runtime, where workflow_runtime drops it before the provider call; context_length is the mirror image, applicable on runtime via contextConfig and meaningless on direct inference where the caller sends the whole message list. A dimension outside its surface list is refused at apply time so it can never reach a benchmark and produce a measurement of nothing. prompt + context_length together carry CONTEXT EFFICIENCY (internally context_reduction): same model, same provider, same sampling parameters, same tools, same output contract, less context.';

COMMENT ON COLUMN public.execution_strategies.dimensions IS
  'Optimization dimensions this strategy differs on. Runtime v1 applies: model, provider, prompt, context_length, fallback_chain. Direct inference applies: model, provider, prompt, temperature, max_tokens, top_p. See optimization/strategy.py SURFACE_APPLICABLE_DIMENSIONS — applicability is per surface, and a dimension this surface cannot vary is refused rather than silently ignored.';


-- ── Candidate bundles: the example in the comment is now real ───────────────

COMMENT ON COLUMN public.optimization_recommendations.bundle_id IS
  'Groups candidates that are meant to be evaluated together (e.g. smaller model + shorter context as one combined change) so the user sees one bundled opportunity plus its parts, instead of a flood of overlapping individual ones. Members of a bundle share bundle_id; the combined candidate is the one whose dimensions is the union. NOTE: the context half of that example is no longer hypothetical — optimization/candidates.py ContextReductionGenerator emits prompt/context_length candidates, and each is measured on its own before any bundle could claim their combination.';


-- ── The consideration funnel gains reason codes, not columns ────────────────
--
-- `benchmark_conclusions.consideration` is JSONB and enumerates nothing in the
-- schema, so the ten new context-reduction exclusion codes need no DDL. They
-- are documented here because the column comment is where a reader looks for
-- the vocabulary, and because an undocumented code entering the API contract is
-- exactly what optimization/domain.py REASON_CODES exists to prevent.

COMMENT ON COLUMN public.benchmark_conclusions.consideration IS
  'Candidate discovery funnel. Three ordered, self-describing lists plus the un-narrated per-candidate records. `stages` counts STAGES REACHED, not stop points: considered, executable, entered_replay, stopped_early, completed_verification, replay_verified_improvement. The entries flagged cumulative=true form a SPINE that is monotonically non-increasing BY CONSTRUCTION; stopped_early is flagged cumulative=false because it is a disjoint exit count (entered_replay minus completed_verification) and asserting monotonicity across it would assert something untrue. `outcomes` carries the disjoint per-disposition counts — where each candidate STOPPED — over the vocabulary considered, incompatible, policy_blocked, provider_not_configured, economically_dominated, eliminated_by_historical_evidence, duplicate, generator_error, benchmarked, not_measured, failed_policy, promising, quality_safe. `exclusions` reports why candidates never reached dispatch at REASON CODE grain rather than at disposition grain, deliberately: several genuinely different pre-dispatch findings exit at the same disposition, and collapsing them into one number erases the evidence that the search was thorough. Every entry carries emitted; emitted=false means nothing in the system can populate that row yet, so its zero must not be read as "we checked and found none". Structured codes and measured facts ONLY — no customer-facing prose; the frontend owns all wording. The context-efficiency static checks are part of the exclusion set and every one of them RAN against real data before any provider request existed: context_reduction_quality_signal_insufficient (the workload has no deterministic/structural/format eval check, so the tail degradation a shortened prompt causes would be invisible — an LLM judge would rate the degraded output fine), context_reduction_case_count_insufficient, context_placeholder_dropped, context_tool_reference_dropped, context_output_contract_dropped, context_budget_below_requirement, context_reduction_changed_other_dimension, context_reduction_unmeasured (the reduction could not be MEASURED, so the variant was refused rather than estimated from character counts), context_reduction_immaterial, context_reduction_variant_not_selected (the benchmark budget cap, reported rather than applied silently). A candidate carrying one of these was NOT benchmarked and NOT failed: it was never dispatched.';
