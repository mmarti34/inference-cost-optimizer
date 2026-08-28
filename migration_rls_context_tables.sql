-- ============================================================================
-- Migration: close direct PostgREST exposure of the context (knowledge-base)
--            tables, and remove the grants that made it reachable.
--
-- RUN AFTER: any migration that creates context_assets / context_chunks /
--            context_asset_snapshots. Order is otherwise irrelevant.
--
-- IDEMPOTENT. ENABLE ROW LEVEL SECURITY is a no-op when already enabled, and
-- REVOKE is a no-op when the privilege is already absent. Re-running changes
-- nothing. NO DATA IS READ, WRITTEN OR DELETED by this file.
--
--
-- ── WHAT WAS WRONG ──────────────────────────────────────────────────────────
--
-- public.context_assets, public.context_chunks and public.context_asset_
-- snapshots sat in the `public` schema — which PostgREST exposes over HTTP —
-- with ROW LEVEL SECURITY DISABLED and SELECT granted to the `anon` and
-- `authenticated` roles.
--
-- The anon role's credential is the project's publishable key. It ships in the
-- frontend bundle and is public by design. So any person on the internet could
-- read EVERY tenant's knowledge-base content — the documents customers upload
-- for their AI to use — straight from the REST API, without authenticating,
-- without touching the backend, and without exercising a single application
-- bug.
--
-- Supabase's own linter reported all three at ERROR level, facing EXTERNAL
-- (lint 0013, rls_disabled_in_public). They were the only ERROR-level findings
-- on the project.
--
-- Measured at the time of the fix: 6 rows in context_assets across 2 distinct
-- orgs, 56 rows in context_chunks, 0 in context_asset_snapshots.
--
-- This matters more than any handler defect found in the same audit. Every
-- backend authorization check can be perfect and the tenant data still walks
-- out the side door.
--
--
-- ── WHY RLS ALONE, WITH NO POLICIES ─────────────────────────────────────────
--
-- Enabling RLS with zero policies denies anon and authenticated outright. That
-- is the intended posture, not an oversight, and it is this codebase's existing
-- deliberate pattern — see cursor_tokens, documented as "RLS enabled with NO
-- policy on purpose".
--
-- DO NOT add a broad policy to make access convenient. Nothing in the product
-- reads these tables from the browser. A policy would exist only to re-open the
-- door this file closes.
--
-- The backend is unaffected: supabase_client authenticates with the
-- SERVICE-ROLE key, which bypasses RLS entirely. Every read of these tables
-- goes through it (context_asset_management.py, context_runtime.py).
--
--
-- ── WHY THE GRANTS GO TOO ───────────────────────────────────────────────────
--
-- RLS is the primary boundary; the grants are defence in depth. With SELECT
-- revoked, a future `ALTER TABLE ... DISABLE ROW LEVEL SECURITY` — or a policy
-- added carelessly — does not silently restore public readability. Two
-- independent things would have to go wrong instead of one.
--
-- Revoking ALL rather than SELECT: these roles have no reason to write either,
-- and INSERT/UPDATE/DELETE on a table with no policies would fail anyway. This
-- makes the intent explicit rather than incidental.
--
--
-- ── VERIFIED BEFORE APPLYING ────────────────────────────────────────────────
--
--   * the frontend reads only user_profiles and organizations directly — a
--     grep for `.from('...')` across lib/, app/ and components/ in
--     inference-app-frontend-3 returns those two tables and nothing else;
--   * the project has ZERO Edge Functions (list_edge_functions returned []);
--   * all backend access is via the service-role client.
--
-- Reversible in one statement per table if a consumer is ever found that needs
-- anon access: ALTER TABLE ... DISABLE ROW LEVEL SECURITY, plus a GRANT. Prefer
-- routing that consumer through the backend instead.
-- ============================================================================


-- ── 1. The primary boundary ─────────────────────────────────────────────────

ALTER TABLE public.context_assets          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.context_chunks          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.context_asset_snapshots ENABLE ROW LEVEL SECURITY;


-- ── 2. Defence in depth: remove the grants that made them reachable ─────────

REVOKE ALL ON public.context_assets          FROM anon, authenticated;
REVOKE ALL ON public.context_chunks          FROM anon, authenticated;
REVOKE ALL ON public.context_asset_snapshots FROM anon, authenticated;


-- ── 3. Verification (read-only; run by hand) ────────────────────────────────
--
--   -- Expect rls_enabled = true and policies = 0 for all three:
--   SELECT c.relname, c.relrowsecurity AS rls_enabled,
--          (SELECT count(*) FROM pg_policies p
--            WHERE p.schemaname = 'public' AND p.tablename = c.relname) AS policies,
--          has_table_privilege('anon', c.oid, 'SELECT')          AS anon_select,
--          has_table_privilege('authenticated', c.oid, 'SELECT') AS authd_select
--     FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
--    WHERE n.nspname = 'public'
--      AND c.relname IN ('context_assets','context_chunks','context_asset_snapshots');
--
--   -- And the rows are still there, read through service-role:
--   SELECT count(*) FROM public.context_assets;   -- 6 at time of writing
--   SELECT count(*) FROM public.context_chunks;   -- 56
--
--   -- Supabase linter should no longer report lint 0013 for these tables.
-- ============================================================================
