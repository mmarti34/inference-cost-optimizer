-- ============================================================================
-- Migration: remove RLS policies that were granted to PUBLIC by mistake
--
-- Context
-- -------
-- Several migrations created policies like:
--
--     CREATE POLICY "Service role full access on org_secrets"
--         ON org_secrets FOR ALL
--         USING (true) WITH CHECK (true);
--
-- with the stated intent "service role only". But a CREATE POLICY with no
-- `TO <role>` clause defaults to role **PUBLIC**, which includes `anon` and
-- `authenticated` — the very roles the browser-facing Supabase anon key
-- authenticates as. `USING (true)` then matches every row.
--
-- So these policies did the exact opposite of their comment: they made the
-- tables fully readable and writable from the browser. That exposed
--   * org_secrets.encrypted_value   (every org's tool/webhook credentials)
--   * webhook_triggers.secret       (HMAC secrets — forge any webhook call)
--   * pending_reviews.*             (HITL payloads, cross-tenant)
--   * monthly_usage.*               (rewrite your own quota)
--   * projects / prompt_templates / api_keys / service_api_keys
--     (from disable_rls_for_backend.sql — including api_keys rows)
--
-- The fix is to DROP them and add nothing back. The backend connects with the
-- service_role key, which BYPASSES row level security entirely; it never
-- needed a policy. RLS enabled with no policy = deny all for anon/authenticated
-- and unchanged behaviour for the backend.
--
-- Affected source files:
--   migration_org_secrets.sql:19-23
--   migration_monthly_usage.sql:42-46
--   migration_webhook_triggers.sql:24-28
--   migration_pending_reviews.sql:24-28
--   disable_rls_for_backend.sql:26-47
--
-- Idempotent. Safe to re-run. Nothing is read, rewritten, or deleted.
-- ============================================================================


-- ── 1. Drop the mislabelled "service role" policies by name ─────────────────

DROP POLICY IF EXISTS "Service role full access on org_secrets"      ON public.org_secrets;
DROP POLICY IF EXISTS "Service role full access on monthly_usage"    ON public.monthly_usage;
DROP POLICY IF EXISTS "Service role full access on webhook_triggers" ON public.webhook_triggers;
DROP POLICY IF EXISTS "Service role full access on pending_reviews"  ON public.pending_reviews;

DROP POLICY IF EXISTS "Service role bypass - projects"          ON public.projects;
DROP POLICY IF EXISTS "Service role bypass - prompt_templates"  ON public.prompt_templates;
DROP POLICY IF EXISTS "Service role bypass - api_keys"          ON public.api_keys;
DROP POLICY IF EXISTS "Service role bypass - service_api_keys"  ON public.service_api_keys;


-- ── 2. Sweep: drop ANY remaining permissive-to-PUBLIC policy on these tables ─
-- Catches copies created by hand in the Supabase dashboard under a different
-- name. A policy is removed when it applies to PUBLIC / anon / authenticated
-- AND its USING clause is the unconditional `true`.

DO $$
DECLARE
    pol record;
    tbl text;
    targets text[] := ARRAY[
        'org_secrets',
        'monthly_usage',
        'webhook_triggers',
        'pending_reviews',
        'projects',
        'prompt_templates',
        'api_keys',
        'service_api_keys'
    ];
BEGIN
    FOREACH tbl IN ARRAY targets LOOP
        IF to_regclass('public.' || tbl) IS NULL THEN
            CONTINUE;
        END IF;

        FOR pol IN
            SELECT policyname, roles, qual, with_check
            FROM pg_policies
            WHERE schemaname = 'public'
              AND tablename = tbl
        LOOP
            IF (pol.roles::text[] && ARRAY['public', 'anon', 'authenticated'])
               AND (
                    coalesce(btrim(pol.qual), 'true') = 'true'
                 OR coalesce(btrim(pol.with_check), '') = 'true'
               )
            THEN
                EXECUTE format('DROP POLICY %I ON public.%I', pol.policyname, tbl);
                RAISE NOTICE 'Dropped permissive-to-PUBLIC policy %% on %%', pol.policyname, tbl;
            END IF;
        END LOOP;
    END LOOP;
END $$;


-- ── 3. Make sure RLS is actually ON for the secret-bearing tables ───────────
-- With no policy, this is a full deny for anon/authenticated. service_role is
-- unaffected — it bypasses RLS.

DO $$
DECLARE
    tbl text;
    targets text[] := ARRAY[
        'org_secrets',
        'monthly_usage',
        'webhook_triggers',
        'pending_reviews'
    ];
BEGIN
    FOREACH tbl IN ARRAY targets LOOP
        IF to_regclass('public.' || tbl) IS NOT NULL THEN
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', tbl);
            EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', tbl);
            EXECUTE format('REVOKE ALL ON public.%I FROM anon', tbl);
            EXECUTE format('REVOKE ALL ON public.%I FROM authenticated', tbl);
        END IF;
    END LOOP;
END $$;

-- Credential and tenant tables. QUICK_RLS_FIX.sql and disable_rls_for_backend.sql
-- both suggested turning RLS OFF on these; if either was ever run against
-- production, the browser-facing anon key has unrestricted access to api_keys
-- and service_api_keys right now. Turn RLS back on.
--
-- Deliberately NO `REVOKE` here: create_missing_tables.sql defines legitimate
-- org-scoped SELECT policies on these tables, and revoking the table grant
-- would override them (no privilege = no access, policy or not). RLS + the
-- existing scoped policies is the intended design, and service_role bypasses
-- both.
DO $$
DECLARE
    tbl text;
    targets text[] := ARRAY['projects', 'prompt_templates', 'api_keys', 'service_api_keys'];
BEGIN
    FOREACH tbl IN ARRAY targets LOOP
        IF to_regclass('public.' || tbl) IS NOT NULL THEN
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', tbl);
        END IF;
    END LOOP;
END $$;


-- ── 4. Verify ───────────────────────────────────────────────────────────────
-- Expect: no rows for the secret-bearing tables. Any row printed here is a
-- policy still handing data to the browser-facing anon key.

SELECT schemaname, tablename, policyname, roles, cmd, qual
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename IN (
      'org_secrets', 'monthly_usage', 'webhook_triggers', 'pending_reviews',
      'projects', 'prompt_templates', 'api_keys', 'service_api_keys',
      'cursor_tokens'
  )
ORDER BY tablename, policyname;
