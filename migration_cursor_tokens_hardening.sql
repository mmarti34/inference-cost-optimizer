-- ============================================================================
-- Migration: lock down cursor_tokens + add revocation
--
-- Context
-- -------
-- migration_cursor_tokens.sql created `cursor_tokens` with NO row level
-- security, and it runs after migration_enable_rls.sql (the authoritative RLS
-- migration), so nothing ever enabled it.
--
-- cursor_tokens rows are bearer credentials: auth_dependency._verify_cursor_token
-- looks a token up by hash and returns an AuthenticatedUser bound to the row's
-- org_id. With RLS off, anyone holding the browser-facing anon key could INSERT
-- a row and mint a valid `optml_` bearer token scoped to ANY organization —
-- and SELECT the table to enumerate existing ones.
--
-- The backend talks to Postgres with the service_role key, which BYPASSES RLS
-- entirely. So enabling RLS with no policy at all is exactly right: the app
-- keeps working, and anon/authenticated get nothing.
--
-- Also adds the `status` column the app assumed existed: cursor_tokens.py told
-- users to "Revoke one in Settings first" when there was no way to revoke
-- anything. auth_dependency now refuses any token whose status is not 'active'.
--
-- Idempotent. Safe to re-run.
-- ============================================================================


-- ── 1. Revocation state ─────────────────────────────────────────────────────

ALTER TABLE public.cursor_tokens
    ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active';

ALTER TABLE public.cursor_tokens
    ADD COLUMN IF NOT EXISTS revoked_at timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'public.cursor_tokens'::regclass
          AND conname = 'cursor_tokens_status_check'
    ) THEN
        ALTER TABLE public.cursor_tokens
            ADD CONSTRAINT cursor_tokens_status_check
            CHECK (status IN ('active', 'revoked'));
    END IF;
END $$;

-- Verification looks tokens up by hash and then filters on status.
CREATE INDEX IF NOT EXISTS idx_cursor_tokens_status
    ON public.cursor_tokens(status);


-- ── 2. Row level security ───────────────────────────────────────────────────

ALTER TABLE public.cursor_tokens ENABLE ROW LEVEL SECURITY;

-- Belt and braces: RLS does not apply to a table owner unless FORCE is set,
-- and these tables are owned by the migration role.
ALTER TABLE public.cursor_tokens FORCE ROW LEVEL SECURITY;

-- Drop any permissive policy that may have been added by hand. There must be
-- NO policy on this table: service_role bypasses RLS, and no other role has
-- any business reading or writing bearer credentials.
DO $$
DECLARE
    pol record;
BEGIN
    FOR pol IN
        SELECT policyname FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'cursor_tokens'
    LOOP
        EXECUTE format('DROP POLICY %I ON public.cursor_tokens', pol.policyname);
        RAISE NOTICE 'Dropped policy %% on cursor_tokens', pol.policyname;
    END LOOP;
END $$;

-- Revoke any direct grants the anon/authenticated roles may hold. RLS with no
-- policy already denies everything, but this removes the grant as well.
REVOKE ALL ON public.cursor_tokens FROM anon;
REVOKE ALL ON public.cursor_tokens FROM authenticated;

COMMENT ON TABLE public.cursor_tokens IS
    'Long-lived API bearer tokens for the Cursor plugin; scoped to one org. RLS enabled with NO policy on purpose: only the backend service_role (which bypasses RLS) may touch this table. status: active | revoked.';
