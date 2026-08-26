-- ============================================================================
-- Migration: widen the stale plan / subscription_tier CHECK constraints
--
-- Context
-- -------
-- create_missing_tables.sql defines:
--     organizations.plan          CHECK (plan IN ('free','pro','enterprise'))
--     user_profiles.subscription_tier
--                                 CHECK (subscription_tier IN ('free','pro','enterprise'))
--
-- Those value sets are stale: they predate the 'pro' -> 'startup'/'team'
-- rename. PLAN_LIMITS in org_access_control.py, and PRICE_TO_TIER in
-- stripe_webhook.py, both use 'free' | 'startup' | 'team' | 'enterprise'.
--
-- Net effect while the old constraint is in place: the Stripe webhook's
--     UPDATE organizations SET plan = 'startup'
-- is rejected by Postgres, the failure was swallowed as "non-critical", and
-- plan_enforcement.get_org_plan_tier() then reads the unchanged row and
-- returns 'free'. A paying Startup/Team customer is enforced at free-tier
-- limits (2 projects / 5 workflows / 1 server key / 1,000 requests per month).
--
-- This migration is idempotent and non-destructive: it only widens the
-- accepted set. No row is read, rewritten, or deleted. Legacy values ('pro',
-- 'starter') stay accepted so the ADD CONSTRAINT cannot fail validation
-- against rows written before the rename.
--
-- Safe to run whether or not the deployed constraint matches the repo SQL —
-- the DO blocks drop whatever single-column CHECK currently guards the column
-- (if any) and then install the correct one.
--
-- Verify afterwards with:
--     SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint
--     WHERE conrelid = 'organizations'::regclass AND contype = 'c';
--     SELECT plan, count(*) FROM organizations GROUP BY plan;
-- ============================================================================


-- ── 1. organizations.plan ───────────────────────────────────────────────────

DO $$
DECLARE
    con_name text;
    plan_attnum smallint;
BEGIN
    SELECT a.attnum INTO plan_attnum
    FROM pg_attribute a
    WHERE a.attrelid = 'public.organizations'::regclass
      AND a.attname = 'plan'
      AND NOT a.attisdropped;

    IF plan_attnum IS NULL THEN
        RAISE NOTICE 'organizations.plan does not exist — skipping';
        RETURN;
    END IF;

    -- Drop every single-column CHECK constraint on organizations.plan,
    -- whatever it happens to be named in this database.
    FOR con_name IN
        SELECT c.conname
        FROM pg_constraint c
        WHERE c.conrelid = 'public.organizations'::regclass
          AND c.contype = 'c'
          AND c.conkey = ARRAY[plan_attnum]
    LOOP
        EXECUTE format('ALTER TABLE public.organizations DROP CONSTRAINT %I', con_name);
        RAISE NOTICE 'Dropped stale constraint %% on organizations.plan', con_name;
    END LOOP;

    ALTER TABLE public.organizations
        ADD CONSTRAINT organizations_plan_check
        CHECK (plan IN ('free', 'startup', 'team', 'enterprise', 'pro', 'starter'));
END $$;

COMMENT ON CONSTRAINT organizations_plan_check ON public.organizations IS
    'Live tiers: free | startup | team | enterprise. ''pro'' and ''starter'' are legacy values kept accepted so existing rows validate; they are not sold and map to free limits in PLAN_LIMITS.';


-- ── 2. user_profiles.subscription_tier ──────────────────────────────────────
-- Same stale set, same rename. Widened here too so the webhook's
-- user_profiles write can never be the thing that fails.

DO $$
DECLARE
    con_name text;
    tier_attnum smallint;
BEGIN
    SELECT a.attnum INTO tier_attnum
    FROM pg_attribute a
    WHERE a.attrelid = 'public.user_profiles'::regclass
      AND a.attname = 'subscription_tier'
      AND NOT a.attisdropped;

    IF tier_attnum IS NULL THEN
        RAISE NOTICE 'user_profiles.subscription_tier does not exist — skipping';
        RETURN;
    END IF;

    FOR con_name IN
        SELECT c.conname
        FROM pg_constraint c
        WHERE c.conrelid = 'public.user_profiles'::regclass
          AND c.contype = 'c'
          AND c.conkey = ARRAY[tier_attnum]
    LOOP
        EXECUTE format('ALTER TABLE public.user_profiles DROP CONSTRAINT %I', con_name);
        RAISE NOTICE 'Dropped stale constraint %% on user_profiles.subscription_tier', con_name;
    END LOOP;

    ALTER TABLE public.user_profiles
        ADD CONSTRAINT user_profiles_subscription_tier_check
        CHECK (subscription_tier IN ('free', 'startup', 'team', 'enterprise', 'pro', 'starter'));
END $$;

COMMENT ON CONSTRAINT user_profiles_subscription_tier_check ON public.user_profiles IS
    'Live tiers: free | startup | team | enterprise. ''pro'' and ''starter'' are legacy values kept accepted so existing rows validate.';
