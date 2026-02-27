-- Fix "Database error saving new user" (500) on signup
-- Run this entire file in Supabase Dashboard → SQL Editor → New query → Run
-- If signup still returns 500, run diagnose_signup_500.sql and check:
--   (2) any NOT NULL column without default must be added to the INSERT below or given a default.

-- 1. Ensure user_profiles has onboarding columns with defaults (so trigger insert cannot fail on them)
ALTER TABLE user_profiles
  ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS onboarding_completed_at TIMESTAMPTZ DEFAULT NULL;

-- 2. Drop any existing trigger on auth.users that might be failing (common names)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP TRIGGER IF EXISTS on_auth_user_created_trigger ON auth.users;
DROP TRIGGER IF EXISTS handle_new_user_trigger ON auth.users;

-- 3. Create a safe trigger function: only inserts user_id (user_profiles may not have email column)
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  BEGIN
    INSERT INTO public.user_profiles (user_id)
    VALUES (new.id);
  EXCEPTION
    WHEN unique_violation THEN NULL;  -- row already exists (e.g. another trigger), ignore
  END;
  RETURN new;
END;
$$;

-- 4. Attach trigger so new signups get a user_profiles row
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();
