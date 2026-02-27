-- Run this in Supabase SQL Editor to diagnose "Database error saving new user" (500).
-- Copy the results; they show what runs on signup and what the user_profiles table looks like.

-- 1. Triggers on auth.users (one of these runs after each new signup)
SELECT tgname AS trigger_name, proname AS function_name
FROM pg_trigger t
JOIN pg_proc p ON t.tgfoid = p.oid
WHERE t.tgrelid = 'auth.users'::regclass
  AND NOT t.tgisinternal
ORDER BY tgname;

-- 2. user_profiles columns (trigger must only set columns that exist and are not null without default)
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'user_profiles'
ORDER BY ordinal_position;

-- 3. RLS on user_profiles (if RLS is enabled, the trigger’s role must be allowed to INSERT)
SELECT relname, relrowsecurity AS rls_enabled
FROM pg_class
WHERE relname = 'user_profiles' AND relnamespace = 'public'::regnamespace;
