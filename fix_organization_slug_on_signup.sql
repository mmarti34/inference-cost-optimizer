-- Fix "null value in column slug of relation organizations" on signup
-- A trigger creates an organization row on signup but doesn't set slug. This ensures slug is set.
-- Run in Supabase Dashboard → SQL Editor → New query → Run

-- 1. Trigger function: set slug from name + id when slug is null (keeps slug unique)
CREATE OR REPLACE FUNCTION public.set_organization_slug_if_null()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  base TEXT;
  short_id TEXT;
BEGIN
  IF NEW.slug IS NULL OR trim(NEW.slug) = '' THEN
    base := lower(regexp_replace(substring(trim(COALESCE(NEW.name, 'org')) from 1 for 50), '[^a-zA-Z0-9]+', '-', 'g'));
    IF base = '' OR base IS NULL THEN base := 'org'; END IF;
    base := trim(both '-' from base);
    short_id := substring(replace(COALESCE(NEW.id::text, gen_random_uuid()::text), '-', '') from 1 for 8);
    NEW.slug := base || '-' || short_id;
  END IF;
  RETURN NEW;
END;
$$;

-- 2. Drop if already exists (e.g. from a previous run), then create
DROP TRIGGER IF EXISTS set_organization_slug_trigger ON public.organizations;
CREATE TRIGGER set_organization_slug_trigger
  BEFORE INSERT ON public.organizations
  FOR EACH ROW EXECUTE PROCEDURE public.set_organization_slug_if_null();
