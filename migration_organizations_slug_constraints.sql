-- organizations.slug: NOT NULL + UNIQUE with safe backfill (enterprise hardening).
-- Run after migration_organizations_slug.sql (or standalone if slug column exists).

-- Ensure slug column exists
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS slug TEXT;

-- 1. Backfill slug where null: slugify(name) if present else slugify(id)
UPDATE organizations
SET slug = regexp_replace(
  regexp_replace(lower(trim(coalesce(name, ''))), '\s+', '-', 'g'),
  '[^a-z0-9-]', '', 'g'
)
WHERE slug IS NULL AND name IS NOT NULL AND trim(coalesce(name, '')) != '';

UPDATE organizations
SET slug = 'org-' || replace(substring(id::text from 1 for 8), '-', '')
WHERE slug IS NULL OR trim(coalesce(slug, '')) = '';

-- 2. Ensure uniqueness: append -2, -3 etc deterministically for collisions
WITH numbered AS (
  SELECT id, slug,
    row_number() OVER (PARTITION BY slug ORDER BY created_at NULLS LAST, id) AS rn
  FROM organizations
)
UPDATE organizations o
SET slug = CASE
  WHEN n.rn = 1 THEN n.slug
  ELSE n.slug || '-' || n.rn::text
END
FROM numbered n
WHERE n.id = o.id;

-- 3. Enforce NOT NULL
ALTER TABLE organizations ALTER COLUMN slug SET NOT NULL;

-- 4. Unique index (drop if exists then create to avoid duplicate name)
DROP INDEX IF EXISTS idx_organizations_slug;
DROP INDEX IF EXISTS organizations_slug_unique;
CREATE UNIQUE INDEX organizations_slug_unique ON organizations(slug);

COMMENT ON COLUMN organizations.slug IS 'Stable URL-safe org identifier for /api/public/{org_slug}/...; do not change lightly.';
