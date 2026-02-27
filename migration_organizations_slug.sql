-- Add slug to organizations for org-scoped public routing: /api/public/{org_slug}/{endpoint_slug}
-- Slug is URL-safe, unique, derived from name (aligned with frontend orgSlug(org)).

-- Add column (nullable first for backfill)
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS slug TEXT;

-- Backfill: slugify name (lowercase, spaces to hyphen, strip non-alphanumeric-hyphen)
-- PostgreSQL: trim, lower, regexp_replace for spaces, then non-[a-z0-9-]
UPDATE organizations
SET slug = regexp_replace(
  regexp_replace(lower(trim(coalesce(name, ''))), '\s+', '-', 'g'),
  '[^a-z0-9-]', '', 'g'
)
WHERE slug IS NULL AND name IS NOT NULL;

-- Empty slug fallback: use first 8 chars of id
UPDATE organizations
SET slug = 'org-' || replace(substring(id::text from 1 for 8), '-', '')
WHERE slug IS NULL OR slug = '';

-- Deduplicate: append short id suffix for collisions
WITH numbered AS (
  SELECT id, slug,
    row_number() OVER (PARTITION BY slug ORDER BY created_at NULLS LAST, id) AS rn
  FROM organizations
)
UPDATE organizations o
SET slug = CASE WHEN n.rn > 1 THEN n.slug || '-' || replace(substring(o.id::text from 1 for 6), '-', '') ELSE n.slug END
FROM numbered n WHERE n.id = o.id;

-- Enforce uniqueness and not null
CREATE UNIQUE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);
ALTER TABLE organizations ALTER COLUMN slug SET NOT NULL;

COMMENT ON COLUMN organizations.slug IS 'URL-safe org identifier for public API path; must match frontend orgSlug(org).';
