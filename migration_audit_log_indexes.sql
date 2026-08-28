-- ============================================================================
-- Migration: indexes for public.audit_log
--
-- NOT RUN. Written for review only, per instruction. There is no DDL here that
-- creates, alters or drops a table: `public.audit_log` ALREADY EXISTS with the
-- schema below, verified against the live project's metadata:
--
--   id           uuid        PK, default gen_random_uuid()
--   org_id       uuid        NOT NULL, FK -> organizations(id)
--   actor_id     uuid        NULL
--   action       text        NOT NULL
--   resource_type text       NULL
--   resource_id  text        NULL
--   metadata     jsonb       NULL, default '{}'
--   ip_address   inet        NULL
--   created_at   timestamptz NOT NULL, default now()
--
-- RLS is ENABLED on the table. The backend writes with the service-role key,
-- which bypasses RLS, so these indexes change nothing about who may read a row.
--
-- WHY INDEXES ARE NEEDED AT ALL
-- ─────────────────────────────
-- The table holds zero rows today and every query against it is currently a
-- seq scan that costs nothing. That stops being true on the first month of
-- production writes, and the moment it matters is the worst possible moment to
-- notice: an incident, where the question is asked under time pressure against
-- a table that has been accumulating rows since. Every index below is shaped by
-- a question that was actually asked and could not be answered.
--
-- CONCURRENTLY, so this cannot take a write lock on a table the application is
-- appending to. Note that CREATE INDEX CONCURRENTLY cannot run inside a
-- transaction block — run these as separate statements, not as one script in a
-- BEGIN/COMMIT.
-- ============================================================================


-- Q: "show me everything that happened in this tenant between these dates"
--    The tenant timeline. Every incident review starts here, and org_id is the
--    first filter in all of them because audit rows are tenant data.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_org_created_at
  ON public.audit_log (org_id, created_at DESC);


-- Q: "who revoked this production key?"  /  "did anyone TRY to?"
--    The question that could not be answered. Answering it means filtering by
--    org and action together — the timeline index alone would scan a month of
--    that tenant's rows to find four.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_org_action_created_at
  ON public.audit_log (org_id, action, created_at DESC);


-- Q: "what is the full history of THIS key / secret / membership row?"
--    Deliberately not prefixed by org_id: the first thing known during an
--    incident is often a resource id and nothing else, and the owning org is
--    what the query is trying to establish. Partial, because resource_id is
--    nullable and a NULL resource is never the subject of this question.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_resource
  ON public.audit_log (resource_type, resource_id, created_at DESC)
  WHERE resource_id IS NOT NULL;


-- Q: "what has this actor done, across every org they touch?"
--    A compromised or departing account is investigated by actor, not by org.
--    Partial: actor_id is NULL for server-key surfaces, where the key is the
--    actor and there is no human to pivot on.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_actor_created_at
  ON public.audit_log (actor_id, created_at DESC)
  WHERE actor_id IS NOT NULL;


-- ============================================================================
-- CONSTRAINTS DELIBERATELY NOT ADDED
-- ============================================================================
--
-- 1. No CHECK constraint pinning `action` to the vocabulary in `audit.py`.
--    It is tempting — the vocabulary IS closed — but it would make the audit
--    trail fail closed against the application: a deploy that adds a new action
--    constant before the constraint is migrated would raise on every write of
--    that action. `audit._write()` swallows the exception, so the visible
--    result would be silently missing audit rows, which is the exact failure
--    this whole change exists to remove. The vocabulary is enforced in
--    `audit.ACTIONS` at the only place that writes the table, where a rejected
--    action is logged rather than lost.
--
-- 2. No NOT NULL on `resource_type` / `resource_id`. Some legitimate events
--    have no single subject row — a refused provider-credential delete is
--    keyed by (org, provider) and has no row id to name.
--
-- 3. No retention/partitioning policy here. It is a real decision (how long
--    must "who revoked this key six months ago" stay answerable?) but it is a
--    policy question for an owner, not a side effect of adding indexes.
--
-- 4. No UPDATE/DELETE revocation on the table. Making audit rows immutable at
--    the database level is worth doing, but the backend connects as
--    service_role and would need a separate append-only role to benefit —
--    a change to the service-role pattern, which is out of scope here.
