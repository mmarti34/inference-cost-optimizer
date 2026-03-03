-- ============================================================================
-- Migration: Enable RLS on all public tables missing it
-- Run this in Supabase SQL Editor
--
-- Context: The backend uses the service_role key which BYPASSES RLS,
-- so all backend queries continue to work unchanged. This migration
-- locks down DIRECT access via the anon key (exposed in the frontend)
-- so that nobody can query Supabase tables directly from the browser.
--
-- Pattern: org members can SELECT rows for their org. All writes go
-- through the backend (service_role), so we only need SELECT policies
-- for the anon key. This is defense-in-depth.
-- ============================================================================

-- Helper: reusable function to check if a user is an active org member
CREATE OR REPLACE FUNCTION public.is_org_member(check_org_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.organization_members
    WHERE org_id = check_org_id
      AND user_id = auth.uid()
      AND status = 'active'
  );
$$;


-- ============================================================================
-- 1. workflow_runs (already has policies but RLS not enabled)
-- ============================================================================
ALTER TABLE public.workflow_runs ENABLE ROW LEVEL SECURITY;
-- Existing policies "Members can create runs" and "Members can view org runs"
-- should already work. No new policies needed.


-- ============================================================================
-- 2. custom_endpoints
-- ============================================================================
ALTER TABLE public.custom_endpoints ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view custom endpoints"
  ON public.custom_endpoints FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 3. model_registry
-- ============================================================================
ALTER TABLE public.model_registry ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view model registry"
  ON public.model_registry FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 4. routing_latency_facts
-- ============================================================================
ALTER TABLE public.routing_latency_facts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view latency facts"
  ON public.routing_latency_facts FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 5. invite_tokens  (sensitive: contains token column)
-- ============================================================================
ALTER TABLE public.invite_tokens ENABLE ROW LEVEL SECURITY;

-- Org admins can see invites for their org
CREATE POLICY "Org members can view invite tokens"
  ON public.invite_tokens FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 6. routing_policies
-- ============================================================================
ALTER TABLE public.routing_policies ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view routing policies"
  ON public.routing_policies FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 7. rollback_rules
-- ============================================================================
ALTER TABLE public.rollback_rules ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view rollback rules"
  ON public.rollback_rules FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 8. custom_metrics
-- ============================================================================
ALTER TABLE public.custom_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view custom metrics"
  ON public.custom_metrics FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 9. experiments
-- ============================================================================
ALTER TABLE public.experiments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view experiments"
  ON public.experiments FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 10. api_request_log
-- ============================================================================
ALTER TABLE public.api_request_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view request logs"
  ON public.api_request_log FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 11. auto_grade_results
-- ============================================================================
ALTER TABLE public.auto_grade_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view auto grade results"
  ON public.auto_grade_results FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 12. auto_graded_metrics
-- ============================================================================
ALTER TABLE public.auto_graded_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view auto graded metrics"
  ON public.auto_graded_metrics FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 13. conversations
-- ============================================================================
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view conversations"
  ON public.conversations FOR SELECT
  USING (public.is_org_member(org_id));


-- ============================================================================
-- 14. conversation_turns (no org_id — scoped via conversations FK)
-- ============================================================================
ALTER TABLE public.conversation_turns ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view conversation turns"
  ON public.conversation_turns FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.conversations c
      WHERE c.id = conversation_turns.conversation_id
        AND public.is_org_member(c.org_id)
    )
  );


-- ============================================================================
-- Verify: list all tables and their RLS status
-- ============================================================================
SELECT
  schemaname,
  tablename,
  rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
