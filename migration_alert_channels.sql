-- ============================================================================
-- Migration: Alert channels + alert history tables
-- Run this in Supabase SQL Editor
--
-- Context: Enables production alert notifications (Slack, webhook, email)
-- when rollback rules trigger or other production issues occur.
-- Backend uses service_role key (bypasses RLS). RLS protects direct
-- anon-key access in the browser.
-- ============================================================================

-- 1. alert_channels — stores Slack/webhook/email configs per org
CREATE TABLE IF NOT EXISTS public.alert_channels (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  channel_type TEXT NOT NULL CHECK (channel_type IN ('slack_webhook', 'webhook', 'email')),
  name TEXT NOT NULL DEFAULT '',
  config JSONB NOT NULL DEFAULT '{}',
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT timezone('utc', now()),
  updated_at TIMESTAMPTZ DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_alert_channels_org_id ON public.alert_channels(org_id);

-- 2. alert_history — log of all dispatched alerts
CREATE TABLE IF NOT EXISTS public.alert_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_id UUID NOT NULL,
  channel_id UUID REFERENCES public.alert_channels(id) ON DELETE SET NULL,
  alert_type TEXT NOT NULL,
  endpoint_slug TEXT,
  payload JSONB NOT NULL DEFAULT '{}',
  delivery_status TEXT NOT NULL DEFAULT 'pending' CHECK (delivery_status IN ('pending', 'sent', 'failed')),
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_alert_history_org_id ON public.alert_history(org_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_created_at ON public.alert_history(created_at DESC);

-- 3. RLS (follows existing is_org_member pattern)
ALTER TABLE public.alert_channels ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view alert channels"
  ON public.alert_channels FOR SELECT
  USING (public.is_org_member(org_id));

ALTER TABLE public.alert_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view alert history"
  ON public.alert_history FOR SELECT
  USING (public.is_org_member(org_id));

-- 4. Verify
SELECT schemaname, tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('alert_channels', 'alert_history')
ORDER BY tablename;
