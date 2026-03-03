-- Migration: Invite tokens table + fix organization_members status constraint
-- Run in Supabase SQL Editor before deploying backend

-- 1. Fix organization_members status CHECK to include 'invited'
ALTER TABLE organization_members DROP CONSTRAINT IF EXISTS organization_members_status_check;
ALTER TABLE organization_members ADD CONSTRAINT organization_members_status_check
  CHECK (status IN ('active', 'pending', 'inactive', 'invited'));

-- 2. Create invite_tokens table
CREATE TABLE IF NOT EXISTS invite_tokens (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  token UUID NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  invited_email TEXT NOT NULL,
  invited_by UUID NOT NULL,
  member_id UUID REFERENCES organization_members(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'expired', 'revoked')),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '7 days'),
  accepted_at TIMESTAMPTZ,
  accepted_by UUID,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for token lookup (primary query path)
CREATE INDEX IF NOT EXISTS idx_invite_tokens_token ON invite_tokens(token);
-- Index for finding pending invites by email (auto-accept on login)
CREATE INDEX IF NOT EXISTS idx_invite_tokens_email_status ON invite_tokens(invited_email, status);
-- One pending invite per email per org
CREATE UNIQUE INDEX IF NOT EXISTS idx_invite_tokens_unique_pending
  ON invite_tokens(org_id, invited_email) WHERE status = 'pending';
