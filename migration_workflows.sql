-- Workflows table for Studio persistence
-- Run this in Supabase SQL editor if workflows table does not exist.

CREATE TABLE IF NOT EXISTS workflows (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  name TEXT NOT NULL DEFAULT 'Untitled workflow',
  graph_json JSONB NOT NULL DEFAULT '{"nodes":[],"edges":[]}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_workflows_org_id ON workflows(org_id);

COMMENT ON TABLE workflows IS 'Studio workflow definitions; graph_json stores nodes and edges.';
