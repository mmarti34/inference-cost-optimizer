-- Cursor tokens: long-lived tokens for Cursor plugin (and other API clients).
-- User creates a token in Settings; plugin uses it as Bearer for parse-import, workflows, deploy.
-- One token is scoped to one org.

CREATE TABLE IF NOT EXISTS cursor_tokens (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  org_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  token_hash text NOT NULL UNIQUE,
  name text NOT NULL DEFAULT 'Cursor',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cursor_tokens_token_hash ON cursor_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_cursor_tokens_user_id ON cursor_tokens(user_id);

COMMENT ON TABLE cursor_tokens IS 'Long-lived API tokens for Cursor plugin; scope to one org.';
