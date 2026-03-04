-- Human-in-the-loop: pause workflow execution, wait for human review, then resume.
-- When a human_review node is reached, execution pauses and a review record is created.
-- The reviewer can approve (resume) or reject (stop) the workflow.

CREATE TABLE IF NOT EXISTS pending_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    workflow_run_id UUID,
    node_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    input_data TEXT,                -- The data/output that needs review (from previous step)
    reviewer_instructions TEXT,     -- What the reviewer should check
    reviewer_response TEXT,         -- Reviewer's comment/modification
    reviewer_email TEXT,            -- Who reviewed it
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- RLS: service role only
ALTER TABLE pending_reviews ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on pending_reviews"
    ON pending_reviews
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_pending_reviews_org_id ON pending_reviews(org_id);
CREATE INDEX IF NOT EXISTS idx_pending_reviews_status ON pending_reviews(status) WHERE status = 'pending';
