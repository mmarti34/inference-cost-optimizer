-- Migration to add project tracking to usage_logs and create optimizer_recommendations table

-- 1. Add project_id column to usage_logs table (if not already exists)
ALTER TABLE usage_logs 
ADD COLUMN IF NOT EXISTS project_id UUID REFERENCES projects(id);

-- 2. Add org_id column to usage_logs table (if not already exists)
ALTER TABLE usage_logs 
ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);

-- 3. Create optimizer_recommendations table
CREATE TABLE IF NOT EXISTS optimizer_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prompt_id UUID REFERENCES prompt_templates(id) NOT NULL,
    project_id UUID REFERENCES projects(id) NOT NULL,
    org_id UUID REFERENCES organizations(id) NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    recommended_provider TEXT NOT NULL,
    recommended_model TEXT NOT NULL,
    estimated_cost_usd DECIMAL(10,6) NOT NULL,
    estimated_input_tokens INTEGER NOT NULL,
    estimated_output_tokens INTEGER NOT NULL,
    full_prompt_text TEXT NOT NULL,
    budget_used_usd DECIMAL(10,6) NOT NULL,
    monthly_budget_usd DECIMAL(10,6) NOT NULL,
    reasoning TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_usage_logs_project_id ON usage_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_usage_logs_org_id ON usage_logs(org_id);
CREATE INDEX IF NOT EXISTS idx_optimizer_recommendations_prompt_id ON optimizer_recommendations(prompt_id);
CREATE INDEX IF NOT EXISTS idx_optimizer_recommendations_project_id ON optimizer_recommendations(project_id);
CREATE INDEX IF NOT EXISTS idx_optimizer_recommendations_created_at ON optimizer_recommendations(created_at);

-- 5. Add comments to explain the new columns
COMMENT ON COLUMN usage_logs.project_id IS 'Reference to the project this usage log belongs to';
COMMENT ON COLUMN usage_logs.org_id IS 'Reference to the organization this usage log belongs to';
COMMENT ON TABLE optimizer_recommendations IS 'Stores AI optimization recommendations for prompts';

-- 6. Update projects table to ensure monthly_budget column exists (if not already added)
ALTER TABLE projects 
ADD COLUMN IF NOT EXISTS monthly_budget DECIMAL(10,2) DEFAULT 0.00;

COMMENT ON COLUMN projects.monthly_budget IS 'Monthly budget limit for the project in USD'; 