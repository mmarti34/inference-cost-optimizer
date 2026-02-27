-- Fix user_id constraints to allow NULL values for organization-based resources
-- This allows resources to be created without requiring a specific user_id

-- Make user_id nullable in projects table
ALTER TABLE projects 
ALTER COLUMN user_id DROP NOT NULL;

-- Make user_id nullable in prompt_templates table  
ALTER TABLE prompt_templates 
ALTER COLUMN user_id DROP NOT NULL;

-- Make user_id nullable in api_keys table
ALTER TABLE api_keys 
ALTER COLUMN user_id DROP NOT NULL;

-- Add comments to explain the change
COMMENT ON COLUMN projects.user_id IS 'Optional user who created this project (can be NULL for org-level projects)';
COMMENT ON COLUMN prompt_templates.user_id IS 'Optional user who created this prompt template (can be NULL for org-level templates)';
COMMENT ON COLUMN api_keys.user_id IS 'Optional user who created this API key (can be NULL for org-level keys)';

-- Verify the changes
SELECT 
    table_name, 
    column_name, 
    is_nullable,
    data_type
FROM information_schema.columns 
WHERE table_name IN ('projects', 'prompt_templates', 'api_keys') 
AND column_name = 'user_id'
ORDER BY table_name;
