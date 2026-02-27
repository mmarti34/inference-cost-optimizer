-- Fix organization-based access for API keys, projects, and prompt templates
-- This ensures all resources are properly associated with organizations

-- 1. Update service_api_keys to ensure all have org_id and remove user_id dependency
UPDATE service_api_keys 
SET user_id = NULL 
WHERE user_id IS NOT NULL;

-- 2. Ensure all prompt_templates have org_id
UPDATE prompt_templates 
SET org_id = (
    SELECT org_id 
    FROM organization_members 
    WHERE user_id = prompt_templates.user_id 
    AND status = 'active' 
    LIMIT 1
)
WHERE org_id IS NULL 
AND user_id IS NOT NULL;

-- 3. Ensure all projects have org_id
UPDATE projects 
SET org_id = (
    SELECT org_id 
    FROM organization_members 
    WHERE user_id = projects.user_id 
    AND status = 'active' 
    LIMIT 1
)
WHERE org_id IS NULL 
AND user_id IS NOT NULL;

-- 4. Add foreign key constraints to ensure data integrity
-- (These will fail if there are orphaned records, which is good for data cleanup)

-- Add foreign key for prompt_templates.org_id -> organizations.id
ALTER TABLE prompt_templates 
ADD CONSTRAINT fk_prompt_templates_org_id 
FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Add foreign key for projects.org_id -> organizations.id  
ALTER TABLE projects 
ADD CONSTRAINT fk_projects_org_id 
FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- Add foreign key for service_api_keys.org_id -> organizations.id
ALTER TABLE service_api_keys 
ADD CONSTRAINT fk_service_api_keys_org_id 
FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE;

-- 5. Verify the changes
SELECT 'prompt_templates' as table_name, COUNT(*) as total_records, 
       COUNT(org_id) as records_with_org_id,
       COUNT(user_id) as records_with_user_id
FROM prompt_templates
UNION ALL
SELECT 'projects' as table_name, COUNT(*) as total_records,
       COUNT(org_id) as records_with_org_id, 
       COUNT(user_id) as records_with_user_id
FROM projects
UNION ALL
SELECT 'service_api_keys' as table_name, COUNT(*) as total_records,
       COUNT(org_id) as records_with_org_id,
       COUNT(user_id) as records_with_user_id
FROM service_api_keys; 