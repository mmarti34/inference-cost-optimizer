-- Fix service API keys by updating user_id for the correct user
-- This will fix the authentication issue in the /v1/prompt endpoint

-- First, let's see which API key belongs to your organization
SELECT 
    id,
    user_id,
    org_id,
    LEFT(api_key, 20) || '...' as api_key_preview,
    created_at
FROM service_api_keys 
WHERE org_id = '05ef4e73-de21-49fe-bf7f-b8303cab31b6';

-- Update the API key for your organization to have the correct user_id
UPDATE service_api_keys 
SET user_id = 'b82d8f65-cecc-43c0-a6b5-5332317643c6'
WHERE org_id = '05ef4e73-de21-49fe-bf7f-b8303cab31b6'
AND user_id IS NULL;

-- Also update the other API key that has a different user_id
UPDATE service_api_keys 
SET user_id = 'b82d8f65-cecc-43c0-a6b5-5332317643c6'
WHERE user_id = '1575796f-303e-4f7b-8018-d463b3177f47'
AND org_id = '5afa11af-a5b2-4ad7-aeee-f78901dda5be';

-- Verify the changes
SELECT 
    id,
    user_id,
    org_id,
    LEFT(api_key, 20) || '...' as api_key_preview,
    created_at
FROM service_api_keys 
WHERE user_id = 'b82d8f65-cecc-43c0-a6b5-5332317643c6'; 