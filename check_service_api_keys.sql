-- Check service_api_keys table structure and data
-- This will help us understand why the API key authentication is failing

-- Check table structure
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'service_api_keys'
ORDER BY ordinal_position;

-- Check if there are any service API keys
SELECT 
    id,
    user_id,
    org_id,
    LEFT(api_key, 20) || '...' as api_key_preview,
    created_at
FROM service_api_keys 
LIMIT 5;

-- Check for the specific user's API keys
SELECT 
    id,
    user_id,
    org_id,
    LEFT(api_key, 20) || '...' as api_key_preview,
    created_at
FROM service_api_keys 
WHERE user_id = 'b82d8f65-cecc-43c0-a6b5-5332317643c6'; 