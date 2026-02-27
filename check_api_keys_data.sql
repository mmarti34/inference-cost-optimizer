-- Check what's actually in the api_keys table
-- Run this in Supabase SQL Editor to see the data structure

-- Check table structure
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'api_keys' 
AND table_schema = 'public'
ORDER BY ordinal_position;

-- Check actual data (first 5 rows, masking the api_key)
SELECT 
    id,
    org_id,
    user_id,
    provider,
    name,
    LEFT(api_key, 20) || '...' as api_key_preview,
    created_at,
    updated_at
FROM api_keys
ORDER BY created_at DESC
LIMIT 5;

-- Count by org_id
SELECT 
    org_id,
    COUNT(*) as key_count,
    COUNT(DISTINCT provider) as provider_count
FROM api_keys
GROUP BY org_id;

-- Check for missing required fields
SELECT 
    COUNT(*) as total_keys,
    COUNT(org_id) as has_org_id,
    COUNT(provider) as has_provider,
    COUNT(api_key) as has_api_key,
    COUNT(user_id) as has_user_id,
    COUNT(name) as has_name
FROM api_keys;
