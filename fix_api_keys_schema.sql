-- Migration to ensure api_keys table uses 'api_key' column (not 'encrypted_key')
-- This aligns the database schema with the code

-- Check if encrypted_key column exists and api_key doesn't
DO $$
BEGIN
    -- If encrypted_key exists but api_key doesn't, rename it
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'api_keys' AND column_name = 'encrypted_key'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'api_keys' AND column_name = 'api_key'
    ) THEN
        ALTER TABLE api_keys RENAME COLUMN encrypted_key TO api_key;
        RAISE NOTICE 'Renamed encrypted_key to api_key in api_keys table';
    END IF;
END $$;

-- Verify the column exists
SELECT 
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'api_keys' 
AND column_name IN ('api_key', 'encrypted_key')
ORDER BY column_name;

