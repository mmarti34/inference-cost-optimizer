-- reload_postgrest_schema.sql
-- Run in Supabase SQL Editor to refresh PostgREST schema cache after schema/table changes.
-- Use after adding exposed schemas or fixing pgrst.db_schemas.

NOTIFY pgrst, 'reload schema';

-- If you added a custom schema and still get PGRST106, also run:
-- ALTER ROLE authenticator SET pgrst.db_schemas = 'public, your_schema_name';
-- or to use dashboard "Exposed schemas" setting:
-- ALTER ROLE authenticator RESET pgrst.db_schemas;
-- Then run NOTIFY again.
