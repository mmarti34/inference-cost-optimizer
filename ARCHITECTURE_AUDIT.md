# Architecture audit: PGRST104/PGRST106 and schema exposure

## Root cause of PGRST104 / PGRST106

These errors mean: **"The schema must be one of the exposed schemas: public, storage, graphql_public"**.

- **Cause**: PostgREST (Supabase Data API) only serves requests for schemas that are (1) listed in the project’s **Exposed schemas** (Dashboard → Project Settings → API) and (2) allowed by the `authenticator` role’s `pgrst.db_schemas` setting. If the client requests a schema that isn’t in that set, or if the schema isn’t exposed, PostgREST returns PGRST104/PGRST106.
- **Not primarily RLS**: RLS affects which rows are returned; PGRST104/PGRST106 are returned before row-level checks. Fix schema exposure/selection first, then RLS if needed.

## What was checked in this codebase

1. **Backend (Python)**  
   - A single Supabase client is created in `supabase_client.py` with `create_client(SUPABASE_URL, SUPABASE_KEY)`.  
   - No `ClientOptions`, `schema`, or `db: { schema: "..." }` is passed; the client uses the library default, which is the **`public`** schema.  
   - **Conclusion**: The backend does not explicitly select a non‑exposed schema. If PGRST104/PGRST106 still appears, the cause is on the Supabase project side (exposed schemas or `pgrst.db_schemas`), not in this code.

2. **Frontend (Next.js)**  
   - Supabase is used with `schema: 'public'` only in Realtime subscription filters (e.g. `postgres_changes` on `user_profiles`).  
   - No REST calls use a custom schema in this repo.  
   - **Conclusion**: Frontend does not request a non‑exposed schema for REST.

3. **Tables**  
   - All app tables are created in the **`public`** schema (e.g. in `create_missing_tables.sql` and migrations).  
   - **Conclusion**: Tables are in `public`. If the project’s exposed schemas include `public`, PostgREST should serve them.

## Fix applied

1. **Structured logging**  
   - All Supabase table access from the backend goes through a wrapper that logs:
     - On each successful `.execute()`: `table`, `key_type` (service_role vs SUPABASE_KEY).
     - On failure: `table`, `key_type`, error message, and extracted PGRST* code.  
   - Implemented in `supabase_logging.py`; used in `supabase_client.py` so every `supabase.table(...).execute()` is logged.  
   - **Use**: Reproduce the failing request and check logs for `supabase_error` with `table=... code=PGRST104` (or PGRST106) to see which table/request triggers the error.

2. **Schema verification and docs**  
   - **`scripts/verify_schema.sql`**: Run in Supabase SQL Editor to list `table_schema` and `table_name` for all app tables. Ensures everything is in `public` (or documents custom schemas).  
   - **`docs/SUPABASE_SCHEMA_EXPOSURE.md`**: Step-by-step instructions for:
     - Confirming table schemas.
     - Setting Exposed schemas in the Dashboard.
     - Granting and exposing custom schemas.
     - Fixing `authenticator`’s `pgrst.db_schemas` and reloading PostgREST (`NOTIFY pgrst, 'reload schema'`).

3. **Service role preference**  
   - `supabase_client.py` now prefers `SUPABASE_SERVICE_ROLE_KEY` over `SUPABASE_KEY` and logs a warning if the key looks like anon. Backend should use the service role for privileged access.

## Current schema exposure configuration (to verify in your project)

- **Exposed schemas**: In Supabase Dashboard → Project Settings → API, ensure **`public`** is in the exposed schemas list (default).  
- **Tables**: All app tables should be in `public` (confirm with `scripts/verify_schema.sql`).  
- **If error persists**: Run `ALTER ROLE authenticator RESET pgrst.db_schemas;` in SQL Editor, then `NOTIFY pgrst, 'reload schema';`. See `docs/SUPABASE_SCHEMA_EXPOSURE.md`.

## No code change for schema selection

No change was made to explicitly pass `schema: "public"` in the Python client, because the default is already `public` and the problem was identified as configuration (exposed schemas / `pgrst.db_schemas`), not the client choosing a different schema.
