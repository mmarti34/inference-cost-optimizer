# Supabase schema exposure and PGRST104 / PGRST106

## What the errors mean

- **PGRST104** / **PGRST106**: "The schema must be one of the exposed schemas: public, storage, graphql_public" (or similar).  
  PostgREST is rejecting the request because the schema you’re targeting is not in the set of schemas it is allowed to serve. This is **not** primarily an RLS issue; fix schema exposure/selection first.

## 1. Confirm where your tables live

Run in **Supabase SQL Editor**:

```sql
-- From repo: scripts/verify_schema.sql
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
  AND table_name IN (
    'api_keys', 'service_api_keys', 'projects', 'prompt_templates',
    'organizations', 'organization_members', 'user_profiles', 'join_requests',
    'usage_logs', 'optimizer_recommendations', 'request_logs', 'request_spans',
    'model_stats_daily', 'replay_logs'
  )
ORDER BY table_schema, table_name;
```

- All these tables should be in the **`public`** schema.  
- If any are in another schema (e.g. `app`), either move them to `public` or expose that schema (see below).

## 2. Exposed schemas in the dashboard

1. Open **[Supabase Dashboard](https://supabase.com/dashboard)** → your project.
2. Go to **Project Settings** (gear) → **API** (or **Settings → API**).
3. Find **“Exposed schemas”** (or “Schema” / API schema setting).
4. Ensure **`public`** is in the list (default).  
   If you use a custom schema, add it there (e.g. `public, myschema`).

## 3. If you use a custom schema

If you keep tables in a custom schema (e.g. `myschema`):

1. Add that schema in **Exposed schemas** in the dashboard (step 2 above).
2. In **Supabase SQL Editor**, run (replace `myschema` with your schema name):

```sql
GRANT USAGE ON SCHEMA myschema TO anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA myschema TO anon, authenticated, service_role;
GRANT ALL ON ALL ROUTINES IN SCHEMA myschema TO anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA myschema TO anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA myschema
  GRANT ALL ON TABLES TO anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA myschema
  GRANT ALL ON ROUTINES TO anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA myschema
  GRANT ALL ON SEQUENCES TO anon, authenticated, service_role;
```

3. If the error persists, the `authenticator` role’s schema list may be out of sync. Run one of:

```sql
-- Option A: Include your schema explicitly (include every schema you expose)
ALTER ROLE authenticator SET pgrst.db_schemas = 'public, myschema';

-- Option B: Use whatever is configured in the dashboard
ALTER ROLE authenticator RESET pgrst.db_schemas;
```

4. Reload PostgREST so it picks up the change (see step 4 below).

## 4. Reload PostgREST schema cache

After changing exposed schemas or `pgrst.db_schemas`, run in the **SQL Editor**:

```sql
NOTIFY pgrst, 'reload schema';
```

(Also in repo: `scripts/reload_postgrest_schema.sql`.)

## 5. Backend / client: don’t request a non‑exposed schema

- **Backend (Python)**: The app uses a single Supabase client and does **not** pass a custom schema; it relies on the default **`public`** schema. Do not set `ClientOptions(schema="...")` to anything other than `public` unless that schema is exposed and granted as above.
- **Frontend (JS)**: If you use `db: { schema: 'myschema' }` or `.schema('myschema')`, that schema must be in Exposed schemas and granted as in step 3.

## 6. If tables are in `public` and the error still appears

- Re-run `scripts/verify_schema.sql` and confirm all app tables are in `public`.
- Ensure **Exposed schemas** in the dashboard includes `public`.
- Run `ALTER ROLE authenticator RESET pgrst.db_schemas;` then `NOTIFY pgrst, 'reload schema';`.
- Check backend logs (e.g. `supabase_error` with `table=... code=PGRST104`) to see which table/request triggers the error.

## References

- [Using custom schemas](https://supabase.com/docs/guides/api/using-custom-schemas)
- [PGRST106 troubleshooting](https://supabase.com/docs/guides/troubleshooting/pgrst106-the-schema-must-be-one-of-the-following-error-when-querying-an-exposed-schema)
- [Refresh PostgREST schema](https://supabase.com/docs/guides/troubleshooting/refresh-postgrest-schema)
