# Database Migration Guide

This guide explains how to run database migrations for the OptiML observability system.

## Quick Start

1. **Get your Supabase direct connection string:**
   - Go to [Supabase Dashboard](https://supabase.com/dashboard)
   - Select your project
   - Go to **Settings > Database**
   - Scroll to **Connection string**
   - Select **URI** or **Direct connection** (NOT Session/Transaction mode)
   - Copy the connection string

2. **Add to your `.env` file:**
   ```bash
   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the migration:**
   ```bash
   python run_migration.py
   ```

That's it! The script will:
- ✅ Connect to your database
- ✅ Check for existing tables
- ✅ Execute the migration
- ✅ Verify tables were created
- ✅ Show you a summary

## Alternative: Using Individual Connection Parameters

Instead of `DATABASE_URL`, you can set individual parameters:

```bash
DB_HOST=db.xxxxx.supabase.co
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password
DB_PORT=5432
```

## What Gets Created

The migration creates 4 new tables:

1. **request_logs** - Main request log table (one row per LLM request)
2. **request_spans** - Trace spans (one row per span/attempt)
3. **model_stats_daily** - Daily aggregated statistics for dashboards
4. **replay_logs** - Offline replay results

Plus indexes for performance.

## Troubleshooting

### "Could not find direct database connection"

**Solution:** Make sure you're using the **Direct connection** string, not the Session/Transaction mode connection string. The direct connection looks like:
```
postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
```

### "Table already exists" warnings

This is normal if you're re-running the migration. The script uses `CREATE TABLE IF NOT EXISTS`, so it's safe to run multiple times.

### Connection timeout

- Check your Supabase project is active
- Verify the connection string is correct
- Make sure your IP isn't blocked (check Supabase dashboard)

### Permission errors

The connection needs to be made with a user that has CREATE TABLE permissions. The default `postgres` user should work. If you're using a custom user, make sure it has the necessary permissions.

## Manual Migration (Alternative)

If you prefer to run the SQL manually:

1. Go to Supabase Dashboard > SQL Editor
2. Copy the contents of `migration_add_observability_tables.sql`
3. Paste and run in the SQL Editor

## Verifying the Migration

After running, you can verify the tables were created:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('request_logs', 'request_spans', 'model_stats_daily', 'replay_logs');
```

You should see all 4 tables listed.

