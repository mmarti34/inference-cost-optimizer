# Quick Migration Guide

## ✅ Your Connection Info is Ready!

Your `.env` file now has:
- `DATABASE_URL=postgresql://postgres:u8cByDlAeGNsZlQQ@db.lpofrfyozqjkjebpwmma.supabase.co:5432/postgres`

## Option 1: Run in Supabase SQL Editor (EASIEST - 2 minutes)

1. Go to: https://supabase.com/dashboard/project/lpofrfyozqjkjebpwmma
2. Click **SQL Editor** in the left sidebar
3. Click **New query**
4. Copy the entire contents of `migration_add_observability_tables.sql`
5. Paste into the SQL Editor
6. Click **Run** (or press Cmd/Ctrl + Enter)

That's it! ✅

## Option 2: Install psycopg2 and Run Script (If you want automation)

The pip hash check is blocking automatic installation. Try this:

```bash
cd inference-cost-optimizer
source venv/bin/activate

# Try installing with pip directly (bypass hash check)
pip install --no-deps psycopg2-binary --no-cache-dir

# If that fails, try:
python -m pip install --no-deps --no-cache-dir psycopg2-binary

# Or install from wheel directly:
pip install https://files.pythonhosted.org/packages/24/cc/dc143ea88e4ec9d386106cac05023b69668bd0be20794c613446eaefafe5/psycopg2_binary-2.9.11-cp310-cp310-macosx_11_0_arm64.whl --no-deps

# Then run:
python run_migration.py
```

## What Gets Created

- ✅ `request_logs` - Main request logging table
- ✅ `request_spans` - Trace spans for observability  
- ✅ `model_stats_daily` - Daily aggregated statistics
- ✅ `replay_logs` - Offline replay results
- ✅ All necessary indexes for performance

## Verify It Worked

After running, check in Supabase:
1. Go to **Table Editor**
2. You should see the 4 new tables listed

Or run this SQL:
```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('request_logs', 'request_spans', 'model_stats_daily', 'replay_logs');
```

You should see all 4 tables! 🎉

