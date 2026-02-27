#!/usr/bin/env python3
"""
Quick script to disable RLS for backend-managed tables
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("psycopg2 not found. Attempting to install...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary", "--no-deps"])
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except Exception as e:
        print(f"Could not install psycopg2: {e}")
        print("\nPlease run the SQL manually in Supabase SQL Editor:")
        print("File: QUICK_RLS_FIX.sql")
        sys.exit(1)

def get_db_connection():
    """Get database connection from environment variables"""
    # Try multiple environment variable names
    database_url = (
        os.getenv("DATABASE_URL") or 
        os.getenv("POSTGRES_URL") or 
        os.getenv("SUPABASE_DB_URL") or
        os.getenv("SUPABASE_DIRECT_URL")
    )
    
    if database_url:
        return psycopg2.connect(database_url)
    
    # Try individual components
    db_host = os.getenv("DB_HOST") or os.getenv("SUPABASE_DB_HOST")
    db_port = os.getenv("DB_PORT") or os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("DB_NAME") or os.getenv("SUPABASE_DB_NAME") or os.getenv("POSTGRES_DB", "postgres")
    db_user = os.getenv("DB_USER") or os.getenv("SUPABASE_DB_USER") or os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD") or os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    
    if not db_host or not db_user or not db_password:
        raise ValueError(
            "Database connection not configured. Set DATABASE_URL or individual DB_* variables.\n"
            "For Supabase, get the direct connection string from:\n"
            "Dashboard > Settings > Database > Connection String > Direct Connection"
        )
    
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )

def run_rls_fix():
    """Disable RLS for backend-managed tables"""
    sql_statements = [
        "ALTER TABLE projects DISABLE ROW LEVEL SECURITY;",
        "ALTER TABLE prompt_templates DISABLE ROW LEVEL SECURITY;",
        "ALTER TABLE api_keys DISABLE ROW LEVEL SECURITY;",
        "ALTER TABLE service_api_keys DISABLE ROW LEVEL SECURITY;",
    ]
    
    print("=" * 60)
    print("Disabling RLS for Backend-Managed Tables")
    print("=" * 60)
    print()
    
    try:
        print("🔌 Connecting to database...")
        conn = get_db_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("✅ Connected successfully")
        print()
        
        print("🚀 Disabling RLS...")
        for i, sql in enumerate(sql_statements, 1):
            table_name = sql.split()[2]  # Extract table name
            print(f"  {i}. Disabling RLS for {table_name}...")
            try:
                cursor.execute(sql)
                print(f"     ✅ Success")
            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg.lower():
                    print(f"     ⚠️  Table {table_name} does not exist (skipping)")
                elif "already" in error_msg.lower() or "not enabled" in error_msg.lower():
                    print(f"     ⚠️  RLS already disabled for {table_name}")
                else:
                    print(f"     ❌ Error: {error_msg}")
        
        print()
        print("🔍 Verifying RLS status...")
        cursor.execute("""
            SELECT 
              tablename,
              rowsecurity as rls_enabled
            FROM pg_tables
            WHERE tablename IN ('projects', 'prompt_templates', 'api_keys', 'service_api_keys')
            AND schemaname = 'public'
            ORDER BY tablename;
        """)
        
        results = cursor.fetchall()
        print()
        print("RLS Status:")
        all_disabled = True
        for table_name, rls_enabled in results:
            status = "❌ ENABLED" if rls_enabled else "✅ DISABLED"
            print(f"  {table_name}: {status}")
            if rls_enabled:
                all_disabled = False
        
        print()
        if all_disabled:
            print("🎉 Success! RLS is disabled for all backend-managed tables.")
            print("   You should now be able to create projects, prompts, and API keys.")
        else:
            print("⚠️  Some tables still have RLS enabled. Check errors above.")
        
        cursor.close()
        conn.close()
        print()
        print("🔌 Database connection closed")
        
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        print()
        print("💡 Alternative: Run the SQL manually in Supabase SQL Editor")
        print("   File: QUICK_RLS_FIX.sql")
        sys.exit(1)

if __name__ == "__main__":
    run_rls_fix()

