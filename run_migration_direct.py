#!/usr/bin/env python3
"""
Direct migration runner - uses connection string from environment.
Bypasses pip hash issues by using subprocess to install psycopg2 if needed.
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def ensure_psycopg2():
    """Ensure psycopg2 is installed, install if needed."""
    try:
        import psycopg2
        return True
    except ImportError:
        print("psycopg2 not found. Attempting to install...")
        try:
            # Try installing without hash checking
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--no-deps', 'psycopg2-binary'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                import psycopg2
                return True
            else:
                print("⚠️  Could not install psycopg2 automatically.")
                print("   Please install manually: pip install psycopg2-binary")
                print("   Or run the SQL manually in Supabase SQL Editor")
                return False
        except Exception as e:
            print(f"Error installing psycopg2: {e}")
            return False

def main():
    if not ensure_psycopg2():
        print("\n💡 Alternative: Run the SQL manually in Supabase SQL Editor")
        print("   File: migration_add_observability_tables.sql")
        sys.exit(1)
    
    # Import and run the actual migration
    import psycopg2
    from run_migration import get_db_connection, read_sql_file, execute_migration, check_table_exists
    
    migration_file = "migration_add_observability_tables.sql"
    
    print("=" * 60)
    print("OptiML Database Migration Runner")
    print("=" * 60)
    print()
    
    if not Path(migration_file).exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)
    
    print(f"📄 Migration file: {migration_file}")
    print()
    
    print("🔌 Connecting to database...")
    try:
        conn = get_db_connection()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)
    
    try:
        print()
        print("🔍 Checking existing tables...")
        tables_to_check = ['request_logs', 'request_spans', 'model_stats_daily', 'replay_logs']
        existing_tables = []
        for table in tables_to_check:
            if check_table_exists(conn, table):
                existing_tables.append(table)
                print(f"   ⚠️  Table '{table}' already exists")
        
        if existing_tables:
            print()
            print(f"✅ Some tables already exist. Migration will skip existing tables.")
        
        print()
        print("📖 Reading migration file...")
        sql_content = read_sql_file(migration_file)
        print(f"✅ Read {len(sql_content)} characters")
        
        print()
        print("🚀 Executing migration...")
        executed, errors = execute_migration(conn, sql_content)
        
        print()
        print("=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"✅ Executed: {executed} statements")
        
        if errors:
            print(f"⚠️  Errors: {len(errors)} (may be expected, e.g., 'already exists')")
        
        print()
        print("🔍 Verifying tables...")
        all_created = True
        for table in tables_to_check:
            if check_table_exists(conn, table):
                print(f"   ✅ {table}")
            else:
                print(f"   ❌ {table} (not found)")
                all_created = False
        
        if all_created:
            print()
            print("🎉 Migration completed successfully!")
        else:
            print()
            print("⚠️  Some tables may not have been created. Check errors above.")
        
    except Exception as e:
        print()
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()
        print()
        print("🔌 Database connection closed")

if __name__ == "__main__":
    main()

