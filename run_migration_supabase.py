#!/usr/bin/env python3
"""
Migration runner using Supabase Python client (no psycopg2 required).
Uses Supabase RPC to execute SQL.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_supabase_client():
    """Get Supabase client."""
    from supabase import create_client
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "❌ Missing Supabase credentials.\n\n"
            "Please set in your .env file:\n"
            "  SUPABASE_URL=your-supabase-url\n"
            "  SUPABASE_KEY=your-supabase-service-role-key\n\n"
            "Note: You need the SERVICE ROLE KEY (not anon key) to execute SQL.\n"
            "Get it from: Dashboard > Settings > API > service_role key"
        )
    
    return create_client(supabase_url, supabase_key)


def read_sql_file(file_path: str) -> str:
    """Read SQL migration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Migration file not found: {file_path}")
    return path.read_text()


def execute_sql_via_rpc(supabase, sql_content: str):
    """
    Execute SQL via Supabase RPC.
    Note: This requires a custom RPC function in Supabase, or we can use the REST API.
    Actually, Supabase doesn't have a built-in RPC for arbitrary SQL.
    Let's use the PostgREST API directly or create a migration function.
    """
    # Split SQL into statements
    statements = []
    current_statement = []
    
    for line in sql_content.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('--'):
            continue
        
        if '--' in line:
            line = line[:line.index('--')]
        
        current_statement.append(line)
        
        if stripped.endswith(';'):
            statement = '\n'.join(current_statement).strip()
            if statement:
                statements.append(statement)
            current_statement = []
    
    # Execute via Supabase REST API (requires service role key)
    # We'll use the supabase-py's postgrest client
    executed = 0
    errors = []
    
    for i, statement in enumerate(statements, 1):
        try:
            print(f"Executing statement {i}/{len(statements)}...")
            # Use Supabase's REST API to execute SQL
            # Note: This requires the SQL to be wrapped in a function or executed via pg_catalog
            # Actually, Supabase doesn't allow arbitrary SQL execution via REST API for security
            
            # Alternative: Use the Supabase management API if available
            # Or we need to tell the user to run this in the SQL editor
            
            # For now, let's try using the supabase-py's raw query capability
            # But supabase-py doesn't support raw SQL execution
            
            # The best approach: Guide user to run in SQL editor, or use psycopg2
            print(f"   ⚠️  Statement {i} requires direct database access")
            print(f"   Please run this SQL in Supabase SQL Editor:")
            print(f"   {statement[:100]}...")
            
        except Exception as e:
            error_msg = f"Error in statement {i}: {str(e)}"
            print(f"⚠️  {error_msg}")
            errors.append((i, statement[:100], str(e)))
    
    return executed, errors


def main():
    """Main migration runner."""
    print("=" * 60)
    print("OptiML Database Migration Runner (Supabase Client)")
    print("=" * 60)
    print()
    print("⚠️  NOTE: Supabase Python client cannot execute arbitrary SQL.")
    print("   This script will show you the SQL to run manually.")
    print()
    print("   For automatic execution, use run_migration.py with psycopg2")
    print("   (requires DATABASE_URL connection string)")
    print()
    print("=" * 60)
    print()
    
    migration_file = "migration_add_observability_tables.sql"
    
    if len(sys.argv) > 1:
        migration_file = sys.argv[1]
    
    if not Path(migration_file).exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)
    
    print(f"📄 Migration file: {migration_file}")
    print()
    
    # Read SQL file
    print("📖 Reading migration file...")
    sql_content = read_sql_file(migration_file)
    print(f"✅ Read {len(sql_content)} characters")
    print()
    
    # Show instructions
    print("=" * 60)
    print("Migration Instructions")
    print("=" * 60)
    print()
    print("To run this migration, you have two options:")
    print()
    print("OPTION 1: Run in Supabase SQL Editor (Easiest)")
    print("  1. Go to: https://supabase.com/dashboard")
    print("  2. Select your project")
    print("  3. Go to: SQL Editor")
    print("  4. Click 'New query'")
    print("  5. Copy and paste the SQL below")
    print("  6. Click 'Run'")
    print()
    print("OPTION 2: Use run_migration.py with direct database connection")
    print("  1. Get your direct connection string from Supabase:")
    print("     Settings > Database > Connection String > Direct Connection")
    print("  2. Add to .env: DATABASE_URL=postgresql://...")
    print("  3. Run: python run_migration.py")
    print()
    print("=" * 60)
    print("SQL Migration Content")
    print("=" * 60)
    print()
    print(sql_content)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()

