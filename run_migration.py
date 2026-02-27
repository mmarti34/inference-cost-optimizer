#!/usr/bin/env python3
"""
Migration runner that connects directly to Supabase Postgres database
and executes SQL migrations with proper error handling.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql

# Load environment variables
load_dotenv()

def get_db_connection():
    """
    Get Postgres connection from Supabase URL or direct connection string.
    Tries multiple methods to connect.
    """
    # Method 1: Direct Postgres connection string (preferred)
    db_url = (
        os.getenv("DATABASE_URL") or 
        os.getenv("POSTGRES_URL") or 
        os.getenv("SUPABASE_DB_URL") or
        os.getenv("SUPABASE_DIRECT_URL")
    )
    
    if db_url:
        # If it's a postgres:// URL, use it directly
        if db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
            try:
                return psycopg2.connect(db_url)
            except Exception as e:
                raise ValueError(f"Failed to connect with DATABASE_URL: {e}")
    
    # Method 2: Try individual connection parameters
    db_host = os.getenv("DB_HOST") or os.getenv("SUPABASE_DB_HOST")
    db_port = os.getenv("DB_PORT") or os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("DB_NAME") or os.getenv("SUPABASE_DB_NAME") or os.getenv("POSTGRES_DB", "postgres")
    db_user = os.getenv("DB_USER") or os.getenv("SUPABASE_DB_USER") or os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD") or os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    
    # If we have individual params, use them
    if db_host and db_user and db_password:
        try:
            return psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password
            )
        except Exception as e:
            raise ValueError(f"Failed to connect with individual parameters: {e}")
    
    # If we have SUPABASE_URL, provide helpful instructions
    supabase_url = os.getenv("SUPABASE_URL", "")
    if supabase_url:
        raise ValueError(
            "❌ Could not find direct database connection.\n\n"
            "The Supabase Python client (SUPABASE_URL/SUPABASE_KEY) is for API access,\n"
            "but migrations need direct Postgres access.\n\n"
            "📋 To get your direct connection string:\n"
            "1. Go to: https://supabase.com/dashboard\n"
            "2. Select your project\n"
            "3. Go to: Settings > Database\n"
            "4. Scroll to 'Connection string'\n"
            "5. Select 'URI' or 'Direct connection' (NOT 'Session mode' or 'Transaction mode')\n"
            "6. Copy the connection string\n\n"
            "Then add to your .env file:\n"
            "DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres\n\n"
            "Or set these individual variables:\n"
            "DB_HOST=db.[PROJECT-REF].supabase.co\n"
            "DB_NAME=postgres\n"
            "DB_USER=postgres\n"
            "DB_PASSWORD=[YOUR-PASSWORD]\n"
        )
    
    raise ValueError(
        "❌ Could not determine database connection.\n\n"
        "Please set one of:\n"
        "  - DATABASE_URL (postgresql://...)\n"
        "  - SUPABASE_DB_URL (postgresql://...)\n"
        "  - Or set DB_HOST, DB_NAME, DB_USER, DB_PASSWORD\n\n"
        "For Supabase, get the direct connection string from:\n"
        "Dashboard > Settings > Database > Connection String > Direct Connection"
    )


def read_sql_file(file_path: str) -> str:
    """Read SQL migration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Migration file not found: {file_path}")
    return path.read_text()


def execute_migration(conn, sql_content: str):
    """
    Execute SQL migration with proper error handling.
    Splits by semicolons and executes statements one by one.
    """
    cursor = conn.cursor()
    
    # Remove comments and split by semicolons
    statements = []
    current_statement = []
    
    for line in sql_content.split('\n'):
        # Skip comment-only lines
        stripped = line.strip()
        if not stripped or stripped.startswith('--'):
            continue
        
        # Remove inline comments
        if '--' in line:
            line = line[:line.index('--')]
        
        current_statement.append(line)
        
        # Check if line ends with semicolon (end of statement)
        if stripped.endswith(';'):
            statement = '\n'.join(current_statement).strip()
            if statement:
                statements.append(statement)
            current_statement = []
    
    # Execute each statement
    executed = 0
    errors = []
    
    for i, statement in enumerate(statements, 1):
        try:
            print(f"Executing statement {i}/{len(statements)}...")
            cursor.execute(statement)
            executed += 1
        except psycopg2.Error as e:
            error_msg = f"Error in statement {i}: {str(e)}"
            print(f"⚠️  {error_msg}")
            errors.append((i, statement[:100], str(e)))
            # Continue with next statement (some errors like "already exists" are OK)
            conn.rollback()
    
    conn.commit()
    cursor.close()
    
    return executed, errors


def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table already exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
    """, (table_name,))
    exists = cursor.fetchone()[0]
    cursor.close()
    return exists


def main():
    """Main migration runner."""
    migration_file = "migration_add_observability_tables.sql"
    
    if len(sys.argv) > 1:
        migration_file = sys.argv[1]
    
    print("=" * 60)
    print("OptiML Database Migration Runner")
    print("=" * 60)
    print()
    
    # Check if migration file exists
    if not Path(migration_file).exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)
    
    print(f"📄 Migration file: {migration_file}")
    print()
    
    # Connect to database
    print("🔌 Connecting to database...")
    try:
        conn = get_db_connection()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print()
        print("💡 Tip: Make sure you have:")
        print("   - DATABASE_URL or SUPABASE_DB_URL set in .env")
        print("   - Or DB_HOST, DB_NAME, DB_USER, DB_PASSWORD set")
        print()
        print("   For Supabase, get the direct connection string from:")
        print("   Dashboard > Settings > Database > Connection String > Direct Connection")
        sys.exit(1)
    
    try:
        # Check existing tables
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
            response = input(f"Some tables already exist. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                return
        
        # Read and execute migration
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
            print("Error details:")
            for i, stmt, err in errors:
                print(f"  Statement {i}: {err}")
                print(f"    {stmt}...")
        else:
            print("✅ No errors")
        
        # Verify tables were created
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

