#!/usr/bin/env python3
"""
Diagnostic script to check api_keys table structure and data
"""
import os
from dotenv import load_dotenv
from supabase_client import supabase

load_dotenv()

def diagnose_api_keys():
    """Check api_keys table structure and sample data"""
    print("=" * 60)
    print("API Keys Table Diagnostic")
    print("=" * 60)
    print()
    
    if not supabase:
        print("❌ Supabase client not initialized")
        return
    
    try:
        # Try to get table structure by querying with limit 0
        print("🔍 Checking table structure...")
        
        # Try to select one row to see what columns exist
        result = supabase.table("api_keys").select("*").limit(1).execute()
        
        if result.data and len(result.data) > 0:
            print("✅ Table exists and has data")
            print()
            print("Sample row columns:")
            sample = result.data[0]
            for key in sample.keys():
                value = sample[key]
                if key == "api_key":
                    value = f"{str(value)[:20]}... (encrypted)" if value else "None"
                print(f"  - {key}: {type(value).__name__} = {value}")
        else:
            print("⚠️  Table exists but is empty")
            print()
            print("Trying to check structure by attempting insert (will rollback)...")
            # Try a test insert to see what columns are required
            try:
                test_result = supabase.table("api_keys").insert({
                    "org_id": "00000000-0000-0000-0000-000000000000",  # Dummy UUID
                    "provider": "test",
                    "api_key": "test"
                }).execute()
                print("✅ Test insert succeeded (structure looks OK)")
                # Delete the test row
                if test_result.data:
                    supabase.table("api_keys").delete().eq("id", test_result.data[0]["id"]).execute()
            except Exception as insert_error:
                print(f"❌ Test insert failed: {insert_error}")
                print("   This shows what columns are required/missing")
        
        print()
        print("🔍 Checking for specific org_id...")
        # Check if we can query by org_id
        test_org_id = "c32d3a3e-1f57-4aed-b7f9-cbef78087166"
        org_result = supabase.table("api_keys").select("*").eq("org_id", test_org_id).execute()
        
        if org_result.data:
            print(f"✅ Found {len(org_result.data)} API keys for org {test_org_id}")
            print()
            print("Sample keys:")
            for i, key in enumerate(org_result.data[:3], 1):
                print(f"  {i}. ID: {key.get('id')}")
                print(f"     Provider: {key.get('provider')}")
                print(f"     Name: {key.get('name', 'N/A')}")
                print(f"     User ID: {key.get('user_id', 'N/A')}")
                print(f"     Created: {key.get('created_at')}")
        else:
            print(f"⚠️  No API keys found for org {test_org_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    diagnose_api_keys()


