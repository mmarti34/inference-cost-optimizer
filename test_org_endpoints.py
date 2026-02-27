#!/usr/bin/env python3
"""
Test script for organization-based endpoints
"""

import requests
import json
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import hashlib

# Configuration
BASE_URL = "https://api.optiml.one"
ENCRYPTION_KEY = "your-secret-key-32-chars-long!!"  # This should match your backend

def decrypt_cryptojs_api_key(encrypted_key: str) -> str:
    """Decrypt a CryptoJS-style AES-encrypted string"""
    try:
        encrypted = base64.b64decode(encrypted_key)
        if encrypted[:8] != b"Salted__":
            raise ValueError("Invalid encrypted data: missing 'Salted__' header")
        salt = encrypted[8:16]
        ciphertext = encrypted[16:]
        password = ENCRYPTION_KEY.encode("utf-8")
        
        # EVP_BytesToKey equivalent
        d = b''
        while len(d) < 48:
            d += hashlib.md5(d + password + salt).digest()
        key, iv = d[:32], d[32:48]
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_debug_api_keys():
    """Test debug API keys endpoint"""
    print("🔍 Testing debug API keys endpoint...")
    response = requests.get(f"{BASE_URL}/debug/api-keys")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    print()
    return data

def test_list_service_api_keys(org_id: str):
    """Test list service API keys endpoint"""
    print(f"🔍 Testing list service API keys for org {org_id}...")
    response = requests.get(f"{BASE_URL}/list-service-api-keys/{org_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_generate_service_api_key(org_id: str):
    """Test generate service API key endpoint"""
    print(f"🔍 Testing generate service API key for org {org_id}...")
    response = requests.post(f"{BASE_URL}/generate-service-api-key/{org_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    return response.json().get("api_key")

def test_prompt_api_direct(api_key: str, org_id: str):
    """Test prompt API with direct call (no prompt template)"""
    print(f"🔍 Testing prompt API direct call for org {org_id}...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "provider": "openai",
        "model": "gpt-3.5-turbo", 
        "input": "Hello, how are you?"
    }
    
    response = requests.post(f"{BASE_URL}/v1/prompt", headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prompt_api_with_template(api_key: str, prompt_id: str):
    """Test prompt API with prompt template"""
    print(f"🔍 Testing prompt API with template {prompt_id}...")
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "prompt_id": prompt_id,
        "input": "Hello, how are you?"
    }
    
    response = requests.post(f"{BASE_URL}/v1/prompt", headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def main():
    print("🚀 Testing Organization-Based Endpoints")
    print("=" * 50)
    
    # Test health
    test_health()
    
    # Test debug endpoint
    debug_data = test_debug_api_keys()
    
    # Get your organization ID and API key
    your_org_id = "05ef4e73-de21-49fe-bf7f-b8303cab31b6"
    your_prompt_id = "8d7dd9dc-2584-40a8-b075-d3df65877708"
    
    # Find your API key
    your_api_key_encrypted = None
    for key in debug_data.get("sample_api_keys", []):
        if key["org_id"] == your_org_id:
            your_api_key_encrypted = key["api_key_preview"]
            break
    
    if your_api_key_encrypted:
        print(f"🔑 Found encrypted API key: {your_api_key_encrypted}")
        
        # Test list service API keys
        test_list_service_api_keys(your_org_id)
        
        # Test generate new API key
        new_api_key = test_generate_service_api_key(your_org_id)
        
        if new_api_key:
            print(f"🔑 Generated new API key: {new_api_key}")
            
            # Test direct prompt API call
            test_prompt_api_direct(new_api_key, your_org_id)
            
            # Test prompt API with template
            test_prompt_api_with_template(new_api_key, your_prompt_id)
        else:
            print("❌ Failed to generate new API key")
    else:
        print(f"❌ No API key found for org {your_org_id}")

if __name__ == "__main__":
    main() 