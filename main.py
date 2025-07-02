from fastapi import FastAPI, HTTPException, Header, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase_client import supabase
from utils.encryption import encrypt_api_key, decrypt_api_key
import secrets
from utils.pricing import get_pricing, suggest_model

from routers import openai_router, anthropic_router, mistral_router, cohere_router, gemini_router

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(openai_router.router, prefix="/openai")
app.include_router(anthropic_router.router, prefix="/anthropic")
app.include_router(mistral_router.router, prefix="/mistral")
app.include_router(cohere_router.router, prefix="/cohere")
app.include_router(gemini_router.router, prefix="/gemini")

# Health check endpoint for Railway
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

class APIKeyPayload(BaseModel):
    user_id: str
    provider: str
    api_key: str

class PromptPayload(BaseModel):
    user_id: str
    provider: str
    model: str
    prompt: str

class DeleteKeyPayload(BaseModel):
    user_id: str
    provider: str

@app.get("/get-keys/{user_id}")
def get_keys(user_id: str):
    try:
        result = supabase.table("api_keys").select("*").eq("user_id", user_id).execute()
        
        # Decrypt API keys before returning
        decrypted_keys = []
        for key in result.data:
            decrypted_key = key.copy()
            try:
                decrypted_key["api_key"] = decrypt_api_key(key["api_key"])
            except:
                # If decryption fails, mask the key
                decrypted_key["api_key"] = "***DECRYPTION_FAILED***"
            decrypted_keys.append(decrypted_key)
        
        return {"status": "success", "keys": decrypted_keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching keys: {str(e)}")

@app.post("/store-key")
def store_api_key(payload: APIKeyPayload):
    try:
        # Encrypt the API key before storing
        encrypted_api_key = encrypt_api_key(payload.api_key)
        
        data = {
            "user_id": payload.user_id,
            "provider": payload.provider,
            "api_key": encrypted_api_key,
        }
        
        # First check if a key for this user/provider already exists
        existing = supabase.table("api_keys").select("*").eq("user_id", payload.user_id).eq("provider", payload.provider).execute()
        
        if existing.data:
            # Update existing key
            result = supabase.table("api_keys").update({"api_key": encrypted_api_key}).eq("user_id", payload.user_id).eq("provider", payload.provider).execute()
            return {"status": "success", "updated": result.data}
        else:
            # Insert new key
            result = supabase.table("api_keys").insert(data).execute()
            return {"status": "success", "inserted": result.data}
            
    except Exception as e:
        print(f"Error storing API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing API key: {str(e)}")

@app.post("/route-prompt")
def route_prompt(payload: PromptPayload):
    provider = payload.provider.lower()

    if provider == "openai":
        return openai_router.handle_prompt(payload)
    elif provider == "anthropic":
        return anthropic_router.handle_prompt(payload)
    elif provider == "mistral":
        return mistral_router.handle_prompt(payload)
    elif provider == "cohere":
        return cohere_router.handle_prompt(payload)
    elif provider == "gemini":
        return gemini_router.handle_prompt(payload)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")

@app.post("/test-prompt")
def test_prompt(payload: PromptPayload):
    """Test endpoint for playground that doesn't use usage logging"""
    provider = payload.provider.lower()

    if provider == "openai":
        return test_openai_call(payload)
    elif provider == "anthropic":
        return test_anthropic_call(payload)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider for testing: {payload.provider}")

def test_openai_call(payload: PromptPayload):
    """Test OpenAI call without usage logging"""
    from openai import OpenAI
    from utils.encryption import decrypt_api_key
    
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("user_id", payload.user_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    if not keys:
        raise HTTPException(status_code=404, detail="API key not found for user/provider.")

    # Decrypt the API key
    try:
        encrypted_api_key = keys[0]["api_key"]
        api_key = decrypt_api_key(encrypted_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decrypt API key: {e}")

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=payload.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": payload.prompt}
            ]
        )
        reply = completion.choices[0].message.content
        
        return {
            "status": "success",
            "response": reply,
            "input_tokens": getattr(completion.usage, "prompt_tokens", 0),
            "output_tokens": getattr(completion.usage, "completion_tokens", 0),
            "total_tokens": getattr(completion.usage, "total_tokens", 0),
            "cost_usd": 0.0  # No cost calculation for testing
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI test call failed: {e}")

def test_anthropic_call(payload: PromptPayload):
    """Test Anthropic call without usage logging"""
    from anthropic import Anthropic
    from utils.encryption import decrypt_api_key
    
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("user_id", payload.user_id) \
        .eq("provider", payload.provider) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="API key not found.")

    # Decrypt the API key
    try:
        encrypted_api_key = result.data[0]["api_key"]
        api_key = decrypt_api_key(encrypted_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decrypt API key: {e}")

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=payload.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": payload.prompt}]
        )

        reply = response.content[0].text
        
        return {
            "status": "success",
            "response": reply,
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
            "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
            "cost_usd": 0.0  # No cost calculation for testing
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic test call failed: {str(e)}")

@app.post("/v1/prompt")
def universal_prompt(request: Request, payload: dict, authorization: str = Header(None)):
    print("--- DEBUG: /v1/prompt called ---")
    print("Payload:", payload)
    print("Authorization header:", authorization)
    # 1. Authenticate API key
    if not authorization or not authorization.startswith("Bearer "):
        print("Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    service_api_key = authorization.split(" ", 1)[1]
    
    # Get all service API keys for this user and check if any match
    key_result = supabase.table("service_api_keys").select("*").execute()
    print("Service API keys lookup result:", key_result.data)
    
    # Find the matching key by decrypting each one
    matching_key = None
    for key_data in key_result.data:
        try:
            decrypted_key = decrypt_api_key(key_data["api_key"])
            if decrypted_key == service_api_key:
                matching_key = key_data
                break
        except Exception as e:
            print(f"Failed to decrypt key {key_data['id']}: {e}")
            continue
    
    if not matching_key:
        print("Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key.")
    
    user_id = matching_key["user_id"]

    # 2. Get prompt_id and input
    prompt_id = payload.get("prompt_id")
    user_input = payload.get("input")
    print("Prompt ID:", prompt_id, "User input:", user_input)
    if not prompt_id or not user_input:
        print("Missing prompt_id or input")
        raise HTTPException(status_code=400, detail="Missing prompt_id or input.")

    # 3. Lookup prompt template
    prompt_result = supabase.table("prompt_templates").select("*").eq("id", prompt_id).single().execute()
    print("Prompt template lookup result:", prompt_result.data)
    if not prompt_result.data:
        print("Prompt template not found")
        raise HTTPException(status_code=404, detail="Prompt template not found.")
    prompt_template = prompt_result.data
    # Allow if user is the prompt owner
    if prompt_template["user_id"] == user_id:
        pass  # allow
    else:
        # Check if both user and prompt are in the same org
        prompt_org_id = prompt_template.get("org_id")
        if not prompt_org_id:
            print("Prompt has no org_id, access denied")
            raise HTTPException(status_code=403, detail="You do not have access to this prompt.")

        # Check if user is a member of the org
        org_member_result = supabase.table("organization_members") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("org_id", prompt_org_id) \
            .eq("status", "active") \
            .execute()
        if not org_member_result.data:
            print("User is not a member of the prompt's org, access denied")
            raise HTTPException(status_code=403, detail="You do not have access to this prompt.")
        # else: allow

    # 4. Prepare prompt
    prompt_text = prompt_template["prompt"].replace("{input}", user_input) if prompt_template["prompt"] else user_input
    provider = prompt_template["provider"].lower()
    model = prompt_template["model"]
    print(f"Using provider: {provider}, model: {model}, prompt_text: {prompt_text}")

    # 5. Call the correct LLM router
    payload_obj = PromptPayload(
        user_id=user_id,
        provider=provider,
        model=model,
        prompt=prompt_text
    )
    print("Calling LLM router with:", payload_obj)
    try:
        if provider == "openai":
            result = openai_router.handle_prompt(payload_obj)
        elif provider == "anthropic":
            result = anthropic_router.handle_prompt(payload_obj)
        elif provider == "mistral":
            result = mistral_router.handle_prompt(payload_obj)
        elif provider == "cohere":
            result = cohere_router.handle_prompt(payload_obj)
        elif provider == "gemini":
            result = gemini_router.handle_prompt(payload_obj)
        else:
            print(f"Unsupported provider: {provider}")
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    except Exception as e:
        print(f"Exception in LLM router: {e}")
        raise

    print("LLM router result:", result)
    # 6. Return response in standard format
    return {
        "status": result.get("status", "success"),
        "response": result.get("response"),
        "provider": provider,
        "model": model,
        "prompt_id": prompt_id
    }

@app.get("/get-service-api-key/{user_id}")
def get_service_api_key(user_id: str):
    result = supabase.table("service_api_keys").select("*").eq("user_id", user_id).single().execute()
    if result.data:
        return {"api_key": result.data["api_key"]}
    else:
        return {"api_key": None}

@app.post("/generate-service-api-key/{user_id}")
def generate_service_api_key(user_id: str):
    # Check if key already exists
    result = supabase.table("service_api_keys").select("*").eq("user_id", user_id).single().execute()
    if result.data:
        return {"api_key": result.data["api_key"]}
    # Generate new key
    api_key = secrets.token_urlsafe(32)
    data = {"user_id": user_id, "api_key": api_key}
    supabase.table("service_api_keys").insert(data).execute()
    return {"api_key": api_key}

@app.get("/list-service-api-keys/{user_id}")
def list_service_api_keys(user_id: str):
    result = supabase.table("service_api_keys").select("id, created_at, api_key").eq("user_id", user_id).execute()
    keys = result.data or []
    # Mask the API key (show only first 4 and last 4 chars)
    for k in keys:
        key = k["api_key"]
        k["api_key_masked"] = f'{key[:4]}...{key[-4:]}'
        del k["api_key"]
    return {"keys": keys}

@app.delete("/delete-service-api-key/{key_id}")
def delete_service_api_key(key_id: str):
    result = supabase.table("service_api_keys").delete().eq("id", key_id).execute()
    return {"status": "deleted", "id": key_id}

@app.delete("/delete-key")
def delete_api_key(payload: DeleteKeyPayload):
    try:
        # Delete the API key for the user and provider
        result = supabase.table("api_keys") \
            .delete() \
            .eq("user_id", payload.user_id) \
            .eq("provider", payload.provider) \
            .execute()
        # result.data will be a list of deleted rows (or empty if nothing deleted)
        if not result.data:
            raise HTTPException(status_code=404, detail="No API key found to delete.")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting API key: {str(e)}")

@app.post("/suggest-model")
def suggest_model_endpoint(data: dict = Body(...)):
    prompt = data.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    suggestion = suggest_model(prompt)
    return {"suggestion": suggestion}
