from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key

router = APIRouter()

class PromptPayload(BaseModel):
    user_id: str
    provider: str
    model: str
    prompt: str
    org_id: str
    project_id: str
    prompt_id: str

def handle_prompt(payload: PromptPayload):
    print(f"[Anthropic Router] Looking up API key for org_id={payload.org_id}, provider={payload.provider}")
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()
    
    print(f"[Anthropic Router] API key query result: {result.data}")
    if not result.data:
        print("[Anthropic Router] No API key found for org/provider.")
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

    # Decrypt the API key
    try:
        encrypted_api_key = result.data[0]["api_key"]
        print("[Anthropic Router] Encrypted API key from DB:", encrypted_api_key)
        api_key = decrypt_api_key(encrypted_api_key)
        print("[Anthropic Router] Decrypted API key:", api_key)
    except Exception as e:
        print("[Anthropic Router] Decryption error:", e)
        raise HTTPException(status_code=500, detail=f"Failed to decrypt API key: {e}")

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=payload.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": payload.prompt}]
        )

        reply = response.content[0].text

        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        pricing = get_pricing("anthropic", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        log_usage(
            user_id=payload.user_id,
            provider="Anthropic",
            model=payload.model,
            prompt=payload.prompt,
            response=reply,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            project_id=getattr(payload, 'project_id', None),
            org_id=payload.org_id,
            prompt_id=getattr(payload, 'prompt_id', None)
        )

        return {
            "status": "success",
            "response": reply,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic call failed: {str(e)}")
