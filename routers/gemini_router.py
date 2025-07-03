from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing

router = APIRouter()

class PromptPayload(BaseModel):
    user_id: str
    org_id: str
    provider: str
    model: str
    prompt: str
    project_id: str
    prompt_id: str

def handle_prompt(payload: PromptPayload):
    print(f"[Gemini Router] Looking up API key for user_id={payload.user_id}, provider={payload.provider}, org_id={payload.org_id}")
    
    # First, try to get user-specific API key
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("user_id", payload.user_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    print(f"[Gemini Router] User API key query result: {keys}")
    
    # If no user-specific key, try organization-level key
    if not keys and payload.org_id:
        print(f"[Gemini Router] No user API key found, checking org_id={payload.org_id}")
        result = supabase.table("api_keys") \
            .select("*") \
            .eq("org_id", payload.org_id) \
            .eq("provider", payload.provider) \
            .execute()
        
        keys = result.data
        print(f"[Gemini Router] Org API key query result: {keys}")
    
    if not keys:
        print("[Gemini Router] No API key found for user/provider or org/provider.")
        raise HTTPException(status_code=404, detail="API key not found for user/provider.")

    # 2. Configure Gemini client
    genai.configure(api_key=keys[0]["api_key"])
    model = genai.GenerativeModel(payload.model)

    try:
        # 3. Generate content
        response = model.generate_content(payload.prompt)
        reply = response.text

        # 4. Get token usage (may be None depending on model/version)
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
        total_tokens = input_tokens + output_tokens

        # 5. Pricing
        pricing = get_pricing("gemini", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        # 6. Log usage
        log_usage(
            user_id=payload.user_id,
            provider="Gemini",
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
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {str(e)}")
