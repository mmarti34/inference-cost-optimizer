from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from supabase_client import supabase
from utils.usage_logger import log_usage
from utils.pricing import get_pricing

router = APIRouter()

class PromptPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    prompt_id: str

def handle_prompt(payload: PromptPayload):
    print(f"[Gemini Router] Looking up API key for org_id={payload.org_id}, provider={payload.provider}")
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    print(f"[Gemini Router] Org API key query result: {keys}")
    
    if not keys:
        print("[Gemini Router] No API key found for org/provider.")
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

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
            payload.org_id,
            "Gemini",
            payload.model,
            payload.prompt,
            reply,
            payload.prompt_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd
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
