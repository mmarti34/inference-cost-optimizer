from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cohere
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
    print(f"[Cohere Router] Looking up API key for org_id={payload.org_id}, provider={payload.provider}")
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    print(f"[Cohere Router] Org API key query result: {keys}")
    
    if not keys:
        print("[Cohere Router] No API key found for org/provider.")
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

    client = cohere.Client(keys[0]["api_key"])

    try:
        # 2. Send prompt
        response = client.chat(
            model=payload.model,
            message=payload.prompt
        )
        reply = response.text

        # 3. Extract token usage (Cohere returns token_count and generation_token_count)
        input_tokens = getattr(response, "token_count", 0)
        output_tokens = getattr(response, "generation_token_count", 0)
        total_tokens = input_tokens + output_tokens

        # 4. Pricing
        pricing = get_pricing("cohere", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        # 5. Log usage
        log_usage(
            payload.org_id,
            "Cohere",
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
        raise HTTPException(status_code=500, detail=f"Cohere call failed: {str(e)}")
