from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mistralai.client import MistralClient
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
    print(f"[Mistral Router] Looking up API key for org_id={payload.org_id}, provider={payload.provider}")
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    print(f"[Mistral Router] Org API key query result: {keys}")
    
    if not keys:
        print("[Mistral Router] No API key found for org/provider.")
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

    client = MistralClient(api_key=keys[0]["api_key"])

    try:
        response = client.chat(
            model=payload.model,
            messages=[{"role": "user", "content": payload.prompt}]
        )

        reply = response.choices[0].message.content

        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        pricing = get_pricing("mistral", payload.model)
        cost_usd = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

        log_usage(
            payload.org_id,
            "Mistral",
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
        raise HTTPException(status_code=500, detail=f"Mistral call failed: {str(e)}")
