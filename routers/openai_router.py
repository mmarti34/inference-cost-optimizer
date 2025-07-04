from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import supabase
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from utils.usage_logger import log_usage
from utils.pricing import get_pricing
from utils.encryption import decrypt_api_key

router = APIRouter()

class PromptPayload(BaseModel):
    org_id: str
    provider: str
    model: str
    prompt: str
    prompt_id: str

def handle_prompt(payload: PromptPayload):
    print(f"[OpenAI Router] Looking up API key for org_id={payload.org_id}, provider={payload.provider}")
    result = supabase.table("api_keys") \
        .select("*") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    print(f"[OpenAI Router] API key query result: {keys}")
    if not keys:
        print("[OpenAI Router] No API key found for org/provider.")
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

    # Decrypt the API key
    try:
        encrypted_api_key = keys[0]["api_key"]
        print("[OpenAI Router] Encrypted API key from DB:", encrypted_api_key)
        api_key = decrypt_api_key(encrypted_api_key)
        print("[OpenAI Router] Decrypted API key:", api_key)
    except Exception as e:
        print("[OpenAI Router] Decryption error:", e)
        raise HTTPException(status_code=500, detail=f"Failed to decrypt API key: {e}")

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI client: {e}")

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": payload.prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model=payload.model,
            messages=messages
        )
        reply = completion.choices[0].message.content
        input_tokens = getattr(completion.usage, "prompt_tokens", 0)
        output_tokens = getattr(completion.usage, "completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        pricing = get_pricing("openai", payload.model)
        cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

        log_usage(
            payload.org_id,
            "OpenAI",
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
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")
