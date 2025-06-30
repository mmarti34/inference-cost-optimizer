from supabase_client import supabase

def log_usage(user_id, provider, model, prompt, response,
              input_tokens=None, output_tokens=None, total_tokens=None, cost_usd=None):
    data = {
        "user_id": user_id,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }
    supabase.table("usage_logs").insert(data).execute()
