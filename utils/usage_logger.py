from supabase_client import supabase
from utils.pricing import get_pricing, suggest_model

def log_usage(user_id, provider, model, prompt, response,
              input_tokens=None, output_tokens=None, total_tokens=None, cost_usd=None,
              project_id=None, org_id=None, prompt_id=None):
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
        "project_id": project_id,
        "org_id": org_id,
        "prompt_id": prompt_id,
    }
    
    # Insert usage log
    result = supabase.table("usage_logs").insert(data).execute()
    
    # Update optimizer recommendations if we have project_id, org_id, and prompt_id
    if project_id and org_id and prompt_id:
        update_optimizer_recommendations(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd)

def update_optimizer_recommendations(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd):
    """
    Update optimizer recommendations based on actual usage data, handling both static and dynamic prompts
    """
    try:
        # Get the prompt template to check if it's dynamic
        prompt_result = supabase.table("prompt_templates") \
            .select("is_dynamic") \
            .eq("id", prompt_id) \
            .single() \
            .execute()
        
        if not prompt_result.data:
            print(f"[Usage Logger] Prompt template {prompt_id} not found, skipping optimization")
            return
            
        is_dynamic = prompt_result.data.get("is_dynamic", False)
        
        if is_dynamic:
            # For dynamic prompts, check if we should update monthly
            update_dynamic_prompt_recommendation(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd)
        else:
            # For static prompts, only update if prompt text changed
            update_static_prompt_recommendation(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd)
        
    except Exception as e:
        print(f"[Usage Logger] Error updating optimizer recommendations: {e}")
        # Don't fail the main usage logging if optimization update fails

def update_static_prompt_recommendation(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd):
    """
    Update optimizer recommendations for static prompts - only if prompt text changed
    """
    # Get the latest recommendation for this prompt_id
    rec_result = supabase.table("optimizer_recommendations") \
        .select("full_prompt_text") \
        .eq("prompt_id", prompt_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    
    if rec_result.data and rec_result.data[0]["full_prompt_text"] == prompt:
        print(f"[Usage Logger] No change in static prompt text for prompt_id {prompt_id}, skipping optimizer recommendation update.")
        return
    
    # Create and insert new recommendation
    create_and_insert_recommendation(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd, "static")

def update_dynamic_prompt_recommendation(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd):
    """
    Update optimizer recommendations for dynamic prompts - monthly aggregation
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Check if we already have a recommendation for this month
    monthly_rec_result = supabase.table("optimizer_recommendations") \
        .select("id") \
        .eq("prompt_id", prompt_id) \
        .gte("created_at", start_of_month.isoformat()) \
        .execute()
    
    if monthly_rec_result.data:
        print(f"[Usage Logger] Monthly recommendation already exists for dynamic prompt {prompt_id}, skipping update.")
        return
    
    # For dynamic prompts, aggregate data from the current month
    monthly_usage_result = supabase.table("usage_logs") \
        .select("input_tokens, output_tokens, cost_usd, prompt") \
        .eq("project_id", project_id) \
        .eq("prompt_id", prompt_id) \
        .gte("created_at", start_of_month.isoformat()) \
        .execute()
    
    if not monthly_usage_result.data:
        print(f"[Usage Logger] No monthly usage data found for dynamic prompt {prompt_id}")
        return
    
    # Calculate averages
    total_requests = len(monthly_usage_result.data)
    avg_input_tokens = sum(log.get("input_tokens", 0) for log in monthly_usage_result.data) / total_requests
    avg_output_tokens = sum(log.get("output_tokens", 0) for log in monthly_usage_result.data) / total_requests
    avg_cost = sum(log.get("cost_usd", 0) for log in monthly_usage_result.data) / total_requests
    
    # Use the most recent prompt as representative
    representative_prompt = monthly_usage_result.data[-1].get("prompt", prompt)
    
    # Create and insert monthly recommendation
    create_and_insert_recommendation(user_id, org_id, project_id, prompt_id, representative_prompt, 
                                   avg_input_tokens, avg_output_tokens, avg_cost, "dynamic")

def create_and_insert_recommendation(user_id, org_id, project_id, prompt_id, prompt, input_tokens, output_tokens, cost_usd, prompt_type):
    """
    Create and insert a new optimizer recommendation
    """
    # Get the project's monthly budget
    project_result = supabase.table("projects").select("monthly_budget").eq("id", project_id).single().execute()
    monthly_budget = project_result.data.get("monthly_budget", 0) if project_result.data else 0
    
    # Get current month's usage for this project
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    usage_result = supabase.table("usage_logs") \
        .select("cost_usd") \
        .eq("project_id", project_id) \
        .gte("created_at", start_of_month.isoformat()) \
        .execute()
    
    budget_used = sum(log.get("cost_usd", 0) for log in usage_result.data) if usage_result.data else 0
    
    # Get optimization suggestion
    suggestion = suggest_model(prompt)
    recommended_provider = suggestion.get("provider", "openai")
    recommended_model = suggestion.get("model", "gpt-3.5-turbo")
    
    # Calculate estimated cost for recommended model
    pricing = get_pricing(recommended_provider, recommended_model)
    estimated_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
    
    # Create recommendation record
    recommendation_data = {
        "prompt_id": prompt_id,
        "project_id": project_id,
        "org_id": org_id,
        "user_id": user_id,
        "recommended_provider": recommended_provider,
        "recommended_model": recommended_model,
        "estimated_cost_usd": estimated_cost,
        "estimated_input_tokens": input_tokens or 0,
        "estimated_output_tokens": output_tokens or 0,
        "full_prompt_text": prompt,
        "budget_used_usd": budget_used,
        "monthly_budget_usd": monthly_budget,
        "reasoning": f"Based on actual usage data ({prompt_type} prompt). Current cost: ${cost_usd:.6f}, Recommended cost: ${estimated_cost:.6f}. Potential savings: ${(cost_usd - estimated_cost):.6f} per request."
    }
    
    # Insert new recommendation
    supabase.table("optimizer_recommendations").insert(recommendation_data).execute()
    
    print(f"[Usage Logger] Updated optimizer recommendations for {prompt_type} prompt {prompt_id}")
