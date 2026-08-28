from fastapi import FastAPI, HTTPException, Header, Request, Body, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from supabase_client import supabase
from auth_dependency import require_auth, require_org_member, AuthenticatedUser, verified_org_id
from utils.encryption import (
    encrypt_api_key,
    decrypt_api_key,
    validate_encryption_key_at_startup,
    hash_service_api_key,
    verify_service_api_key,
)
import os
import secrets
import logging
from utils.pricing import get_pricing, suggest_model
import time
from datetime import datetime, timezone

from routers import (
    openai_router,
    anthropic_router,
    mistral_router,
    cohere_router,
    gemini_router,
    groq_router,
    together_router,
    deepseek_router,
    fireworks_router,
)
from routers.base import Outcome, FailureReason
from org_access_control import router as org_access_router
from stripe_webhook import router as stripe_webhook_router
from prompt_management import router as prompt_management_router
from project_management import router as project_management_router
from user_management import router as user_management_router
from api_key_management import router as api_key_management_router
from organization_management import router as organization_management_router
from usage_management import router as usage_management_router
from workflow_management import router as workflow_management_router
from custom_model_management import router as custom_model_router
from model_registry_management import router as model_registry_router
from alert_management import router as alert_management_router
from secrets_management import router as secrets_management_router
from context_asset_management import router as context_asset_management_router
from hitl_management import router as hitl_management_router
from webhook_management import router as webhook_management_router
from routers.public_execution import router as public_execution_router
from routers.openai_compat import router as openai_compat_router
from parse_import import router as parse_import_router
from cursor_tokens import router as cursor_tokens_router
from cursor_deploy import router as cursor_deploy_router
from workflow_draft_generator import router as workflow_draft_generator_router
from github_migration import router as github_migration_router
from synthetic_mind.router import router as synthetic_mind_router
from routers.optimization_router import router as optimization_router
from routers.optimization_router import public_router as optimization_public_router
from middleware.observability_middleware import ObservabilityMiddleware
from utils.observability import log_request, log_span, generate_request_id, generate_trace_id

# Model definitions
class APIKeyPayload(BaseModel):
    user_id: Optional[str] = None
    org_id: str
    provider: str
    api_key: str
    name: Optional[str] = None

class PromptPayload(BaseModel):
    user_id: str
    provider: str
    model: str
    prompt: str
    prompt_id: str
    org_id: str | None = None
    project_id: str | None = None

class OptimizePayload(BaseModel):
    prompt_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    project_id: str
    org_id: str
    user_id: str

class DeleteKeyPayload(BaseModel):
    org_id: str
    provider: str

app = FastAPI()
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_validate():
    """Fail fast if encryption key is missing or invalid."""
    try:
        validate_encryption_key_at_startup()
    except ValueError as e:
        logger.critical("Startup validation failed: %s", e)
        raise

# Observability middleware (adds request_id and trace_id)
app.add_middleware(ObservabilityMiddleware)

# CORS setup — must include frontend origin (optiml.one) or browser blocks API calls
_default_origins = [
    "https://optiml.one",
    "https://www.optiml.one",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3020",
]
_ALLOWED_ORIGINS = os.environ.get("OPTIML_CORS_ORIGINS", "").strip()
if _ALLOWED_ORIGINS:
    _ALLOWED_ORIGINS = [o.strip() for o in _ALLOWED_ORIGINS.split(",") if o.strip()]
else:
    _ALLOWED_ORIGINS = _default_origins
logger.info("CORS allow_origins: %s", _ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount routers
app.include_router(openai_router.router, prefix="/openai")
app.include_router(anthropic_router.router, prefix="/anthropic")
app.include_router(mistral_router.router, prefix="/mistral")
app.include_router(cohere_router.router, prefix="/cohere")
app.include_router(gemini_router.router, prefix="/gemini")
app.include_router(org_access_router, prefix="/org-access")
app.include_router(stripe_webhook_router)
app.include_router(prompt_management_router, prefix="/api")
app.include_router(project_management_router, prefix="/api")
app.include_router(user_management_router, prefix="/api")
app.include_router(api_key_management_router, prefix="/api")
app.include_router(organization_management_router, prefix="/api")
app.include_router(usage_management_router, prefix="/api")
app.include_router(workflow_management_router, prefix="/api")
app.include_router(parse_import_router, prefix="/api")
app.include_router(cursor_tokens_router)
app.include_router(cursor_deploy_router)
app.include_router(public_execution_router, prefix="/api/public", tags=["public"])
app.include_router(openai_compat_router)
app.include_router(custom_model_router, prefix="/api")
app.include_router(model_registry_router, prefix="/api")
app.include_router(alert_management_router, prefix="/api")
app.include_router(secrets_management_router, prefix="/api")
app.include_router(hitl_management_router, prefix="/api")
app.include_router(webhook_management_router, prefix="/api")
app.include_router(context_asset_management_router, prefix="/api")
app.include_router(workflow_draft_generator_router, prefix="/api")
app.include_router(github_migration_router, prefix="/api")
app.include_router(synthetic_mind_router, prefix="/api")
app.include_router(optimization_router, prefix="/api")
# Customer-facing outcome ingestion alias: POST /v1/outcomes
app.include_router(optimization_public_router)

# Control loop: runs the rollback monitor and experiment auto-conclude on a timer,
# and exposes machine-callable POST /api/control-loop/run for external cron.
# Without this, automatic rollback never fires. See background_jobs.py.
from background_jobs import register_background_jobs  # noqa: E402
register_background_jobs(app)

# Health check endpoint for Railway
@app.get("/health")
def health_check():
    try:
        # Simple health check that doesn't depend on external services
        return Response(
            content='{"status": "healthy", "message": "API is running", "timestamp": "' + str(__import__("datetime").datetime.now()) + '"}',
            media_type="application/json"
        )
    except Exception as e:
        # Even if there's an error, return a basic health response
        return Response(
            content='{"status": "healthy", "message": "API is running (basic mode)"}',
            media_type="application/json"
        )

@app.post("/optimize")
def optimize_prompt(payload: OptimizePayload, _user: AuthenticatedUser = Depends(require_org_member)):
    """
    Optimize prompt by analyzing recent usage and budget constraints
    """
    # CROSS-TENANT GUARD. require_org_member proves the caller belongs to the
    # org they NAMED in the body — it does not constrain prompt_id or
    # project_id, which are separate caller-supplied identifiers. Every query
    # below must therefore be re-filtered by the verified org, exactly as
    # verified_org_id's own docstring instructs.
    #
    # Without this, a member of any org could pass another tenant's prompt_id
    # and have that prompt's TEXT read into the system prompt below and
    # described back in the recommendation's `reasoning`. supabase_client uses
    # the service-role key, so RLS does not backstop it.
    org_id = verified_org_id(_user)

    try:
        # 1. Get the two most recent prompts for this prompt_id
        prompt_result = supabase.table("prompt_templates") \
            .select("id, org_id, prompt, provider, model, name, project_id, created_at") \
            .eq("id", payload.prompt_id) \
            .eq("org_id", org_id) \
            .order("created_at", desc=True) \
            .limit(2) \
            .execute()
        
        if not prompt_result.data:
            raise HTTPException(status_code=404, detail="Prompt template not found.")
        
        prompts = prompt_result.data
        
        # 2. Check if the most recent prompt differs from the previous one
        if len(prompts) >= 2:
            latest_prompt = prompts[0]["prompt"]
            previous_prompt = prompts[1]["prompt"]
            
            if latest_prompt == previous_prompt:
                return {
                    "status": "unchanged",
                    "message": "Prompt has not changed. Skipping optimizer call."
                }
        
        # 3. Fetch the project's monthly_budget_usd
        project_result = supabase.table("projects") \
            .select("monthly_budget") \
            .eq("id", payload.project_id) \
            .eq("org_id", org_id) \
            .limit(1) \
            .execute()

        # Same 404 whether the project belongs to another tenant or does not
        # exist: a foreign project's existence must not be confirmable, and
        # .single() would have raised on a miss rather than returning cleanly.
        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found.")
        project_row = project_result.data[0]
        
        monthly_budget_usd = project_row.get("monthly_budget", 0.0)
        
        # 4. Calculate how much budget the project has used so far this month
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        usage_result = supabase.table("usage_logs") \
            .select("cost_usd") \
            .eq("project_id", payload.project_id) \
            .eq("org_id", org_id) \
            .gte("created_at", start_of_month.isoformat()) \
            .execute()
        
        budget_used_usd = sum(log.get("cost_usd", 0) for log in usage_result.data or [])
        
        # 5. Get the current prompt text
        current_prompt = prompts[0]["prompt"]
        
        # 6. Call OpenAI for optimization recommendation
        import os
        from openai import OpenAI
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = f"""
You are an AI cost optimization expert. Analyze the following prompt and provide a recommendation for the best provider/model combination that balances cost, latency, and quality while staying within budget constraints.

CURRENT PROMPT:
{current_prompt}

ESTIMATED TOKENS:
- Input tokens: {payload.estimated_input_tokens}
- Output tokens: {payload.estimated_output_tokens}

BUDGET INFORMATION:
- Monthly budget: ${monthly_budget_usd}
- Budget used this month: ${budget_used_usd}
- Budget remaining: ${monthly_budget_usd - budget_used_usd}

AVAILABLE PROVIDERS AND MODELS:
- OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo
- Anthropic: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
- Mistral: mistral-large-latest, mistral-medium-latest, mistral-small-latest
- Cohere: command, command-light
- Gemini: gemini-pro, gemini-flash

Consider the following factors:
1. Cost efficiency (tokens per dollar)
2. Quality requirements for the task
3. Latency requirements
4. Budget constraints
5. Model capabilities for the specific prompt type

Return your recommendation as a JSON object with the following structure:
{{
  "recommended_provider": "provider_name",
  "recommended_model": "model_name", 
  "estimated_cost_usd": 0.00,
  "reasoning": "detailed explanation of your recommendation"
}}
"""
        
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please provide a cost optimization recommendation for this prompt."}
            ],
            response_format={"type": "json_object"}
        )
        
        recommendation = completion.choices[0].message.content
        import json
        if recommendation:
            recommendation_data = json.loads(recommendation)
        else:
            raise HTTPException(status_code=500, detail="Failed to get recommendation from OpenAI")
        
        # 7. Insert recommendation into optimizer_recommendations table
        recommendation_record = {
            "prompt_id": payload.prompt_id,
            "project_id": payload.project_id,
            # The VERIFIED org, not the body's. `full_prompt_text` below
            # persists the prompt, so a body-supplied org_id here would have
            # written the row — and the prompt text — under an org the caller
            # merely claimed.
            "org_id": org_id,
            "user_id": payload.user_id,
            "recommended_provider": recommendation_data["recommended_provider"],
            "recommended_model": recommendation_data["recommended_model"],
            "estimated_cost_usd": recommendation_data["estimated_cost_usd"],
            "estimated_input_tokens": payload.estimated_input_tokens,
            "estimated_output_tokens": payload.estimated_output_tokens,
            "full_prompt_text": current_prompt,
            "budget_used_usd": budget_used_usd,
            "monthly_budget_usd": monthly_budget_usd,
            "reasoning": recommendation_data["reasoning"]
        }
        
        supabase.table("optimizer_recommendations").insert(recommendation_record).execute()
        
        return {
            "status": "success",
            "recommendation": recommendation_data
        }
        
    except HTTPException:
        # Re-raise deliberate status codes. Without this the 404 above was DEAD
        # CODE: the bare `except Exception` swallowed it and answered 500 with
        # the detail interpolated into the message. Same dead-404 pattern as
        # /v1/prompt. A 500 is both wrong and noisier than the 404 it replaced.
        raise
    except Exception as e:
        # Do not echo the exception to the caller: it can carry row contents and
        # identifiers from whatever query failed. Operator-side log keeps it.
        logger.exception("Error in optimize endpoint: %s", e)
        raise HTTPException(status_code=500, detail="Optimization failed.")

# REMOVED: `GET /get-keys/{org_id}`.
# It returned every provider API key for an org fully DECRYPTED to any org
# member — a single request exfiltrated the org's OpenAI/Anthropic/etc.
# credentials in plaintext over HTTP. Provider keys are decrypted server-side
# at call time only. The frontend never used this route; it reads the masked
# list from `GET /api/api-keys/{org_id}` (api_key_management.get_api_keys).

@app.post("/store-key")
def store_api_key(payload: APIKeyPayload, _user: AuthenticatedUser = Depends(require_org_member)):
    try:
        # Encrypt the API key before storing
        encrypted_api_key = encrypt_api_key(payload.api_key)
        
        data = {
            "org_id": payload.org_id,
            "provider": payload.provider,
            "api_key": encrypted_api_key,
        }
        # Add optional fields if provided
        if payload.user_id:
            data["user_id"] = payload.user_id
        if payload.name:
            data["name"] = payload.name
        
        # First check if a key for this org/provider already exists
        existing = supabase.table("api_keys").select("id, org_id, provider, api_key, name, user_id, created_at").eq("org_id", payload.org_id).eq("provider", payload.provider).execute()
        
        if existing.data:
            # Update existing key
            result = supabase.table("api_keys").update({"api_key": encrypted_api_key}).eq("org_id", payload.org_id).eq("provider", payload.provider).execute()
            # Invalidate cached decrypted key so next request picks up the new one
            from api_key_cache import invalidate_provider_key
            invalidate_provider_key(payload.org_id, payload.provider)
            return {"status": "success", "updated": result.data}
        else:
            # Insert new key
            result = supabase.table("api_keys").insert(data).execute()
            return {"status": "success", "inserted": result.data}
            
    except Exception as e:
        print(f"Error storing API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing API key: {str(e)}")

@app.post("/route-prompt")
def route_prompt(payload: PromptPayload, _user: AuthenticatedUser = Depends(require_org_member)):
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
    elif provider == "groq":
        return groq_router.handle_prompt(payload)
    elif provider == "together":
        return together_router.handle_prompt(payload)
    elif provider == "deepseek":
        return deepseek_router.handle_prompt(payload)
    elif provider == "fireworks":
        return fireworks_router.handle_prompt(payload)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")

@app.post("/test-prompt")
def test_prompt(payload: PromptPayload, _user: AuthenticatedUser = Depends(require_org_member)):
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
        .select("id, org_id, provider, api_key, name, user_id, created_at") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()

    keys = result.data
    if not keys:
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

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
        .select("id, org_id, provider, api_key, name, user_id, created_at") \
        .eq("org_id", payload.org_id) \
        .eq("provider", payload.provider) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="API key not found for org/provider.")

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
    # Security: Never log authorization headers or sensitive payload data
    # Only log non-sensitive request metadata if needed for debugging
    
    # Observability: Track request start
    import time
    from datetime import datetime, timezone
    from utils.observability_integration import (
        log_request_with_observability,
        log_gateway_span,
        log_provider_call_span,
    )
    from routers.base import Outcome, FailureReason
    
    request_start_time = time.time()
    gateway_start = datetime.now(timezone.utc)
    log_gateway_span(request, gateway_start)
    
    try:
        # 1. Authenticate API key — reuse the shared validate_api_key function
        # which checks status=active, uses hash-then-query, and updates last_used_at.
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
        bearer_token = authorization.split(" ", 1)[1]
        from api_key_validation import validate_api_key as _validate_key
        from rate_limiting import check_and_increment_usage
        api_ctx = _validate_key(bearer_token)
        org_id = api_ctx.org_id
        check_and_increment_usage(org_id, "_v1_prompt", api_ctx.rate_limit_per_minute)
        logger.info("Authenticated org for inference: %s", org_id)

        # 2. Check subscription-based access control
        org_result = supabase.table("organizations").select("id, name, type, plan, created_by, logo, created_at").eq("id", org_id).single().execute()
        if not org_result.data:
            raise HTTPException(status_code=404, detail="Organization not found.")

        org = org_result.data
        org_type = org.get("type", "Organization")
        
        # Get the admin's subscription plan
        admin_result = supabase.table("organization_members").select("user_id").eq("org_id", org_id).eq("role", "admin").eq("status", "active").single().execute()
        if not admin_result.data:
            print("Organization admin not found")
            raise HTTPException(status_code=404, detail="Organization admin not found.")
        
        admin_user_id = admin_result.data["user_id"]
        admin_profile_result = supabase.table("user_profiles").select("subscription_tier, subscription_status").eq("user_id", admin_user_id).single().execute()
        admin_plan = "free"
        subscription_status = "active"
        if admin_profile_result.data:
            admin_plan = admin_profile_result.data.get("subscription_tier") or "free"
            subscription_status = (admin_profile_result.data.get("subscription_status") or "active").lower()

        # Teams (non-Personal) require active Stripe subscription
        if org_type != "Personal" and subscription_status not in ("active", "trialing"):
            raise HTTPException(
                status_code=402,
                detail="Subscription required. This team's subscription is inactive or past due. Please update billing to continue.",
            )

        # Check access based on admin's plan and organization type
        can_access = True
        access_message = "Access granted"

        if admin_plan == "free" and org_type != "Personal":
            can_access = False
            access_message = "API key access denied. This organization's admin is on a Free plan. Upgrade to use team API keys."
        
        # Check user limits for organizations (not personal workspace)
        if org_type != "Personal" and can_access:
            # Get current member count
            members_result = supabase.table("organization_members").select("id").eq("org_id", org_id).eq("status", "active").execute()
            current_member_count = len(members_result.data) if members_result.data else 0
            
            # Define plan limits
            plan_limits = {
                "free": 1,
                "starter": 3,
                "team": 20,
                "pro": 99999,  # unlimited
                "enterprise": 99999  # unlimited
            }
            
            max_members = plan_limits.get(admin_plan, 1)
            print(f"Max members for {admin_plan} plan: {max_members}")
            
            if current_member_count > max_members:
                can_access = False
                access_message = f"API key access denied. This organization has {current_member_count} members but the {admin_plan} plan only allows {max_members} members. Please upgrade your subscription or remove members."
        
        if not can_access:
            print(f"Access denied: {access_message}")
            raise HTTPException(status_code=403, detail=access_message)
        
        print(f"Access granted: {access_message}")

        # 3. Get prompt_id and input
        prompt_id = payload.get("prompt_id")
        user_input = payload.get("input")
        print("Prompt ID:", prompt_id, "User input:", user_input)
        
        # Handle case where no prompt_id is provided (direct API call)
        if not prompt_id:
            # Direct API call without prompt template
            provider = payload.get("provider", "openai").lower()
            model = payload.get("model", "gpt-3.5-turbo")
            prompt_text = user_input
            
            if not user_input:
                print("Missing input")
                raise HTTPException(status_code=400, detail="Missing input.")
            
            print(f"Direct API call - provider: {provider}, model: {model}")
            
            # No routing decision is made here: the caller named the provider and
            # model and we execute exactly that. The heuristic BaselineRouter that
            # used to run at this point had its result thrown away one line later
            # (the payload below is built from `provider`/`model`, not from the
            # decision), so all it did was stamp a fictional reason_code onto
            # request_logs. Removed rather than left half-wired.
            
            # Call the correct LLM router directly
            payload_obj = PromptPayload(
                user_id="",  # Not needed for org-level access
                provider=provider,
                model=model,
                prompt=prompt_text,
                prompt_id="direct_call",
                org_id=org_id
            )
            
            provider_start = datetime.now(timezone.utc)
            provider_start_time = time.time()
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
                elif provider == "groq":
                    result = groq_router.handle_prompt(payload_obj)
                elif provider == "together":
                    result = together_router.handle_prompt(payload_obj)
                elif provider == "deepseek":
                    result = deepseek_router.handle_prompt(payload_obj)
                elif provider == "fireworks":
                    result = fireworks_router.handle_prompt(payload_obj)
                else:
                    print(f"Unsupported provider: {provider}")
                    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
                
                # Observability: Log successful provider call
                provider_duration = int((time.time() - provider_start_time) * 1000)
                log_provider_call_span(
                    request, provider, model, provider_start, provider_duration, "success",
                    result.get("input_tokens"), result.get("output_tokens"), result.get("total_tokens"),
                    result.get("cost_usd")
                )
                
                # Observability: Log complete request
                log_request_with_observability(
                    request, org_id, None, None, "direct_call",
                    provider, model, prompt_text, result, request_start_time,
                    Outcome.SUCCESS
                )
                
            except HTTPException as e:
                # Observability: Log failed provider call
                provider_duration = int((time.time() - provider_start_time) * 1000)
                log_provider_call_span(
                    request, provider, model, provider_start, provider_duration, "failure",
                    error_message=str(e.detail)
                )
                # Log failed request
                log_request_with_observability(
                    request, org_id, None, None, "direct_call",
                    provider, model, prompt_text, {}, request_start_time,
                    Outcome.FAILURE, FailureReason.UNKNOWN
                )
                raise
            except Exception as e:
                print(f"Exception in LLM router: {e}")
                # Observability: Log failed provider call
                provider_duration = int((time.time() - provider_start_time) * 1000)
                log_provider_call_span(
                    request, provider, model, provider_start, provider_duration, "failure",
                    error_message=str(e)
                )
                # Log failed request
                log_request_with_observability(
                    request, org_id, None, None, "direct_call",
                    provider, model, prompt_text, {}, request_start_time,
                    Outcome.FAILURE, FailureReason.UNKNOWN
                )
                raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")
            
            return {
                "status": result.get("status", "success"),
                "response": result.get("response"),
                "provider": provider,
                "model": model,
                "prompt_id": None
            }
        
        if not user_input:
            print("Missing input")
            raise HTTPException(status_code=400, detail="Missing input.")

        # 3. Lookup prompt template.
        #
        # TENANT OPACITY — same rule as POST /v1/outcomes and
        # POST /api/public/{org_slug}/{endpoint_slug}. This is a server-API-key
        # surface taking a caller-supplied prompt_id, and it used to answer
        #   403 "You do not have access to this prompt."  -> the id EXISTS, in
        #                                                    another org
        #   500 "Error accessing prompt template."        -> the id exists
        #                                                    nowhere (`.single()`
        #                                                    raised on 0 rows and
        #                                                    the except below
        #                                                    swallowed the 404)
        # which let any keyholder test whether a prompt_template id belonged to
        # another tenant. Both now return one 404, byte-identical, echoing
        # nothing the caller supplied. The specific reason goes to the operator
        # log only.
        #
        # The lookup itself is unchanged (one query, same columns) so neither
        # branch does more work than the other; a foreign row is read exactly as
        # before and is now discarded unread instead of being described back to
        # the caller.
        _PROMPT_NOT_FOUND_DETAIL = "Prompt template not found."
        try:
            prompt_result = supabase.table("prompt_templates").select("id, org_id, prompt, provider, model, name, project_id, created_at").eq("id", prompt_id).limit(1).execute()
            rows = prompt_result.data or []
            prompt_template = rows[0] if rows else None
        except Exception as e:
            logger.warning("Error looking up prompt template: %s", e)
            raise HTTPException(status_code=500, detail="Error accessing prompt template.")

        # Uniform to the caller, specific to us.
        prompt_org_id = (prompt_template or {}).get("org_id")
        if prompt_template is None:
            logger.warning("prompt_id %s: no such prompt template", prompt_id)
            raise HTTPException(status_code=404, detail=_PROMPT_NOT_FOUND_DETAIL)
        if not prompt_org_id:
            logger.warning("prompt_id %s: template has no org_id; refusing", prompt_id)
            raise HTTPException(status_code=404, detail=_PROMPT_NOT_FOUND_DETAIL)
        if str(prompt_org_id) != str(org_id):
            logger.warning(
                "prompt_id %s belongs to org %s, not the key's org %s; refusing",
                prompt_id, prompt_org_id, org_id,
            )
            raise HTTPException(status_code=404, detail=_PROMPT_NOT_FOUND_DETAIL)

        # 4. Prepare prompt
        prompt_text = prompt_template["prompt"].replace("{input}", user_input) if prompt_template["prompt"] else user_input
        provider = prompt_template["provider"].lower()
        model = prompt_template["model"]
        project_id = prompt_template.get("project_id")
        # Never log prompt content: these logs are shipped off-box and prompts
        # routinely carry customer data. Length only, plus the full text behind
        # an explicit debug opt-in.
        print(f"Using provider: {provider}, model: {model}, prompt_chars: {len(prompt_text or '')}")
        if _debug_logging_enabled():
            print(f"[OPTIML_DEBUG] prompt_text: {prompt_text}")

        # No routing decision is made here either: provider/model come from the
        # prompt template and the payload below is built from them directly.
        # See the note on the direct-call path above.

        # 5. Call the correct LLM router
        payload_obj = PromptPayload(
            user_id="",  # Not needed for org-level access
            provider=provider,
            model=model,
            prompt=prompt_text,
            prompt_id=prompt_id,
            org_id=org_id
        )
        print("Calling LLM router with:", payload_obj)
        
        provider_start = datetime.now(timezone.utc)
        provider_start_time = time.time()
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
            elif provider == "groq":
                result = groq_router.handle_prompt(payload_obj)
            elif provider == "together":
                result = together_router.handle_prompt(payload_obj)
            elif provider == "deepseek":
                result = deepseek_router.handle_prompt(payload_obj)
            elif provider == "fireworks":
                result = fireworks_router.handle_prompt(payload_obj)
            else:
                print(f"Unsupported provider: {provider}")
                raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
            
            # Observability: Log successful provider call
            provider_duration = int((time.time() - provider_start_time) * 1000)
            log_provider_call_span(
                request, provider, model, provider_start, provider_duration, "success",
                result.get("input_tokens"), result.get("output_tokens"), result.get("total_tokens"),
                result.get("cost_usd")
            )
            
            # Observability: Log complete request
            log_request_with_observability(
                request, org_id, project_id, None, prompt_id,
                provider, model, prompt_text, result, request_start_time,
                Outcome.SUCCESS
            )
            
        except HTTPException as e:
            # Observability: Log failed provider call
            provider_duration = int((time.time() - provider_start_time) * 1000)
            log_provider_call_span(
                request, provider, model, provider_start, provider_duration, "failure",
                error_message=str(e.detail)
            )
            # Log failed request
            log_request_with_observability(
                request, org_id, project_id, None, prompt_id,
                provider, model, prompt_text, {}, request_start_time,
                Outcome.FAILURE, FailureReason.UNKNOWN
            )
            raise
        except Exception as e:
            print(f"Exception in LLM router: {e}")
            # Observability: Log failed provider call
            provider_duration = int((time.time() - provider_start_time) * 1000)
            log_provider_call_span(
                request, provider, model, provider_start, provider_duration, "failure",
                error_message=str(e)
            )
            # Log failed request
            log_request_with_observability(
                request, org_id, project_id, None, prompt_id,
                provider, model, prompt_text, {}, request_start_time,
                Outcome.FAILURE, FailureReason.UNKNOWN
            )
            raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")

        # Model responses can echo the prompt and its data — same rule as above.
        if _debug_logging_enabled():
            print("[OPTIML_DEBUG] LLM router result:", result)
        else:
            print(f"LLM router result: status={result.get('status', 'success')} response_chars={len(str(result.get('response') or ''))}")
        # 6. Return response in standard format
        return {
            "status": result.get("status", "success"),
            "response": result.get("response"),
            "provider": provider,
            "model": model,
            "prompt_id": prompt_id
        }
        
    except HTTPException as e:
        # Try to log the error if we have enough context
        try:
            org_id = locals().get('org_id')
            if org_id:
                from utils.observability_integration import log_request_with_observability
                from routers.base import Outcome, FailureReason
                log_request_with_observability(
                    request, org_id, None, None, None,
                    "unknown", "unknown", "", {}, request_start_time,
                    Outcome.FAILURE, FailureReason.UNKNOWN
                )
        except Exception:
            pass  # Don't fail if logging fails
        raise
    except Exception as e:
        print(f"Unexpected error in /v1/prompt: {e}")
        # Try to log the error if we have enough context
        try:
            org_id = locals().get('org_id')
            if org_id:
                from utils.observability_integration import log_request_with_observability
                from routers.base import Outcome, FailureReason
                log_request_with_observability(
                    request, org_id, None, None, None,
                    "unknown", "unknown", "", {}, request_start_time,
                    Outcome.FAILURE, FailureReason.UNKNOWN
                )
        except Exception:
            pass  # Don't fail if logging fails
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/get-service-api-key/{org_id}")
def get_service_api_key(org_id: str, _user: AuthenticatedUser = Depends(require_org_member)):
    """Universal key is shown only once at creation. This returns it only for legacy (stored encrypted) keys."""
    result = supabase.table("service_api_keys").select("id, api_key, hashed_key").eq("org_id", org_id).single().execute()
    if not result.data:
        return {"api_key": None}
    row = result.data
    if row.get("hashed_key"):
        return {"api_key": None, "message": "Key was shown only at creation; it is stored as a hash and cannot be retrieved."}
    if row.get("api_key"):
        try:
            return {"api_key": decrypt_api_key(row["api_key"])}
        except Exception:
            return {"api_key": None}
    return {"api_key": None}

@app.post("/generate-service-api-key/{org_id}")
def generate_service_api_key(org_id: str, body: Optional[dict] = Body(None), _user: AuthenticatedUser = Depends(require_org_member)):
    """Create a new server API key. Multiple keys per org allowed. Body: { \"name\": \"Production\" }. Returns plaintext once."""
    name = "Server Key"
    if body and isinstance(body.get("name"), str) and body["name"].strip():
        name = body["name"].strip()[:255]
    plaintext_key = secrets.token_urlsafe(32)
    hashed = hash_service_api_key(plaintext_key)
    insert_payload = {"org_id": org_id, "hashed_key": hashed, "name": name, "status": "active"}
    try:
        supabase.table("service_api_keys").insert(insert_payload).execute()
    except Exception as e:
        if "null value" in str(e).lower() and "api_key" in str(e).lower():
            insert_payload["api_key"] = ""
            supabase.table("service_api_keys").insert(insert_payload).execute()
        elif "column" in str(e).lower() and "name" in str(e).lower():
            insert_payload.pop("name", None)
            insert_payload.pop("status", None)
            supabase.table("service_api_keys").insert(insert_payload).execute()
        else:
            raise
    return {"api_key": plaintext_key}

@app.get("/list-service-api-keys/{org_id}")
def list_service_api_keys(org_id: str, _user: AuthenticatedUser = Depends(require_org_member)):
    result = supabase.table("service_api_keys").select("id, created_at").eq("org_id", org_id).execute()
    keys = result.data or []
    # Don't expose the actual key, just show it's encrypted
    for k in keys:
        k["api_key_masked"] = "***encrypted***"
    return {"keys": keys}

@app.delete("/delete-service-api-key/{key_id}")
def delete_service_api_key(key_id: str, _user: AuthenticatedUser = Depends(require_auth)):
    """
    Revoke one server API key belonging to an org the caller is a member of.

    CROSS-TENANT DESTRUCTIVE WRITE, fixed. This used `require_auth` — any
    authenticated user, with no org membership proven — and deleted from
    `service_api_keys` filtered ONLY by `key_id`. The service-role client
    bypasses RLS, so any signed-in user who knew a key id could revoke ANOTHER
    TENANT'S production key and take their traffic down. `/delete-key` directly
    below has always scoped its delete by org; this one simply did not.

    The guard cannot be `require_org_member`: there is no org in the path and a
    DELETE carries no body, so that dependency would demand an X-Org-Id header
    the existing frontend does not send here. Membership is therefore proven
    against the KEY'S OWN org, which is stricter than a caller-supplied one —
    the caller cannot name the org they are checked against.

    "No such key" and "not your key" return the SAME 404: a key id belonging to
    another tenant must not be confirmable, matching this file's other
    org-scoped surfaces.
    """
    _NOT_FOUND = HTTPException(status_code=404, detail="Service API key not found.")

    try:
        found = (
            supabase.table("service_api_keys")
            .select("id, org_id")
            .eq("id", key_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("service key lookup failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Error revoking service API key.") from exc

    rows = found.data or []
    if not rows:
        raise _NOT_FOUND
    key_org_id = str(rows[0].get("org_id") or "")
    if not key_org_id:
        raise _NOT_FOUND

    # A cursor token is scoped to exactly one org and proves nothing elsewhere.
    cursor_org = getattr(_user, "_cursor_org_id", None)
    if cursor_org:
        if str(cursor_org) != key_org_id:
            logger.warning(
                "cross-tenant service-key revoke refused: cursor token org mismatch"
            )
            raise _NOT_FOUND
    else:
        try:
            member = (
                supabase.table("organization_members")
                .select("role")
                .eq("org_id", key_org_id)
                .eq("user_id", _user.user_id)
                .eq("status", "active")
                .limit(1)
                .execute()
            )
        except Exception as exc:
            logger.warning("membership check failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Error verifying organization access.") from exc
        if not (member.data or []):
            # Same 404 as "no such key" — do not confirm that this id exists.
            logger.warning(
                "cross-tenant service-key revoke refused: user=%s key_org=%s",
                _user.user_id, key_org_id,
            )
            raise _NOT_FOUND

    # Belt and braces: scope the DELETE itself, so even a logic slip above
    # cannot widen it beyond the org the key belongs to.
    supabase.table("service_api_keys").delete() \
        .eq("id", key_id) \
        .eq("org_id", key_org_id) \
        .execute()
    return {"status": "deleted", "id": key_id}

@app.delete("/delete-key")
def delete_api_key(payload: DeleteKeyPayload, _user: AuthenticatedUser = Depends(require_org_member)):
    try:
        # Delete the API key for the org and provider
        result = supabase.table("api_keys") \
            .delete() \
            .eq("org_id", payload.org_id) \
            .eq("provider", payload.provider) \
            .execute()
        # result.data will be a list of deleted rows (or empty if nothing deleted)
        if not result.data:
            raise HTTPException(status_code=404, detail="No API key found to delete.")
        # Invalidate cached decrypted key immediately
        from api_key_cache import invalidate_provider_key
        invalidate_provider_key(payload.org_id, payload.provider)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting API key: {str(e)}")

@app.post("/suggest-model")
def suggest_model_endpoint(data: dict = Body(...), _user: AuthenticatedUser = Depends(require_auth)):
    prompt = data.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    suggestion = suggest_model(prompt)
    return {"suggestion": suggestion}

def _debug_logging_enabled() -> bool:
    """True only when OPTIML_DEBUG=true. Gates logging of prompt/response bodies."""
    return os.environ.get("OPTIML_DEBUG", "false").lower() == "true"


# REMOVED: `GET /debug/api-keys`.
# It had NO authentication dependency. It was gated only by OPTIML_DEBUG=true,
# and a configuration flag is not an authorization boundary: flipping one
# environment variable turned it into an unauthenticated, cross-tenant metadata
# endpoint returning service_api_keys ids and org_ids for EVERY organization,
# plus prompt-template ids, names and org_ids. It selected the `prompt` text
# column into memory as well, though it did not return it.
#
# The ids it handed out were exactly the input DELETE /delete-service-api-key
# accepted, which until this release deleted by key_id alone — an unauthenticated
# listing feeding an authenticated cross-tenant destructive write.
#
# Deleted rather than protected. The diagnostics it offered (is Supabase
# reachable, how many rows exist) are a local/admin script's job, not a route on
# the production API surface. Do not reintroduce it behind an env var.

# ============================================================================
# Observability Endpoints
# ============================================================================

@app.get("/observability/summary")
def get_observability_summary(
    org_id: str,
    _user: AuthenticatedUser = Depends(require_org_member),
    project_id: str = None,
    start_date: str = None,
    end_date: str = None,
    provider: str = None,
    model: str = None,
):
    """
    Get observability summary (daily aggregates) for dashboards.

    org_id is REQUIRED and membership-checked. It used to be an optional filter
    behind require_auth only: any signed-in user could pass someone else's
    org_id, and omitting it entirely returned every organization's daily cost
    and latency aggregates in one response.

    Query parameters:
    - org_id: Organization to report on (required; must be one you belong to)
    - project_id: Filter by project
    - start_date: Start date (YYYY-MM-DD)
    - end_date: End date (YYYY-MM-DD)
    - provider: Filter by provider
    - model: Filter by model
    """
    # Filter by the org the guard actually verified, never the raw query value.
    caller_org_id = verified_org_id(_user)
    try:
        query = supabase.table("model_stats_daily").select("date, org_id, project_id, provider, model, request_count, success_count, failure_count, fallback_count, total_prompt_tokens, total_completion_tokens, total_tokens, total_cost_usd, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms, timeout_count, rate_limit_count, provider_5xx_count, created_at")

        query = query.eq("org_id", caller_org_id)
        if project_id:
            query = query.eq("project_id", project_id)
        if start_date:
            query = query.gte("date", start_date)
        if end_date:
            query = query.lte("date", end_date)
        if provider:
            query = query.eq("provider", provider)
        if model:
            query = query.eq("model", model)
        
        result = query.order("date", desc=True).limit(1000).execute()
        
        return {
            "status": "success",
            "data": result.data or [],
            "count": len(result.data) if result.data else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch observability summary: {str(e)}")


@app.get("/observability/requests")
def get_observability_requests(
    org_id: str,
    _user: AuthenticatedUser = Depends(require_org_member),
    project_id: str = None,
    start_date: str = None,
    end_date: str = None,
    provider: str = None,
    model: str = None,
    outcome: str = None,
    limit: int = 100,
):
    """
    Get recent request logs for one organization.

    org_id is REQUIRED and membership-checked. It used to be an optional filter
    behind require_auth only, so any signed-in user could list another org's
    request logs — and harvest the request_ids that make the trace endpoint
    below exploitable.

    Query parameters:
    - org_id: Organization to list (required; must be one you belong to)
    - project_id: Filter by project
    - start_date: Start date (ISO format)
    - end_date: End date (ISO format)
    - provider: Filter by provider
    - model: Filter by model
    - outcome: Filter by outcome (success, fallback_success, failure, refused)
    - limit: Maximum number of results (default 100, max 1000)
    """
    # Filter by the org the guard actually verified, never the raw query value.
    caller_org_id = verified_org_id(_user)
    try:
        limit = min(limit, 1000)  # Cap at 1000

        cols = "id, request_id, trace_id, org_id, project_id, user_id, prompt_id, route_policy_name, route_policy_version, chosen_provider, chosen_model, reason_code, prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, outcome, failure_reason, created_at"
        query = supabase.table("request_logs").select(cols)

        query = query.eq("org_id", caller_org_id)
        if project_id:
            query = query.eq("project_id", project_id)
        if start_date:
            query = query.gte("created_at", start_date)
        if end_date:
            query = query.lte("created_at", end_date)
        if provider:
            query = query.eq("chosen_provider", provider)
        if model:
            query = query.eq("chosen_model", model)
        if outcome:
            query = query.eq("outcome", outcome)
        
        result = query.order("created_at", desc=True).limit(limit).execute()
        
        return {
            "status": "success",
            "data": result.data or [],
            "count": len(result.data) if result.data else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch request logs: {str(e)}")


@app.get("/observability/requests/{request_id}/trace")
def get_request_trace(
    request_id: str,
    _user: AuthenticatedUser = Depends(require_org_member),
):
    """
    Get full trace for a specific request (request + all spans).

    Was completely unauthenticated: any anonymous caller who had a request_id
    got that request's org_id, user_id, provider, model, token counts, cost and
    outcome. A UUID is obscurity, not an access control — and the requests list
    endpoint above was handing those UUIDs out to any signed-in user.

    There is no {org_id} in this path, so the guard resolves the caller's org
    from the X-Org-Id header (which is what the frontend already sends) and the
    request row is re-filtered by that same verified org. A request belonging to
    another tenant reads as 404, not 403, so this cannot be used to probe which
    request_ids exist.
    """
    caller_org_id = verified_org_id(_user)
    try:
        # Get request log — scoped to the caller's verified org.
        req_cols = "id, request_id, trace_id, org_id, project_id, user_id, prompt_id, route_policy_name, route_policy_version, chosen_provider, chosen_model, reason_code, prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, outcome, failure_reason, created_at"
        request_result = supabase.table("request_logs") \
            .select(req_cols) \
            .eq("request_id", request_id) \
            .eq("org_id", caller_org_id) \
            .limit(1) \
            .execute()

        rows = request_result.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Request not found")
        request_row = rows[0]

        # Defence in depth: never serve a row whose org_id is missing or does
        # not match, even if the filter above were ever loosened.
        if str(request_row.get("org_id") or "") != str(caller_org_id):
            raise HTTPException(status_code=404, detail="Request not found")

        # Spans are keyed only by request_id and carry no org_id of their own,
        # so they are fetched only after the parent request has been confirmed
        # to belong to the caller's org.
        span_cols = "id, request_id, trace_id, span_type, provider, model, attempt_number, start_time, end_time, duration_ms, status, error_message, prompt_tokens, completion_tokens, total_tokens, cost_usd, created_at"
        spans_result = supabase.table("request_spans") \
            .select(span_cols) \
            .eq("request_id", request_id) \
            .order("start_time", desc=False) \
            .execute()

        return {
            "status": "success",
            "request": request_row,
            "spans": spans_result.data or []
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch request trace: {str(e)}")


@app.post("/observability/replay")
def replay_request(
    request_id: str,
    provider: str = None,
    model: str = None,
    prompt_text: str = None,
    authorization: str = Header(None),
):
    """
    Not implemented here. Replay/benchmarking lives in the `optimization` package.

    This used to carry ~95 lines of unreachable body below an unconditional
    `raise HTTPException(501)` — including an OPTIML_ENABLE_REPLAY gate that
    could never be evaluated and a log_request() call that wrote a
    reason_code="offline_replay" row with every metric set to None, i.e. a
    fabricated replay record for a provider call that never happened. Deleted
    rather than left to look like a shipped feature.
    """
    raise HTTPException(
        status_code=501,
        detail="Replay is not available on this endpoint; use the optimization replay/benchmark API.",
    )
