"""
Webhook trigger management: CRUD for webhook triggers + incoming webhook handler.
External services can POST to /api/webhooks/trigger/{endpoint_path} to execute a workflow.
"""
import logging
import hmac
import hashlib
import json
import secrets as secrets_module
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from supabase_client import supabase
from auth_dependency import require_org_member, AuthenticatedUser, verified_org_id
from rate_limiting import check_and_increment_usage
from plan_enforcement import check_monthly_request_limit, increment_monthly_usage

logger = logging.getLogger(__name__)

router = APIRouter()

#: Requests per minute allowed per (org, webhook). Incoming webhooks execute
#: workflows against the org's provider keys, so an unbounded firehose burns
#: the org's money.
WEBHOOK_RATE_LIMIT_PER_MINUTE = 60


# Pydantic models
class WebhookCreate(BaseModel):
    org_id: str
    workflow_id: str
    name: str
    payload_template: str = "{{body}}"
    secret: Optional[str] = None  # If not provided, auto-generate


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    payload_template: Optional[str] = None
    secret: Optional[str] = None
    is_active: Optional[bool] = None


# CRUD Endpoints

@router.get("/webhooks/{org_id}")
async def list_webhooks(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """List all webhook triggers for an org."""
    try:
        result = (
            supabase.table("webhook_triggers")
            .select(
                "id, org_id, workflow_id, name, endpoint_path, secret, payload_template, "
                "is_active, last_triggered_at, trigger_count, created_at, updated_at"
            )
            .eq("org_id", org_id)
            .order("created_at", desc=False)
            .execute()
        )
        raw = result.data or []
        # Build response with masked secrets (don't mutate; Supabase rows may be read-only)
        webhooks = []
        for w in raw:
            row = dict(w) if not isinstance(w, dict) else w.copy()
            secret = row.pop("secret", None)
            if secret and isinstance(secret, str) and len(secret) > 8:
                row["secret_masked"] = secret[:4] + "••••" + secret[-4:]
            else:
                row["secret_masked"] = "••••••••" if secret else None
            webhooks.append(row)
        return webhooks
    except Exception as e:
        logger.error("Failed to list webhooks for org %s: %s", org_id, e)
        raise HTTPException(status_code=500, detail="Failed to list webhooks")


@router.post("/webhooks")
async def create_webhook(
    body: WebhookCreate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new webhook trigger in the caller's verified org."""
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="Webhook name is required")

    org_id = verified_org_id(auth_user)

    # The workflow this webhook fires must belong to the same org — otherwise a
    # webhook in org A could be pointed at org B's workflow.
    wf = (
        supabase.table("workflows")
        .select("id")
        .eq("id", body.workflow_id)
        .eq("org_id", org_id)
        .limit(1)
        .execute()
    )
    if not wf.data:
        raise HTTPException(status_code=404, detail="Workflow not found in this organization")

    # Generate unique endpoint path
    endpoint_path = secrets_module.token_urlsafe(16)

    # Auto-generate secret if not provided
    secret = body.secret if body.secret else secrets_module.token_hex(32)

    try:
        result = (
            supabase.table("webhook_triggers")
            .insert({
                "org_id": org_id,
                "workflow_id": body.workflow_id,
                "name": body.name.strip(),
                "endpoint_path": endpoint_path,
                "secret": secret,
                "payload_template": body.payload_template or "{{body}}",
                "is_active": True,
            })
            .execute()
        )
        row = result.data[0] if result.data else {}
        return {
            "id": row.get("id"),
            "org_id": row.get("org_id"),
            "workflow_id": row.get("workflow_id"),
            "name": row.get("name"),
            "endpoint_path": row.get("endpoint_path"),
            "webhook_url": f"/api/webhooks/trigger/{endpoint_path}",
            "secret": secret,  # Return full secret on creation only
            "payload_template": row.get("payload_template"),
            "is_active": row.get("is_active"),
            "created_at": row.get("created_at"),
        }
    except Exception as e:
        logger.error("Failed to create webhook: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create webhook")


@router.put("/webhooks/{org_id}/{webhook_id}")
async def update_webhook(
    org_id: str,
    webhook_id: str,
    body: WebhookUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update a webhook trigger."""
    patch = {"updated_at": "now()"}
    if body.name is not None:
        patch["name"] = body.name.strip()
    if body.payload_template is not None:
        patch["payload_template"] = body.payload_template
    if body.secret is not None:
        patch["secret"] = body.secret
    if body.is_active is not None:
        patch["is_active"] = body.is_active

    try:
        result = (
            supabase.table("webhook_triggers")
            .update(patch)
            .eq("id", webhook_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Webhook not found")
        row = result.data[0]
        return {
            "id": row.get("id"),
            "org_id": row.get("org_id"),
            "name": row.get("name"),
            "endpoint_path": row.get("endpoint_path"),
            "payload_template": row.get("payload_template"),
            "is_active": row.get("is_active"),
            "updated_at": row.get("updated_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update webhook: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update webhook")


@router.delete("/webhooks/{org_id}/{webhook_id}")
async def delete_webhook(
    org_id: str,
    webhook_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a webhook trigger."""
    try:
        result = (
            supabase.table("webhook_triggers")
            .delete()
            .eq("id", webhook_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete webhook: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete webhook")


# Incoming webhook handler (no auth - uses webhook secret for verification)

@router.post("/webhooks/trigger/{endpoint_path}")
async def trigger_webhook(
    endpoint_path: str,
    request: Request,
):
    """
    Handle incoming webhook. Verifies signature if secret is set,
    extracts input from payload using template, and executes the workflow.
    """
    # Look up webhook trigger
    try:
        result = (
            supabase.table("webhook_triggers")
            .select("*")
            .eq("endpoint_path", endpoint_path)
            .eq("is_active", True)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Webhook not found or inactive")
        trigger = result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to look up webhook: %s", e)
        raise HTTPException(status_code=500, detail="Webhook lookup failed")

    # Read request body
    try:
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
    except Exception:
        body_str = ""

    # Verify signature if a secret is configured.
    #
    # This used to be `if signature:` — omitting the header entirely skipped
    # verification and executed the workflow unauthenticated. When a secret is
    # configured the signature is REQUIRED.
    secret = trigger.get("secret")
    if not secret:
        # A trigger with no secret has no authentication at all. create_webhook
        # always generates one, so this only happens for rows written outside
        # the API. Refuse rather than execute a workflow for an anonymous caller.
        logger.error(
            "Webhook %s has no secret configured — refusing to execute unauthenticated",
            trigger.get("id"),
        )
        raise HTTPException(
            status_code=503,
            detail="This webhook has no signing secret configured. Set one before using it.",
        )
    if secret:
        signature = request.headers.get("x-webhook-signature") or request.headers.get("x-hub-signature-256") or ""
        if not signature:
            logger.warning(
                "Webhook %s rejected: signature header missing but a secret is configured",
                trigger.get("id"),
            )
            raise HTTPException(
                status_code=401,
                detail="Missing webhook signature. Send X-Webhook-Signature (HMAC-SHA256 hex of the raw body).",
            )
        expected = hmac.new(
            secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        # Support "sha256=..." prefix
        sig_value = signature.replace("sha256=", "").strip()
        if not hmac.compare_digest(expected, sig_value):
            logger.warning("Webhook %s rejected: invalid signature", trigger.get("id"))
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Parse JSON body
    try:
        body_json = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        body_json = {"raw": body_str}

    # Apply payload template to extract workflow input
    template = trigger.get("payload_template") or "{{body}}"
    if template == "{{body}}":
        input_text = json.dumps(body_json) if isinstance(body_json, dict) else str(body_json)
    elif "{{" in template:
        # Simple template replacement
        input_text = template
        input_text = input_text.replace("{{body}}", json.dumps(body_json) if isinstance(body_json, dict) else str(body_json))
        # Replace {{body.key}} patterns
        if isinstance(body_json, dict):
            for key, val in body_json.items():
                input_text = input_text.replace(f"{{{{body.{key}}}}}", str(val))
    else:
        input_text = template

    # Execute the workflow
    org_id = trigger.get("org_id")
    workflow_id = trigger.get("workflow_id")

    # Rate limit + monthly quota. This path spends the org's provider tokens,
    # so it gets the same accounting as the public execution endpoint.
    check_and_increment_usage(
        org_id=org_id,
        endpoint_slug=f"webhook:{endpoint_path}",
        rate_limit_per_minute=WEBHOOK_RATE_LIMIT_PER_MINUTE,
    )
    check_monthly_request_limit(org_id)

    try:
        # Fetch workflow graph — re-filtered by the trigger's own org so a
        # webhook can never execute another tenant's workflow.
        wf_result = (
            supabase.table("workflows")
            .select("graph_json, variables")
            .eq("id", workflow_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not wf_result.data:
            raise HTTPException(status_code=404, detail="Workflow not found")

        wf = wf_result.data[0]
        graph = wf.get("graph_json") or {}
        if isinstance(graph, str):
            graph = json.loads(graph)

        from workflow_runtime import execute_workflow
        result = execute_workflow(
            graph=graph,
            input_text=input_text,
            org_id=org_id,
            user_id="webhook",
            workflow_id=workflow_id,
            execution_mode="production",
        )

        increment_monthly_usage(org_id)

        # Update trigger stats
        try:
            supabase.table("webhook_triggers").update({
                "last_triggered_at": "now()",
                "trigger_count": trigger.get("trigger_count", 0) + 1,
            }).eq("id", trigger["id"]).execute()
        except Exception:
            pass  # Non-critical

        return {
            "status": "success",
            "output": result.get("final_output"),
            "total_cost": result.get("total_cost"),
            "total_latency_ms": result.get("total_latency"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Webhook execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
