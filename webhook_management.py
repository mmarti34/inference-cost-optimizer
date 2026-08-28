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
from resource_access import get_workflow_for_org
import audit

logger = logging.getLogger(__name__)

router = APIRouter()

#: Requests per minute allowed per (org, webhook). Incoming webhooks execute
#: workflows against the org's provider keys, so an unbounded firehose burns
#: the org's money.
WEBHOOK_RATE_LIMIT_PER_MINUTE = 60


# ─── The receiver's pre-authentication response ──────────────────────────────
#
# ENUMERATION, and why there is exactly one failure response below.
#
# ``POST /api/webhooks/trigger/{endpoint_path}`` has no auth dependency by
# design: the sender is an external service and the shared secret IS the
# credential. But the handler used to answer an *unauthenticated* caller three
# different ways depending on what the lookup found:
#
#     unknown path                    -> 404 "Webhook not found or inactive"
#     known path, no signature sent   -> 401 "Missing webhook signature..."
#     known path, no secret on the row-> 503 "This webhook has no signing..."
#
# Any one of those, contrasted with the 404, answers "does this endpoint_path
# exist and is it active?" for an anonymous caller — for EVERY tenant, since the
# lookup is necessarily global. Endpoint paths are frequently meaningful
# (``stripe-customer-acme-foo``), so the oracle leaks customer names and vendor
# relationships, not just row existence.
#
# The fix is not to stop looking the webhook up — the receiver cannot work
# without that — but to make the lookup's RESULT unable to reach the caller
# before the signature verifies. Every pre-authentication outcome now produces
# this one response, byte for byte.
#
# AFTER a valid signature, normal receiver semantics resume and are deliberately
# NOT flattened: a handler 500 stays a 500 so the sender retries, a 2xx stays a
# 2xx. A sender that cannot tell "your handler broke, retry" from "rejected,
# stop" is worse off than before this change.
WEBHOOK_AUTH_FAILED_STATUS = 401
WEBHOOK_AUTH_FAILED_DETAIL = (
    "Invalid or missing webhook signature. Send X-Webhook-Signature "
    "(HMAC-SHA256 hex of the raw request body)."
)

#: A secret generated once per process that no caller can ever hold, used in
#: place of a real one when the path does not exist or the row has no secret.
#: It exists so those cases run the SAME hmac computation and the SAME
#: comparison as a real webhook with a wrong signature, rather than taking a
#: visibly (and measurably) shorter route to a different status code.
#: ``token_hex(32)`` is 256 bits of ``secrets``-grade entropy: no signature a
#: caller can construct will ever verify against it.
_SYNTHETIC_WEBHOOK_SECRET = secrets_module.token_hex(32)


def _webhook_auth_failed() -> HTTPException:
    """The single pre-authentication failure. Same status, detail and headers
    for an unknown path, a bad signature, a missing signature and a webhook
    with no secret configured."""
    return HTTPException(
        status_code=WEBHOOK_AUTH_FAILED_STATUS,
        detail=WEBHOOK_AUTH_FAILED_DETAIL,
    )


def _signature_matches(secret: str, raw_body: bytes, signature: str) -> bool:
    """Constant-time check of an HMAC-SHA256 hex signature over the raw body.

    Runs identically for a real secret and for ``_SYNTHETIC_WEBHOOK_SECRET``,
    and for a missing header (which compares against the empty string) — one
    code path, so the branch taken cannot be read off the response or the clock.
    Compares BYTES: ``hmac.compare_digest`` rejects non-ASCII ``str`` inputs
    with a TypeError, and headers arrive latin-1 decoded.
    """
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    presented = (signature or "").replace("sha256=", "").strip()
    return hmac.compare_digest(expected.encode("utf-8"), presented.encode("utf-8"))


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
    # The guard refuses a request that names two different orgs, so the path
    # value and the verified value are equal by construction — but read the
    # verified one anyway, so the filter is trusted first-hand rather than by
    # an argument about the guard.
    org_id = verified_org_id(auth_user)
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
    request: Request,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new webhook trigger in the caller's verified org."""
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="Webhook name is required")

    org_id = verified_org_id(auth_user)

    # The workflow this webhook fires must belong to the same org — otherwise a
    # webhook in org A could be pointed at org B's workflow. `resource_access`
    # applies the org filter itself and answers "another tenant's workflow" and
    # "no such workflow" with one opaque 404; the old detail here said "not
    # found in this organization", which hinted that it might exist elsewhere.
    try:
        get_workflow_for_org(body.workflow_id, auth_user)
    except HTTPException:
        # Pointing a webhook at a workflow that is not yours is exactly the
        # attempt the audit trail exists to answer questions about.
        audit.record(
            audit.WEBHOOK_CREATE_REFUSED,
            principal=auth_user,
            resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
            metadata={
                "workflow_id": body.workflow_id,
                "reason_code": audit.REASON_NOT_FOUND,
            },
            request=request,
        )
        raise

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
        # A webhook is a customer-controlled egress path into their systems and
        # carries a live signing secret. Never the secret or the endpoint path
        # in the row — only identifiers.
        audit.record(
            audit.WEBHOOK_CREATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
            resource_id=row.get("id"),
            metadata={"workflow_id": body.workflow_id},
            request=request,
        )
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
    request: Request,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update a webhook trigger."""
    org_id = verified_org_id(auth_user)  # see list_webhooks
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
            # The id does not exist, or it belongs to another tenant — one
            # answer for both, and a recorded attempt either way.
            audit.record(
                audit.WEBHOOK_UPDATE_REFUSED,
                principal=auth_user,
                resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
                resource_id=webhook_id,
                metadata={"reason_code": audit.REASON_NOT_FOUND},
                request=request,
            )
            raise HTTPException(status_code=404, detail="Webhook not found")
        row = result.data[0]

        _audit_meta = {}
        if body.is_active is not None:
            _audit_meta["new_status"] = "active" if body.is_active else "inactive"
        audit.record(
            audit.WEBHOOK_UPDATED,
            principal=auth_user,
            resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
            resource_id=row.get("id"),
            metadata=_audit_meta,
            request=request,
        )
        if body.secret is not None:
            # Its own action, because "who last changed this webhook's signing
            # secret, and when" is a question of its own. The VALUE is never
            # passed to the writer — only the fact that it was replaced.
            audit.record(
                audit.WEBHOOK_SECRET_ROTATED,
                principal=auth_user,
                resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
                resource_id=row.get("id"),
                request=request,
            )
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
    request: Request,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete a webhook trigger."""
    org_id = verified_org_id(auth_user)  # see list_webhooks
    try:
        result = (
            supabase.table("webhook_triggers")
            .delete()
            .eq("id", webhook_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            audit.record(
                audit.WEBHOOK_DELETE_REFUSED,
                principal=auth_user,
                resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
                resource_id=webhook_id,
                metadata={"reason_code": audit.REASON_NOT_FOUND},
                request=request,
            )
            raise HTTPException(status_code=404, detail="Webhook not found")
        audit.record(
            audit.WEBHOOK_DELETED,
            principal=auth_user,
            resource_type=audit.RESOURCE_WEBHOOK_TRIGGER,
            resource_id=webhook_id,
            request=request,
        )
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
    Handle incoming webhook: verify the sender's HMAC signature, extract the
    workflow input from the payload template, and execute the workflow.

    Every failure BEFORE the signature verifies is the one response built by
    ``_webhook_auth_failed()`` — see the note beside it. Everything AFTER keeps
    ordinary receiver semantics so senders' retry logic still works.
    """
    # ── 1. Look the webhook up. This still happens, and must: the receiver
    # cannot verify a signature without the row's secret. What changed is that
    # nothing below lets the RESULT of this lookup reach the caller before the
    # signature verifies.
    #
    # The lookup is deliberately not org-scoped: `endpoint_path` is the only
    # locator an external sender has, and the org is derived FROM the row it
    # finds (never from the request) once the sender has authenticated.
    trigger = None
    try:
        result = (
            supabase.table("webhook_triggers")
            .select("*")
            .eq("endpoint_path", endpoint_path)
            .eq("is_active", True)
            .execute()
        )
        rows = result.data or []
        if isinstance(rows, dict):  # a .single()-shaped response, defensively
            rows = [rows]
        if rows:
            trigger = rows[0]
    except Exception as e:
        # An infrastructure failure, not an answer about this path. It fires for
        # EVERY request while the lookup is broken — existing path or not — so
        # it cannot separate the cases above, and a 5xx is the honest reply that
        # keeps a real sender retrying instead of giving up on a 401.
        logger.error("Webhook lookup failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="Webhook lookup failed")

    # ── 2. Read the body unconditionally: it is the signed material, and both
    # the real and the synthetic verification need it.
    raw_body = b""
    try:
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
    except Exception:
        body_str = ""

    # ── 3. Authenticate the SENDER.
    #
    # `real_secret` is None for an unknown path AND for a row written outside
    # the API with no secret — a row that can never authenticate anyone, so
    # refusing it costs no legitimate sender anything. Both substitute the
    # synthetic secret and run the identical hmac + compare below, so all three
    # pre-auth failures converge on one response instead of 404 / 401 / 503.
    real_secret = trigger.get("secret") if trigger else None
    if trigger is not None and not real_secret:
        # Operator-side only, and the one place the misconfiguration is visible.
        # The anonymous caller is told nothing it could not already guess.
        logger.error(
            "Webhook %s has no secret configured — refusing to execute unauthenticated",
            trigger.get("id"),
        )

    secret = real_secret or _SYNTHETIC_WEBHOOK_SECRET
    signature = (
        request.headers.get("x-webhook-signature")
        or request.headers.get("x-hub-signature-256")
        or ""
    )
    signature_ok = _signature_matches(secret, raw_body, signature)

    if trigger is None or not real_secret or not signature_ok:
        # Belt and braces: a synthetic secret can never verify, so the first two
        # conditions are already implied. They stay explicit so a future edit to
        # `_signature_matches` cannot turn "no such webhook" into an execution.
        logger.warning(
            "Webhook delivery rejected (webhook=%s)",
            (trigger or {}).get("id") or "unknown",
        )
        raise _webhook_auth_failed()

    # ── The sender is authenticated from here on. Responses below are ordinary
    # receiver semantics and are NOT flattened.

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
            # `org_id` here is the trigger row's own, read by the server —
            # never from the request. Redundant given the id, and kept so the
            # statement is scoped on its face.
            supabase.table("webhook_triggers").update({
                "last_triggered_at": "now()",
                "trigger_count": trigger.get("trigger_count", 0) + 1,
            }).eq("id", trigger["id"]).eq("org_id", org_id).execute()
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
