import stripe
import os
import logging
from fastapi import APIRouter, Request, HTTPException, Header
from supabase_client import supabase
import json

logger = logging.getLogger(__name__)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

router = APIRouter()


def _propagate_tier_to_orgs(user_id: str, tier: str):
    """When a user's subscription tier changes, update the plan on any org where they are admin."""
    try:
        admin_orgs = supabase.table("organization_members") \
            .select("org_id") \
            .eq("user_id", user_id) \
            .eq("role", "admin") \
            .eq("status", "active") \
            .execute()
        if admin_orgs.data:
            for membership in admin_orgs.data:
                supabase.table("organizations") \
                    .update({"plan": tier}) \
                    .eq("id", membership["org_id"]) \
                    .execute()
            logger.info("Propagated tier=%s to %d org(s) for user_id=%s", tier, len(admin_orgs.data), user_id)
    except Exception as e:
        # Non-critical — log but don't fail the webhook
        logger.warning("Failed to propagate tier to orgs for user_id=%s: %s", user_id, e)


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """
    Handle Stripe webhook events for subscription management.
    Requires STRIPE_WEBHOOK_SECRET. Logs event IDs and outcome only (no secrets).
    """
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing stripe signature")

    body = await request.body()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        logger.error("Stripe webhook: STRIPE_WEBHOOK_SECRET not set")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(body, stripe_signature, webhook_secret)
    except ValueError as e:
        logger.warning("Stripe webhook invalid payload: %s", type(e).__name__)
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.warning("Stripe webhook invalid signature: %s", type(e).__name__)
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_id = event.get("id", "")
    event_type = event.get("type", "")
    logger.info("Stripe webhook event_id=%s type=%s", event_id, event_type)
    
    try:
        # Handle the event
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            user_id = session['metadata'].get('user_id')
            tier = session['metadata'].get('tier')
            
            if user_id and tier:
                result = supabase.table("user_profiles").update({
                    "subscription_tier": tier,
                    "subscription_status": "active",
                    "stripe_subscription_id": session.get("subscription"),
                }).eq("user_id", user_id).execute()
                _propagate_tier_to_orgs(user_id, tier)
                logger.info("Stripe checkout.session.completed event_id=%s user_id=%s tier=%s updated=%s", event_id, user_id, tier, bool(result.data))
        
        elif event['type'] in ['customer.subscription.created', 'customer.subscription.updated']:
            subscription = event['data']['object']
            user_id = subscription['metadata'].get('user_id')
            tier = subscription['metadata'].get('tier')
            
            if user_id and tier:
                status = subscription['status']
                current_period_start = subscription['current_period_start']
                current_period_end = subscription['current_period_end']
                canceled_at = subscription.get('canceled_at')
                cancel_at_period_end = subscription.get('cancel_at_period_end', False)
                trial_end = subscription.get('trial_end')
                
                # Convert timestamps to ISO strings
                period_start_iso = None
                period_end_iso = None
                canceled_at_iso = None
                trial_end_iso = None
                
                if current_period_start:
                    period_start_iso = f"{current_period_start}T00:00:00.000Z"
                if current_period_end:
                    period_end_iso = f"{current_period_end}T00:00:00.000Z"
                if canceled_at:
                    canceled_at_iso = f"{canceled_at}T00:00:00.000Z"
                if trial_end:
                    trial_end_iso = f"{trial_end}T00:00:00.000Z"
                
                result = supabase.table("user_profiles").update({
                    "subscription_tier": tier,
                    "subscription_status": status,
                    "stripe_subscription_id": subscription['id'],
                    "current_period_start": period_start_iso,
                    "current_period_end": period_end_iso,
                    "canceled_at": canceled_at_iso,
                    "cancel_at_period_end": cancel_at_period_end,
                    "trial_end": trial_end_iso
                }).eq("user_id", user_id).execute()
                _propagate_tier_to_orgs(user_id, tier)
                logger.info("Stripe subscription updated event_id=%s user_id=%s tier=%s status=%s", event_id, user_id, tier, status)
        
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            user_id = subscription['metadata'].get('user_id')
            
            if user_id:
                # Downgrade to free plan and clear subscription data
                result = supabase.table("user_profiles").update({
                    "subscription_tier": "free",
                    "subscription_status": "canceled",
                    "stripe_subscription_id": None,
                    "current_period_start": None,
                    "current_period_end": None,
                    "canceled_at": f"{subscription['canceled_at']}T00:00:00.000Z" if subscription.get('canceled_at') else None,
                    "cancel_at_period_end": False,
                    "trial_end": None
                }).eq("user_id", user_id).execute()
                _propagate_tier_to_orgs(user_id, "free")
                logger.info("Stripe subscription deleted event_id=%s user_id=%s", event_id, user_id)
        
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            subscription_id = invoice.get('subscription')
            
            if subscription_id:
                # Get subscription details
                subscription = stripe.Subscription.retrieve(subscription_id)
                user_id = subscription['metadata'].get('user_id')
                
                if user_id:
                    result = supabase.table("user_profiles").update({
                        "subscription_status": "past_due"
                    }).eq("user_id", user_id).execute()
                    logger.info("Stripe invoice.payment_failed event_id=%s user_id=%s", event_id, user_id)
        
        elif event['type'] == 'invoice.payment_succeeded':
            invoice = event['data']['object']
            subscription_id = invoice.get('subscription')
            
            if subscription_id:
                # Get subscription details
                subscription = stripe.Subscription.retrieve(subscription_id)
                user_id = subscription['metadata'].get('user_id')
                
                if user_id:
                    result = supabase.table("user_profiles").update({
                        "subscription_status": "active"
                    }).eq("user_id", user_id).execute()
                    logger.info("Stripe invoice.payment_succeeded event_id=%s user_id=%s", event_id, user_id)
        
        else:
            logger.info("Stripe webhook unhandled type event_id=%s type=%s", event_id, event_type)
        
        return {"received": True}
        
    except Exception as e:
        logger.exception("Stripe webhook processing error event_id=%s: %s", event_id, e)
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


