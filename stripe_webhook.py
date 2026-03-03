import stripe
import os
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException, Header
from supabase_client import supabase
import json

logger = logging.getLogger(__name__)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

router = APIRouter()

# Price ID → tier mapping (must match STRIPE_PRODUCTS in frontend lib/stripe.ts)
# When users switch plans via the Stripe Customer Portal, Stripe does NOT update
# the subscription metadata — only the price changes. So we must resolve the tier
# from the price ID, not from metadata (which is stale after plan switches).
PRICE_TO_TIER = {
    "price_1T6eNiPsb79p0xBWvsNR5v4X": "startup",
    "price_1T6ePNPsb79p0xBWrw9e6tD0": "team",
}


def _unix_to_iso(ts) -> str | None:
    """Convert a Unix epoch timestamp (int/float) to an ISO 8601 UTC string.

    Stripe sends timestamps as integer seconds since epoch (e.g. 1740614400).
    Supabase TIMESTAMP WITH TIME ZONE columns need proper ISO format like
    '2025-02-27T00:00:00+00:00', NOT '1740614400T00:00:00.000Z'.
    Returns None if the input is None or invalid.
    """
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return None


def _resolve_tier_from_subscription(subscription: dict) -> str | None:
    """Resolve the plan tier from the subscription's current price ID.

    The price ID is the authoritative source because Stripe updates it on plan
    switches but leaves metadata unchanged. Falls back to metadata only if the
    price ID is not recognized (e.g. a new price created in Stripe Dashboard
    that hasn't been added to PRICE_TO_TIER yet).
    """
    try:
        items = subscription.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id")
            if price_id and price_id in PRICE_TO_TIER:
                return PRICE_TO_TIER[price_id]
            if price_id:
                logger.warning("Unknown price_id=%s in subscription %s — falling back to metadata",
                               price_id, subscription.get("id"))
    except (KeyError, IndexError, TypeError):
        pass
    # Fallback to metadata (works for first checkout before any plan switch)
    return subscription.get("metadata", {}).get("tier")


def _lookup_user_id_by_customer(customer_id: str) -> str | None:
    """Look up user_id from stripe_customer_id in user_profiles.

    Fallback for when subscription metadata doesn't include user_id
    (e.g. subscriptions created directly in Stripe Dashboard).
    """
    if not customer_id:
        return None
    try:
        result = supabase.table("user_profiles") \
            .select("user_id") \
            .eq("stripe_customer_id", customer_id) \
            .limit(1) \
            .execute()
        if result.data:
            return result.data[0]["user_id"]
    except Exception as e:
        logger.warning("Failed to look up user by customer_id=%s: %s", customer_id, e)
    return None


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
    except Exception as e:
        # stripe.SignatureVerificationError (v6+) or stripe.error.SignatureVerificationError (v5)
        if "SignatureVerification" in type(e).__name__:
            logger.warning("Stripe webhook invalid signature: %s", type(e).__name__)
            raise HTTPException(status_code=400, detail="Invalid signature")
        raise

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
                update_data = {
                    "subscription_tier": tier,
                    "subscription_status": "active",
                    "stripe_subscription_id": session.get("subscription"),
                }
                # Save stripe_customer_id from the checkout session
                customer_id = session.get("customer")
                if customer_id:
                    update_data["stripe_customer_id"] = customer_id
                result = supabase.table("user_profiles").update(update_data).eq("user_id", user_id).execute()
                _propagate_tier_to_orgs(user_id, tier)
                logger.info("Stripe checkout.session.completed event_id=%s user_id=%s tier=%s customer=%s updated=%s", event_id, user_id, tier, customer_id, bool(result.data))
        
        elif event['type'] in ['customer.subscription.created', 'customer.subscription.updated']:
            subscription = event['data']['object']
            user_id = subscription['metadata'].get('user_id')
            # Fallback: look up user_id from stripe_customer_id if metadata is missing
            if not user_id:
                customer_id = subscription.get('customer')
                user_id = _lookup_user_id_by_customer(customer_id)
                if user_id:
                    logger.info("Resolved user_id=%s from customer_id=%s (metadata was empty)", user_id, customer_id)
            # Resolve tier from price ID (authoritative) instead of metadata (stale after plan switches)
            tier = _resolve_tier_from_subscription(subscription)

            if user_id and tier:
                status = subscription['status']
                current_period_start = subscription['current_period_start']
                current_period_end = subscription['current_period_end']
                canceled_at = subscription.get('canceled_at')
                cancel_at_period_end = subscription.get('cancel_at_period_end', False)
                trial_end = subscription.get('trial_end')
                
                # Convert Unix timestamps to proper ISO 8601 strings
                # Stripe sends epoch seconds (e.g. 1740614400), Supabase needs ISO format
                result = supabase.table("user_profiles").update({
                    "subscription_tier": tier,
                    "subscription_status": status,
                    "stripe_subscription_id": subscription['id'],
                    "current_period_start": _unix_to_iso(current_period_start),
                    "current_period_end": _unix_to_iso(current_period_end),
                    "canceled_at": _unix_to_iso(canceled_at),
                    "cancel_at_period_end": cancel_at_period_end,
                    "trial_end": _unix_to_iso(trial_end),
                }).eq("user_id", user_id).execute()
                _propagate_tier_to_orgs(user_id, tier)
                logger.info("Stripe subscription updated event_id=%s user_id=%s tier=%s status=%s", event_id, user_id, tier, status)
        
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            user_id = subscription['metadata'].get('user_id')
            if not user_id:
                user_id = _lookup_user_id_by_customer(subscription.get('customer'))
            # On deletion we always go to free, but log what tier was resolved for debugging
            deleted_tier = _resolve_tier_from_subscription(subscription)
            logger.info("Subscription deletion: resolved tier was %s", deleted_tier)

            if user_id:
                # Downgrade to free plan and clear subscription data
                result = supabase.table("user_profiles").update({
                    "subscription_tier": "free",
                    "subscription_status": "canceled",
                    "stripe_subscription_id": None,
                    "current_period_start": None,
                    "current_period_end": None,
                    "canceled_at": _unix_to_iso(subscription.get('canceled_at')),
                    "cancel_at_period_end": False,
                    "trial_end": None
                }).eq("user_id", user_id).execute()
                _propagate_tier_to_orgs(user_id, "free")
                logger.info("Stripe subscription deleted event_id=%s user_id=%s", event_id, user_id)
        
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            subscription_id = invoice.get('subscription')

            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                user_id = subscription['metadata'].get('user_id')
                if not user_id:
                    user_id = _lookup_user_id_by_customer(subscription.get('customer'))

                if user_id:
                    result = supabase.table("user_profiles").update({
                        "subscription_status": "past_due"
                    }).eq("user_id", user_id).execute()
                    logger.info("Stripe invoice.payment_failed event_id=%s user_id=%s", event_id, user_id)

        elif event['type'] == 'invoice.payment_succeeded':
            invoice = event['data']['object']
            subscription_id = invoice.get('subscription')

            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                user_id = subscription['metadata'].get('user_id')
                if not user_id:
                    user_id = _lookup_user_id_by_customer(subscription.get('customer'))

                if user_id:
                    # Resolve tier from the subscription's price (it may have changed)
                    tier = _resolve_tier_from_subscription(subscription)
                    update_data: dict = {"subscription_status": "active"}
                    if tier:
                        update_data["subscription_tier"] = tier
                    result = supabase.table("user_profiles").update(update_data).eq("user_id", user_id).execute()
                    if tier:
                        _propagate_tier_to_orgs(user_id, tier)
                    logger.info("Stripe invoice.payment_succeeded event_id=%s user_id=%s tier=%s", event_id, user_id, tier)
        
        else:
            logger.info("Stripe webhook unhandled type event_id=%s type=%s", event_id, event_type)
        
        return {"received": True}
        
    except Exception as e:
        logger.exception("Stripe webhook processing error event_id=%s: %s", event_id, e)
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


