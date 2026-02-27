# Stripe webhook (Railway)

## Required env (Railway)

- `STRIPE_WEBHOOK_SECRET` – from Stripe Dashboard → Developers → Webhooks → your endpoint → Signing secret
- `STRIPE_SECRET_KEY` – Stripe API key (for optional Subscription.retrieve in handlers)

## Webhook URL

Use your **public backend URL** plus the route:

```
https://<your-railway-backend-host>/stripe/webhook
```

Example: `https://your-app.railway.app/stripe/webhook`

In **Stripe Dashboard**: Developers → Webhooks → Add endpoint → paste URL.

## Events to subscribe to

Select at least:

- `checkout.session.completed` – after checkout, set subscription_tier and status
- `customer.subscription.created`
- `customer.subscription.updated` – keep subscription_status in sync (active, canceled, past_due, etc.)
- `customer.subscription.deleted` – downgrade to free, clear subscription fields
- `invoice.payment_failed` – set subscription_status to `past_due` (backend returns 402 for inference)
- `invoice.payment_succeeded` – set subscription_status to `active`

## Behavior

- Signature is verified with `STRIPE_WEBHOOK_SECRET`; invalid requests return 400.
- Event ID and type are logged (no secrets). Subscription updates to `user_profiles` are logged (user_id, tier, status).
- Backend uses `subscription_status` to block `/v1/inference` for teams with inactive/past_due subscriptions (402).
