# inference-cost-optimizer

Railway deployment fix.

## Known good checklist

Use this to get the app reliable end-to-end (Backend on Railway, Frontend on Vercel, DB/Auth on Supabase).

### Local env vars (backend)

- `SUPABASE_URL` – project URL (e.g. `https://xxx.supabase.co`)
- `SUPABASE_SERVICE_ROLE_KEY` – **required** for backend (not anon key)
- `MASTER_ENCRYPTION_KEY` or `ENCRYPTION_KEY` – for encrypting provider API keys (min 32 bytes)
- `STRIPE_WEBHOOK_SECRET` – for Stripe webhooks (if using billing)
- `STRIPE_SECRET_KEY` – for Stripe API (if using billing)

### Railway env vars (backend)

- Same as above. Prefer `SUPABASE_SERVICE_ROLE_KEY` over `SUPABASE_KEY`.
- Ensure no `SUPABASE_KEY` anon key is set if you rely on service role.

### Vercel env vars (frontend)

- `NEXT_PUBLIC_SUPABASE_URL` – project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` – anon key (frontend only; never use for backend)
- Backend API URL for inference (e.g. `NEXT_PUBLIC_API_URL` or similar)

### Supabase settings checks

1. **Exposed schemas**: Project Settings → API → Exposed schemas must include `public` (default).
2. **Tables in public**: Run `scripts/verify_schema.sql` in SQL Editor; all app tables should be in `public`.
3. **If PGRST104/PGRST106**: See `docs/SUPABASE_SCHEMA_EXPOSURE.md` and run `scripts/reload_postgrest_schema.sql` after fixing.
4. **RLS**: Backend uses service role to bypass RLS for sensitive tables; ensure RLS policies match your design or disable RLS on backend‑only tables if intended.

### Stripe (teams billing)

- **Railway**: `STRIPE_WEBHOOK_SECRET`, `STRIPE_SECRET_KEY`.
- **Webhook URL**: `https://<your-backend>/stripe/webhook`. See `docs/STRIPE_WEBHOOK.md` for events and setup.
