# Fix: "Database error saving new user" (500 on signup)

When signup returns **500** and the client shows **`AuthApiError: Database error saving new user`**, Supabase Auth is failing during the database transaction that creates the user.

## Cause

Almost always one of:

1. **Trigger on `auth.users`** – A trigger runs after insert (e.g. to create a row in `public.user_profiles`). If that insert fails (NOT NULL column missing, constraint, RLS), Auth returns 500.
2. **Constraint on `auth.users`** – A check or FK that isn’t satisfied.
3. **Permissions** – e.g. Prisma or another tool changed permissions on `auth.users`.

## Steps to fix

### 1. See the real error

- **Supabase Dashboard** → **Logs** → **Postgres logs**
- Trigger a signup again and look for the **database error** (e.g. `null value in column "onboarding_completed"` or `violates not-null constraint`).

### 2. Make the trigger/table safe

- If the error points to **`user_profiles`** (e.g. missing or null column), run **`fix_signup_database_error.sql`** in the SQL Editor. It adds `onboarding_completed` / `onboarding_completed_at` if missing and ensures they have defaults or are nullable.
- If the error is **`null value in column "slug" of relation "organizations"`**, a trigger is creating an organization row on signup without setting `slug`. Run **`fix_organization_slug_on_signup.sql`** in the SQL Editor. It adds a BEFORE INSERT trigger on `organizations` that sets `slug` from the org name + id when null.
- If you have a custom “on signup” trigger that inserts into `user_profiles`, ensure it either:
  - Sets every NOT NULL column, or
  - The table has DEFAULTs or nullable for those columns.

### 3. Optional: use the suggested trigger

The comment block in `fix_signup_database_error.sql` shows a **safe** `handle_new_user()` that only inserts `user_id` and `email`. If you want to (re)create that trigger:

- Uncomment the block in the SQL file.
- Run it in the Supabase SQL Editor (with the rest of the file).

### 4. RLS

If the trigger runs as a role that can’t insert into `user_profiles` because of RLS, either:

- Make the trigger function **SECURITY DEFINER** (as in the example), and/or
- Add an RLS policy that allows insert for the trigger context (e.g. `service_role` or the function owner).

After fixing the failing constraint/trigger/column, signup should succeed.
