"""
Pytest bootstrap.

Several test modules import `main`/`routers.*`, which validate the encryption
key and construct a Supabase client AT IMPORT TIME. Without these variables the
import raises and the tests fail — but only in a clean shell, so they pass for a
developer who happens to have them exported and fail in CI. That divergence
caused a real measurement dispute in this repo: the same commit reported 7
failures locally and 13 in a clean environment.

conftest is the only place early enough to fix it: pytest imports this before
collecting any test module, whereas setting the vars inside a single test file
runs after an alphabetically-earlier module has already imported (and cached)
the app.

`setdefault`, so a real environment always wins. These are placeholders that
allow import — no test performs real crypto or reaches a real database.
"""
import base64
import os

os.environ.setdefault("MASTER_ENCRYPTION_KEY", base64.b64encode(b"0" * 32).decode())
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("SUPABASE_KEY", "test-service-role-key")
os.environ.setdefault("OPTIML_CONTROL_LOOP_ENABLED", "false")
