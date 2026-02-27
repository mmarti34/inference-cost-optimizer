# supabase_client.py
# Backend uses a single DB layer: service role only for all Supabase access.
# Never use anon key for api_keys, service_api_keys, or other privileged tables.

from supabase import create_client, Client
import os
import logging
from dotenv import load_dotenv
from typing import Optional

from supabase_logging import wrap_supabase_client

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Backend must use service role for privileged reads/writes.
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or os.getenv("SUPABASE_KEY")

# Single client: admin (service role) only. No anon client in backend.
_raw_supabase: Optional[Client] = None
supabase_admin: Optional[Client] = None
# Alias for backward compatibility; all code should use service role.
supabase: Optional[Client] = None

try:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Set SUPABASE_SERVICE_ROLE_KEY for backend.")
    else:
        if not SUPABASE_SERVICE_ROLE_KEY and os.getenv("SUPABASE_KEY"):
            logger.warning("Using SUPABASE_KEY. Prefer SUPABASE_SERVICE_ROLE_KEY for backend.")
        if SUPABASE_KEY and len(SUPABASE_KEY) < 100:
            logger.warning("Key looks like anon. Use service_role key to avoid PGRST/RLS errors.")
        _raw_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_admin = wrap_supabase_client(_raw_supabase)
        supabase = supabase_admin
        logger.info("Supabase client initialized (service role, with query logging).")
except Exception as e:
    logger.exception("Error initializing Supabase client: %s", e)
