# supabase_logging.py
# Temporary structured logging for all Supabase queries. Never logs secrets.
# Use this to locate PGRST104/PGRST106 and verify service role usage.

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("supabase")
if not logger.handlers and not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# Detect which key is in use (never log key values)
def _key_type() -> str:
    if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        return "service_role"
    if os.getenv("SUPABASE_KEY"):
        return "SUPABASE_KEY"
    return "none"


def wrap_supabase_client(raw_client: Any) -> Any:
    """Wrap the Supabase client so every .table(...).execute() is logged."""
    if raw_client is None:
        return None

    key_type = _key_type()
    logger.info("Supabase client wrapper active; key_type=%s", key_type)

    class TableWrapper:
        """Wraps the request builder to log table name and errors on execute()."""
        def __init__(self, real_builder: Any, table_name: str):
            self._real = real_builder
            self._table = table_name

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._real, name)
            if not callable(attr):
                return attr
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                out = attr(*args, **kwargs)
                # Wrap any builder (same or new) so execute() is logged
                if hasattr(out, "execute") and callable(getattr(out, "execute", None)):
                    return TableWrapper(out, self._table)
                return out
            return wrapper

        def execute(self) -> Any:
            table, key = self._table, key_type
            try:
                result = self._real.execute()
                logger.info(
                    "supabase_ok table=%s key_type=%s",
                    table, key, extra={"table": table, "key_type": key}
                )
                return result
            except Exception as e:
                err_msg = str(e)
                # Extract PostgREST error code (e.g. PGRST104, PGRST106) from message or details
                err_code = ""
                if hasattr(e, "details") and isinstance(getattr(e, "details"), dict):
                    err_code = (e.details.get("code") or e.details.get("message") or "")
                if not err_code and "PGRST" in err_msg:
                    for token in err_msg.split():
                        if token.startswith("PGRST") and ":" in token:
                            err_code = token.split(":")[0]
                            break
                logger.error(
                    "supabase_error table=%s key_type=%s code=%s error=%s",
                    table, key, err_code or type(e).__name__, err_msg,
                    extra={"table": table, "key_type": key, "error": err_msg, "code": err_code}
                )
                raise

    original_table = raw_client.table

    def logged_table(name: str) -> Any:
        return TableWrapper(original_table(name), name)

    raw_client.table = logged_table
    return raw_client
