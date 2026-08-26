"""
VENDOR side of the layer: what a vendor CLAIMS about a thing that performs work.

Hard boundary, enforced by keeping this module free of any query against
workflow_runs / outcomes / cost_events / optimization_benchmarks:

    A vendor claim is NEVER performance evidence.

Published token price, advertised context window, declared region and stated
retention policy all live here. What a model actually cost, how long it
actually took and whether it actually worked on THIS customer's workload lives
in optimization/evidence.py. A cheaper price sheet is a reason to BENCHMARK a
candidate. It is never a reason to promote one.

An executor is anything that can perform work — a model, an agent or external
tool, deterministic software, or (schema-only, nothing built) a human. Adding
Claude Code, Devin or an internal classifier later is an INSERT into
`executors`, not a migration.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase
from utils.pricing import get_all_providers, get_pricing

from optimization import domain

logger = logging.getLogger(__name__)

EXECUTOR_COLS = (
    "id, org_id, executor_type, vendor, external_id, display_name, version, "
    "capabilities, configuration, cost_model, policy_metadata, "
    "integration_source, enabled, created_at, updated_at"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Vendor price sheet
# ---------------------------------------------------------------------------

def vendor_cost_model(provider: str, model: str) -> dict:
    """
    The vendor's LIST PRICE for a model, from shared/providers.json.

    This is a price sheet entry, not a measurement. `source` is stamped on the
    result so that a caller which mistakenly hands this to an evidence field is
    at least auditable after the fact.
    """
    pricing = get_pricing(provider, model)
    return {
        "unit": "usd_per_1k_tokens",
        "input": pricing.get("input"),
        "output": pricing.get("output"),
        "source": "vendor_list_price:shared/providers.json",
        "is_measurement": False,
    }


def blended_vendor_price(
    provider: str,
    model: str,
    *,
    input_output_ratio: float = 3.0,
) -> Optional[float]:
    """
    A single comparable USD/1k-token number for ranking price sheets.

    `input_output_ratio` is how many input tokens are assumed per output token.
    It is an ASSUMPTION, not a measurement, and it is only ever used to order
    candidates for benchmarking — never to state a saving. Callers that have a
    measured ratio for the workload should pass it.
    """
    cm = vendor_cost_model(provider, model)
    inp, out = cm.get("input"), cm.get("output")
    if inp is None or out is None:
        return None
    r = max(0.0, float(input_output_ratio))
    return (float(inp) * r + float(out)) / (r + 1.0)


def vendor_catalog() -> list[dict]:
    """
    Every model in shared/providers.json, flattened, as vendor claims.

    Returned fields are all vendor-published: display name, price, category,
    context window. No field here has been measured by OptiML.
    """
    out: list[dict] = []
    try:
        providers = get_all_providers()
    except Exception as exc:  # pragma: no cover - missing providers.json
        logger.warning("vendor_catalog unavailable: %s", type(exc).__name__)
        return []

    for provider_id, provider in (providers or {}).items():
        for model_id, meta in (provider.get("models") or {}).items():
            out.append({
                "executor_type": "model",
                "vendor": provider_id,
                "external_id": model_id,
                "display_name": meta.get("display_name") or model_id,
                "capabilities": {
                    "context_window": meta.get("context_window"),
                    "category": meta.get("category"),
                    "api_style": provider.get("api_style"),
                    "source": "vendor_claim:shared/providers.json",
                },
                "cost_model": {
                    "unit": "usd_per_1k_tokens",
                    "input": meta.get("input_per_1k"),
                    "output": meta.get("output_per_1k"),
                    "source": "vendor_list_price:shared/providers.json",
                    "is_measurement": False,
                },
                "policy_metadata": {
                    # Deliberately empty rather than guessed. Region, retention
                    # and certification claims are not in providers.json, so we
                    # record that they are unknown instead of assuming them.
                    "region": None,
                    "zero_data_retention": None,
                    "stores_prompts": None,
                    "source": "unknown",
                },
                "integration_source": "providers_json",
            })
    return out


# ---------------------------------------------------------------------------
# Executor registry
# ---------------------------------------------------------------------------

def list_executors(
    org_id: str,
    *,
    executor_type: Optional[str] = None,
    vendor: Optional[str] = None,
    enabled_only: bool = True,
    limit: int = 500,
) -> list[dict]:
    """Registered executors for an org. Always re-filtered by org_id."""
    try:
        q = supabase.table("executors").select(EXECUTOR_COLS).eq("org_id", org_id)
        if executor_type:
            q = q.eq("executor_type", executor_type)
        if vendor:
            q = q.eq("vendor", vendor)
        if enabled_only:
            q = q.eq("enabled", True)
        resp = q.order("created_at", desc=True).limit(max(1, min(limit, 1000))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_executors failed: %s", type(exc).__name__)
        return []


def upsert_executor(org_id: str, executor: dict) -> Optional[dict]:
    """
    Register or refresh one executor for an org.

    Identity is (org_id, executor_type, vendor, external_id, version). Only
    vendor-claim fields are written; there is no code path here that can write a
    measured number into `executors`.
    """
    executor_type = (executor.get("executor_type") or "model").strip().lower()
    if executor_type not in domain.EXECUTOR_TYPES:
        raise ValueError(f"Unknown executor_type '{executor_type}'.")
    external_id = (executor.get("external_id") or "").strip()
    if not external_id:
        raise ValueError("executor.external_id is required.")

    vendor = (executor.get("vendor") or "").strip() or None
    version = (executor.get("version") or "").strip() or None

    row = {
        "org_id": org_id,
        "executor_type": executor_type,
        "vendor": vendor,
        "external_id": external_id,
        "display_name": executor.get("display_name") or external_id,
        "version": version,
        "capabilities": executor.get("capabilities") or {},
        "configuration": executor.get("configuration") or {},
        "cost_model": executor.get("cost_model") or {},
        "policy_metadata": executor.get("policy_metadata") or {},
        "integration_source": executor.get("integration_source") or "manual",
        "enabled": bool(executor.get("enabled", True)),
        "updated_at": _utc_now_iso(),
    }

    try:
        q = (
            supabase.table("executors")
            .select("id")
            .eq("org_id", org_id)
            .eq("executor_type", executor_type)
            .eq("external_id", external_id)
        )
        q = q.is_("vendor", "null") if vendor is None else q.eq("vendor", vendor)
        q = q.is_("version", "null") if version is None else q.eq("version", version)
        existing = q.limit(1).execute()

        if existing.data:
            executor_id = existing.data[0]["id"]
            updated = (
                supabase.table("executors")
                .update(row)
                .eq("id", executor_id)
                .eq("org_id", org_id)
                .execute()
            )
            return (updated.data or [None])[0]

        inserted = supabase.table("executors").insert(row).execute()
        return (inserted.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("upsert_executor failed for %s: %s", external_id, type(exc).__name__)
        return None


def sync_model_executors(org_id: str) -> dict:
    """
    Populate `executors` from the vendor catalog for this org.

    Real and idempotent. Registers only executor_type='model' — that is the
    only executor kind OptiML can execute today. Agent, software and human
    executors are registered by their own connectors when those exist; nothing
    here fabricates a placeholder for them.
    """
    catalog = vendor_catalog()
    registered, failed = 0, 0
    for entry in catalog:
        try:
            if upsert_executor(org_id, entry) is not None:
                registered += 1
            else:
                failed += 1
        except Exception:  # pragma: no cover
            failed += 1

    return {
        "org_id": org_id,
        "catalog_size": len(catalog),
        "registered": registered,
        "failed": failed,
        "source": "shared/providers.json",
        "note": (
            "Vendor metadata only: list prices and published capabilities. "
            "Nothing here is evidence of performance on your workloads."
        ),
    }


def find_executor(
    org_id: str,
    *,
    executor_type: str,
    external_id: str,
    vendor: Optional[str] = None,
) -> Optional[dict]:
    """Look up one registered executor. None when it is not registered."""
    try:
        q = (
            supabase.table("executors")
            .select(EXECUTOR_COLS)
            .eq("org_id", org_id)
            .eq("executor_type", executor_type)
            .eq("external_id", external_id)
        )
        if vendor:
            q = q.eq("vendor", vendor)
        resp = q.limit(1).execute()
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("find_executor failed: %s", type(exc).__name__)
        return None


def executor_row_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "executor_type": row.get("executor_type"),
        "vendor": row.get("vendor"),
        "external_id": row.get("external_id"),
        "display_name": row.get("display_name"),
        "version": row.get("version"),
        "capabilities": row.get("capabilities") or {},
        "configuration": row.get("configuration") or {},
        "cost_model": row.get("cost_model") or {},
        "policy_metadata": row.get("policy_metadata") or {},
        "integration_source": row.get("integration_source"),
        "enabled": bool(row.get("enabled", True)),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        # Stated on every response so a frontend cannot present these numbers
        # as OptiML measurements.
        "data_class": "vendor_metadata",
    }
