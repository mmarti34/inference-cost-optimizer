"""
Optimization policies: constraints that make a strategy INVALID, not merely
worse.

Optimization is "find the best strategy WITHIN constraints". A strategy that
violates a policy is not a lower-ranked option — it is not an option. The
optimizer must never trade a policy away for savings, so `evaluate` returns
violations separately from rankings and callers gate on them.

Two things this module refuses to do:

  * Assume an unenforceable constraint is satisfied. If a policy demands
    EU-only processing and nothing in the executor registry records a region,
    that constraint is reported as `unenforced` — never as `satisfied`. A
    silent pass would be a compliance claim we cannot back.

  * Apply a global definition of success. Which signal decides whether a
    workload is doing well is declared per workload in `success_signal`, not
    baked into a constant here. See domain.resolve_success_signal.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import domain

logger = logging.getLogger(__name__)

POLICY_COLS = (
    "id, org_id, policy_key, version, is_current, superseded_at, workload_id, name, "
    "description, enabled, priority, constraints, automation, success_signal, "
    "materiality, created_at, updated_at"
)

#: Conservative by default: OptiML does nothing on its own and a human approves
#: anything that touches production. auto_rollback defaults TRUE because
#: rolling back is the safe direction.
DEFAULT_AUTOMATION = {
    "auto_benchmark": False,
    "auto_shadow": False,
    "auto_canary": False,
    "auto_promote": False,
    "require_human_approval": True,
    "auto_rollback": True,
}

#: Constraints this codebase can actually check against measured evidence.
ENFORCEABLE_CONSTRAINTS = (
    "min_quality",
    "max_error_rate",
    "max_latency_p95_ms",
    "max_cost_per_task_usd",
    "allowed_vendors",
    "blocked_vendors",
    "require_human_approval",
)

#: Constraints that are accepted, stored and REPORTED, but cannot be verified
#: today. Each entry says what would be needed to enforce it. They are never
#: reported as satisfied.
UNENFORCEABLE_CONSTRAINTS = {
    "require_zero_data_retention": (
        "executors.policy_metadata.zero_data_retention has no source: "
        "shared/providers.json does not publish retention terms. Enforcing this "
        "requires a vendor attestation feed or customer-entered attestations."
    ),
    "allow_prompt_storage": (
        "executors.policy_metadata.stores_prompts has no source. Same "
        "requirement as require_zero_data_retention."
    ),
    "data_region": (
        "executors.policy_metadata.region has no source, and OptiML does not "
        "currently pin provider requests to a region."
    ),
    "require_certifications": (
        "No certification registry exists. Would need per-vendor attestations."
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def list_policies(org_id: str, *, workload_id: Optional[str] = None) -> list[dict]:
    try:
        q = supabase.table("optimization_policies").select(POLICY_COLS).eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        resp = q.order("priority", desc=False).limit(200).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_policies failed: %s", type(exc).__name__)
        return []


def get_effective_policy(org_id: str, workload_id: Optional[str]) -> Optional[dict]:
    """
    The policy in force for a workload.

    A workload-scoped policy wins over an org-wide one; within a scope the
    lowest `priority` value wins. Returns None when the org has no policy — and
    None means "no constraints declared", which is reported as such rather than
    quietly treated as "everything is allowed".
    """
    try:
        q = (
            supabase.table("optimization_policies")
            .select(POLICY_COLS)
            .eq("org_id", org_id)
            .eq("enabled", True)
            .eq("is_current", True)
        )
        rows = getattr(q.order("priority", desc=False).limit(200).execute(), "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("get_effective_policy failed: %s", type(exc).__name__)
        return None

    if not rows:
        return None

    if workload_id:
        scoped = [r for r in rows if r.get("workload_id") and str(r["workload_id"]) == str(workload_id)]
        if scoped:
            return scoped[0]

    org_wide = [r for r in rows if not r.get("workload_id")]
    return org_wide[0] if org_wide else None


def constraints_of(policy: Optional[dict]) -> dict:
    if not policy:
        return {}
    c = policy.get("constraints")
    return c if isinstance(c, dict) else {}


def automation_of(policy: Optional[dict]) -> dict:
    if not policy:
        return dict(DEFAULT_AUTOMATION)
    a = policy.get("automation")
    if not isinstance(a, dict):
        return dict(DEFAULT_AUTOMATION)
    return {**DEFAULT_AUTOMATION, **a}


def materiality_of(policy: Optional[dict], objective: str = "cost") -> dict:
    """
    Materiality threshold in force, normalised to the objective's own metric
    and units, with its source and the exact policy VERSION stamped on. The
    stamp is what makes a historical conclusion reproducible after the policy
    changes.
    """
    normalized = domain.normalize_materiality((policy or {}).get("materiality"), objective)
    normalized["policy_id"] = str(policy["id"]) if policy and policy.get("id") else None
    normalized["policy_key"] = str(policy["policy_key"]) if policy and policy.get("policy_key") else None
    normalized["policy_version"] = (policy or {}).get("version")
    return normalized


def success_signal_of(policy: Optional[dict]) -> dict:
    s = (policy or {}).get("success_signal")
    return s if isinstance(s, dict) else {}


def approval_required(policy: Optional[dict]) -> bool:
    """
    Human approval is the DEFAULT. It is only waived when an org has explicitly
    turned off require_human_approval AND turned on the relevant automation.
    """
    automation = automation_of(policy)
    constraints = constraints_of(policy)
    if constraints.get("require_human_approval") is True:
        return True
    return bool(automation.get("require_human_approval", True))


def may_auto(policy: Optional[dict], action: str) -> bool:
    """Whether OptiML may take `action` (benchmark/shadow/canary/promote/rollback) unattended."""
    automation = automation_of(policy)
    key = f"auto_{action}"
    if key not in automation:
        return False
    if action in ("canary", "promote") and approval_required(policy):
        return False
    return bool(automation.get(key, False))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    policy: Optional[dict],
    *,
    measured: dict,
    executor_refs: Optional[list[dict]] = None,
    quality_provenance: Optional[str] = None,
) -> dict:
    """
    Check a candidate's MEASURED metrics against a policy.

    `measured` accepts: quality, error_rate, latency_p95_ms, cost_per_task_usd.
    A None value means "not measured" and can never satisfy a constraint — it
    produces an `unmeasured` entry, which is treated as a failure to satisfy,
    not as a pass.

    Returns:
        {
          "policy_id": str|None,
          "eligible": bool,        # no violations AND no unmeasured requirement
          "satisfied": [str],
          "violated":  [{constraint, required, measured, shortfall}],
          "unmeasured":[{constraint, required, reason}],
          "unenforced":[{constraint, required, reason}],
        }
    """
    constraints = constraints_of(policy)
    satisfied: list[str] = []
    violated: list[dict] = []
    unmeasured: list[dict] = []
    unenforced: list[dict] = []

    for key, reason in UNENFORCEABLE_CONSTRAINTS.items():
        if key in constraints:
            unenforced.append({
                "constraint": key,
                "required": constraints[key],
                "reason": reason,
            })

    def _check_max(key: str, measured_key: str) -> None:
        if key not in constraints:
            return
        required = constraints[key]
        actual = measured.get(measured_key)
        if actual is None:
            unmeasured.append({
                "constraint": key,
                "required": required,
                "reason": f"{measured_key} was not measured in this run.",
            })
            return
        if float(actual) <= float(required):
            satisfied.append(key)
        else:
            violated.append({
                "constraint": key,
                "required": float(required),
                "measured": float(actual),
                "shortfall": round(float(actual) - float(required), 8),
                "direction": "exceeded_maximum",
            })

    def _check_min(key: str, measured_key: str) -> None:
        if key not in constraints:
            return
        required = constraints[key]
        actual = measured.get(measured_key)
        if actual is None:
            unmeasured.append({
                "constraint": key,
                "required": required,
                "reason": f"{measured_key} was not measured in this run.",
            })
            return
        if float(actual) >= float(required):
            satisfied.append(key)
        else:
            violated.append({
                "constraint": key,
                "required": float(required),
                "measured": float(actual),
                "shortfall": round(float(required) - float(actual), 8),
                "direction": "below_minimum",
            })

    # min_quality has an extra requirement: the quality signal must be strong
    # enough to be worth constraining on. A number produced by an LLM judge
    # cannot satisfy a hard quality floor on its own.
    if "min_quality" in constraints:
        rank = domain.provenance_rank(quality_provenance)
        if measured.get("quality") is None:
            unmeasured.append({
                "constraint": "min_quality",
                "required": constraints["min_quality"],
                "reason": (
                    "Quality was not measured. A min_quality constraint cannot be "
                    "satisfied on cost evidence alone."
                ),
            })
        elif rank < domain.MIN_QUALITY_PROVENANCE_RANK_FOR_CONSTRAINT:
            unmeasured.append({
                "constraint": "min_quality",
                "required": constraints["min_quality"],
                "reason": (
                    f"Quality signal provenance '{quality_provenance}' (rank {rank}) is "
                    f"below the minimum rank {domain.MIN_QUALITY_PROVENANCE_RANK_FOR_CONSTRAINT} "
                    "required to satisfy a hard quality floor."
                ),
            })
        else:
            _check_min("min_quality", "quality")

    _check_max("max_error_rate", "error_rate")
    _check_max("max_latency_p95_ms", "latency_p95_ms")
    _check_max("max_cost_per_task_usd", "cost_per_task_usd")

    vendors = {
        (r.get("vendor") or "").strip().lower()
        for r in (executor_refs or [])
        if r.get("vendor")
    }
    allowed = constraints.get("allowed_vendors")
    if isinstance(allowed, list) and allowed:
        allowed_set = {str(v).strip().lower() for v in allowed}
        bad = sorted(vendors - allowed_set)
        if not vendors:
            unmeasured.append({
                "constraint": "allowed_vendors",
                "required": allowed,
                "reason": "No executor vendors could be determined for this strategy.",
            })
        elif bad:
            violated.append({
                "constraint": "allowed_vendors",
                "required": sorted(allowed_set),
                "measured": bad,
                "shortfall": None,
                "direction": "vendor_not_approved",
            })
        else:
            satisfied.append("allowed_vendors")

    blocked = constraints.get("blocked_vendors")
    if isinstance(blocked, list) and blocked:
        blocked_set = {str(v).strip().lower() for v in blocked}
        bad = sorted(vendors & blocked_set)
        if bad:
            violated.append({
                "constraint": "blocked_vendors",
                "required": sorted(blocked_set),
                "measured": bad,
                "shortfall": None,
                "direction": "vendor_blocked",
            })
        else:
            satisfied.append("blocked_vendors")

    return {
        "policy_id": (str(policy["id"]) if policy and policy.get("id") else None),
        "eligible": (not violated) and (not unmeasured),
        "satisfied": satisfied,
        "violated": violated,
        "unmeasured": unmeasured,
        "unenforced": unenforced,
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def create_policy(org_id: str, payload: dict) -> Optional[dict]:
    row = {
        "org_id": org_id,
        "version": 1,
        "is_current": True,
        "workload_id": payload.get("workload_id"),
        "name": payload.get("name") or "Default policy",
        "description": payload.get("description"),
        "enabled": bool(payload.get("enabled", True)),
        "priority": int(payload.get("priority") or 100),
        "constraints": payload.get("constraints") or {},
        "automation": {**DEFAULT_AUTOMATION, **(payload.get("automation") or {})},
        "success_signal": payload.get("success_signal") or {},
        "materiality": payload.get("materiality") or domain.copy_default_materiality("cost"),
        "updated_at": _utc_now_iso(),
    }
    try:
        result = supabase.table("optimization_policies").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:  # pragma: no cover
        logger.warning("create_policy failed: %s", type(exc).__name__)
        return None


def update_policy(org_id: str, policy_id: str, payload: dict) -> Optional[dict]:
    """
    "Edit" a policy by INSERTING a new version. Rows are never updated in place.

    A historical benchmark conclusion must stay reproducible as
    (evidence + policy version + objective). If relaxing quality >= 0.95 to
    >= 0.94 rewrote the row that yesterday's 'candidates_failed_policy' was
    judged under, that verdict would become unexplainable. So the old version
    is retained with is_current=false, and evaluation moves to the new one.
    Re-evaluating retained evidence under the new version produces a NEW
    conclusion; it does not mean the original was wrong — it was correct under
    the policy in force at the time.
    """
    try:
        current = (
            supabase.table("optimization_policies")
            .select(POLICY_COLS)
            .eq("id", policy_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(current, "data", None) or []
        if not rows:
            return None
        prev = rows[0]

        merged = {
            "org_id": org_id,
            "policy_key": prev.get("policy_key") or prev["id"],
            "version": int(prev.get("version") or 1) + 1,
            "is_current": True,
            "supersedes_policy_id": prev["id"],
            "workload_id": payload.get("workload_id", prev.get("workload_id")),
            "name": payload.get("name", prev.get("name")),
            "description": payload.get("description", prev.get("description")),
            "enabled": bool(payload.get("enabled", prev.get("enabled", True))),
            "priority": int(payload.get("priority", prev.get("priority") or 100)),
            "constraints": payload.get("constraints", prev.get("constraints") or {}),
            "automation": {
                **DEFAULT_AUTOMATION,
                **(prev.get("automation") or {}),
                **(payload.get("automation") or {}),
            },
            "success_signal": payload.get("success_signal", prev.get("success_signal") or {}),
            "materiality": payload.get("materiality", prev.get("materiality") or {}),
            "updated_at": _utc_now_iso(),
        }

        inserted = supabase.table("optimization_policies").insert(merged).execute()
        new_row = (inserted.data or [None])[0]
        if new_row is None:
            return None

        # Retire the previous version. This flips a flag; it never edits the
        # constraint values that a historical conclusion was judged against.
        supabase.table("optimization_policies").update({
            "is_current": False,
            "superseded_at": _utc_now_iso(),
        }).eq("id", prev["id"]).eq("org_id", org_id).execute()

        return new_row
    except Exception as exc:  # pragma: no cover
        logger.warning("update_policy failed: %s", type(exc).__name__)
        return None


def get_policy_version(org_id: str, policy_id: str) -> Optional[dict]:
    """Fetch one specific policy VERSION by id, for reproducing an old verdict."""
    try:
        resp = (
            supabase.table("optimization_policies")
            .select(POLICY_COLS)
            .eq("id", policy_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("get_policy_version failed: %s", type(exc).__name__)
        return None


def policy_row_to_response(row: dict) -> dict:
    constraints = row.get("constraints") or {}
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "policy_key": (str(row["policy_key"]) if row.get("policy_key") else None),
        "version": row.get("version"),
        "is_current": bool(row.get("is_current", True)),
        "superseded_at": row.get("superseded_at"),
        "workload_id": (str(row["workload_id"]) if row.get("workload_id") else None),
        "name": row.get("name"),
        "description": row.get("description"),
        "enabled": bool(row.get("enabled", True)),
        "priority": row.get("priority"),
        "constraints": constraints,
        "automation": automation_of(row),
        "success_signal": row.get("success_signal") or {},
        "materiality": row.get("materiality") or domain.copy_default_materiality("cost"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        # Which declared constraints OptiML can actually verify. Surfaced so a
        # UI never shows an unenforceable constraint as if it were protecting
        # the customer.
        "enforceable": [k for k in constraints if k in ENFORCEABLE_CONSTRAINTS],
        "unenforceable": [
            {"constraint": k, "reason": UNENFORCEABLE_CONSTRAINTS[k]}
            for k in constraints
            if k in UNENFORCEABLE_CONSTRAINTS
        ],
    }
