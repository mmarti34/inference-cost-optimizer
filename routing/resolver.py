"""
Version resolution for public API: explicit pin → active experiment → routing policy → latest promoted.
"""
import asyncio
import random
from typing import Optional, Tuple, Any

from fastapi import HTTPException
from supabase_client import supabase

_COLS = "id, workflow_id, project_id, org_id, version, endpoint_slug, graph_json, created_at"

#: The single detail returned when no promoted deployment matches.
#: Every resolver miss uses this exact string, and routers/openai_compat.py
#: imports it so its tenant-boundary rejection is byte-identical to an ordinary
#: miss. A caller must not be able to tell "no such deployment anywhere" from
#: "a deployment that is not yours".
NO_PROMOTED_DEPLOYMENT_DETAIL = "No promoted deployment found."


def _get_promoted_deployment_sync(org_id: str, endpoint_slug: str, version: int) -> Optional[dict]:
    result = (
        supabase.table("workflow_deployments")
        .select(_COLS)
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("version", version)
        .eq("status", "promoted")
        .limit(1)
        .execute()
    )
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


def _get_latest_promoted_sync(org_id: str, endpoint_slug: str) -> Optional[dict]:
    result = (
        supabase.table("workflow_deployments")
        .select(_COLS)
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("status", "promoted")
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


def _get_active_experiment_sync(org_id: str, endpoint_slug: str) -> Optional[dict]:
    result = (
        supabase.table("experiments")
        .select("id, status, variants")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("status", "running")
        .limit(1)
        .execute()
    )
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


def _get_active_routing_policy_sync(org_id: str, endpoint_slug: str) -> Optional[dict]:
    result = (
        supabase.table("routing_policies")
        .select("id, policy_type, rules, active")
        .eq("org_id", org_id)
        .eq("endpoint_slug", endpoint_slug)
        .eq("active", True)
        .limit(1)
        .execute()
    )
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


async def get_promoted_deployment_by_version(
    org_id: str, endpoint_slug: str, version: int
) -> Optional[dict]:
    return await asyncio.to_thread(_get_promoted_deployment_sync, org_id, endpoint_slug, version)


async def get_latest_promoted_deployment(org_id: str, endpoint_slug: str) -> Optional[dict]:
    return await asyncio.to_thread(_get_latest_promoted_sync, org_id, endpoint_slug)


def pick_weighted_version(rules: list) -> int:
    """Weighted random selection. rules: [{"version": int, "weight": int}, ...]"""
    if not rules:
        raise ValueError("rules empty")
    total_weight = sum(r.get("weight", 0) for r in rules)
    if total_weight <= 0:
        return rules[0].get("version", 0)
    rand = random.uniform(0, total_weight)
    cumulative = 0
    for rule in rules:
        cumulative += rule.get("weight", 0)
        if rand <= cumulative:
            return rule.get("version", 0)
    return rules[-1].get("version", 0)


def pick_header_version(rules: list, headers: dict) -> int:
    """Match request header to route to a specific version. rules: [{"header", "value", "version", "default_version"}, ...]"""
    if not rules:
        raise ValueError("rules empty")
    for rule in rules:
        header_val = (headers.get(rule.get("header", "")) or "").strip().lower()
        if header_val == (rule.get("value") or "").strip().lower():
            return rule.get("version", 0)
    return rules[0].get("default_version", rules[0].get("version", 0))


def pick_experiment_variant(experiment: dict) -> Tuple[int, str]:
    """Pick a variant based on experiment weights. experiment has 'variants': [{"version", "weight", "name"}, ...]"""
    variants = experiment.get("variants") or []
    if not variants:
        raise ValueError("experiment has no variants")
    total = sum(v.get("weight", 0) for v in variants)
    if total <= 0:
        v = variants[0]
        return (v.get("version", 0), v.get("name", "control"))
    rand = random.uniform(0, total)
    cumulative = 0
    for v in variants:
        cumulative += v.get("weight", 0)
        if rand <= cumulative:
            return (v.get("version", 0), v.get("name", "control"))
    v = variants[-1]
    return (v.get("version", 0), v.get("name", "control"))


async def resolve_version(
    org_id: str,
    endpoint_slug: str,
    request_headers: dict,
    pinned_version: Optional[int] = None,
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Returns: (version_number, experiment_id, variant_name).
    Resolution order:
    1. Explicit version pin (?version=v2)
    2. Active experiment (running)
    3. Active routing policy (weighted or header)
    4. Default: latest promoted
    """
    # 1. Explicit pin
    if pinned_version is not None:
        deployment = await get_promoted_deployment_by_version(org_id, endpoint_slug, pinned_version)
        if deployment:
            return (pinned_version, None, None)

    # 2 + 3: Fetch experiment and routing policy in parallel (saves one round-trip)
    experiment_task = asyncio.to_thread(_get_active_experiment_sync, org_id, endpoint_slug)
    policy_task = asyncio.to_thread(_get_active_routing_policy_sync, org_id, endpoint_slug)
    experiment, policy = await asyncio.gather(experiment_task, policy_task)

    # 2. Active experiment
    if experiment and (experiment.get("status") or "") == "running":
        try:
            version, variant_name = pick_experiment_variant(experiment)
            return (version, str(experiment["id"]), variant_name)
        except (ValueError, KeyError):
            pass

    # 3. Active routing policy
    if policy and policy.get("active"):
        rules = policy.get("rules") or []
        ptype = (policy.get("policy_type") or "latest").strip()
        if ptype == "weighted" and rules:
            try:
                version = pick_weighted_version(rules)
                return (version, None, None)
            except ValueError:
                pass
        elif ptype == "header" and rules:
            try:
                version = pick_header_version(rules, request_headers)
                return (version, None, None)
            except ValueError:
                pass

    # 4. Default: latest promoted
    deployment = await get_latest_promoted_deployment(org_id, endpoint_slug)
    if deployment:
        return (deployment.get("version", 0), None, None)

    raise HTTPException(status_code=404, detail=NO_PROMOTED_DEPLOYMENT_DETAIL)


async def resolve_version_and_deployment(
    org_id: str,
    endpoint_slug: str,
    request_headers: dict,
    pinned_version: Optional[int] = None,
) -> Tuple[int, Optional[str], Optional[str], dict]:
    """
    Combined resolver that returns (version, experiment_id, variant_name, deployment).
    Eliminates the redundant second query for the deployment after version resolution.
    """
    # 1. Explicit pin
    if pinned_version is not None:
        deployment = await get_promoted_deployment_by_version(org_id, endpoint_slug, pinned_version)
        if deployment:
            return (pinned_version, None, None, deployment)

    # 2 + 3: Fetch experiment and routing policy in parallel
    experiment_task = asyncio.to_thread(_get_active_experiment_sync, org_id, endpoint_slug)
    policy_task = asyncio.to_thread(_get_active_routing_policy_sync, org_id, endpoint_slug)
    experiment, policy = await asyncio.gather(experiment_task, policy_task)

    resolved_version: Optional[int] = None
    experiment_id: Optional[str] = None
    variant_name: Optional[str] = None

    # 2. Active experiment
    if experiment and (experiment.get("status") or "") == "running":
        try:
            resolved_version, variant_name = pick_experiment_variant(experiment)
            experiment_id = str(experiment["id"])
        except (ValueError, KeyError):
            pass

    # 3. Active routing policy (only if experiment didn't resolve)
    if resolved_version is None and policy and policy.get("active"):
        rules = policy.get("rules") or []
        ptype = (policy.get("policy_type") or "latest").strip()
        if ptype == "weighted" and rules:
            try:
                resolved_version = pick_weighted_version(rules)
            except ValueError:
                pass
        elif ptype == "header" and rules:
            try:
                resolved_version = pick_header_version(rules, request_headers)
            except ValueError:
                pass

    # If experiment or routing resolved a version, fetch that specific deployment
    if resolved_version is not None:
        deployment = await get_promoted_deployment_by_version(org_id, endpoint_slug, resolved_version)
        if deployment:
            return (resolved_version, experiment_id, variant_name, deployment)

    # 4. Default: latest promoted
    deployment = await get_latest_promoted_deployment(org_id, endpoint_slug)
    if deployment:
        return (deployment.get("version", 0), None, None, deployment)

    raise HTTPException(status_code=404, detail=NO_PROMOTED_DEPLOYMENT_DETAIL)


async def update_workflow_run_routing(
    run_id: str,
    served_version: int,
    experiment_id: Optional[str] = None,
    variant_name: Optional[str] = None,
) -> None:
    """Tag a workflow_run with routing metadata after execution."""
    if not run_id:
        return
    update = {
        "served_version": served_version,
        "experiment_id": experiment_id,
        "variant_name": variant_name,
    }
    await asyncio.to_thread(
        lambda: supabase.table("workflow_runs").update(update).eq("id", run_id).execute()
    )
