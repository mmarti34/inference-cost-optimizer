"""
Workload discovery and identity.

A Workload is a piece or CLASS of work with an intended outcome. Nothing here
assumes an LLM performs it.

IDENTITY HAS THREE LEVELS, and the product must work at all three:

  explicit    The customer names the workload on the request ('support-refund',
              via `external_key`). Always wins when present. Customers are NEVER
              REQUIRED to do this — discovery works without it — but they must
              be ABLE to, and naming is the most reliable identity available.

  structural  Derived from what OptiML can see: endpoint slug, workflow id,
              prompt template, or the model target of a direct-inference call.
              This is what `discover_workloads` produces. Implemented.

  learned     Repeated semantically-similar work clustered into one workload.
              DOCUMENTED EXTENSION POINT — see `discover_learned_workloads`.
              Nothing produces it today, and `identity_level='learned'` is never
              written by this module.

Discovery is deliberately simple and defensible: group production executions by
endpoint identity, INDEPENDENT of served_version, because v5 and v6 of the same
endpoint are the same work. Anything cleverer would be guessing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import domain

logger = logging.getLogger(__name__)

WORKLOAD_COLS = (
    "id, org_id, project_id, name, description, surface, identity_kind, identity_ref, "
    "identity_level, external_key, intended_outcome, grain, default_objective, tags, "
    "metadata, created_at, updated_at"
)

_MAX_SCAN_ROWS = 2000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def list_workloads(
    org_id: str,
    *,
    surface: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    try:
        q = supabase.table("workloads").select(WORKLOAD_COLS).eq("org_id", org_id)
        if surface:
            q = q.eq("surface", surface)
        if project_id:
            q = q.eq("project_id", project_id)
        resp = q.order("created_at", desc=True).limit(max(1, min(limit, 500))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_workloads failed: %s", type(exc).__name__)
        return []


def get_workload(org_id: str, workload_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("workloads")
            .select(WORKLOAD_COLS)
            .eq("id", workload_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as exc:  # pragma: no cover
        logger.warning("get_workload failed: %s", type(exc).__name__)
        return None


def resolve_workflow_id(org_id: str, workload: dict) -> Optional[str]:
    """
    The workflow behind a workload, when there is one.

    Returns None for a direct-inference workload — which has no workflow and no
    deployment. Callers must handle None rather than assuming a deployment.
    """
    kind = workload.get("identity_kind")
    ref = workload.get("identity_ref")
    if not ref:
        return None
    if kind == "workflow":
        return str(ref)
    if kind == "endpoint":
        try:
            resp = (
                supabase.table("workflow_deployments")
                .select("workflow_id, version")
                .eq("org_id", org_id)
                .eq("endpoint_slug", str(ref))
                .order("version", desc=True)
                .limit(1)
                .execute()
            )
            rows = getattr(resp, "data", None) or []
            return str(rows[0]["workflow_id"]) if rows else None
        except Exception as exc:  # pragma: no cover
            logger.warning("resolve_workflow_id failed: %s", type(exc).__name__)
            return None
    return None


# ---------------------------------------------------------------------------
# Identity resolution — the entry point other surfaces call
# ---------------------------------------------------------------------------

def resolve_workload(
    org_id: str,
    *,
    external_key: Optional[str] = None,
    surface: str = "runtime",
    endpoint_slug: Optional[str] = None,
    workflow_id: Optional[str] = None,
    prompt_template_id: Optional[str] = None,
    model_target: Optional[str] = None,
    project_id: Optional[str] = None,
    name: Optional[str] = None,
    create: bool = True,
) -> Optional[dict]:
    """
    Resolve (and optionally create) the workload for a unit of work.

    THIS IS THE INTERFACE OTHER SURFACES SHOULD CALL — including the Direct
    Inference path (POST /v1/chat/completions), which has no workflow and no
    deployment. It never requires the customer to have named anything.

    Precedence:
      1. `external_key` — the customer named it. Wins over everything, and is
         matched across surfaces so one name can cover both Studio traffic and
         direct-inference traffic.
      2. Structural identity, in order: workflow_id, endpoint_slug,
         prompt_template_id, model_target.
      3. None (or a newly created workload when `create`).
    """
    if external_key:
        key = str(external_key).strip()
        existing = _find_by_external_key(org_id, key)
        if existing:
            return existing
        if not create:
            return None
        return _insert_workload({
            "org_id": org_id,
            "project_id": project_id,
            "name": name or key,
            "surface": surface,
            "identity_kind": "explicit",
            "identity_ref": key,
            "identity_level": "explicit",
            "external_key": key,
            "grain": "task_class",
        })

    structural: list[tuple[str, Optional[str], str]] = [
        ("workflow", workflow_id, "workflow"),
        ("endpoint", endpoint_slug, "endpoint"),
        ("prompt_template", prompt_template_id, "task_class"),
        ("model_endpoint", model_target, "task_class"),
    ]
    for identity_kind, ref, grain in structural:
        if not ref:
            continue
        existing = _find_by_identity(org_id, surface, identity_kind, str(ref))
        if existing:
            return existing
        if not create:
            continue
        return _insert_workload({
            "org_id": org_id,
            "project_id": project_id,
            "name": name or f"{identity_kind}:{ref}",
            "surface": surface,
            "identity_kind": identity_kind,
            "identity_ref": str(ref),
            "identity_level": "structural",
            "grain": grain,
        })

    return None


def _find_by_external_key(org_id: str, external_key: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("workloads")
            .select(WORKLOAD_COLS)
            .eq("org_id", org_id)
            .eq("external_key", external_key)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception:  # pragma: no cover
        return None


def _find_by_identity(
    org_id: str, surface: str, identity_kind: str, identity_ref: str
) -> Optional[dict]:
    try:
        resp = (
            supabase.table("workloads")
            .select(WORKLOAD_COLS)
            .eq("org_id", org_id)
            .eq("surface", surface)
            .eq("identity_kind", identity_kind)
            .eq("identity_ref", identity_ref)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception:  # pragma: no cover
        return None


def _insert_workload(row: dict) -> Optional[dict]:
    row = {**row, "updated_at": _iso(_utc_now())}
    try:
        result = supabase.table("workloads").insert(row).execute()
        return (result.data or [None])[0]
    except Exception as exc:
        # Lost a race, or the unique index rejected a duplicate: re-read.
        if row.get("external_key"):
            found = _find_by_external_key(row["org_id"], row["external_key"])
            if found:
                return found
        if row.get("identity_ref"):
            found = _find_by_identity(
                row["org_id"], row.get("surface") or "runtime",
                row.get("identity_kind") or "endpoint", row["identity_ref"],
            )
            if found:
                return found
        logger.warning("_insert_workload failed: %s", type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# Structural discovery
# ---------------------------------------------------------------------------

def discover_workloads(org_id: str, *, lookback_days: int = 30) -> dict:
    """
    Discover runtime workloads from observed production traffic.

    Groups `workflow_runs` with execution_mode='production' by endpoint_slug,
    INDEPENDENT of served_version: v5 and v6 of an endpoint are the same work,
    and treating them as different workloads would fragment every measurement.

    Upserts one workload per endpoint. Existing workloads are never downgraded:
    a workload the customer has explicitly named keeps its explicit identity.

    Returns measured counts plus a `coverage` block. Runs with no endpoint_slug
    (draft/eval executions) are counted and excluded rather than silently
    dropped.
    """
    since = _utc_now() - timedelta(days=max(1, lookback_days))
    coverage: dict[str, Any] = {
        "window_days": lookback_days,
        "since": _iso(since),
        "row_cap": _MAX_SCAN_ROWS,
        "truncated": False,
        "source": "workflow_runs(execution_mode=production)",
    }

    try:
        resp = (
            supabase.table("workflow_runs")
            .select("id, workflow_id, endpoint_slug, total_cost, created_at")
            .eq("org_id", org_id)
            .eq("execution_mode", "production")
            .gte("created_at", _iso(since))
            .order("created_at", desc=True)
            .limit(_MAX_SCAN_ROWS)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
    except Exception as exc:
        logger.warning("discover_workloads scan failed: %s", type(exc).__name__)
        return {
            "discovered": [],
            "created": 0,
            "existing": 0,
            "coverage": {**coverage, "error": "query_failed"},
        }

    if len(rows) >= _MAX_SCAN_ROWS:
        coverage["truncated"] = True

    groups: dict[str, dict] = {}
    without_endpoint = 0
    for r in rows:
        slug = (r.get("endpoint_slug") or "").strip()
        if not slug:
            without_endpoint += 1
            continue
        g = groups.setdefault(slug, {
            "endpoint_slug": slug,
            "workflow_ids": set(),
            "run_count": 0,
            "observed_cost_usd": 0.0,
        })
        g["run_count"] += 1
        if r.get("workflow_id"):
            g["workflow_ids"].add(str(r["workflow_id"]))
        if r.get("total_cost") is not None:
            g["observed_cost_usd"] += float(r["total_cost"])

    project_by_workflow = _project_ids_for_workflows(
        org_id, {wid for g in groups.values() for wid in g["workflow_ids"]}
    )

    discovered: list[dict] = []
    created = existing = 0

    for slug, g in groups.items():
        workflow_ids = sorted(g["workflow_ids"])
        project_id = next(
            (project_by_workflow.get(w) for w in workflow_ids if project_by_workflow.get(w)),
            None,
        )

        found = _find_by_identity(org_id, "runtime", "endpoint", slug)
        if found:
            existing += 1
            workload = found
            if project_id and not found.get("project_id"):
                try:
                    supabase.table("workloads").update({
                        "project_id": project_id,
                        "updated_at": _iso(_utc_now()),
                    }).eq("id", found["id"]).eq("org_id", org_id).execute()
                    workload = {**found, "project_id": project_id}
                except Exception:  # pragma: no cover
                    pass
        else:
            workload = _insert_workload({
                "org_id": org_id,
                "project_id": project_id,
                "name": slug,
                "description": None,
                "surface": "runtime",
                "identity_kind": "endpoint",
                "identity_ref": slug,
                "identity_level": "structural",
                "grain": "endpoint",
                "default_objective": domain.DEFAULT_OBJECTIVE,
                "metadata": {"discovered_from": "workflow_runs", "workflow_ids": workflow_ids},
            })
            if workload is None:
                continue
            created += 1

        discovered.append({
            "workload_id": str(workload["id"]),
            "endpoint_slug": slug,
            "workflow_ids": workflow_ids,
            "project_id": project_id,
            "identity_level": workload.get("identity_level"),
            # Measured over the window; not extrapolated.
            "observed_run_count": g["run_count"],
            "observed_cost_usd": round(g["observed_cost_usd"], 8),
        })

    coverage["runs_scanned"] = len(rows)
    coverage["runs_without_endpoint_slug"] = without_endpoint
    coverage["note"] = (
        "Structural discovery only: production runs grouped by endpoint_slug, "
        "version-independent. Runs with no endpoint_slug (draft/eval) are excluded. "
        "Semantic ('learned') clustering is not performed — see "
        "discover_learned_workloads."
    )

    return {
        "discovered": sorted(discovered, key=lambda d: -d["observed_cost_usd"]),
        "created": created,
        "existing": existing,
        "coverage": coverage,
    }


def _project_ids_for_workflows(org_id: str, workflow_ids: set[str]) -> dict[str, Optional[str]]:
    if not workflow_ids:
        return {}
    try:
        resp = (
            supabase.table("workflows")
            .select("id, project_id")
            .eq("org_id", org_id)
            .in_("id", sorted(workflow_ids))
            .execute()
        )
        return {
            str(r["id"]): (str(r["project_id"]) if r.get("project_id") else None)
            for r in (getattr(resp, "data", None) or [])
        }
    except Exception:  # pragma: no cover
        return {}


def discover_learned_workloads(org_id: str, **_kwargs) -> dict:
    """
    EXTENSION POINT — not implemented, and deliberately not faked.

    Learned identity means clustering repeated semantically-similar work into
    one workload even when it arrives through different endpoints, prompts or
    surfaces. It is the third identity level and the schema already supports it
    (`identity_level='learned'`, `identity_kind='inferred'`, `grain='cluster'`).

    What would be required to build it honestly:

      1. A stable representation of "what this work is": input embeddings plus
         the prompt/tool shape. The repo has embedding infrastructure
         (context_embeddings.py) but nothing that embeds workload inputs.
      2. A clustering pass with a measured stability criterion — a cluster that
         reshuffles between runs would make every longitudinal metric on that
         workload meaningless.
      3. A merge/split protocol for when a cluster is discovered to be two
         kinds of work, including what happens to benchmarks and realized
         savings already attributed to the old cluster.
      4. Human confirmation, because a wrong cluster silently mixes unrelated
         work into one optimization decision.

    Until those exist, this raises rather than returning a plausible-looking
    grouping. A guessed cluster would corrupt every measurement attached to it.
    """
    raise NotImplementedError(
        "Learned (semantic) workload discovery is not implemented. "
        "Requires input embeddings, a stability criterion, a merge/split "
        "protocol and human confirmation. See the docstring."
    )


def workload_row_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "project_id": (str(row["project_id"]) if row.get("project_id") else None),
        "name": row.get("name"),
        "description": row.get("description"),
        "surface": row.get("surface"),
        "identity_kind": row.get("identity_kind"),
        "identity_ref": row.get("identity_ref"),
        "identity_level": row.get("identity_level"),
        "external_key": row.get("external_key"),
        "intended_outcome": row.get("intended_outcome"),
        "grain": row.get("grain"),
        "default_objective": row.get("default_objective"),
        "tags": row.get("tags") or [],
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


# ---------------------------------------------------------------------------
# Structural identity for Direct Inference
# ---------------------------------------------------------------------------
# `resolve_workload(model_target=...)` above needs a STRING that identifies a
# direct-inference workload structurally. On the Studio surface that string is
# obvious — the endpoint slug. Direct Inference has no endpoint, so the string
# has to be derived from the shape of the request itself.
#
# This lives here, next to resolve_workload, because there must be exactly one
# workload-identity implementation in the codebase. The Direct Inference router
# does not compute identity of its own; it calls
# `direct_inference_identity()` and hands the result to `resolve_workload`.
#
# WHY THESE FOUR COMPONENTS
#   model family      the customer's current model IS their baseline strategy.
#                     Included as the FAMILY only: `gpt-4o-2024-08-06` and
#                     `gpt-4o` must be one workload, or a model upgrade would
#                     look like a brand-new workload instead of a strategy
#                     change on an existing one — which is the entire product.
#   system prompt     the strongest available signal of what job this is. Two
#                     calls sharing a system prompt are nearly always the same
#                     work; two with different ones nearly never are.
#   tool signature    the contract the caller expects back, which constrains
#                     which candidate models are even eligible.
#   response_format   same reason.
#
# DELIBERATELY EXCLUDED: user messages, conversation content, end-user ids,
# conversation ids. Those vary per request; including them would mint a new
# workload per call and make discovery useless. They are also customer data
# with no business being hashed into a long-lived identifier.
#
# Tool DESCRIPTIONS are excluded too: rewording a tool description is a prompt
# change (a strategy change on this workload), not different work.

import hashlib as _hashlib
import json as _json
import re as _re

_WHITESPACE_RE = _re.compile(r"\s+")

#: Trailing variant suffixes stripped so dated snapshots collapse to a family.
_MODEL_VARIANT_SUFFIX_RE = _re.compile(
    r"(-\d{4}-\d{2}-\d{2}|-\d{8}|-latest|-preview|-turbo-preview)$", _re.IGNORECASE
)

#: Bumped only if the components change meaning. A bump re-mints every
#: structural direct-inference workload, so it is not done casually.
DIRECT_INFERENCE_IDENTITY_VERSION = 1


def normalize_model_family(model: str) -> str:
    """`gpt-4o-2024-08-06` -> `gpt-4o`; `claude-sonnet-4-5-20250929` -> `claude-sonnet-4-5`."""
    m = (model or "").strip().lower()
    prev = None
    while prev != m:
        prev = m
        m = _MODEL_VARIANT_SUFFIX_RE.sub("", m)
    return m or "unknown"


def _system_text(messages: Any) -> str:
    parts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").lower() not in ("system", "developer"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
    return _WHITESPACE_RE.sub(" ", "\n".join(parts)).strip()


def _schema_keys(schema: Any, prefix: str = "", depth: int = 0) -> list[str]:
    """Flatten a JSON-Schema's property paths — the tool's contract, not its prose."""
    if depth > 6 or not isinstance(schema, dict):
        return []
    out: list[str] = []
    props = schema.get("properties")
    if isinstance(props, dict):
        for name in sorted(props):
            path = f"{prefix}{name}"
            child = props[name]
            ctype = child.get("type") if isinstance(child, dict) else None
            out.append(f"{path}:{ctype or 'any'}")
            out.extend(_schema_keys(child, f"{path}.", depth + 1))
    items = schema.get("items")
    if isinstance(items, dict):
        out.extend(_schema_keys(items, f"{prefix}[].", depth + 1))
    return out


def tool_signature(tools: Any) -> tuple[str, tuple[str, ...]]:
    """Order-independent fingerprint of an OpenAI `tools` array."""
    if not tools:
        return "none", ()
    entries: list[str] = []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        names.append(name)
        entries.append(name + "(" + ",".join(sorted(_schema_keys(fn.get("parameters") or {}))) + ")")
    if not entries:
        return "none", ()
    return (
        _hashlib.sha256("|".join(sorted(entries)).encode("utf-8")).hexdigest()[:16],
        tuple(sorted(names)),
    )


def _response_format_key(response_format: Any) -> str:
    if not isinstance(response_format, dict):
        return "none"
    rtype = str(response_format.get("type") or "").strip() or "none"
    if rtype == "json_schema":
        name = str((response_format.get("json_schema") or {}).get("name") or "").strip()
        return f"json_schema:{name or 'unnamed'}"
    return rtype


def direct_inference_identity(
    *,
    provider: str,
    model: str,
    messages: Any = None,
    tools: Any = None,
    response_format: Any = None,
) -> dict:
    """
    Derive the STRUCTURAL identity of one direct-inference request.

    Pure: no I/O, safe on the request hot path. Returns::

        {
          "model_target": "di_<20 hex>",   # pass to resolve_workload(model_target=)
          "name":         "gpt-4o - tools: get_weather - sys:57e8f4",
          "components":   {...}            # what went into the hash, for debugging
        }

    Persisting and de-duplicating this is `resolve_workload`'s job, not this
    function's. Naming a workload explicitly still wins over everything here —
    pass `external_key` to `resolve_workload` and this identity is not used.
    """
    prov = (provider or "unknown").strip().lower()
    family = normalize_model_family(model)
    system = _system_text(messages)
    system_fp = (
        _hashlib.sha256(system.encode("utf-8")).hexdigest()[:16] if system else "none"
    )
    tool_fp, tool_names = tool_signature(tools)
    rf = _response_format_key(response_format)

    canonical = _json.dumps(
        {
            "v": DIRECT_INFERENCE_IDENTITY_VERSION,
            "provider": prov,
            "model_family": family,
            "system": system_fp,
            "tools": tool_fp,
            "response_format": rf,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    model_target = "di_" + _hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    bits = [family]
    if tool_names:
        head = ", ".join(tool_names[:2])
        more = f" +{len(tool_names) - 2}" if len(tool_names) > 2 else ""
        bits.append(f"tools: {head}{more}")
    if rf not in ("none", "text"):
        bits.append(rf)
    if system_fp != "none":
        bits.append(f"sys:{system_fp[:6]}")

    return {
        "model_target": model_target,
        "name": " - ".join(bits)[:120],
        "components": {
            "provider": prov,
            "model_family": family,
            "system_fingerprint": system_fp,
            "system_chars": len(system),
            "tool_signature": tool_fp,
            "tool_names": list(tool_names),
            "response_format": rf,
        },
    }
