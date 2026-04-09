"""
Synthetic Mind — Hierarchical Scope Management.

Manages WHAT understanding is available to WHICH requests.

Scope hierarchy:
  Platform → Organization → Team → Workflow → User → Session

Lower scopes inherit from higher scopes.
Lower scopes can override higher scopes (explicit, auditable).
Scope isolation: Org A's understanding never leaks to Org B.
"""
import logging
from typing import Any, Optional

from synthetic_mind.memory_store import (
    ensure_scope,
    get_active_memories,
)

logger = logging.getLogger(__name__)


def resolve_scope_chain(
    org_id: str,
    *,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
) -> list[dict]:
    """
    Build the scope chain for a request context.
    Returns scopes from broadest (org) to narrowest (user/workflow).

    Each scope dict contains:
      - scope_type: str
      - scope_id: str (from sm_memory_scopes)
      - filter_kwargs: dict (for querying memory units)
    """
    chain = []

    # Organization scope (always present)
    org_scope_id = ensure_scope(org_id, "organization", entity_id=org_id)
    chain.append({
        "scope_type": "organization",
        "scope_id": org_scope_id,
        "filter_kwargs": {"org_id": org_id},
    })

    # Workflow scope
    if workflow_id:
        wf_scope_id = ensure_scope(
            org_id, "workflow",
            entity_id=workflow_id,
            entity_slug=endpoint_slug,
            parent_scope_id=org_scope_id,
        )
        chain.append({
            "scope_type": "workflow",
            "scope_id": wf_scope_id,
            "filter_kwargs": {"org_id": org_id, "workflow_id": workflow_id},
        })

    # User scope
    if user_id:
        user_scope_id = ensure_scope(
            org_id, "user",
            entity_id=user_id,
            parent_scope_id=org_scope_id,
        )
        chain.append({
            "scope_type": "user",
            "scope_id": user_scope_id,
            "filter_kwargs": {"org_id": org_id, "user_id": user_id},
        })

    return chain


def load_scoped_memories(
    org_id: str,
    *,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    endpoint_slug: Optional[str] = None,
    min_confidence: float = 0.2,
    limit_per_scope: int = 20,
) -> list[dict]:
    """
    Load memories from all applicable scopes, with inheritance.

    Returns a flat list of memory units ordered by:
    1. Scope specificity (narrower scopes first — they override)
    2. Confidence (higher first)

    Deduplication: if a narrower scope has a memory with the same
    (subject, predicate), it overrides the broader one.
    """
    chain = resolve_scope_chain(
        org_id,
        workflow_id=workflow_id,
        user_id=user_id,
        endpoint_slug=endpoint_slug,
    )

    # Collect memories from broadest to narrowest
    seen_keys: set[tuple[str, str]] = set()
    result: list[dict] = []

    # Process in reverse (narrowest first) so narrow scopes win
    for scope in reversed(chain):
        memories = get_active_memories(
            min_confidence=min_confidence,
            limit=limit_per_scope,
            **scope["filter_kwargs"],
        )
        for mem in memories:
            key = (mem["subject"], mem["predicate"])
            if key not in seen_keys:
                seen_keys.add(key)
                mem["_scope_type"] = scope["scope_type"]
                result.append(mem)

    # Sort: narrower scope first, then by confidence
    scope_order = {"session": 0, "user": 1, "workflow": 2, "team": 3, "organization": 4, "platform": 5}
    result.sort(key=lambda m: (scope_order.get(m.get("_scope_type", ""), 5), -m.get("confidence", 0)))

    return result
