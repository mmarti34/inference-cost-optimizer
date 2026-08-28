"""
Ownership-checked resource lookups for tenant-scoped tables.

WHY THIS MODULE EXISTS
──────────────────────
``supabase_client`` uses the SERVICE-ROLE key, so Postgres RLS never applies to
anything the backend does. The query filters ARE the authorization. A query
written as::

    supabase.table("workflow_deployments").select("*").eq("id", deployment_id)

is therefore an unauthenticated read of that row, no matter which guard the
route is decorated with: ``require_org_member`` proves the caller belongs to
the org they named, and says nothing whatsoever about whether the id they also
supplied lives in that org. Reading ``org_id`` back out of the fetched row and
trusting it inverts the check — the caller's own input decides which tenant the
handler operates as.

The anti-pattern this module replaces, and which must not come back::

    row = table.select("*").eq("id", rid).single().execute()   # disclosure
    org_id = row.data["org_id"]                                # attacker-chosen
    require_membership(user, org_id)                           # too late

Authorization must precede disclosure.

THE HARD INVARIANT
──────────────────
Every helper here takes the ``AuthenticatedUser`` produced by
``require_org_member``/``require_org_admin`` and derives the org from
``verified_org_id()`` itself. None of them accepts an org id as an argument.
A caller therefore *cannot* pass a naked, caller-supplied org string in — there
is no parameter to put it in — and cannot forget to filter, because the filter
is applied inside the helper. Passing anything that is not an
``AuthenticatedUser`` is a programming error and raises immediately.

TENANT OPACITY
──────────────
"this id belongs to another org" and "this id does not exist" must be
indistinguishable, or the endpoint becomes an existence oracle for other
tenants' resource ids. Both raise the SAME ``HTTPException`` — same status,
same detail string — per resource class. This matches what
``POST /v1/outcomes``, ``POST /v1/prompt`` and the public execution surface
already do.

MUTATIONS
─────────
These helpers authorize a *read*. Mutations must ALSO carry the org filter, so
that the check and the write are one atomic statement rather than a
check-then-act window::

    supabase.table(...).delete().eq("id", rid).eq("org_id", verified_org(user))
"""
import logging
from typing import Any, Iterable, Optional

from fastapi import HTTPException

from auth_dependency import AuthenticatedUser, verified_org_id
from supabase_client import supabase

logger = logging.getLogger(__name__)


# ─── Opaque failure details, one per resource class ──────────────────────────
# Reused verbatim for BOTH "not yours" and "does not exist". Never add the id,
# the owning org, or a distinguishing suffix to any of these.
WORKFLOW_NOT_FOUND = "Workflow not found"
DEPLOYMENT_NOT_FOUND = "Deployment not found"
PROJECT_NOT_FOUND = "Project not found"
PROVIDER_CREDENTIAL_NOT_FOUND = "API key not found for org/provider."

_LOOKUP_FAILED = "Error verifying resource access."


def verified_org(auth_user: AuthenticatedUser) -> str:
    """The org ``require_org_member`` actually proved membership of.

    Thin re-export of ``auth_dependency.verified_org_id`` so a handler needs
    only one import to write a correctly scoped query.
    """
    return verified_org_id(auth_user)


def _require_verified_identity(auth_user: Any) -> str:
    """Reject anything that is not a guard-produced identity.

    This is the invariant in executable form. Without it a future caller could
    pass ``payload.org_id`` positionally and quietly reintroduce a
    caller-controlled filter that *looks* scoped.
    """
    if not isinstance(auth_user, AuthenticatedUser):
        raise TypeError(
            "resource_access helpers require the AuthenticatedUser from "
            "require_org_member — never a caller-supplied org id."
        )
    return verified_org_id(auth_user)


def fetch_owned_row(
    table: str,
    resource_id: Optional[str],
    auth_user: AuthenticatedUser,
    columns: str,
    not_found_detail: str,
) -> dict:
    """Fetch one row by id, constrained to the caller's verified org.

    Returns the row, or raises the resource class's opaque 404. The org filter
    is applied here, in the same statement as the id filter, so there is no
    point at which the row's contents exist in the handler without ownership
    having been established.
    """
    org_id = _require_verified_identity(auth_user)

    rid = str(resource_id or "").strip()
    if not rid:
        # An empty id cannot own anything. Answer exactly as for a foreign id.
        raise HTTPException(status_code=404, detail=not_found_detail)

    try:
        result = (
            supabase.table(table)
            .select(columns)
            .eq("id", rid)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        # Never echo the driver error: it can carry row contents and ids from
        # the failed query. Operator-side log keeps the detail.
        logger.warning("Ownership lookup on %s failed: %s", table, type(e).__name__)
        raise HTTPException(status_code=500, detail=_LOOKUP_FAILED) from e

    rows = result.data or []
    if isinstance(rows, dict):  # a .single()-shaped response, defensively
        rows = [rows]
    if not rows:
        raise HTTPException(status_code=404, detail=not_found_detail)
    return rows[0]


# ─── Per resource class ──────────────────────────────────────────────────────


def get_workflow_for_org(
    workflow_id: Optional[str],
    auth_user: AuthenticatedUser,
    columns: str = "id, org_id",
) -> dict:
    """The workflow, if it belongs to the caller's verified org. Else 404."""
    return fetch_owned_row(
        "workflows", workflow_id, auth_user, columns, WORKFLOW_NOT_FOUND
    )


def get_deployment_for_org(
    deployment_id: Optional[str],
    auth_user: AuthenticatedUser,
    columns: str = "id, org_id",
) -> dict:
    """The deployment, if it belongs to the caller's verified org. Else 404."""
    return fetch_owned_row(
        "workflow_deployments", deployment_id, auth_user, columns, DEPLOYMENT_NOT_FOUND
    )


def get_project_for_org(
    project_id: Optional[str],
    auth_user: AuthenticatedUser,
    columns: str = "id, org_id",
) -> dict:
    """The project, if it belongs to the caller's verified org. Else 404."""
    return fetch_owned_row(
        "projects", project_id, auth_user, columns, PROJECT_NOT_FOUND
    )


def get_provider_credential_for_org(
    provider: str,
    auth_user: AuthenticatedUser,
    columns: str = "id, org_id, provider, api_key, name, user_id, created_at",
) -> dict:
    """The org's stored credential for one provider.

    Keyed by (org_id, provider) rather than by row id, but the same rule holds:
    the org comes from the guard, never from the request body. Raises the
    opaque 404 when the org has no key for that provider.
    """
    org_id = _require_verified_identity(auth_user)
    prov = (provider or "").strip()
    if not prov:
        raise HTTPException(status_code=404, detail=PROVIDER_CREDENTIAL_NOT_FOUND)

    try:
        result = (
            supabase.table("api_keys")
            .select(columns)
            .eq("org_id", org_id)
            .eq("provider", prov)
            .execute()
        )
    except Exception as e:
        logger.warning("Provider credential lookup failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail=_LOOKUP_FAILED) from e

    rows = result.data or []
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        raise HTTPException(status_code=404, detail=PROVIDER_CREDENTIAL_NOT_FOUND)
    return rows[0]


def get_context_assets_for_org(
    asset_ids: Iterable[str],
    auth_user: AuthenticatedUser,
    columns: str = "id, content, metadata",
) -> list:
    """Context assets from a caller-supplied id list, restricted to their org.

    A set-valued lookup, so the correct failure is SILENT OMISSION rather than
    404: ids belonging to other tenants simply do not come back. The caller
    must treat "fewer rows than ids" as normal and must not fall back to an
    unfiltered read.
    """
    org_id = _require_verified_identity(auth_user)
    ids = [str(a).strip() for a in (asset_ids or []) if str(a or "").strip()]
    if not ids:
        return []

    try:
        result = (
            supabase.table("context_assets")
            .select(columns)
            .in_("id", ids)
            .eq("org_id", org_id)
            .execute()
        )
    except Exception as e:
        logger.warning("Context asset lookup failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail=_LOOKUP_FAILED) from e

    rows = result.data or []
    if isinstance(rows, dict):
        rows = [rows]
    return list(rows)
