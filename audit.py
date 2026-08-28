"""
The security audit trail: one writer for `public.audit_log`.

WHY THIS MODULE EXISTS
──────────────────────
`public.audit_log` was designed and never written to. The table exists with a
well-formed schema and zero rows, and a repo-wide search for its name returned
nothing. The consequence was concrete: when asked "did anyone exploit the
endpoint that let any authenticated user revoke any tenant's production API
key?", the only available answer was "the database cannot tell us."

This module exists so that question has an answer. Every security-relevant
mutation — credential minted or revoked, secret written or destroyed, member
invited or removed, production deployment moved — passes through `record()`.

NOT REDACTION. NOT COLLECTING.
──────────────────────────────
`evidence_redaction.py` solves a different problem: customer request content
*must* be persisted for replay, so it is scrubbed on the way to the row. That is
redaction — collect, then remove what is dangerous.

This module does the opposite and stricter thing: it NEVER COLLECTS. There is no
scrubbing pass here because no prompt, response body, request body, header,
credential, token, or secret value is ever handed to the writer in the first
place. The metadata allow-list below is a closed set of identifiers and codes;
anything not on it is dropped before the insert. A redactor can be wrong about
what it recognises. An allow-list that never sees the value cannot be.

Consequence, stated plainly: an audit row can tell you WHO did WHAT to WHICH
resource and WHEN. It can never tell you what the secret was, what the prompt
said, or what the customer sent. That is the intended limit, not a gap.

BEST EFFORT, ALWAYS
───────────────────
An audit write must never turn a successful operation into a customer-facing
error. Every public function here swallows every exception. If the audit table
is unreachable, the caller's work still commits and the request still succeeds;
the failure is visible operator-side in the log and nowhere else. This is
deliberate: an audit trail that can take the product down would be turned off.

THE ORG IS NEVER CALLER-SUPPLIED
────────────────────────────────
Same invariant as `resource_access.py`, for the same reason. `supabase_client`
uses the SERVICE-ROLE key, so an org id taken from a request body is an
attacker-chosen tenant label. An audit row filed under the wrong tenant is worse
than no row: it is a false alibi.

`record()` therefore takes the guard-produced `AuthenticatedUser` and derives
the org itself. There is no `org_id` parameter to put a body value in, and
anything that is not an `AuthenticatedUser` raises `TypeError` — the same
executable invariant as `resource_access._require_verified_identity`.

Three surfaces have no verified org because they authenticate differently. Each
gets its own narrow, loudly-named entry point, and each still derives the org
from the SERVER, never from the request:

    record_for_server_key(...)      org from OrgContext.org_id (the key row)
    record_server_derived(...)      org read from a database row the server
                                    fetched, e.g. the org that owns the service
                                    key being revoked, or the org named on a
                                    redeemed invite token

`record_server_derived` is the one to be suspicious of in review. Its contract
is that `org_id` is a value the SERVER read out of the database, never a value
the client sent. Callers pass `derived_from=` naming the row it came from, so
the provenance of every such call is greppable.

REFUSALS ARE RECORDED
─────────────────────
A refused cross-tenant attempt is at least as interesting as a successful
action — it is the shape of an attack in progress, and the successful-only log
is the one that cannot answer "did anyone TRY?". Refusals are recorded as their
own action constants, all suffixed `.refused` and all listed in
`REFUSAL_ACTIONS`, and `record()` stamps `metadata["outcome"] = "refused"`
automatically so a typo in the action name cannot produce a refusal row that
reads as a success. Query either way:

    SELECT * FROM audit_log WHERE action LIKE '%.refused';
    SELECT * FROM audit_log WHERE metadata->>'outcome' = 'refused';

CLOSED ACTION VOCABULARY
────────────────────────
Actions are module-level constants, never free strings at the call site. The set
is therefore greppable and complete, and a typo is an ImportError rather than a
silently invented action name that no dashboard will ever count.
"""
from __future__ import annotations

import ipaddress
import logging
import uuid as _uuid
from typing import Any, Mapping, Optional

from auth_dependency import AuthenticatedUser
from supabase_client import supabase

logger = logging.getLogger(__name__)

#: The table. Named once so a rename is one edit.
AUDIT_TABLE = "audit_log"


# ─── Action vocabulary ───────────────────────────────────────────────────────
# Closed set. Add a constant here before adding a call site; never pass a
# literal. `<noun>.<verb>` throughout, `.refused` for a denied attempt.

# Server API keys (the credential that authenticates the public execution API)
SERVER_KEY_CREATED = "server_api_key.created"
SERVER_KEY_REVOKED = "server_api_key.revoked"
SERVER_KEY_REVOKE_REFUSED = "server_api_key.revoke.refused"
SERVER_KEY_UPDATED = "server_api_key.updated"

# Cursor tokens (long-lived org-scoped bearer tokens for the editor plugin)
CURSOR_TOKEN_CREATED = "cursor_token.created"
CURSOR_TOKEN_REVOKED = "cursor_token.revoked"
CURSOR_TOKEN_REVOKE_REFUSED = "cursor_token.revoke.refused"

# Provider credentials (the org's OpenAI/Anthropic/... keys, encrypted at rest)
PROVIDER_CREDENTIAL_CREATED = "provider_credential.created"
PROVIDER_CREDENTIAL_OVERWRITTEN = "provider_credential.overwritten"
PROVIDER_CREDENTIAL_DELETED = "provider_credential.deleted"
PROVIDER_CREDENTIAL_DELETE_REFUSED = "provider_credential.delete.refused"

# Org secrets ({{secrets.NAME}} referenced from workflow graphs)
ORG_SECRET_CREATED = "org_secret.created"
ORG_SECRET_UPDATED = "org_secret.updated"
ORG_SECRET_DELETED = "org_secret.deleted"
ORG_SECRET_UPDATE_REFUSED = "org_secret.update.refused"
ORG_SECRET_DELETE_REFUSED = "org_secret.delete.refused"

# Membership
MEMBER_INVITED = "org_member.invited"
MEMBER_INVITE_REVOKED = "org_member.invite_revoked"
MEMBER_INVITE_ACCEPTED = "org_member.invite_accepted"
MEMBER_REMOVED = "org_member.removed"
MEMBER_REMOVE_REFUSED = "org_member.remove.refused"

# Organization lifecycle and settings
ORGANIZATION_CREATED = "organization.created"
ORGANIZATION_UPDATED = "organization.updated"

# Optimization decisions that can reach production
RECOMMENDATION_ACCEPTED = "optimization_recommendation.accepted"
RECOMMENDATION_REJECTED = "optimization_recommendation.rejected"

# Webhook triggers (a customer-controlled egress path into their own systems,
# authenticated by a signing secret this table stores). Creating one opens that
# path; rotating the secret changes who can drive it; deleting one closes it.
WEBHOOK_CREATED = "webhook_trigger.created"
WEBHOOK_UPDATED = "webhook_trigger.updated"
WEBHOOK_SECRET_ROTATED = "webhook_trigger.secret_rotated"
WEBHOOK_DELETED = "webhook_trigger.deleted"
WEBHOOK_CREATE_REFUSED = "webhook_trigger.create.refused"
WEBHOOK_UPDATE_REFUSED = "webhook_trigger.update.refused"
WEBHOOK_DELETE_REFUSED = "webhook_trigger.delete.refused"

# Deployments
DEPLOYMENT_PROMOTED = "workflow_deployment.promoted"
DEPLOYMENT_ACTIVATED = "workflow_deployment.activated"
DEPLOYMENT_ROLLED_BACK = "workflow_deployment.rolled_back"
DEPLOYMENT_DELETED = "workflow_deployment.deleted"

#: Every action this module may write. `record()` refuses anything else, so a
#: hand-typed string cannot create a new action name by accident.
ACTIONS = frozenset({
    SERVER_KEY_CREATED,
    SERVER_KEY_REVOKED,
    SERVER_KEY_REVOKE_REFUSED,
    SERVER_KEY_UPDATED,
    CURSOR_TOKEN_CREATED,
    CURSOR_TOKEN_REVOKED,
    CURSOR_TOKEN_REVOKE_REFUSED,
    PROVIDER_CREDENTIAL_CREATED,
    PROVIDER_CREDENTIAL_OVERWRITTEN,
    PROVIDER_CREDENTIAL_DELETED,
    PROVIDER_CREDENTIAL_DELETE_REFUSED,
    ORG_SECRET_CREATED,
    ORG_SECRET_UPDATED,
    ORG_SECRET_DELETED,
    ORG_SECRET_UPDATE_REFUSED,
    ORG_SECRET_DELETE_REFUSED,
    MEMBER_INVITED,
    MEMBER_INVITE_REVOKED,
    MEMBER_INVITE_ACCEPTED,
    MEMBER_REMOVED,
    MEMBER_REMOVE_REFUSED,
    ORGANIZATION_CREATED,
    ORGANIZATION_UPDATED,
    RECOMMENDATION_ACCEPTED,
    RECOMMENDATION_REJECTED,
    DEPLOYMENT_PROMOTED,
    DEPLOYMENT_ACTIVATED,
    DEPLOYMENT_ROLLED_BACK,
    DEPLOYMENT_DELETED,
    WEBHOOK_CREATED,
    WEBHOOK_UPDATED,
    WEBHOOK_SECRET_ROTATED,
    WEBHOOK_DELETED,
    WEBHOOK_CREATE_REFUSED,
    WEBHOOK_UPDATE_REFUSED,
    WEBHOOK_DELETE_REFUSED,
})

#: Actions that record a DENIED attempt. `record()` stamps outcome=refused for
#: these, so the two ways of querying refusals can never disagree.
REFUSAL_ACTIONS = frozenset({
    SERVER_KEY_REVOKE_REFUSED,
    CURSOR_TOKEN_REVOKE_REFUSED,
    PROVIDER_CREDENTIAL_DELETE_REFUSED,
    ORG_SECRET_UPDATE_REFUSED,
    ORG_SECRET_DELETE_REFUSED,
    MEMBER_REMOVE_REFUSED,
    WEBHOOK_CREATE_REFUSED,
    WEBHOOK_UPDATE_REFUSED,
    WEBHOOK_DELETE_REFUSED,
})


# ─── Resource types ──────────────────────────────────────────────────────────

RESOURCE_SERVER_API_KEY = "service_api_key"
RESOURCE_CURSOR_TOKEN = "cursor_token"
RESOURCE_PROVIDER_CREDENTIAL = "provider_credential"
RESOURCE_ORG_SECRET = "org_secret"
RESOURCE_ORG_MEMBER = "organization_member"
RESOURCE_ORGANIZATION = "organization"
RESOURCE_RECOMMENDATION = "optimization_recommendation"
RESOURCE_DEPLOYMENT = "workflow_deployment"
RESOURCE_WEBHOOK_TRIGGER = "webhook_trigger"


# ─── Refusal reason codes ────────────────────────────────────────────────────
# Closed set, so "why was this denied" is countable rather than grepped out of
# prose. Never a message: a detail string can carry row contents.

REASON_CROSS_TENANT = "cross_tenant"          # caller is not in the owning org
REASON_NOT_OWNER = "not_resource_owner"        # in-org, but not this row's owner
REASON_NOT_ADMIN = "not_admin"                 # in-org member, admin required
REASON_NOT_FOUND = "not_found"                 # no such row for this org

REASON_CODES = frozenset({
    REASON_CROSS_TENANT,
    REASON_NOT_OWNER,
    REASON_NOT_ADMIN,
    REASON_NOT_FOUND,
})


# ─── Metadata: a closed ALLOW-LIST, not a denylist ───────────────────────────
#
# THE POINT OF THIS LIST is that a caller cannot widen it. A future call site
# that passes `{"api_key": ..., "prompt": ...}` writes NEITHER — not because a
# redactor recognised them, but because they are not on the list. Every entry
# below is an identifier, an enum value, or a code. None of them is, or can
# contain, customer content:
#
#   provider        a provider slug ("openai"), never the credential
#   outcome         "success" | "refused"
#   reason_code     one of REASON_CODES
#   target_user_id  a uuid; WHICH member was invited/removed. Deliberately NOT
#                   the invitee's email: the organization_members row resolves
#                   it, and an audit table is the wrong place to duplicate PII.
#   role            "admin" | "member"
#   prior_status    a lifecycle enum, e.g. "verified"
#   new_status      a lifecycle enum, e.g. "canary"
#   *_id / version  server-generated identifiers
#
_ALLOWED_METADATA_KEYS = frozenset({
    "provider",
    "outcome",
    "reason_code",
    "target_user_id",
    "role",
    "prior_status",
    "new_status",
    "deployment_id",
    "workflow_id",
    "workload_id",
    "recommendation_id",
    "version",
})

#: Scalars only. A dict or list is a container for content we did not inspect.
_ALLOWED_METADATA_TYPES = (str, int, float, bool)

#: Nothing on the allow-list is legitimately longer than this. A value that is
#: gets dropped rather than truncated: a truncated secret is still a secret.
_MAX_METADATA_VALUE_LEN = 128

#: Set on the row when anything was dropped, so a filtered write is visible
#: without the dropped keys or values ever reaching the database.
_FILTERED_MARKER = "_metadata_filtered"


def _safe_metadata(metadata: Optional[Mapping[str, Any]], *, action: str) -> dict:
    """Reduce caller metadata to the allow-list. Never raises.

    Deny-by-default in both directions: an unknown KEY is dropped, and an
    allowed key holding a non-scalar or oversized VALUE is dropped too. What
    survives is exactly what this module promised to collect.
    """
    out: dict = {}
    if not metadata:
        return out

    filtered = False
    try:
        items = list(metadata.items())
    except Exception:
        logger.warning("audit: metadata for %s was not a mapping; dropped", action)
        return {_FILTERED_MARKER: True}

    for key, value in items:
        if key not in _ALLOWED_METADATA_KEYS:
            # Log the KEY only. The value is exactly the thing we refuse to
            # handle, so it does not go to the log either.
            logger.warning("audit: dropped metadata key %r on action %s", key, action)
            filtered = True
            continue
        if value is None:
            continue
        if not isinstance(value, _ALLOWED_METADATA_TYPES):
            # A dict, list or object is a container for something we did not
            # inspect. Refuse the whole value rather than walk into it.
            logger.warning("audit: dropped non-scalar metadata %r on %s", key, action)
            filtered = True
            continue
        if isinstance(value, (bool, int, float)):
            # Already a bounded scalar; jsonb keeps the type.
            out[key] = value
            continue
        text = value.strip()
        if not text:
            continue
        if len(text) > _MAX_METADATA_VALUE_LEN:
            # Dropped, never truncated: the front of a secret is still a secret.
            logger.warning("audit: dropped oversized metadata %r on %s", key, action)
            filtered = True
            continue
        out[key] = text

    if filtered:
        out[_FILTERED_MARKER] = True
    return out


def _as_uuid_text(value: Any) -> Optional[str]:
    """Canonical uuid string, or None. `org_id`/`actor_id` are uuid columns."""
    if value is None:
        return None
    try:
        return str(_uuid.UUID(str(value).strip()))
    except Exception:
        return None


def _as_resource_id(value: Any) -> Optional[str]:
    """`resource_id` is text, but is still an identifier — bounded, no content."""
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text or len(text) > 128:
        return None
    return text


def _client_ip(request: Any) -> Optional[str]:
    """The peer address of the connection, or None.

    DELIBERATELY NOT `X-Forwarded-For`. That header is set by the client and
    nothing in this deployment strips or verifies it, so trusting it would let
    the actor of a recorded event choose the address the audit trail blames.
    A NULL ip_address is a known-unknown; a forged one is a false lead.

    Behind a proxy this is therefore the proxy's address. Recording the true
    client IP needs a trusted-proxy configuration first; until then this column
    is honest about what it knows. `ip_address` is `inet`, so anything that does
    not parse as an IP is dropped rather than sent to Postgres.
    """
    if request is None:
        return None
    try:
        client = getattr(request, "client", None)
        host = getattr(client, "host", None)
        if not host:
            return None
        return str(ipaddress.ip_address(str(host).strip()))
    except Exception:
        return None


def _org_from_principal(principal: AuthenticatedUser) -> Optional[str]:
    """The org the AUTH LAYER established for this principal.

    Both sources are written by `auth_dependency` and neither is reachable from
    a request body:

      `_verified_org_id`  set by require_org_member/require_org_admin after it
                          proved membership (and after it refused a request that
                          named two different orgs).
      `_cursor_org_id`    read out of the cursor_tokens row the presented bearer
                          token hashed to.
    """
    verified = getattr(principal, "_verified_org_id", None)
    if verified:
        return str(verified)
    cursor_org = getattr(principal, "_cursor_org_id", None)
    if cursor_org:
        return str(cursor_org)
    return None


def _write(
    *,
    action: str,
    org_id: Optional[str],
    actor_id: Optional[str],
    resource_type: Optional[str],
    resource_id: Optional[str],
    metadata: Optional[Mapping[str, Any]],
    ip_address: Optional[str],
) -> None:
    """The single insert. Never raises, never returns a value.

    Every public entry point funnels through here so there is exactly one place
    that touches the table and exactly one place that swallows failures.
    """
    if action not in ACTIONS:
        # A literal slipped past the constants. Refuse rather than invent an
        # action name no query will ever look for.
        logger.error("audit: refusing unknown action %r", action)
        return

    org = _as_uuid_text(org_id)
    if org is None:
        # `audit_log.org_id` is NOT NULL with an FK to organizations. A row we
        # cannot file under a real tenant is not a row worth forging.
        logger.error("audit: no usable org_id for action %s; row not written", action)
        return

    payload = {
        "org_id": org,
        "actor_id": _as_uuid_text(actor_id),
        "action": action,
        "resource_type": resource_type or None,
        "resource_id": _as_resource_id(resource_id),
        "metadata": _safe_metadata(metadata, action=action),
        "ip_address": ip_address,
    }

    try:
        if supabase is None:
            logger.warning("audit: no database client; %s not recorded", action)
            return
        supabase.table(AUDIT_TABLE).insert(payload).execute()
    except Exception as exc:
        # BEST EFFORT. The caller's operation has already succeeded (or already
        # been refused) and must not be turned into a 500 because the audit
        # table was unreachable. Operator-side only; never re-raised.
        logger.error(
            "audit: failed to record %s: %s", action, type(exc).__name__, exc_info=True
        )


def _stamp_outcome(action: str, metadata: Optional[Mapping[str, Any]]) -> dict:
    """Attach outcome=success|refused derived from the ACTION, not the caller."""
    merged = dict(metadata or {})
    merged["outcome"] = "refused" if action in REFUSAL_ACTIONS else "success"
    return merged


def record(
    action: str,
    *,
    principal: AuthenticatedUser,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    request: Any = None,
) -> None:
    """Record one security-relevant event for a guard-authenticated caller.

    `principal` is the `AuthenticatedUser` a dependency produced. The org and
    the actor are read off it here; there is no parameter for either, so a
    handler cannot pass `payload.org_id` in by mistake.

    Never raises — including on a programming error, which is logged and
    dropped rather than allowed to fail the caller's request.
    """
    try:
        if not isinstance(principal, AuthenticatedUser):
            # Same executable invariant as resource_access, but it must not
            # take the request down: log loudly, write nothing.
            logger.error(
                "audit: record() requires the AuthenticatedUser from the auth "
                "guard, never a caller-supplied org id (got %s) — %s not recorded",
                type(principal).__name__, action,
            )
            return
        _write(
            action=action,
            org_id=_org_from_principal(principal),
            actor_id=getattr(principal, "user_id", None),
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=_stamp_outcome(action, metadata),
            ip_address=_client_ip(request),
        )
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.error("audit: record() failed for %s: %s", action, type(exc).__name__)


def record_for_server_key(
    action: str,
    *,
    ctx: Any,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    request: Any = None,
) -> None:
    """Record an event on a surface authenticated by a server API key.

    The org comes from `OrgContext.org_id`, which `validate_api_key` read out of
    the `service_api_keys` row the presented bearer token hashed to. There is no
    human actor, so `actor_id` is NULL — the key itself is the actor, and the
    caller should pass its id as `resource_id` where that is the subject.
    """
    try:
        _write(
            action=action,
            org_id=getattr(ctx, "org_id", None),
            actor_id=None,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=_stamp_outcome(action, metadata),
            ip_address=_client_ip(request),
        )
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.error(
            "audit: record_for_server_key() failed for %s: %s", action, type(exc).__name__
        )


def record_server_derived(
    action: str,
    *,
    org_id: Optional[str],
    derived_from: str,
    actor_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    request: Any = None,
) -> None:
    """Record an event whose org the SERVER read out of the database.

    THE ONE ENTRY POINT THAT TAKES AN ORG ID, and therefore the one to read
    carefully in review. It exists for the surfaces that have no verified org
    because they authenticate differently:

      * `DELETE /delete-service-api-key/{key_id}` — no org in the path and no
        body to carry one. The org is the one on the `service_api_keys` row the
        server fetched by key id. Filing the refusal under the VICTIM's org is
        the point: "did anyone try to revoke MY production key" is the question
        that could not be answered.
      * `POST /api/organizations/accept-invite` — the org is the one named on
        the `invite_tokens` row the presented token matched.
      * `POST /api/organizations/create` — the org did not exist until the
        server inserted it.

    `derived_from` names the row the org came from and is written nowhere; it is
    documentation that a reviewer can grep. The contract this argument records:
    **`org_id` must be a value the server read from the database, never a value
    the client sent.** A call passing a request-body org id here is a bug of the
    same class this whole module exists to make visible.
    """
    try:
        if not derived_from:
            logger.error("audit: record_server_derived requires derived_from; %s not recorded", action)
            return
        _write(
            action=action,
            org_id=org_id,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=_stamp_outcome(action, metadata),
            ip_address=_client_ip(request),
        )
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.error(
            "audit: record_server_derived() failed for %s: %s", action, type(exc).__name__
        )
