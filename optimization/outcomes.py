"""
Outcomes: what actually happened.

Three properties drive the whole design, and each is a first-class requirement
rather than an edge case:

  DELAYED ARRIVAL.  A request at 10:00 may be resolved by a customer at 13:00.
  A PR merges tomorrow and is reverted next week. `occurred_at` is when it
  happened in the world; `recorded_at` is when OptiML learned. Analyses window
  on occurred_at.

  PLURAL AND NAMED.  One support response accumulates 'thumbs_up',
  'ticket_resolved', 'escalation', 'reopened_7d'. `outcome_type` is an OPEN
  vocabulary with per-workload meaning, deliberately not collapsed into a
  single quality score — the distinct names are what let a policy declare which
  one decides success for THAT workload.

  CORRECTABLE.  Business data gets revised. A correction is a NEW row that
  supersedes the old one; the original is retained and marked, never
  overwritten, because savings math may already have consumed the old value and
  an audit must be able to see what it consumed.

Idempotency is required, not optional: outcome feeds are webhooks and webhooks
retry.
"""
from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Optional

from supabase_client import supabase

from optimization import attempts as attempts_mod
from optimization import domain

logger = logging.getLogger(__name__)

OUTCOME_COLS = (
    "id, org_id, workload_id, attempt_ref, external_attempt_ref, attempt_source, "
    "outcome_type, outcome_category, "
    "outcome_key, outcome_value, outcome_value_text, unit, success, source, provenance, "
    "provenance_rank, signal_strength, confidence, occurred_at, recorded_at, metadata, "
    "idempotency_key, revision, is_current, supersedes_outcome_id, "
    "superseded_by_outcome_id, superseded_at, correction_reason, created_at"
)


class OutcomeError(ValueError):
    """Invalid outcome payload."""


class AttemptNotFoundError(OutcomeError):
    """
    The referenced attempt does not exist *for this org*.

    Deliberately a SUBCLASS of OutcomeError: every existing caller (the
    dashboard endpoint) keeps mapping it to the same 400 it always did, while
    the customer-facing endpoint can map it to a 404 that reads identically
    whether the id belongs to another tenant or to nobody at all. A caller
    holding one org's key must not be able to use this endpoint to discover
    which request ids exist in another org.
    """


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_outcome(
    org_id: str,
    *,
    outcome_type: str,
    idempotency_key: str,
    attempt_ref: Optional[str] = None,
    attempt_source: str = "workflow_run",
    workload_id: Optional[str] = None,
    value: Optional[float] = None,
    value_text: Optional[str] = None,
    unit: Optional[str] = None,
    success: Optional[bool] = None,
    outcome_category: Optional[str] = None,
    outcome_key: Optional[str] = None,
    source: str = "api",
    provenance: str = "unknown",
    signal_strength: Optional[float] = None,
    confidence: Optional[float] = None,
    occurred_at: Optional[Any] = None,
    metadata: Optional[dict] = None,
    verify_attempt: bool = True,
) -> tuple[dict, bool]:
    """
    Attach an outcome to an earlier attempt (or to a workload as a whole).

    Returns (row, created). `created` is False when the idempotency key had
    already been used, in which case the EXISTING row is returned unchanged —
    a retrying webhook must not create a second outcome or mutate the first.

    Raises OutcomeError when the payload is invalid or the referenced attempt
    does not belong to this org. Attempt ownership is verified rather than
    trusted, because attempt ids come from customer systems.
    """
    otype = (outcome_type or "").strip()
    if not domain.is_valid_outcome_type(otype):
        raise OutcomeError("outcome_type is required (1-120 characters).")
    if not (idempotency_key or "").strip():
        raise OutcomeError("idempotency_key is required.")
    if attempt_source not in domain.ATTEMPT_SOURCES:
        raise OutcomeError(f"attempt_source must be one of {domain.ATTEMPT_SOURCES}.")
    prov = (provenance or "unknown").strip().lower()
    if prov not in domain.OUTCOME_PROVENANCES:
        raise OutcomeError(f"provenance must be one of {domain.OUTCOME_PROVENANCES}.")
    if source not in domain.OUTCOME_SOURCES:
        raise OutcomeError(f"source must be one of {domain.OUTCOME_SOURCES}.")
    if attempt_ref is None and workload_id is None:
        raise OutcomeError(
            "Provide attempt_ref (the attempt this describes) or workload_id "
            "(when the outcome describes the workload over a period)."
        )

    key = str(idempotency_key).strip()

    # Idempotency first: never do the work twice.
    existing = _find_by_idempotency_key(org_id, key)
    if existing is not None:
        return existing, False

    resolved_workload = workload_id
    if attempt_ref and verify_attempt:
        # Ownership is VERIFIED, never trusted: attempt ids come from customer
        # systems. get_attempt routes on the shape of the id, so a mislabelled
        # attempt_source still resolves rather than silently failing.
        attempt = attempts_mod.get_attempt(org_id, str(attempt_ref), attempt_source=attempt_source)
        if attempt is None:
            raise AttemptNotFoundError(
                "attempt_ref does not identify an attempt belonging to this organization."
            )
        resolved_workload = resolved_workload or attempt.workload_id

    occurred = _parse_ts(occurred_at) or _utc_now()

    # A direct-inference attempt is addressed by its customer-visible
    # ``chatcmpl-...`` id, which is not a UUID and so cannot go in the UUID
    # column. Route it to the TEXT column instead of coercing it.
    is_external_ref = bool(attempt_ref) and str(attempt_ref).startswith("chatcmpl-")

    row = {
        "org_id": org_id,
        "workload_id": resolved_workload,
        "attempt_ref": (None if is_external_ref else attempt_ref),
        "external_attempt_ref": (str(attempt_ref) if is_external_ref else None),
        "attempt_source": attempt_source,
        "outcome_type": otype,
        "outcome_category": outcome_category,
        "outcome_key": outcome_key,
        "outcome_value": value,
        "outcome_value_text": value_text,
        "unit": unit,
        "success": success,
        "source": source,
        "provenance": prov,
        "signal_strength": signal_strength,
        "confidence": confidence,
        # occurred_at may be well before recorded_at. That gap is the point.
        "occurred_at": _iso(occurred),
        "recorded_at": _iso(_utc_now()),
        "metadata": metadata or {},
        "idempotency_key": key,
        "revision": 1,
        "is_current": True,
    }

    try:
        result = supabase.table("outcomes").insert(row).execute()
        created = (result.data or [None])[0]
        if created is None:
            raise OutcomeError("Failed to record outcome.")
        return created, True
    except OutcomeError:
        raise
    except Exception as exc:
        # A concurrent request may have won the idempotency race.
        again = _find_by_idempotency_key(org_id, key)
        if again is not None:
            return again, False
        logger.warning("record_outcome failed: %s", type(exc).__name__)
        raise OutcomeError("Failed to record outcome.") from exc


def correct_outcome(
    org_id: str,
    outcome_id: str,
    *,
    idempotency_key: str,
    correction_reason: str,
    value: Optional[float] = None,
    value_text: Optional[str] = None,
    success: Optional[bool] = None,
    provenance: Optional[str] = None,
    signal_strength: Optional[float] = None,
    confidence: Optional[float] = None,
    occurred_at: Optional[Any] = None,
    metadata: Optional[dict] = None,
) -> tuple[dict, bool]:
    """
    Revise a previously recorded outcome.

    Inserts a NEW row carrying the corrected values, links it to the original
    via supersedes_outcome_id, and marks the original is_current=false with
    superseded_at and superseded_by_outcome_id. The original values remain
    readable forever.

    This is not pedantry: savings math may already have consumed the superseded
    value, and an audit of a realized-savings claim has to be able to see both
    what we believed then and what we believe now.
    """
    if not (correction_reason or "").strip():
        raise OutcomeError("correction_reason is required for a correction.")

    key = str(idempotency_key or "").strip()
    if not key:
        raise OutcomeError("idempotency_key is required.")

    existing = _find_by_idempotency_key(org_id, key)
    if existing is not None:
        return existing, False

    try:
        resp = (
            supabase.table("outcomes")
            .select(OUTCOME_COLS)
            .eq("id", outcome_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("correct_outcome lookup failed: %s", type(exc).__name__)
        raise OutcomeError("Failed to load the outcome being corrected.") from exc

    if not rows:
        raise OutcomeError("Outcome not found for this organization.")
    prev = rows[0]

    if not prev.get("is_current", True):
        raise OutcomeError(
            "That outcome has already been superseded. Correct the current "
            "revision instead, so the chain stays linear."
        )

    occurred = _parse_ts(occurred_at) or _parse_ts(prev.get("occurred_at")) or _utc_now()

    row = {
        "org_id": org_id,
        "workload_id": prev.get("workload_id"),
        "attempt_ref": prev.get("attempt_ref"),
        "external_attempt_ref": prev.get("external_attempt_ref"),
        "attempt_source": prev.get("attempt_source") or "workflow_run",
        "outcome_type": prev.get("outcome_type"),
        "outcome_category": prev.get("outcome_category"),
        "outcome_key": prev.get("outcome_key"),
        "outcome_value": value if value is not None else prev.get("outcome_value"),
        "outcome_value_text": value_text if value_text is not None else prev.get("outcome_value_text"),
        "unit": prev.get("unit"),
        "success": success if success is not None else prev.get("success"),
        "source": prev.get("source") or "api",
        "provenance": (provenance or prev.get("provenance") or "unknown"),
        "signal_strength": (
            signal_strength if signal_strength is not None else prev.get("signal_strength")
        ),
        "confidence": confidence if confidence is not None else prev.get("confidence"),
        "occurred_at": _iso(occurred),
        "recorded_at": _iso(_utc_now()),
        "metadata": {**(prev.get("metadata") or {}), **(metadata or {})},
        "idempotency_key": key,
        "revision": int(prev.get("revision") or 1) + 1,
        "is_current": True,
        "supersedes_outcome_id": prev["id"],
        "correction_reason": correction_reason.strip(),
    }

    try:
        inserted = supabase.table("outcomes").insert(row).execute()
        new_row = (inserted.data or [None])[0]
        if new_row is None:
            raise OutcomeError("Failed to record correction.")

        supabase.table("outcomes").update({
            "is_current": False,
            "superseded_at": _iso(_utc_now()),
            "superseded_by_outcome_id": new_row["id"],
        }).eq("id", prev["id"]).eq("org_id", org_id).execute()

        return new_row, True
    except OutcomeError:
        raise
    except Exception as exc:  # pragma: no cover
        logger.warning("correct_outcome failed: %s", type(exc).__name__)
        raise OutcomeError("Failed to record correction.") from exc


def _find_by_idempotency_key(org_id: str, key: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("outcomes")
            .select(OUTCOME_COLS)
            .eq("org_id", org_id)
            .eq("idempotency_key", key)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def list_outcomes(
    org_id: str,
    *,
    workload_id: Optional[str] = None,
    attempt_ref: Optional[str] = None,
    outcome_type: Optional[str] = None,
    current_only: bool = True,
    since: Optional[datetime] = None,
    limit: int = 200,
) -> list[dict]:
    """Outcomes for an org, newest OCCURRENCE first (not newest recording)."""
    try:
        q = supabase.table("outcomes").select(OUTCOME_COLS).eq("org_id", org_id)
        if workload_id:
            q = q.eq("workload_id", workload_id)
        if attempt_ref:
            if str(attempt_ref).startswith("chatcmpl-"):
                q = q.eq("external_attempt_ref", str(attempt_ref))
            else:
                q = q.eq("attempt_ref", attempt_ref)
        if outcome_type:
            q = q.eq("outcome_type", outcome_type)
        if current_only:
            q = q.eq("is_current", True)
        if since is not None:
            q = q.gte("occurred_at", _iso(since))
        resp = q.order("occurred_at", desc=True).limit(max(1, min(limit, 1000))).execute()
        return getattr(resp, "data", None) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("list_outcomes failed: %s", type(exc).__name__)
        return []


def revision_chain(org_id: str, outcome_id: str) -> list[dict]:
    """Every revision of one outcome, oldest first — the correction audit trail."""
    chain: list[dict] = []
    seen: set[str] = set()
    cursor: Optional[str] = outcome_id
    while cursor and cursor not in seen:
        seen.add(cursor)
        try:
            resp = (
                supabase.table("outcomes")
                .select(OUTCOME_COLS)
                .eq("id", cursor)
                .eq("org_id", org_id)
                .limit(1)
                .execute()
            )
            rows = getattr(resp, "data", None) or []
        except Exception:  # pragma: no cover
            break
        if not rows:
            break
        chain.append(rows[0])
        cursor = rows[0].get("supersedes_outcome_id")
    return list(reversed(chain))


def aggregate_by_signal(
    outcomes: list[dict],
    signal: domain.SuccessSignal,
) -> Optional[dict]:
    """
    Aggregate outcomes for ONE named signal, never across signals.

    Returns None when the signal's outcome_type is absent from the data — that
    is a legitimate "we cannot judge this", reported as the
    `missing_primary_outcome` reason code rather than as a zero.
    """
    if not signal.outcome_type:
        return None

    rows = [
        o for o in outcomes
        if (o.get("outcome_type") or "").strip() == signal.outcome_type
        and o.get("is_current", True)
    ]
    if signal.provenance:
        rows = [o for o in rows if (o.get("provenance") or "") == signal.provenance]
    if not rows:
        return None

    # Never mix provenance tiers inside one number.
    buckets = domain.group_outcomes_by_provenance(rows)
    chosen_prov = signal.provenance or domain.strongest_provenance(rows)
    chosen = buckets.get(chosen_prov or "", [])
    if not chosen:
        return None

    values = [float(o["outcome_value"]) for o in chosen if o.get("outcome_value") is not None]
    successes = [bool(o["success"]) for o in chosen if o.get("success") is not None]

    if signal.aggregate == "rate":
        value = (sum(1 for s in successes if s) / len(successes)) if successes else None
    else:
        value = domain.mean(values)

    return {
        "outcome_type": signal.outcome_type,
        "provenance": chosen_prov,
        "aggregate": signal.aggregate,
        "value": (round(value, 6) if value is not None else None),
        "n": len(chosen),
        "variation": domain.coefficient_of_variation(values),
        "signal_strength": (
            round(sum(domain.signal_strength(o) for o in chosen) / len(chosen), 4)
        ),
        "other_provenances_present": sorted(k for k in buckets if k != chosen_prov),
    }


def outcome_row_to_response(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "org_id": str(row["org_id"]),
        "workload_id": (str(row["workload_id"]) if row.get("workload_id") else None),
        "attempt_ref": (
            str(row["attempt_ref"]) if row.get("attempt_ref")
            else (row.get("external_attempt_ref") or None)
        ),
        "attempt_source": row.get("attempt_source"),
        "outcome_type": row.get("outcome_type"),
        "outcome_category": row.get("outcome_category"),
        "outcome_key": row.get("outcome_key"),
        "value": row.get("outcome_value"),
        "value_text": row.get("outcome_value_text"),
        "unit": row.get("unit"),
        "success": row.get("success"),
        "source": row.get("source"),
        "provenance": row.get("provenance"),
        "provenance_rank": row.get("provenance_rank"),
        "signal_strength": row.get("signal_strength"),
        "confidence": row.get("confidence"),
        "occurred_at": row.get("occurred_at"),
        "recorded_at": row.get("recorded_at"),
        "metadata": row.get("metadata") or {},
        "idempotency_key": row.get("idempotency_key"),
        "revision": row.get("revision"),
        "is_current": bool(row.get("is_current", True)),
        "supersedes_outcome_id": (
            str(row["supersedes_outcome_id"]) if row.get("supersedes_outcome_id") else None
        ),
        "superseded_by_outcome_id": (
            str(row["superseded_by_outcome_id"]) if row.get("superseded_by_outcome_id") else None
        ),
        "superseded_at": row.get("superseded_at"),
        "correction_reason": row.get("correction_reason"),
        "created_at": row.get("created_at"),
    }
