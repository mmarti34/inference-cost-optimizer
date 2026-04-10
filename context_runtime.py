"""
Context resolution and packaging for workflow nodes.

Resolves contextConfig from node data, fetches knowledge assets,
collects sources, packages them, and returns injectable text with trace metadata.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from openai import OpenAI
from supabase_client import supabase

logger = logging.getLogger(__name__)

# Rough token estimate (same as conversation_service)
_CHARS_PER_TOKEN = 4


def resolve_node_context(
    node: dict,
    context: dict,
    variables: dict | None,
    input_text: str,
    org_id: str,
    execution_mode: str,
    deployment_id: str | None = None,
) -> dict | None:
    """
    Entry point. Returns None if context not enabled on this node.

    Returns:
        {
            "final_text": str,
            "items_used": list[dict],
            "truncated": bool,
            "total_chars": int,
            "mode": str,
            "injection_location": str,
        }
    """
    data = node.get("data") or {}
    config = data.get("contextConfig")
    if not config or not config.get("enabled"):
        return None

    sources_config = config.get("sources") or []
    if not sources_config:
        return None

    packaging_config = config.get("packaging") or {}
    injection_config = config.get("injection") or {}

    candidates = _collect_sources(sources_config, context, org_id, deployment_id=deployment_id)
    if not candidates:
        return None

    # SM v2 Phase 2: try to replace raw KB candidates with a consolidated summary
    _sm_scope_value = (variables or {}).get("_sm_scope_value")
    candidates = _maybe_use_kb_consolidation(candidates, org_id, scope_value=_sm_scope_value)

    packaged = _package_context(candidates, packaging_config, org_id=org_id)

    if not packaged["final_text"].strip():
        return None

    return {
        **packaged,
        "mode": config.get("mode", "prepacked"),
        "injection_location": injection_config.get("location", "prepend_to_system"),
    }


def build_context_trace(resolved: dict | None, node_data: dict) -> dict:
    """Build trace dict for node_results. Metadata only, no raw content."""
    config = (node_data or {}).get("contextConfig")
    if not resolved or not config or not config.get("enabled"):
        return {"enabled": False}

    return {
        "enabled": True,
        "mode": resolved.get("mode", "prepacked"),
        "source_count": len(resolved.get("items_used", [])),
        "items_used": [
            {
                "source_type": item.get("source_type", ""),
                "label": item.get("label", ""),
                "chars": item.get("estimated_chars", 0),
            }
            for item in resolved.get("items_used", [])
        ],
        "truncated": resolved.get("truncated", False),
        "total_chars": resolved.get("total_chars", 0),
    }


def _collect_sources(
    sources_config: list[dict],
    context: dict,
    org_id: str,
    deployment_id: str | None = None,
) -> list[dict]:
    """
    Resolve each source config to a normalized candidate.
    Batch-fetches knowledge assets in a single DB query.
    """
    # Batch-fetch all knowledge_asset IDs in one query
    asset_ids = [
        s["assetId"]
        for s in sources_config
        if s.get("type") == "knowledge_asset" and s.get("assetId")
    ]
    assets_by_id: dict[str, dict] = {}
    if asset_ids:
        try:
            result = (
                supabase.table("context_assets")
                .select("id, name, content, asset_type, status")
                .in_("id", asset_ids)
                .eq("org_id", org_id)
                .execute()
            )
            for row in (result.data or []):
                assets_by_id[row["id"]] = row
        except Exception:
            logger.exception("Failed to fetch context assets")

    candidates: list[dict] = []

    for src in sources_config:
        src_type = src.get("type", "")
        label = src.get("label", src_type)
        required = src.get("required", False)
        max_chars = src.get("maxChars")
        raw_text = ""

        if src_type == "knowledge_asset":
            asset_id = src.get("assetId")
            if not asset_id:
                if required:
                    logger.error("Required knowledge_asset source missing assetId")
                continue
            asset = assets_by_id.get(asset_id)
            if not asset:
                if required:
                    logger.error("Required context asset %s not found", asset_id)
                else:
                    logger.warning("Optional context asset %s not found, skipping", asset_id)
                continue
            if asset.get("status") == "deleted":
                logger.warning("Context asset %s has been deleted, skipping", asset_id)
                continue
            if not asset.get("content"):
                logger.warning(
                    "Context asset %s has no content (type '%s'), skipping",
                    asset_id, asset.get("asset_type"),
                )
                if required:
                    raw_text = ""
                else:
                    continue
            else:
                raw_text = asset.get("content") or ""

            # Prefer snapshot content for versioned deployments
            if deployment_id:
                try:
                    snap = (
                        supabase.table("context_asset_snapshots")
                        .select("content")
                        .eq("asset_id", asset_id)
                        .eq("deployment_id", deployment_id)
                        .maybe_single()
                        .execute()
                    )
                    if snap.data and snap.data.get("content"):
                        raw_text = snap.data["content"]
                except Exception:
                    logger.warning(
                        "Failed to fetch snapshot for asset %s deployment %s, using live content",
                        asset_id, deployment_id,
                    )

            label = label or asset.get("name", "asset")

        elif src_type == "inline_text":
            raw_text = src.get("value") or ""

        elif src_type == "previous_node_output":
            node_id = src.get("nodeId")
            if not node_id:
                if required:
                    logger.error("Required previous_node_output source missing nodeId")
                continue
            out = context.get(node_id)
            if out is None:
                if required:
                    logger.error("Required previous node %s has no output", node_id)
                else:
                    logger.warning("Previous node %s has no output yet, skipping", node_id)
                continue
            if isinstance(out, dict):
                raw_text = out.get("output") or out.get("response") or str(out)
            else:
                raw_text = str(out)

        else:
            logger.warning("Unknown context source type '%s', skipping", src_type)
            continue

        if not raw_text and not required:
            continue

        # Per-source char limit
        if max_chars and len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars]

        candidates.append({
            "source_type": src_type,
            "label": label,
            "raw_text": raw_text,
            "source_ref": src.get("assetId") or src.get("nodeId"),
            "required": required,
            "estimated_chars": len(raw_text),
        })

    return candidates


def _summarize_context(text: str, max_chars: int, org_id: str | None = None) -> str:
    """Summarize text using gpt-4o-mini to fit within max_chars."""
    api_key = os.environ.get("SYSTEM_OPENAI_API_KEY")
    if not api_key:
        logger.warning("SYSTEM_OPENAI_API_KEY not set, falling back to truncation")
        return text[:max_chars]

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Summarize the following context to fit within {max_chars} characters. "
                        "Preserve the most important facts, data points, and key information. "
                        "Be concise but accurate."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=max_chars // 3,
            temperature=0.2,
        )
        summary = response.choices[0].message.content or ""
        if len(summary) > max_chars:
            summary = summary[:max_chars]
        return summary
    except Exception:
        logger.exception("Summarization failed (org=%s), falling back to truncation", org_id)
        return text[:max_chars]


def _maybe_use_kb_consolidation(
    candidates: list[dict],
    org_id: str,
    scope_value: str | None = None,
) -> list[dict]:
    """
    SM v2 Phase 2: replace raw KB asset candidates with a consolidated summary
    if one exists with sufficient confidence. Returns the (possibly modified)
    candidates list.

    Only replaces knowledge_asset candidates — inline_text and previous_node_output
    are left untouched. If consolidation isn't found or wouldn't save tokens,
    returns the original candidates unchanged. Zero risk.
    """
    # Separate KB assets from other source types
    kb_candidates = [c for c in candidates if c["source_type"] == "knowledge_asset" and c.get("source_ref")]
    other_candidates = [c for c in candidates if c["source_type"] != "knowledge_asset" or not c.get("source_ref")]

    if len(kb_candidates) < 1:
        return candidates  # Nothing to consolidate

    asset_ids = [c["source_ref"] for c in kb_candidates]

    try:
        from synthetic_mind.memory_store import get_kb_consolidation, bump_kb_consolidation_hit

        consolidation = get_kb_consolidation(org_id, asset_ids, min_confidence=0.7, scope_value=scope_value)
        if not consolidation:
            return candidates

        # Validate freshness: check asset_versions_hash
        current_hash = _compute_asset_versions_hash(asset_ids, org_id)
        if current_hash and consolidation.get("asset_versions_hash") != current_hash:
            logger.info("SM KB consolidation stale (hash mismatch), serving raw candidates")
            return candidates

        # Check compression ratio — only use if it actually saves tokens
        consolidated_tokens = max(1, len(consolidation["consolidated_text"]) // _CHARS_PER_TOKEN)
        raw_tokens = sum(max(1, c["estimated_chars"] // _CHARS_PER_TOKEN) for c in kb_candidates)

        if consolidated_tokens >= raw_tokens * 0.8:
            return candidates  # Not enough savings

        # Replace KB candidates with consolidated text
        bump_kb_consolidation_hit(consolidation["id"])

        consolidated_candidate = {
            "source_type": "knowledge_asset",
            "label": "Consolidated Knowledge",
            "raw_text": consolidation["consolidated_text"],
            "source_ref": f"sm_consolidation:{consolidation['id']}",
            "required": False,
            "estimated_chars": len(consolidation["consolidated_text"]),
        }

        logger.info(
            "SM KB consolidation: %d raw tokens → %d consolidated (%d%% saved), org=%s",
            raw_tokens, consolidated_tokens,
            int((1 - consolidated_tokens / max(raw_tokens, 1)) * 100),
            org_id[:8],
        )

        return other_candidates + [consolidated_candidate]

    except Exception as e:
        logger.warning("SM KB consolidation lookup failed, using raw candidates: %s", e)
        return candidates


def _compute_asset_versions_hash(asset_ids: list[str], org_id: str) -> str | None:
    """Compute a hash of updated_at timestamps for a set of assets to detect staleness."""
    if not asset_ids:
        return None
    try:
        result = (
            supabase.table("context_assets")
            .select("id, updated_at")
            .in_("id", asset_ids)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            return None
        timestamps = sorted(f"{r['id']}:{r['updated_at']}" for r in result.data)
        return hashlib.sha256("|".join(timestamps).encode()).hexdigest()[:16]
    except Exception:
        return None


def _package_context(
    candidates: list[dict],
    packaging_config: dict,
    org_id: str | None = None,
) -> dict:
    """Assemble candidates into final text with trimming."""
    strategy = packaging_config.get("strategy", "concat")
    max_chars = packaging_config.get("maxChars")
    include_labels = packaging_config.get("includeSourceLabels", False)

    # Build sections
    sections: list[str] = []
    for c in candidates:
        text = c["raw_text"]
        if include_labels and c.get("label"):
            text = f"## {c['label']}\n{text}"
        sections.append(text)

    separator = "\n\n---\n\n"
    final_text = separator.join(sections)

    truncated = False
    if max_chars and len(final_text) > max_chars:
        if strategy == "summarize":
            final_text = _summarize_context(final_text, max_chars, org_id=org_id)
            truncated = len(final_text) >= max_chars
        else:
            final_text = final_text[:max_chars]
            truncated = True

    return {
        "final_text": final_text,
        "items_used": [
            {
                "source_type": c["source_type"],
                "label": c["label"],
                "source_ref": c.get("source_ref"),
                "estimated_chars": c["estimated_chars"],
            }
            for c in candidates
        ],
        "truncated": truncated,
        "total_chars": len(final_text),
    }
