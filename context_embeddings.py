"""
Chunking, embedding, and semantic search for context assets via pgvector.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import openai
from supabase_client import supabase

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536


def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks respecting paragraph/sentence boundaries."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at paragraph boundary
        segment = text[start:end]
        para_break = segment.rfind("\n\n")
        if para_break > chunk_size // 2:
            end = start + para_break + 2
        else:
            # Try sentence boundary
            sent_break = segment.rfind(". ")
            if sent_break > chunk_size // 2:
                end = start + sent_break + 2
            # else hard cut at chunk_size

        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0

    return [c for c in chunks if c]


def embed_chunks(chunks: list[str]) -> list[list[float] | None]:
    """Embed a list of text chunks using text-embedding-3-small.
    Returns list of embeddings (or None for failed chunks)."""
    if not chunks:
        return []

    api_key = os.environ.get("SYSTEM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key available for embedding")
        return [None] * len(chunks)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=chunks,
        )
        result: list[list[float] | None] = [None] * len(chunks)
        for item in response.data:
            result[item.index] = item.embedding
        return result
    except Exception:
        logger.exception("Failed to embed chunks")
        return [None] * len(chunks)


def index_asset(asset_id: str, org_id: str, content: str) -> bool:
    """Full pipeline: chunk text, embed, store in context_chunks.
    Returns True if successful."""
    # Delete existing chunks first
    delete_asset_chunks(asset_id)

    if not content or not content.strip():
        return True

    chunks = chunk_text(content)
    if not chunks:
        return True

    embeddings = embed_chunks(chunks)

    rows = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        row: dict[str, Any] = {
            "asset_id": asset_id,
            "org_id": org_id,
            "chunk_index": i,
            "content": chunk,
            "metadata": {"char_count": len(chunk)},
        }
        if embedding:
            row["embedding"] = embedding
        rows.append(row)

    try:
        # Insert in batches of 50
        for i in range(0, len(rows), 50):
            batch = rows[i:i + 50]
            supabase.table("context_chunks").insert(batch).execute()
        return True
    except Exception:
        logger.exception("Failed to store chunks for asset %s", asset_id)
        return False


def delete_asset_chunks(asset_id: str) -> None:
    """Remove all chunks for an asset."""
    try:
        supabase.table("context_chunks").delete().eq("asset_id", asset_id).execute()
    except Exception:
        logger.exception("Failed to delete chunks for asset %s", asset_id)


def search_similar(
    query: str,
    org_id: str,
    limit: int = 5,
    asset_ids: list[str] | None = None,
) -> list[dict]:
    """Embed query, search pgvector for nearest chunks.
    Optionally scope to specific asset IDs."""
    if not query or not query.strip():
        return []

    api_key = os.environ.get("SYSTEM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key for search embedding")
        return []

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(model=_EMBEDDING_MODEL, input=[query])
        query_embedding = response.data[0].embedding
    except Exception:
        logger.exception("Failed to embed search query")
        return []

    try:
        # Use Supabase RPC for pgvector similarity search
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_org_id": org_id,
            "match_count": limit,
        }
        if asset_ids:
            params["filter_asset_ids"] = asset_ids

        # Call a Postgres function for vector similarity search
        # We'll create this as part of the migration or use raw SQL
        result = supabase.rpc("search_context_chunks", params).execute()

        results = []
        for row in (result.data or []):
            results.append({
                "content": row.get("content", ""),
                "asset_id": row.get("asset_id", ""),
                "asset_name": row.get("asset_name", ""),
                "chunk_index": row.get("chunk_index", 0),
                "similarity": row.get("similarity", 0),
            })
        return results
    except Exception:
        logger.exception("pgvector search failed")
        return []
