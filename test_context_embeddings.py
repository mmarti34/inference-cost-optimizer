"""Tests for context_embeddings module."""

import sys
import types
from unittest.mock import patch, MagicMock

_mock_supabase_mod = types.ModuleType("supabase_client")
_mock_supabase_mod.supabase = MagicMock()
sys.modules.setdefault("supabase_client", _mock_supabase_mod)

# Mock openai — only when the real package is absent. A bare ModuleType has no
# __path__, so installing it unconditionally makes `openai` a non-package for the
# whole pytest session and breaks every later module doing `from openai.types...`.
try:  # pragma: no cover - depends on the environment, not the code under test
    import openai  # noqa: F401
except ImportError:
    _mock_openai = types.ModuleType("openai")
    _mock_openai.__path__ = []
    _mock_openai.OpenAI = MagicMock()
    sys.modules.setdefault("openai", _mock_openai)

from context_embeddings import chunk_text


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text():
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=1000)
    assert chunks == ["Hello world"]


def test_chunk_respects_size():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 1200  # allow slack for boundary finding


def test_chunk_paragraph_boundary():
    text = "A" * 600 + "\n\n" + "B" * 600
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) >= 2


def test_chunk_single_chunk_boundary():
    """Text exactly at chunk_size should be one chunk."""
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_overlap_produces_more_chunks():
    text = "x" * 2000
    chunks_no_overlap = chunk_text(text, chunk_size=1000, overlap=0)
    chunks_with_overlap = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_chunk_preserves_all_content():
    """All content should appear in at least one chunk."""
    text = "The quick brown fox jumps over the lazy dog. " * 50
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    combined = " ".join(chunks)
    # Key phrases should be present
    assert "quick brown fox" in combined
    assert "lazy dog" in combined
