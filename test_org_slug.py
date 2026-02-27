"""
Tests for org slug resolution: get_org_id_from_slug (used by public execution).
- Org slug resolves to correct org_id.
- Unknown slug returns None (no fallback to org_id).
"""
import sys
from unittest.mock import patch, MagicMock

if "Crypto" not in sys.modules:
    import types
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        _m = types.ModuleType("Crypto." + sub)
        sys.modules["Crypto." + sub] = _m
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from routers.public_execution import get_org_id_from_slug

ORG_ID = "11111111-1111-1111-1111-111111111111"


@patch("routers.public_execution.supabase")
def test_org_slug_resolves_to_org_id(mock_supabase):
    """Valid slug returns corresponding org id."""
    mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"id": ORG_ID}]
    )
    assert get_org_id_from_slug("my-org") == ORG_ID
    mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.assert_called_once()


@patch("routers.public_execution.supabase")
def test_org_slug_missing_returns_none(mock_supabase):
    """Unknown slug returns None (no fallback to org_id)."""
    mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[]
    )
    assert get_org_id_from_slug("nonexistent") is None


@patch("routers.public_execution.supabase")
def test_org_slug_collision_resolves_first_match(mock_supabase):
    """Slug lookup returns first row (deterministic); collision case does not break lookup."""
    # With UNIQUE(slug) there is only one row per slug; if we had duplicates before migration, DB returns one
    mock_supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"id": ORG_ID}]
    )
    assert get_org_id_from_slug("acme") == ORG_ID
