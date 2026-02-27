"""
Tests for service API key validation: active-only enforcement, revoked returns 401.
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

import pytest
from fastapi import HTTPException
from api_key_validation import validate_api_key

ORG_ID = "11111111-1111-1111-1111-111111111111"


@patch("api_key_validation._update_last_used")
@patch("api_key_validation.verify_service_api_key")
@patch("api_key_validation.supabase")
def test_revoked_key_returns_401(mock_supabase, mock_verify, mock_update):
    """Revoked key is rejected with 401 (active-only enforcement)."""
    revoked_row = {
        "id": "key-1",
        "org_id": ORG_ID,
        "hashed_key": "abc123",
        "api_key": None,
        "key_type": "live",
        "rate_limit_per_minute": 60,
        "status": "revoked",
    }
    call_count = [0]

    def select(*args, **kwargs):
        call_count[0] += 1
        chain = MagicMock()
        chain.eq.return_value = chain
        if call_count[0] == 1:
            # First call: .select(cols).eq("status", "active").execute() -> raise so we use fallback
            chain.execute.side_effect = Exception("status column missing")
        else:
            # Fallback: .select(cols).execute() -> return revoked row
            chain.execute.return_value = MagicMock(data=[revoked_row])
        return chain
    mock_supabase.table.return_value.select.side_effect = select
    mock_verify.return_value = True  # token matches hash

    with pytest.raises(HTTPException) as exc_info:
        validate_api_key("sk_live_fake")
    assert exc_info.value.status_code == 401
    assert "Invalid" in exc_info.value.detail or "invalid" in exc_info.value.detail.lower()


@patch("api_key_validation._update_last_used")
@patch("api_key_validation.verify_service_api_key")
@patch("api_key_validation.supabase")
def test_active_key_returns_org_context(mock_supabase, mock_verify, mock_update):
    """Active key with matching hash returns OrgContext."""
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.execute.return_value = MagicMock(
        data=[{
            "id": "key-1",
            "org_id": ORG_ID,
            "hashed_key": "abc123",
            "api_key": None,
            "key_type": "live",
            "rate_limit_per_minute": 60,
            "status": "active",
        }]
    )
    mock_supabase.table.return_value.select.return_value = chain
    mock_verify.return_value = True

    ctx = validate_api_key("sk_live_fake")
    assert ctx.org_id == ORG_ID
    assert ctx.key_type == "live"
    assert ctx.rate_limit_per_minute == 60
