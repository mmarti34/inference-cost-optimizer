# Provider API keys: AES-256-GCM at rest. Universal keys: hashed only, never stored plaintext.
# Requires MASTER_ENCRYPTION_KEY or ENCRYPTION_KEY (min 32 chars). Never log keys.

import os
import base64
import hashlib
import hmac
import logging
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

logger = logging.getLogger(__name__)

# Prefer Railway-style env; support ENCRYPTION_KEY for backward compat
def _get_master_key() -> str:
    key = os.getenv("MASTER_ENCRYPTION_KEY") or os.getenv("ENCRYPTION_KEY")
    if not key:
        raise ValueError(
            "MASTER_ENCRYPTION_KEY or ENCRYPTION_KEY is required. "
            "Set it in Railway (or .env). Use at least 32 characters."
        )
    if len(key) < 32:
        logger.warning("Encryption key should be at least 32 characters; current length=%s", len(key))
    return key

# Lazy so tests can mock env before import
_ENCRYPTION_KEY: str | None = None

def _key() -> str:
    global _ENCRYPTION_KEY
    if _ENCRYPTION_KEY is None:
        _ENCRYPTION_KEY = _get_master_key()
    return _ENCRYPTION_KEY


def validate_encryption_key_at_startup() -> None:
    """Call at app startup to fail fast if key is missing or invalid."""
    k = _get_master_key()
    if len(k) < 32:
        raise ValueError("MASTER_ENCRYPTION_KEY/ENCRYPTION_KEY must be at least 32 characters")


# --- AES-256-GCM (preferred for new provider key encryption) ---

GCM_NONCE_LEN = 12
GCM_TAG_LEN = 16

def _derived_key_gcm() -> bytes:
    return hashlib.sha256(_key().encode()).digest()

def encrypt_api_key_gcm(plaintext: str) -> str:
    """Encrypt with AES-256-GCM. Returns base64(nonce + ciphertext + tag). Never log plaintext."""
    try:
        key = _derived_key_gcm()
        nonce = get_random_bytes(GCM_NONCE_LEN)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
        payload = nonce + ciphertext + tag
        return base64.b64encode(payload).decode()
    except Exception as e:
        logger.error("Encryption failed: %s", type(e).__name__, exc_info=False)
        raise ValueError("Encryption failed") from e

def decrypt_api_key_gcm(encoded: str) -> str:
    """Decrypt AES-256-GCM payload. Raises if invalid."""
    try:
        payload = base64.b64decode(encoded)
        if len(payload) < GCM_NONCE_LEN + GCM_TAG_LEN:
            raise ValueError("Payload too short")
        nonce = payload[:GCM_NONCE_LEN]
        tag = payload[-GCM_TAG_LEN:]
        ciphertext = payload[GCM_NONCE_LEN:-GCM_TAG_LEN]
        key = _derived_key_gcm()
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")
    except Exception as e:
        logger.debug("GCM decryption failed: %s", type(e).__name__)
        raise ValueError("Failed to decrypt") from e


# --- Legacy AES-256-CBC (CryptoJS-compatible) for existing rows ---

def get_aes_key():
    return hashlib.sha256(_key().encode()).digest()

def _evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int):
    d = b''
    last = b''
    while len(d) < key_len + iv_len:
        last = hashlib.md5(last + password + salt).digest()
        d += last
    return d[:key_len], d[key_len:key_len+iv_len]

def encrypt_api_key(api_key: str) -> str:
    """Encrypt provider API key. Prefers GCM; use for all new inserts."""
    try:
        return encrypt_api_key_gcm(api_key)
    except Exception:
        try:
            return _encrypt_cbc(api_key)
        except Exception as e:
            logger.error("Encryption failed: %s", type(e).__name__, exc_info=False)
            raise ValueError("Encryption failed") from e

def _encrypt_cbc(api_key: str) -> str:
    password = _key().encode("utf-8")
    salt = get_random_bytes(8)
    key, iv = _evp_bytes_to_key(password, salt, 32, 16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(api_key.encode(), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    encrypted = b"Salted__" + salt + ciphertext
    return base64.b64encode(encrypted).decode()

def decrypt_cryptojs_api_key(encrypted_key: str) -> str:
    try:
        encrypted = base64.b64decode(encrypted_key)
        if encrypted[:8] != b"Salted__":
            raise ValueError("Invalid encrypted data")
        salt = encrypted[8:16]
        ciphertext = encrypted[16:]
        password = _key().encode("utf-8")
        key, iv = _evp_bytes_to_key(password, salt, 32, 16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted.decode("utf-8")
    except Exception as e:
        logger.debug("CBC decryption failed: %s", type(e).__name__)
        raise ValueError("Failed to decrypt CryptoJS API key") from e

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt provider API key. Tries GCM then CBC then legacy fallbacks."""
    try:
        return decrypt_api_key_gcm(encrypted_key)
    except Exception:
        pass
    try:
        return decrypt_cryptojs_api_key(encrypted_key)
    except Exception:
        pass
    try:
        return simple_decrypt(encrypted_key)
    except Exception:
        pass
    try:
        return base64.b64decode(encrypted_key).decode()
    except Exception as e:
        logger.debug("All decryption methods failed: %s", type(e).__name__)
        raise ValueError("Failed to decrypt API key") from e

def simple_encrypt(text: str) -> str:
    key = _key().encode()
    encrypted = bytearray()
    for i, char in enumerate(text):
        encrypted.append(ord(char) ^ key[i % len(key)])
    return base64.b64encode(encrypted).decode()

def simple_decrypt(encrypted_text: str) -> str:
    encrypted = base64.b64decode(encrypted_text.encode())
    key = _key().encode()
    decrypted = "".join(chr(byte ^ key[i % len(key)]) for i, byte in enumerate(encrypted))
    return decrypted


# --- Universal API keys: hash only, never store plaintext ---

def hash_service_api_key(plaintext: str) -> str:
    """SHA-256 hex of the bearer token. Use for storage and lookup only."""
    return hashlib.sha256(plaintext.encode()).hexdigest()

def verify_service_api_key(plaintext: str, stored_hash: str) -> bool:
    """Constant-time comparison to avoid timing side-channels."""
    computed = hashlib.sha256(plaintext.encode()).hexdigest()
    return hmac.compare_digest(computed, stored_hash)
