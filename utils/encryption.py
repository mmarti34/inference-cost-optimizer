import os
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib

# Get encryption key from environment variable
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "your-secret-key-here")

def get_aes_key():
    """Generate a 32-byte AES key from the encryption key"""
    # Use SHA-256 to hash the encryption key to get a 32-byte key
    return hashlib.sha256(ENCRYPTION_KEY.encode()).digest()

def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key using AES-256-CBC"""
    try:
        key = get_aes_key()
        # Use a static IV for compatibility (in production, use random IV)
        iv = b'\x00' * 16  # 16 bytes of zeros
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_data = pad(api_key.encode(), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        # Return base64 encoded encrypted data
        return base64.b64encode(encrypted_data).decode()
    except Exception as e:
        print(f"Encryption error: {e}")
        # Fallback to simple base64 encoding if encryption fails
        return base64.b64encode(api_key.encode()).decode()

def _evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int):
    d = b''
    last = b''
    while len(d) < key_len + iv_len:
        last = hashlib.md5(last + password + salt).digest()
        d += last
    return d[:key_len], d[key_len:key_len+iv_len]

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt a CryptoJS/OpenSSL-style AES-encrypted string (CBC, PKCS7)"""
    try:
        encrypted = base64.b64decode(encrypted_key)
        if encrypted[:8] != b"Salted__":
            raise ValueError("Invalid encrypted data: missing 'Salted__' header")
        salt = encrypted[8:16]
        ciphertext = encrypted[16:]
        password = ENCRYPTION_KEY.encode("utf-8")
        key, iv = _evp_bytes_to_key(password, salt, 32, 16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        raise ValueError("Failed to decrypt API key")

# Alternative simple encryption for development (less secure but simpler)
def simple_encrypt(text: str) -> str:
    """Simple XOR encryption for development"""
    key = ENCRYPTION_KEY.encode()
    encrypted = bytearray()
    for i, char in enumerate(text):
        encrypted.append(ord(char) ^ key[i % len(key)])
    return base64.b64encode(encrypted).decode()

def simple_decrypt(encrypted_text: str) -> str:
    """Simple XOR decryption for development"""
    try:
        encrypted = base64.b64decode(encrypted_text.encode())
        key = ENCRYPTION_KEY.encode()
        decrypted = ""
        for i, byte in enumerate(encrypted):
            decrypted += chr(byte ^ key[i % len(key)])
        return decrypted
    except Exception as e:
        print(f"Simple decryption error: {e}")
        raise ValueError("Failed to decrypt API key") 