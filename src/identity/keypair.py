"""Ed25519 keypair generation, serialization, signing, and verification.

Fernet encryption scheme for private key:
  salt   = SHA-256(public_key_pem)[:16]   — deterministic, derived from public key
  fernet_key = PBKDF2HMAC(SHA256, length=32, salt=salt, iterations=600_000)
             → base64url-encode → Fernet key

Signature format: base64 raw bytes (not PEM, not hex).
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)


def generate() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 keypair.

    Returns ``(private_key_pem, public_key_pem)`` — both as UTF-8 bytes.
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    public_pem = public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


def _derive_fernet_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    raw = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(raw)


def _salt_from_public_pem(public_pem: bytes) -> bytes:
    return hashlib.sha256(public_pem).digest()[:16]


def encrypt_private(private_pem: bytes, passphrase: str, public_pem: bytes) -> bytes:
    """Encrypt ``private_pem`` using Fernet + PBKDF2 with salt derived from ``public_pem``."""
    salt = _salt_from_public_pem(public_pem)
    fernet = Fernet(_derive_fernet_key(passphrase, salt))
    return fernet.encrypt(private_pem)


def load_private(
    encrypted: bytes,
    passphrase: str,
    public_pem: bytes | str,
) -> Ed25519PrivateKey:
    """Decrypt and load Ed25519PrivateKey from Fernet ciphertext."""
    if isinstance(public_pem, str):
        public_pem = public_pem.encode("utf-8")
    salt = _salt_from_public_pem(public_pem)
    fernet = Fernet(_derive_fernet_key(passphrase, salt))
    try:
        pem = fernet.decrypt(encrypted)
    except InvalidToken as exc:
        raise ValueError("Wrong passphrase or corrupted key material.") from exc
    return load_pem_private_key(pem, password=None)  # type: ignore[return-value]


def load_public(pem: str | bytes) -> Ed25519PublicKey:
    """Load Ed25519PublicKey from PEM bytes/string."""
    if isinstance(pem, str):
        pem = pem.encode("utf-8")
    return load_pem_public_key(pem)  # type: ignore[return-value]


def sign(private_key: Ed25519PrivateKey, content: str) -> str:
    """Sign UTF-8 content string. Returns base64-encoded raw signature."""
    raw = private_key.sign(content.encode("utf-8"))
    return base64.b64encode(raw).decode("ascii")


def verify(public_key: Ed25519PublicKey, content: str, signature: str) -> bool:
    """Verify signature against content using only the public key.

    Returns False on any failure (bad sig, wrong key, tampered content).
    Raises nothing.
    """
    try:
        raw_sig = base64.b64decode(signature)
        public_key.verify(raw_sig, content.encode("utf-8"))
        return True
    except Exception:
        return False
