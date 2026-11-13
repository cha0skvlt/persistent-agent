"""High-level IdentitySigner — loads key once, signs/verifies content."""

from __future__ import annotations

import logging

import psycopg

from src.config import settings
from src.identity import keypair as kp
from src.identity.keychain import PhylacteryError, load_passphrase

_log = logging.getLogger(__name__)


def _pg_conninfo() -> str:
    url = settings.database_url
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


class IdentitySigner:
    """Loads Ed25519 private key on first use; caches for process lifetime.

    sign() / verify() return None / False gracefully when identity is absent
    so callers never need to guard against missing-identity crashes.
    """

    def __init__(self) -> None:
        self._private_key = None
        self._public_key = None
        self._loaded = False

    def _load(self) -> bool:
        """Attempt to load keys from DB + Keychain. Returns True on success.

        Transient DB/Keychain failures do not stick ``_loaded`` — retry next call.
        Permanent misses (no row / bad key) stick so we do not hammer every sign.
        """
        if self._loaded:
            return self._private_key is not None
        try:
            with psycopg.connect(_pg_conninfo(), connect_timeout=3) as conn:
                row = conn.execute(
                    "SELECT public_key_pem, private_key_encrypted "
                    "FROM agent_identity WHERE agent = %s LIMIT 1",
                    (settings.default_agent_id,),
                ).fetchone()
            if row is None:
                self._loaded = True
                _log.info(
                    "identity.signer: no identity row found (run make init-identity)."
                )
                return False
            public_pem, encrypted = row[0], bytes(row[1])
        except Exception as exc:
            _log.warning("identity.signer: DB unreachable, will retry: %s", exc)
            return False

        try:
            passphrase = load_passphrase()
        except PhylacteryError as exc:
            _log.warning("identity.signer: Keychain unavailable, will retry: %s", exc)
            return False

        try:
            self._private_key = kp.load_private(encrypted, passphrase, public_pem)
            self._public_key = kp.load_public(public_pem)
        except Exception as exc:
            self._loaded = True
            _log.warning("identity.signer: key decryption failed: %s", exc)
            return False

        self._loaded = True
        _log.info("identity.signer: Ed25519 key loaded.")
        return True

    def sign(self, content: str) -> str | None:
        """Sign content. Returns base64 signature string or None if identity unavailable."""
        if not self._load():
            return None
        try:
            return kp.sign(self._private_key, content)
        except Exception as exc:
            _log.warning("identity.signer.sign failed: %s", exc)
            return None

    def verify(self, content: str, signature: str) -> bool:
        """Verify signature. Returns False if identity unavailable or bad signature."""
        if not self._load():
            return False
        try:
            return kp.verify(self._public_key, content, signature)
        except Exception:
            return False

    def public_key_pem(self) -> str | None:
        """Return loaded public key as PEM string, or None."""
        if not self._load() or self._public_key is None:
            return None
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        return self._public_key.public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        ).decode()
