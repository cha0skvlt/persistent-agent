"""Phylactery cryptographic identity — Ed25519 signing infrastructure."""

from __future__ import annotations

from src.identity.keychain import PhylacteryError
from src.identity.signer import IdentitySigner

_signer: IdentitySigner | None = None


def get_signer() -> IdentitySigner:
    """Process-wide singleton IdentitySigner. Lazy-loads key on first call."""
    global _signer
    if _signer is None:
        _signer = IdentitySigner()
    return _signer


def reset_signer() -> None:
    """Force reload of signer (tests / process restart simulation)."""
    global _signer
    _signer = None


__all__ = ["get_signer", "reset_signer", "PhylacteryError"]
