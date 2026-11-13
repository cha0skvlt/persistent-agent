"""Authenticated encryption for phylactery backups.

Format ``PAM1``: ``magic(4) || salt(16) || Fernet token``.

The key is PBKDF2-HMAC-SHA256 over the passphrase (600 000 iterations, 16-byte
salt). The payload is a Fernet token — AES-128-CBC plus HMAC-SHA256 — so the
archive is both confidential and authenticated. No custom construction.

Legacy archives written by ``openssl enc -aes-256-cbc`` have no magic and are
still readable: they are decrypted with ``openssl enc -d`` for the transition
period. They are not authenticated; re-encrypt them to ``PAM1`` when convenient.
"""

from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

MAGIC = b"PAM1"
_SALT_LEN = 16
_ITERATIONS = 600_000


class BackupCryptoError(Exception):
    """Wrong passphrase, corrupted archive, or an unreadable file."""


def _derive(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def is_authenticated(path: Path) -> bool:
    """True when the archive carries the ``PAM1`` authenticated header."""
    try:
        with open(path, "rb") as fh:
            return fh.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(path, 0o600)


def encrypt_file(src: Path, dst: Path, passphrase: str) -> None:
    """Encrypt ``src`` into ``dst`` with authenticated encryption."""
    salt = os.urandom(_SALT_LEN)
    key = _derive(passphrase, salt)
    token = Fernet(key).encrypt(Path(src).read_bytes())
    _write_private(Path(dst), MAGIC + salt + token)


def decrypt_file(src: Path, dst: Path, passphrase: str) -> None:
    """Decrypt ``src`` into ``dst``; verify authenticity before writing."""
    src = Path(src)
    with open(src, "rb") as fh:
        head = fh.read(len(MAGIC))
        if head == MAGIC:
            salt = fh.read(_SALT_LEN)
            token = fh.read()
        else:
            salt = b""
            token = b""

    if head == MAGIC:
        if len(salt) != _SALT_LEN or not token:
            raise BackupCryptoError("truncated PAM1 archive")
        key = _derive(passphrase, salt)
        try:
            plain = Fernet(key).decrypt(token)
        except InvalidToken as exc:
            raise BackupCryptoError(
                "wrong passphrase or corrupted archive (authentication failed)"
            ) from exc
        _write_private(Path(dst), plain)
        return

    # Legacy openssl format: no authentication, decrypt for the transition period.
    try:
        subprocess.run(
            [
                "openssl",
                "enc",
                "-aes-256-cbc",
                "-d",
                "-pbkdf2",
                "-iter",
                "600000",
                "-pass",
                "stdin",
                "-in",
                str(src),
                "-out",
                str(dst),
            ],
            input=passphrase,
            text=True,
            check=True,
            timeout=7200,
        )
    except subprocess.CalledProcessError as exc:
        raise BackupCryptoError(
            "legacy archive failed to decrypt (wrong passphrase or corruption)"
        ) from exc
    os.chmod(dst, 0o600)


__all__ = [
    "MAGIC",
    "BackupCryptoError",
    "decrypt_file",
    "encrypt_file",
    "is_authenticated",
]
