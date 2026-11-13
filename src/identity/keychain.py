"""macOS Keychain wrapper via python-keyring.

Passphrase lives ONLY in Keychain — never in .env, disk files, or env vars.
"""

from __future__ import annotations

import keyring

_SERVICE_SIGNING = "agent.phylactery"
_USERNAME_MASTER = "master"
_SERVICE_BACKUP = "agent.phylactery.backup"


class PhylacteryError(Exception):
    """Raised when identity is missing, corrupted, or duplicate init is attempted."""


def store_passphrase(passphrase: str) -> None:
    """Persist signing passphrase in macOS Keychain (service=agent.phylactery, user=master)."""
    keyring.set_password(_SERVICE_SIGNING, _USERNAME_MASTER, passphrase)


def load_passphrase() -> str:
    """Load signing passphrase from Keychain. Raises PhylacteryError if absent."""
    pp = keyring.get_password(_SERVICE_SIGNING, _USERNAME_MASTER)
    if pp is None:
        raise PhylacteryError(
            "No passphrase found in Keychain (service=agent.phylactery, user=master). "
            "Run: make init-identity"
        )
    return pp


def delete_passphrase() -> None:
    """Remove signing passphrase from Keychain (for tests / uninstall)."""
    try:
        keyring.delete_password(_SERVICE_SIGNING, _USERNAME_MASTER)
    except keyring.errors.PasswordDeleteError:
        pass


def store_backup_passphrase(passphrase: str) -> None:
    """Persist backup encryption passphrase (separate from signing passphrase)."""
    keyring.set_password(_SERVICE_BACKUP, _USERNAME_MASTER, passphrase)


def load_backup_passphrase() -> str:
    """Load backup passphrase from Keychain."""
    pp = keyring.get_password(_SERVICE_BACKUP, _USERNAME_MASTER)
    if pp is None:
        raise PhylacteryError(
            "No backup passphrase in Keychain (service=agent.phylactery.backup, user=master). "
            "Run: python -m src.identity.cli store-backup-passphrase"
        )
    return pp


def delete_backup_passphrase() -> None:
    """Remove backup passphrase from Keychain (tests)."""
    try:
        keyring.delete_password(_SERVICE_BACKUP, _USERNAME_MASTER)
    except keyring.errors.PasswordDeleteError:
        pass
