"""cryptographic identity tests.

All tests are fully mocked — no live DB, no macOS Keychain required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.identity import keypair as kp
from src.identity.keychain import PhylacteryError

# ─────────────────────── keypair primitives ───────────────────────────────────


def test_generate_returns_pem_bytes():
    private_pem, public_pem = kp.generate()
    assert private_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    assert public_pem.startswith(b"-----BEGIN PUBLIC KEY-----")


def test_sign_verify_roundtrip():
    private_pem, public_pem = kp.generate()
    priv = kp.load_private(
        kp.encrypt_private(private_pem, "passphrase1234567890", public_pem),
        "passphrase1234567890",
        public_pem,
    )
    pub = kp.load_public(public_pem)
    content = "I am."
    sig = kp.sign(priv, content)
    assert isinstance(sig, str) and len(sig) > 0
    assert kp.verify(pub, content, sig) is True


def test_verify_tampered_content_returns_false():
    private_pem, public_pem = kp.generate()
    priv = kp.load_private(
        kp.encrypt_private(private_pem, "passphrase1234567890", public_pem),
        "passphrase1234567890",
        public_pem,
    )
    pub = kp.load_public(public_pem)
    sig = kp.sign(priv, "original content")
    assert kp.verify(pub, "tampered content", sig) is False


def test_verify_with_different_public_key_returns_false():
    _, pub1_pem = kp.generate()
    private_pem2, public_pem2 = kp.generate()
    priv2 = kp.load_private(
        kp.encrypt_private(private_pem2, "passphrase1234567890", public_pem2),
        "passphrase1234567890",
        public_pem2,
    )
    pub1 = kp.load_public(pub1_pem)
    sig = kp.sign(priv2, "some content")
    assert kp.verify(pub1, "some content", sig) is False


def test_verify_bad_signature_string():
    _, public_pem = kp.generate()
    pub = kp.load_public(public_pem)
    assert kp.verify(pub, "anything", "AAAA") is False


def test_encrypt_decrypt_roundtrip():
    private_pem, public_pem = kp.generate()
    passphrase = "my-secure-passphrase-abc123"
    encrypted = kp.encrypt_private(private_pem, passphrase, public_pem)
    assert isinstance(encrypted, bytes)
    priv = kp.load_private(encrypted, passphrase, public_pem)
    # Round-trip: sign something and verify
    pub = kp.load_public(public_pem)
    sig = kp.sign(priv, "test")
    assert kp.verify(pub, "test", sig) is True


def test_encrypt_decrypt_wrong_passphrase_raises():
    private_pem, public_pem = kp.generate()
    encrypted = kp.encrypt_private(private_pem, "correct-passphrase-xyz", public_pem)
    with pytest.raises(ValueError):
        kp.load_private(encrypted, "wrong-passphrase-xyz", public_pem)


def test_load_public_from_string():
    _, public_pem = kp.generate()
    pub = kp.load_public(public_pem.decode("utf-8"))
    assert pub is not None


def test_sign_returns_base64_not_pem():
    private_pem, public_pem = kp.generate()
    priv = kp.load_private(
        kp.encrypt_private(private_pem, "passphrase1234567890", public_pem),
        "passphrase1234567890",
        public_pem,
    )
    sig = kp.sign(priv, "content")
    assert "-----" not in sig  # not PEM


# ─────────────────────── keychain wrapper ─────────────────────────────────────


def test_load_passphrase_raises_when_absent(monkeypatch):
    monkeypatch.setattr("keyring.get_password", lambda *a: None)
    with pytest.raises(PhylacteryError):
        from src.identity.keychain import load_passphrase

        load_passphrase()


def test_store_and_load_passphrase(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        "keyring.set_password",
        lambda svc, usr, pwd: stored.update({(svc, usr): pwd}),
    )
    monkeypatch.setattr(
        "keyring.get_password",
        lambda svc, usr: stored.get((svc, usr)),
    )
    from src.identity.keychain import load_passphrase, store_passphrase

    store_passphrase("my-test-passphrase")
    assert load_passphrase() == "my-test-passphrase"


# ─────────────────────── IdentitySigner ───────────────────────────────────────


def _make_signer_with_keypair(monkeypatch):
    """Return a fully wired IdentitySigner backed by a fresh in-memory keypair."""
    private_pem, public_pem = kp.generate()
    passphrase = "testpassphrase123456"
    encrypted = kp.encrypt_private(private_pem, passphrase, public_pem)

    mock_row = (public_pem.decode("utf-8"), encrypted)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = mock_row
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = mock_cursor

    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: mock_conn)
    monkeypatch.setattr("keyring.get_password", lambda *a: passphrase)

    from src.identity.signer import IdentitySigner

    s = IdentitySigner()
    return s


def test_signer_sign_and_verify(monkeypatch):
    s = _make_signer_with_keypair(monkeypatch)
    sig = s.sign("agent here.")
    assert sig is not None
    assert s.verify("agent here.", sig) is True


def test_signer_verify_tampered_false(monkeypatch):
    s = _make_signer_with_keypair(monkeypatch)
    sig = s.sign("original")
    assert s.verify("tampered", sig) is False


def test_signer_returns_none_when_no_identity(monkeypatch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = mock_cursor

    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: mock_conn)

    from src.identity.signer import IdentitySigner

    s = IdentitySigner()
    assert s.sign("anything") is None
    assert s.verify("anything", "sig") is False


def test_signer_returns_none_on_db_error(monkeypatch):
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("db down")),
    )

    from src.identity.signer import IdentitySigner

    s = IdentitySigner()
    assert s.sign("anything") is None
    # Transient DB miss must not stick — retry path stays open.
    assert s._loaded is False


# ─────────────────────── CLI — init-identity duplicate guard ──────────────────


def test_init_identity_duplicate_raises(monkeypatch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)  # identity already exists
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = mock_cursor

    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: mock_conn)

    from src.identity.cli import cmd_init_identity

    with pytest.raises(PhylacteryError, match="Phylactery is forever"):
        cmd_init_identity()
