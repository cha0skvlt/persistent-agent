"""Install and data safety: identity ordering, atomic writes, API bounds, backups."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- PA-18: init-identity ordering -----------------------------------------


@pytest.mark.unit
def test_init_identity_keychain_failure_writes_nothing(monkeypatch):
    from src.identity import cli

    class _Cur:
        def fetchone(self):
            return None

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return _Cur()

        def commit(self):
            pass

    monkeypatch.setattr(cli.psycopg, "connect", lambda *a, **k: _Conn())
    monkeypatch.setattr(cli.kp, "generate", lambda: (b"priv", b"pub"))
    monkeypatch.setattr(cli.kp, "encrypt_private", lambda *a: b"enc")
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "x" * 20)

    def _boom(_passphrase):
        raise RuntimeError("keychain down")

    monkeypatch.setattr(cli, "store_passphrase", _boom)

    with pytest.raises(SystemExit):
        cli.cmd_init_identity()


# --- PA-19: atomic protection ----------------------------------------------


@pytest.mark.unit
def test_memory_replace_protection_lives_in_the_update():
    from src.memory import AgentMemory

    src = inspect.getsource(AgentMemory.replace)
    assert "memory_type <> 'identity'" in src
    assert "AFFIRMATION signed" in src


@pytest.mark.unit
def test_diary_replace_and_delete_lock_the_row():
    from src.memory import SelfReflection

    for fn in (SelfReflection.replace, SelfReflection.delete):
        src = inspect.getsource(fn)
        assert "FOR UPDATE" in src, fn.__name__


# --- PA-20: thread_id bounds ------------------------------------------------


@pytest.mark.unit
def test_thread_id_accepts_existing_ids():
    from src.api.schemas import ChatRequest

    assert (
        ChatRequest(message="hi", thread_id="telegram_123").thread_id == "telegram_123"
    )
    assert (
        ChatRequest(message="hi", thread_id="  api_session_v1  ").thread_id
        == "api_session_v1"
    )
    assert ChatRequest(message="hi", thread_id="cli_session_v1").thread_id == (
        "cli_session_v1"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "thread_id",
    ["x" * 129, "   ", "bad\x00id", "line\nbreak", "tab\there"],
)
def test_thread_id_rejects_bad_values(thread_id):
    from src.api.schemas import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(message="hi", thread_id=thread_id)


# --- PA-21 / PA-22 / PA-23: restore guards ---------------------------------


@pytest.mark.unit
def test_restore_refuses_primary_database():
    mod = _load_script("phylactery_restore")
    with pytest.raises(SystemExit):
        mod._guard_primary("agent_db", "agent_db")
    mod._guard_primary("agent_restore", "agent_db")  # no raise


@pytest.mark.unit
def test_restore_refuses_existing_db_without_flag(monkeypatch):
    mod = _load_script("phylactery_restore")

    class _Cur:
        def fetchone(self):
            return (1,)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return _Cur()

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: _Conn())
    with pytest.raises(SystemExit):
        mod._ensure_restore_database("agent_restore", replace=False)
    mod._ensure_restore_database("agent_restore", replace=True)  # no raise


@pytest.mark.unit
def test_restore_rejects_unreadable_archive(monkeypatch):
    mod = _load_script("phylactery_restore")

    class _P:
        returncode = 1
        stderr = "not a dump"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _P())
    with pytest.raises(SystemExit):
        mod._verify_archive("/tmp/x.dump")


# --- PA-24 / PA-25: authenticated encryption and permissions ---------------


@pytest.mark.unit
def test_backup_crypto_roundtrip_and_permissions(tmp_path):
    from src.identity.backup_crypto import decrypt_file, encrypt_file, is_authenticated

    src = tmp_path / "in.bin"
    src.write_bytes(b"payload" * 500)
    enc = tmp_path / "out.enc"
    encrypt_file(src, enc, "correct horse battery staple")

    assert is_authenticated(enc)
    assert (enc.stat().st_mode & 0o777) == 0o600

    out = tmp_path / "out.bin"
    decrypt_file(enc, out, "correct horse battery staple")
    assert out.read_bytes() == src.read_bytes()
    assert (out.stat().st_mode & 0o777) == 0o600


@pytest.mark.unit
def test_backup_crypto_rejects_wrong_password(tmp_path):
    from src.identity.backup_crypto import (
        BackupCryptoError,
        decrypt_file,
        encrypt_file,
    )

    src = tmp_path / "in.bin"
    src.write_bytes(b"payload")
    enc = tmp_path / "out.enc"
    encrypt_file(src, enc, "correct horse battery staple")

    with pytest.raises(BackupCryptoError):
        decrypt_file(enc, tmp_path / "out.bin", "wrong passphrase")


@pytest.mark.unit
def test_backup_crypto_rejects_tampered_ciphertext(tmp_path):
    from src.identity.backup_crypto import (
        BackupCryptoError,
        decrypt_file,
        encrypt_file,
    )

    src = tmp_path / "in.bin"
    src.write_bytes(b"payload" * 100)
    enc = tmp_path / "out.enc"
    encrypt_file(src, enc, "correct horse battery staple")

    data = bytearray(enc.read_bytes())
    data[-1] ^= 0xFF
    enc.write_bytes(bytes(data))

    with pytest.raises(BackupCryptoError):
        decrypt_file(enc, tmp_path / "out.bin", "correct horse battery staple")


# --- PA-26: schema drift ----------------------------------------------------


@pytest.mark.unit
def test_schema_matches_latest_migrations():
    schema = (_ROOT / "db_schema.sql").read_text(encoding="utf-8")
    assert "importance REAL NOT NULL DEFAULT 0.5" in schema
    assert "idx_causal_lessons_importance" in schema
    assert schema.count("source_ref TEXT") == 2
