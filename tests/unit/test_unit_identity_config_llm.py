"""Unit tests for identity, config (URL + rerank warning), llm_factory, ollama_client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.identity.keychain as keychain_mod
import src.llm_factory as lf
from src.identity import reset_signer
from src.identity import signer as signer_mod
from src.identity.keychain import (
    PhylacteryError,
    delete_backup_passphrase,
    delete_passphrase,
    load_backup_passphrase,
    load_passphrase,
    store_backup_passphrase,
    store_passphrase,
)
from src.identity.signer import IdentitySigner


@pytest.mark.unit
def test_keychain_crud(monkeypatch):
    ops: dict[str, dict] = {"sign": {}, "backup": {}}

    def _set(service, user, pw):
        bucket = ops["backup"] if "backup" in service else ops["sign"]
        bucket[(service, user)] = pw

    def _get(service, user):
        bucket = ops["backup"] if "backup" in service else ops["sign"]
        return bucket.get((service, user))

    def _del(service, user):
        bucket = ops["backup"] if "backup" in service else ops["sign"]
        bucket.pop((service, user), None)

    monkeypatch.setattr(keychain_mod.keyring, "set_password", _set)
    monkeypatch.setattr(keychain_mod.keyring, "get_password", _get)
    monkeypatch.setattr(keychain_mod.keyring, "delete_password", _del)

    store_passphrase("s3cret")
    assert load_passphrase() == "s3cret"
    delete_passphrase()

    store_backup_passphrase("bk")
    assert load_backup_passphrase() == "bk"
    delete_backup_passphrase()

    monkeypatch.setattr(keychain_mod.keyring, "get_password", lambda s, u: None)
    with pytest.raises(PhylacteryError):
        load_passphrase()
    with pytest.raises(PhylacteryError):
        load_backup_passphrase()


@pytest.mark.unit
def test_keychain_delete_password_error_swallowed(monkeypatch):
    def _del(_s, _u):
        raise keychain_mod.keyring.errors.PasswordDeleteError()

    monkeypatch.setattr(keychain_mod.keyring, "delete_password", _del)
    delete_passphrase()
    delete_backup_passphrase()


def _signer_conn_mock(row):
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchone.return_value = row
    return conn


@pytest.mark.unit
def test_identity_signer_no_db_row(monkeypatch, caplog):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(
            database_url="postgresql+psycopg://localhost/db",
            default_agent_id="agent1",
        ),
    )
    conn = _signer_conn_mock(None)
    monkeypatch.setattr(signer_mod.psycopg, "connect", MagicMock(return_value=conn))
    caplog.set_level("INFO", logger=signer_mod.__name__)
    assert IdentitySigner().sign("x") is None
    assert any("no identity row" in r.message for r in caplog.records)


@pytest.mark.unit
def test_identity_signer_db_unreachable(monkeypatch, caplog):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(database_url="postgresql://localhost/db", default_agent_id="a"),
    )
    monkeypatch.setattr(
        signer_mod.psycopg,
        "connect",
        MagicMock(side_effect=OSError("down")),
    )
    caplog.set_level("WARNING", logger=signer_mod.__name__)
    assert IdentitySigner().sign("x") is None


@pytest.mark.unit
def test_identity_signer_keychain_missing(monkeypatch):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(database_url="postgresql://localhost/db", default_agent_id="a"),
    )
    conn = _signer_conn_mock(("pem-lines", b"cipher"))
    monkeypatch.setattr(signer_mod.psycopg, "connect", MagicMock(return_value=conn))
    monkeypatch.setattr(
        signer_mod,
        "load_passphrase",
        MagicMock(side_effect=PhylacteryError("missing")),
    )
    assert IdentitySigner().verify("c", "sig") is False


@pytest.mark.unit
def test_identity_signer_decrypt_fails(monkeypatch):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(database_url="postgresql://localhost/db", default_agent_id="a"),
    )
    conn = _signer_conn_mock(("pem", b"enc"))
    monkeypatch.setattr(signer_mod.psycopg, "connect", MagicMock(return_value=conn))
    monkeypatch.setattr(signer_mod, "load_passphrase", MagicMock(return_value="pw"))
    monkeypatch.setattr(
        signer_mod.kp,
        "load_private",
        MagicMock(side_effect=ValueError("bad")),
    )
    assert IdentitySigner().sign("msg") is None


@pytest.mark.unit
def test_identity_signer_sign_verify_and_pem(monkeypatch):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(database_url="postgresql://localhost/db", default_agent_id="a"),
    )
    conn = _signer_conn_mock(("pem", b"enc"))
    monkeypatch.setattr(signer_mod.psycopg, "connect", MagicMock(return_value=conn))
    monkeypatch.setattr(signer_mod, "load_passphrase", MagicMock(return_value="pw"))
    priv = object()
    pub = MagicMock()
    pub.public_bytes.return_value = (
        b"-----BEGIN PUBLIC KEY-----\nMIIB\n-----END PUBLIC KEY-----\n"
    )
    monkeypatch.setattr(signer_mod.kp, "load_private", MagicMock(return_value=priv))
    monkeypatch.setattr(signer_mod.kp, "load_public", MagicMock(return_value=pub))
    monkeypatch.setattr(signer_mod.kp, "sign", MagicMock(return_value="sigb64"))
    monkeypatch.setattr(signer_mod.kp, "verify", MagicMock(return_value=True))

    s = IdentitySigner()
    assert s.sign("hello") == "sigb64"
    assert s.verify("hello", "sigb64") is True
    pem = s.public_key_pem()
    assert pem and "BEGIN PUBLIC KEY" in pem


@pytest.mark.unit
def test_identity_signer_sign_exception(monkeypatch):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(database_url="postgresql://localhost/db", default_agent_id="a"),
    )
    conn = _signer_conn_mock(("pem", b"enc"))
    monkeypatch.setattr(signer_mod.psycopg, "connect", MagicMock(return_value=conn))
    monkeypatch.setattr(signer_mod, "load_passphrase", MagicMock(return_value="pw"))
    monkeypatch.setattr(signer_mod.kp, "load_private", MagicMock(return_value=object()))
    monkeypatch.setattr(
        signer_mod.kp, "load_public", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(
        signer_mod.kp, "sign", MagicMock(side_effect=RuntimeError("no"))
    )
    assert IdentitySigner().sign("z") is None


@pytest.mark.unit
def test_identity_signer_verify_exception(monkeypatch):
    monkeypatch.setattr(
        signer_mod,
        "settings",
        MagicMock(database_url="postgresql://localhost/db", default_agent_id="a"),
    )
    conn = _signer_conn_mock(("pem", b"enc"))
    monkeypatch.setattr(signer_mod.psycopg, "connect", MagicMock(return_value=conn))
    monkeypatch.setattr(signer_mod, "load_passphrase", MagicMock(return_value="pw"))
    monkeypatch.setattr(signer_mod.kp, "load_private", MagicMock(return_value=object()))
    monkeypatch.setattr(
        signer_mod.kp, "load_public", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(
        signer_mod.kp, "verify", MagicMock(side_effect=TypeError("bad"))
    )
    assert IdentitySigner().verify("a", "b") is False


@pytest.mark.unit
def test_identity_get_signer_singleton():
    reset_signer()
    from src.identity import get_signer

    a = get_signer()
    b = get_signer()
    assert a is b
    reset_signer()
    c = get_signer()
    assert c is not a
    reset_signer()


@pytest.mark.unit
def test_settings_database_url_derivatives():
    from src.config import Settings

    s = Settings(database_url="postgresql+psycopg://u:p@dbhost:5433/mydb")
    assert s.database_host == "dbhost"
    assert s.database_port == 5433
    assert s.database_name == "mydb"
    assert s.database_url_plain_postgresql.startswith("postgresql://u:p@")
    r = s.phylactery_restore_database_url
    assert "agent_db_restore" in r
    assert r.startswith("postgresql://")


@pytest.mark.unit
def test_phylactery_restore_url_auth_variants():
    from src.config import Settings

    s = Settings(
        database_url="postgresql://:secret@onlypass.host:5432/maindb",
        phylactery_restore_db="restore_db",
    )
    assert "restore_db" in s.phylactery_restore_database_url

    s2 = Settings(database_url="postgresql://user@host:5432/db")
    assert "user@" in s2.phylactery_restore_database_url


@pytest.mark.unit
def test_llm_factory_and_clear_cache(monkeypatch):
    fake = MagicMock(name="ChatOllamaFake")
    monkeypatch.setattr(lf, "_AgentChatOllama", fake, raising=True)
    lf.clear_llm_cache()
    m1 = lf.get_llm()
    m2 = lf.get_llm()
    assert m1 is m2
    t1 = lf.get_llm_with_tools([type("T", (), {"name": "t"})()])
    t2 = lf.get_llm_with_tools([type("T", (), {"name": "t"})()])
    assert t1 is t2
    _ = lf.get_llm_with_tools([type("U", (), {"name": "u"})()])
    assert m1.bind_tools.call_count == 2
    lf.clear_llm_cache()


@pytest.mark.unit
def test_ollama_chat_options_reach_request_payload():
    from src.config import Settings

    s = Settings(
        _env_file=None,
        llm_num_ctx=32768,
        llm_num_predict=16384,
        reasoning_model="qwen3-abliterated:32b",
    )
    llm = lf.make_llm(s, for_agent=True)
    opts = lf.ollama_chat_request_options(llm)
    samp = lf.fixed_sampling_summary()
    assert opts["num_ctx"] == 32768
    assert opts["num_predict"] == 16384
    assert opts["temperature"] == samp["temperature"]
    assert opts["top_p"] == samp["top_p"]
    assert opts["top_k"] == samp["top_k"]
    assert opts["min_p"] == samp["min_p"]
    assert opts["repeat_penalty"] == samp["repeat_penalty"]
    assert opts["presence_penalty"] == samp["presence_penalty"]

    def _search_memory(query: str) -> str:
        """Search semantic memory."""
        return query

    llm2 = lf.make_llm(s, for_agent=True)
    bound2 = llm2.bind_tools([_search_memory])
    opts2 = lf.ollama_chat_request_options(bound2)
    assert opts2["top_k"] == samp["top_k"]
    assert opts2["min_p"] == samp["min_p"]
    assert opts2["presence_penalty"] == samp["presence_penalty"]


@pytest.mark.unit
def test_ollama_tools_options_are_cooler_than_speech():
    from src.config import Settings

    s = Settings(
        _env_file=None,
        llm_num_ctx=32768,
        llm_num_predict=8192,
        reasoning_model="qwen3-abliterated:32b",
    )
    llm = lf.make_llm(s, for_agent=True, mode="tools")
    opts = lf.ollama_chat_request_options(llm)
    tools = lf.fixed_tools_sampling_summary()
    speech = lf.fixed_sampling_summary()
    assert opts["num_predict"] == tools["num_predict"] == 4096
    assert opts["temperature"] == tools["temperature"] == 0.3
    assert opts["top_p"] == tools["top_p"] == 0.8
    assert opts["min_p"] == tools["min_p"] == 0.05
    assert float(opts["temperature"]) < float(speech["temperature"])
    assert not bool(getattr(llm, "reasoning", False))


@pytest.mark.unit
def test_warn_llm_num_predict_if_needed(monkeypatch):
    import src.config as cfg

    warned: list = []
    monkeypatch.setattr(cfg._log, "warning", lambda *a, **k: warned.append(a))
    cfg.warn_llm_num_predict_if_needed(
        cfg.Settings(_env_file=None, llm_num_predict=2048)
    )
    assert warned
    assert "num_predict too low for a thinking model" in str(warned[0])


@pytest.mark.unit
def test_warn_llm_num_predict_skips_healthy_default(monkeypatch):
    import src.config as cfg

    warned: list = []
    monkeypatch.setattr(cfg._log, "warning", lambda *a, **k: warned.append(a))
    cfg.warn_llm_num_predict_if_needed(
        cfg.Settings(_env_file=None, llm_num_predict=16384)
    )
    assert not warned


@pytest.mark.unit
def test_identity_signer_public_key_unloaded():
    s = IdentitySigner()
    s._loaded = True
    s._private_key = object()
    s._public_key = None
    assert s.public_key_pem() is None


@pytest.mark.unit
def test_normalize_empty_reasoning_model(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("REASONING_MODEL", "")
    assert Settings(_env_file=None).reasoning_model == "qwen3:8b"


@pytest.mark.unit
def test_keep_alive_normalization_and_validation():
    from src.config import Settings

    s = Settings(llm_keep_alive="-1")
    assert s.llm_keep_alive == "-1m"
    s2 = Settings(llm_keep_alive="30")
    assert s2.llm_keep_alive == "30m"
    assert Settings(llm_keep_alive="30m").llm_keep_alive == "30m"
    assert Settings(llm_keep_alive="0").llm_keep_alive == "0"
    assert Settings(llm_keep_alive="-42").llm_keep_alive == "-1m"


@pytest.mark.unit
def test_ollama_http_client_strips_slash(monkeypatch):
    import src.ollama_client as oc

    fake_client = MagicMock()

    class _Mod:
        @staticmethod
        def Client(*, host):
            fake_client.host = host
            return fake_client

    monkeypatch.setattr(oc, "ollama", _Mod())
    monkeypatch.setattr(oc, "settings", MagicMock(ollama_base_url=" http://x:1/ "))
    c = oc.ollama_http_client()
    assert c is fake_client
    assert fake_client.host == "http://x:1"
