"""Unit tests for individual selftest checks (mocked IO)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.selftest.checks import (
    EMBED_DIM,
    _model_in_tags,
    check_agent_tools,
    check_config_sanity,
    check_embed_model,
    check_heartbeat,
    check_identity,
    check_memory_stats,
    check_ollama_reachable,
    check_pgvector,
    check_postgres,
    check_reasoning_model,
    check_tables,
    check_vitals_freshness,
    check_web_tools,
)
from src.selftest.core import CheckStatus


@pytest.fixture
def settings():
    s = MagicMock()
    s.database_url = "postgresql://localhost/test"
    s.ollama_base_url = "http://127.0.0.1:11434"
    s.reasoning_model = "test-reason:32b"
    s.embed_model = "nomic-embed-text"
    s.llm_keep_alive = "30m"
    s.default_agent_id = "agent"
    s.checkpointer = "postgres"
    s.vitals_trust_mode = "strict"
    s.heart_interval_sec = 30
    s.web_fetch_enabled = True
    s.web_search_enabled = True
    return s


def test_model_in_tags_prefix_match():
    assert _model_in_tags("llama3", ["llama3:latest"]) is True
    assert _model_in_tags("missing", ["other"]) is False


def test_check_postgres_ok(settings):
    with patch("src.selftest.checks.assert_database_ready"):
        r = check_postgres(settings)
    assert r.status == CheckStatus.OK


def test_check_postgres_fail(settings):
    with patch(
        "src.selftest.checks.assert_database_ready",
        side_effect=RuntimeError("down"),
    ):
        r = check_postgres(settings)
    assert r.status == CheckStatus.FAIL


def test_check_pgvector_ok(settings):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (1,)
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_pgvector(settings)
    assert r.status == CheckStatus.OK


def test_check_pgvector_missing(settings):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = None
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_pgvector(settings)
    assert r.status == CheckStatus.FAIL


def test_check_tables_fail_missing(settings):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = [("memories",)]
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_tables(settings)
    assert r.status == CheckStatus.FAIL
    assert "missing" in r.detail


def test_check_tables_warn_checkpoint(settings):
    from src.selftest.checks import REQUIRED_TABLES

    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = [(t,) for t in REQUIRED_TABLES]
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_tables(settings)
    assert r.status == CheckStatus.WARN


def test_check_ollama_reachable_ok(settings):
    with (
        patch("src.selftest.checks.assert_ollama_ready"),
        patch("src.selftest.checks._ollama_tags", return_value=["a"]),
    ):
        r = check_ollama_reachable(settings)
    assert r.status == CheckStatus.OK


def test_check_reasoning_model_fail(settings):
    with patch("src.selftest.checks._ollama_tags", return_value=["other"]):
        r = check_reasoning_model(settings)
    assert r.status == CheckStatus.FAIL


def test_check_embed_model_ok(settings):
    client = MagicMock()
    client.embeddings.return_value = {"embedding": [0.1] * EMBED_DIM}
    with (
        patch("src.selftest.checks._ollama_tags", return_value=[settings.embed_model]),
        patch("src.selftest.checks.ollama_http_client", return_value=client),
    ):
        r = check_embed_model(settings)
    assert r.status == CheckStatus.OK


def test_check_embed_model_ollama_response_object(settings):
    """Ollama SDK returns EmbeddingsResponse (dict-like), not a plain dict."""
    client = MagicMock()
    client.embeddings.return_value = MagicMock(
        get=MagicMock(return_value=[0.1] * EMBED_DIM)
    )
    with (
        patch("src.selftest.checks._ollama_tags", return_value=[settings.embed_model]),
        patch("src.selftest.checks.ollama_http_client", return_value=client),
    ):
        r = check_embed_model(settings)
    assert r.status == CheckStatus.OK


def test_check_embed_model_wrong_dim(settings):
    client = MagicMock()
    client.embeddings.return_value = {"embedding": [0.1] * 64}
    with (
        patch("src.selftest.checks._ollama_tags", return_value=[settings.embed_model]),
        patch("src.selftest.checks.ollama_http_client", return_value=client),
    ):
        r = check_embed_model(settings)
    assert r.status == CheckStatus.FAIL
    assert "dim=" in r.detail


def test_check_agent_tools_ok(settings):
    import src.graph.workflow as w
    from src.selftest.checks import EXPECTED_AGENT_TOOLS

    tools = []
    for n in EXPECTED_AGENT_TOOLS:
        m = MagicMock()
        m.name = n
        tools.append(m)
    with patch.object(w, "tools", tools):
        r = check_agent_tools(settings)
    assert r.status == CheckStatus.OK


def test_check_agent_tools_extra_warn(settings):
    import src.graph.workflow as w
    from src.selftest.checks import EXPECTED_AGENT_TOOLS

    tools = []
    for n in EXPECTED_AGENT_TOOLS:
        m = MagicMock()
        m.name = n
        tools.append(m)
    extra = MagicMock()
    extra.name = "mystery_tool"
    tools.append(extra)
    with patch.object(w, "tools", tools):
        r = check_agent_tools(settings)
    assert r.status == CheckStatus.WARN


def test_check_agent_tools_missing(settings):
    import src.graph.workflow as w

    with patch.object(w, "tools", []):
        r = check_agent_tools(settings)
    assert r.status == CheckStatus.FAIL


def test_check_identity_ok(settings):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("agent",)
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_identity(settings)
    assert r.status == CheckStatus.OK


def test_check_identity_warn_empty(settings):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = None
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_identity(settings)
    assert r.status == CheckStatus.WARN


def test_check_heartbeat_ok_with_pid(settings):
    import os

    tick = datetime.now(timezone.utc) - timedelta(seconds=10)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": "sig", "heart_pid": os.getpid()},
    ):
        r = check_heartbeat(settings)
    assert r.status == CheckStatus.OK
    assert r.name == "heartbeat"
    assert "heart beating" in r.detail
    assert str(os.getpid()) in r.detail


def test_check_vitals_freshness_alias_ok(settings):
    tick = datetime.now(timezone.utc) - timedelta(seconds=10)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": "sig"},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.OK


def test_check_vitals_freshness_fail_strict(settings):
    with patch("src.heart.vitals.fetch_latest_vitals_sync", return_value=None):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.FAIL


def test_check_vitals_freshness_warn_dev(settings):
    settings.vitals_trust_mode = "dev"
    with patch("src.heart.vitals.fetch_latest_vitals_sync", return_value=None):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.WARN


def test_check_vitals_stale_strict(settings):
    tick = datetime.now(timezone.utc) - timedelta(seconds=600)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": "sig"},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.FAIL


def test_check_memory_stats_ok(settings):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.side_effect = [(3,), (5,), (2,)]
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_memory_stats(settings)
    assert r.status == CheckStatus.OK
    assert "memories=3" in r.detail


def test_check_config_sanity_ok(settings):
    r = check_config_sanity(settings)
    assert r.status == CheckStatus.OK


def test_check_config_sanity_warn(settings):
    settings.llm_keep_alive = "30"
    r = check_config_sanity(settings)
    assert r.status == CheckStatus.WARN


def test_connect(settings):
    conn = MagicMock()
    with patch("src.selftest.checks.psycopg.connect", return_value=conn) as connect:
        from src.selftest.checks import _connect

        assert _connect(settings) is conn
    connect.assert_called_once_with(
        settings.database_url_plain_postgresql, connect_timeout=5
    )


def test_check_web_tools_skip(settings):
    r = check_web_tools(settings, probe_network=False)
    assert r.status == CheckStatus.SKIP


def test_check_web_tools_probe_ok(settings):
    with patch(
        "src.tools.web_core.fetch_url_text",
        return_value="URL: https://example.com/\n---\nExample Domain",
    ):
        r = check_web_tools(settings, probe_network=True)
    assert r.status == CheckStatus.OK


def test_check_web_tools_probe_fail(settings):
    with patch(
        "src.tools.web_core.fetch_url_text",
        return_value="Fetch failed: timeout",
    ):
        r = check_web_tools(settings, probe_network=True)
    assert r.status == CheckStatus.FAIL


def test_model_in_tags_empty():
    assert _model_in_tags("", ["x"]) is False
    assert _model_in_tags("exact", ["exact"]) is True


def test_ollama_tags_parses_response(settings):
    payload = b'{"models":[{"name":"a"},{"name":""},{"x":1}]}'
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = payload
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    with patch("src.selftest.checks.urllib_request.urlopen", return_value=resp):
        tags = __import__(
            "src.selftest.checks", fromlist=["_ollama_tags"]
        )._ollama_tags(settings)
    assert tags == ["a"]


def test_check_pgvector_db_error(settings):
    with patch("src.selftest.checks._connect", side_effect=OSError("no db")):
        r = check_pgvector(settings)
    assert r.status == CheckStatus.FAIL


def test_check_tables_all_present(settings):
    from src.selftest.checks import CHECKPOINT_TABLES, REQUIRED_TABLES

    all_tables = REQUIRED_TABLES | CHECKPOINT_TABLES
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = [(t,) for t in all_tables]
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("src.selftest.checks._connect", return_value=conn):
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        r = check_tables(settings)
    assert r.status == CheckStatus.OK


def test_required_tables_match_db_schema_snapshot():
    import re
    from pathlib import Path

    from src.selftest.checks import REQUIRED_TABLES

    schema = Path(__file__).resolve().parents[3] / "db_schema.sql"
    names = set(
        re.findall(
            r"CREATE TABLE(?: IF NOT EXISTS)?\s+(\w+)",
            schema.read_text(encoding="utf-8"),
            flags=re.IGNORECASE,
        )
    )
    assert REQUIRED_TABLES == names


def test_check_tables_connect_error(settings):
    with patch("src.selftest.checks._connect", side_effect=OSError("down")):
        r = check_tables(settings)
    assert r.status == CheckStatus.FAIL


def test_check_ollama_fail(settings):
    with patch(
        "src.selftest.checks.assert_ollama_ready",
        side_effect=RuntimeError("offline"),
    ):
        r = check_ollama_reachable(settings)
    assert r.status == CheckStatus.FAIL


def test_check_reasoning_model_ok(settings):
    with patch(
        "src.selftest.checks._ollama_tags", return_value=[settings.reasoning_model]
    ):
        r = check_reasoning_model(settings)
    assert r.status == CheckStatus.OK


def test_check_embed_model_not_in_tags(settings):
    with patch("src.selftest.checks._ollama_tags", return_value=["other"]):
        r = check_embed_model(settings)
    assert r.status == CheckStatus.FAIL


def test_check_embed_model_no_vector(settings):
    client = MagicMock()
    client.embeddings.return_value = {}
    with (
        patch("src.selftest.checks._ollama_tags", return_value=[settings.embed_model]),
        patch("src.selftest.checks.ollama_http_client", return_value=client),
    ):
        r = check_embed_model(settings)
    assert r.status == CheckStatus.FAIL


def test_check_agent_tools_exception(settings):
    import src.graph.workflow as w

    broken = MagicMock()
    broken.__iter__ = MagicMock(side_effect=RuntimeError("boom"))
    with patch.object(w, "tools", broken):
        r = check_agent_tools(settings)
    assert r.status == CheckStatus.FAIL
    assert "boom" in r.detail


def test_check_web_tools_both_disabled(settings):
    settings.web_fetch_enabled = False
    settings.web_search_enabled = False
    r = check_web_tools(settings, probe_network=True)
    assert r.status == CheckStatus.WARN


def test_check_web_tools_probe_exception(settings):
    with patch(
        "src.tools.web_core.fetch_url_text",
        side_effect=TimeoutError("slow"),
    ):
        r = check_web_tools(settings, probe_network=True)
    assert r.status == CheckStatus.FAIL


def test_check_identity_db_error(settings):
    with patch("src.selftest.checks._connect", side_effect=OSError("no")):
        r = check_identity(settings)
    assert r.status == CheckStatus.WARN


def test_check_vitals_unsigned_strict(settings):
    tick = datetime.now(timezone.utc) - timedelta(seconds=5)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": ""},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.FAIL


def test_check_vitals_unsigned_dev(settings):
    settings.vitals_trust_mode = "dev"
    tick = datetime.now(timezone.utc) - timedelta(seconds=5)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": ""},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.WARN


def test_check_vitals_no_tick_at(settings):
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": None, "signature": "s"},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.FAIL


def test_check_vitals_stale_dev(settings):
    settings.vitals_trust_mode = "dev"
    tick = datetime.now(timezone.utc) - timedelta(seconds=600)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": "sig"},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.WARN


def test_check_vitals_exception_dev(settings):
    settings.vitals_trust_mode = "dev"
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        side_effect=RuntimeError("pg"),
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.WARN


def test_check_memory_stats_db_error(settings):
    with patch("src.selftest.checks._connect", side_effect=OSError("x")):
        r = check_memory_stats(settings)
    assert r.status == CheckStatus.FAIL
    assert "unavailable" in r.detail


def test_web_fetch_probe_failed():
    from src.selftest.checks import _web_fetch_probe_failed

    assert _web_fetch_probe_failed("")
    assert _web_fetch_probe_failed("   ")
    assert _web_fetch_probe_failed("HTTP 404 for https://example.com/")
    assert _web_fetch_probe_failed("URL: x\n(no readable text extracted)")
    assert _web_fetch_probe_failed("Web fetch is disabled (WEB_FETCH_ENABLED=0).")
    assert _web_fetch_probe_failed("URL response too large (>100 bytes)")
    assert not _web_fetch_probe_failed("URL: https://example.com/\n---\nExample Domain")


def test_check_web_tools_fetch_disabled(settings):
    settings.web_fetch_enabled = False
    settings.web_search_enabled = True
    r = check_web_tools(settings, probe_network=True)
    assert r.status == CheckStatus.WARN
    assert "WEB_FETCH_ENABLED=0" in r.detail


def test_check_web_tools_probe_no_text(settings):
    with patch(
        "src.tools.web_core.fetch_url_text",
        return_value="URL: https://example.com/\n(no readable text extracted)",
    ):
        r = check_web_tools(settings, probe_network=True)
    assert r.status == CheckStatus.FAIL


def test_check_config_empty_keep_alive(settings):
    settings.llm_keep_alive = ""
    r = check_config_sanity(settings)
    assert r.status == CheckStatus.WARN


def test_check_config_no_reply_languages(settings, monkeypatch):
    import src.selftest.checks as chk

    monkeypatch.setattr(chk, "ALLOWED_LANGUAGES", set())
    r = check_config_sanity(settings)
    assert r.status == CheckStatus.WARN


def test_ollama_tags_http_error(settings):
    resp = MagicMock()
    resp.status = 503
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    with patch("src.selftest.checks.urllib_request.urlopen", return_value=resp):
        import src.selftest.checks as chk

        with pytest.raises(RuntimeError, match="HTTP 503"):
            chk._ollama_tags(settings)


def test_check_reasoning_model_tags_error(settings):
    with patch("src.selftest.checks._ollama_tags", side_effect=OSError("net")):
        r = check_reasoning_model(settings)
    assert r.status == CheckStatus.FAIL
    assert "net" in r.detail


def test_check_embed_model_client_error(settings):
    client = MagicMock()
    client.embeddings.side_effect = RuntimeError("embed down")
    with (
        patch("src.selftest.checks._ollama_tags", return_value=[settings.embed_model]),
        patch("src.selftest.checks.ollama_http_client", return_value=client),
    ):
        r = check_embed_model(settings)
    assert r.status == CheckStatus.FAIL


def test_check_vitals_no_tick_at_dev(settings):
    settings.vitals_trust_mode = "dev"
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": None, "signature": "s"},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.WARN


def test_check_vitals_naive_datetime(settings):
    tick = datetime.now() - timedelta(seconds=5)
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        return_value={"tick_at": tick, "signature": "sig"},
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.OK


def test_check_vitals_exception_strict(settings):
    with patch(
        "src.heart.vitals.fetch_latest_vitals_sync",
        side_effect=RuntimeError("pg"),
    ):
        r = check_vitals_freshness(settings)
    assert r.status == CheckStatus.FAIL
