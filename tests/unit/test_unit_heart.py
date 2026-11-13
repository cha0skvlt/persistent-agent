"""Unit tests for src.heart.body and src.heart.vitals."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.heart import body as body_mod
from src.heart import vitals as vitals_mod


@pytest.mark.unit
def test_disk_usage_nonzero():
    du = MagicMock(total=100, used=40)
    with patch.object(body_mod.shutil, "disk_usage", return_value=du):
        assert body_mod.disk_usage_pct("/tmp") == 40.0


@pytest.mark.unit
def test_disk_usage_zero_total():
    with patch.object(
        body_mod.shutil, "disk_usage", return_value=MagicMock(total=0, used=0)
    ):
        assert body_mod.disk_usage_pct() == 0.0


@pytest.mark.unit
def test_battery_pct_macos_and_errors():
    with patch.object(
        body_mod.subprocess,
        "check_output",
        return_value="InternalBattery-0  81%; charging;\n",
    ):
        assert body_mod.battery_pct() == 81
    with patch.object(
        body_mod.subprocess,
        "check_output",
        side_effect=FileNotFoundError(),
    ):
        assert body_mod.battery_pct() is None


@pytest.mark.unit
def test_ollama_running_and_network():
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.object(body_mod.urllib.request, "urlopen", return_value=Resp()):
        assert body_mod.ollama_running("http://x") is True
    with patch.object(
        body_mod.urllib.request,
        "urlopen",
        side_effect=body_mod.URLError(reason="x"),
    ):
        assert body_mod.ollama_running() is False

    with patch.object(body_mod.socket, "gethostbyname", return_value="127.0.0.1"):
        assert body_mod.network_up() is True
    with patch.object(body_mod.socket, "gethostbyname", side_effect=OSError()):
        assert body_mod.network_up() is False


@pytest.mark.unit
def test_pg_conninfo_strip_driver(monkeypatch):
    monkeypatch.setattr(
        body_mod.settings, "database_url", "postgresql+psycopg://localhost/x"
    )
    assert body_mod._pg_conninfo().startswith("postgresql://")


@pytest.mark.unit
def test_pg_alive_and_phylactery_counts(monkeypatch):
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    f = MagicMock()
    f.fetchone.side_effect = [(1,), (2,), (3,), (4,)]
    conn.execute.return_value = f

    monkeypatch.setattr(body_mod.settings, "database_url", "postgresql://localhost/db")
    monkeypatch.setattr(body_mod.psycopg, "connect", MagicMock(return_value=conn))
    assert body_mod.pg_alive() is True

    monkeypatch.setattr(
        body_mod.psycopg,
        "connect",
        MagicMock(side_effect=RuntimeError("down")),
    )
    assert body_mod.pg_alive() is False

    conn2 = MagicMock()
    conn2.__enter__ = lambda s: s
    conn2.__exit__ = MagicMock(return_value=False)
    f2 = MagicMock()
    f2.fetchone.side_effect = [(1,), (2,), (3,), (4,)]
    conn2.execute.return_value = f2
    monkeypatch.setattr(body_mod.psycopg, "connect", MagicMock(return_value=conn2))
    counts = body_mod.phylactery_counts()
    assert counts == {
        "memories": 1,
        "entities": 2,
        "knowledge_graph": 3,
        "self_reflections": 4,
    }


@pytest.mark.unit
def test_collect_body_metrics_includes_battery_when_present(monkeypatch):
    monkeypatch.setattr(body_mod, "disk_usage_pct", lambda path="/": 10.0)
    monkeypatch.setattr(body_mod, "network_up", lambda host="x": True)
    monkeypatch.setattr(body_mod, "ollama_running", lambda base_url=None: False)
    monkeypatch.setattr(body_mod, "battery_pct", lambda: 44)
    m = body_mod.collect_body_metrics()
    assert m["battery_pct"] == 44
    monkeypatch.setattr(body_mod, "battery_pct", lambda: None)
    m2 = body_mod.collect_body_metrics()
    assert "battery_pct" not in m2


@pytest.mark.unit
def test_vitals_fetch_with_pool():
    row = (1, "t", 9, {}, "ok", None)
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    pool = MagicMock()

    @contextmanager
    def _pc():
        yield conn

    pool.connection = _pc
    out = vitals_mod.fetch_latest_vitals_sync(pool=pool)
    assert out and out["tick_id"] == 1


@pytest.mark.unit
def test_vitals_fetch_success_default_conninfo(monkeypatch):
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchone.return_value = (1, "t", 2, {}, "s", None)
    monkeypatch.setattr(vitals_mod.psycopg, "connect", MagicMock(return_value=conn))
    with patch(
        "src.config.settings", MagicMock(database_url_plain_postgresql="postgresql://z")
    ):
        out = vitals_mod.fetch_latest_vitals_sync(conninfo=None)
    assert out and out["tick_id"] == 1


@pytest.mark.unit
def test_vitals_fetch_conninfo_exception(monkeypatch, caplog):
    caplog.set_level("DEBUG", logger="src.heart.vitals")
    monkeypatch.setattr(
        vitals_mod.psycopg,
        "connect",
        MagicMock(side_effect=RuntimeError("no db")),
    )
    assert vitals_mod.fetch_latest_vitals_sync(conninfo="postgresql://bad") is None


@pytest.mark.unit
def test_vitals_fetch_uses_settings_when_no_conninfo(monkeypatch):
    monkeypatch.setattr(
        vitals_mod.psycopg,
        "connect",
        MagicMock(side_effect=OSError("x")),
    )
    with patch(
        "src.config.settings", MagicMock(database_url_plain_postgresql="postgresql://z")
    ):
        assert vitals_mod.fetch_latest_vitals_sync(conninfo=None) is None


@pytest.mark.unit
def test_write_tick_autocommit(caplog):
    """write_tick relies on autocommit — no explicit commit/rollback."""
    caplog.set_level("ERROR", logger="src.heart.vitals")
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (42,)
    assert (
        vitals_mod.write_tick(conn, pid=1, body_metrics={}, phylactery_status="ok")
        == 42
    )
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()

    conn2 = MagicMock()
    conn2.execute.side_effect = RuntimeError("fail")
    assert (
        vitals_mod.write_tick(conn2, pid=1, body_metrics={}, phylactery_status="x")
        is None
    )


@pytest.mark.unit
def test_latest_tick_and_history_errors(caplog):
    caplog.set_level("ERROR", logger="src.heart.vitals")
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("sql")
    assert vitals_mod.latest_tick(conn) is None

    conn2 = MagicMock()
    conn2.execute.return_value.fetchone.return_value = None
    assert vitals_mod.latest_tick(conn2) is None

    conn3 = MagicMock()
    conn3.execute.return_value.fetchone.return_value = (1, "t", 2, {}, "s", None)
    got = vitals_mod.latest_tick(conn3)
    assert got["tick_id"] == 1 and got["phylactery_status"] == "s"

    conn4 = MagicMock()
    conn4.execute.side_effect = RuntimeError("hist")
    assert vitals_mod.tick_history(conn4) == []

    conn5 = MagicMock()
    conn5.execute.return_value.fetchall.return_value = [
        (1, "t1", 2, {}, "s", None),
        (2, "t2", 3, {}, "s2", None),
    ]
    hist = vitals_mod.tick_history(conn5, limit=2)
    assert len(hist) == 2 and hist[0]["tick_id"] == 1
