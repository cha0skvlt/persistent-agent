"""heart daemon unit tests.

All DB/OS calls are mocked — no live Postgres or filesystem required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.heart import body as body_mod
from src.heart import vitals as vitals_mod
from src.heart.heartbeat import (
    _compute_phylactery_status,
    _reconnect_loop,
    _require_advisory_lock,
    tick,
)

# ─────────────────────── body.py helpers ──────────────────────────────────────


def test_disk_usage_pct_returns_float(monkeypatch):
    from collections import namedtuple

    DU = namedtuple("DU", ["total", "used", "free"])
    monkeypatch.setattr("shutil.disk_usage", lambda _p: DU(total=100, used=30, free=70))
    from src.heart.body import disk_usage_pct

    assert disk_usage_pct() == 30.0


def test_disk_usage_pct_zero_total(monkeypatch):
    from collections import namedtuple

    DU = namedtuple("DU", ["total", "used", "free"])
    monkeypatch.setattr("shutil.disk_usage", lambda _p: DU(total=0, used=0, free=0))
    from src.heart.body import disk_usage_pct

    assert disk_usage_pct() == 0.0


def test_battery_pct_parsed(monkeypatch):
    output = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=123456)\t72%; charging; 0:00 remaining present: true\n"
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: output)
    from src.heart.body import battery_pct

    result = battery_pct()
    assert result == 72


def test_battery_pct_none_on_error(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
    )
    from src.heart.body import battery_pct

    assert battery_pct() is None


def test_ollama_running_true(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: mock_resp)
    from src.heart.body import ollama_running

    assert ollama_running() is True


def test_ollama_running_false_on_error(monkeypatch):
    from urllib.error import URLError

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(URLError("refused")),
    )
    from src.heart.body import ollama_running

    assert ollama_running() is False


def test_network_up_true(monkeypatch):
    monkeypatch.setattr("socket.gethostbyname", lambda _h: "1.2.3.4")
    from src.heart.body import network_up

    assert network_up() is True


def test_network_up_false(monkeypatch):
    import socket

    monkeypatch.setattr(
        "socket.gethostbyname",
        lambda _h: (_ for _ in ()).throw(socket.gaierror("fail")),
    )
    from src.heart.body import network_up

    assert network_up() is False


def test_collect_body_metrics_keys(monkeypatch):
    monkeypatch.setattr(body_mod, "disk_usage_pct", lambda *a, **kw: 42.0)
    monkeypatch.setattr(body_mod, "network_up", lambda *a, **kw: True)
    monkeypatch.setattr(body_mod, "ollama_running", lambda *a, **kw: False)
    monkeypatch.setattr(body_mod, "battery_pct", lambda: 80)

    metrics = body_mod.collect_body_metrics()
    assert "disk_usage_pct" in metrics
    assert "network_up" in metrics
    assert "ollama_running" in metrics
    assert "battery_pct" in metrics


# ─────────────────────── vitals.py helpers ─────────────────────────────────────


def _make_mock_conn(fetchone_val=None, fetchall_val=None, rowcount=1):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_val
    cursor.fetchall.return_value = fetchall_val or []
    conn.execute.return_value = cursor
    conn.commit = MagicMock()
    conn.rollback = MagicMock()
    return conn


def test_write_tick_returns_tick_id():
    conn = _make_mock_conn(fetchone_val=(7,))
    result = vitals_mod.write_tick(
        conn,
        pid=1234,
        body_metrics={"disk_usage_pct": 10.0},
        phylactery_status="ok",
    )
    assert result == 7


def test_write_tick_returns_none_on_error():
    conn = MagicMock()
    conn.execute.side_effect = Exception("DB error")
    result = vitals_mod.write_tick(
        conn,
        pid=1234,
        body_metrics={},
        phylactery_status="ok",
    )
    assert result is None


def test_latest_tick_returns_dict():
    import datetime

    row = (1, datetime.datetime.now(), 111, {"disk_usage_pct": 20.0}, "ok", None)
    conn = _make_mock_conn(fetchone_val=row)
    result = vitals_mod.latest_tick(conn)
    assert result is not None
    assert result["tick_id"] == 1
    assert result["phylactery_status"] == "ok"
    assert result["signature"] is None


def test_latest_tick_returns_none_on_empty():
    conn = _make_mock_conn(fetchone_val=None)
    assert vitals_mod.latest_tick(conn) is None


def test_tick_history_returns_list():
    import datetime

    rows = [(i, datetime.datetime.now(), 111, {}, "ok", None) for i in range(3)]
    conn = _make_mock_conn(fetchall_val=rows)
    result = vitals_mod.tick_history(conn, limit=3)
    assert len(result) == 3
    assert result[0]["tick_id"] == 0


# ─────────────────────── heartbeat.py tick logic ──────────────────────────────


def test_compute_phylactery_status_fail_when_pg_down(monkeypatch):
    monkeypatch.setattr("src.heart.heartbeat.pg_alive", lambda: False)
    status, counts = _compute_phylactery_status()
    assert status == "fail"
    assert counts == {}


def test_compute_phylactery_status_ok(monkeypatch):
    monkeypatch.setattr("src.heart.heartbeat.pg_alive", lambda: True)
    monkeypatch.setattr(
        "src.heart.heartbeat.phylactery_counts",
        lambda: {
            "memories": 10,
            "entities": 5,
            "knowledge_graph": 3,
            "self_reflections": 1,
        },
    )
    status, counts = _compute_phylactery_status()
    assert status == "ok"
    assert counts["memories"] == 10


def test_compute_phylactery_status_degraded_empty_table(monkeypatch):
    monkeypatch.setattr("src.heart.heartbeat.pg_alive", lambda: True)
    monkeypatch.setattr(
        "src.heart.heartbeat.phylactery_counts",
        lambda: {
            "memories": 10,
            "entities": 0,
            "knowledge_graph": 3,
            "self_reflections": 1,
        },
    )
    status, counts = _compute_phylactery_status()
    assert status == "degraded"


def test_compute_phylactery_status_degraded_on_count_exception(monkeypatch):
    monkeypatch.setattr("src.heart.heartbeat.pg_alive", lambda: True)
    monkeypatch.setattr(
        "src.heart.heartbeat.phylactery_counts",
        lambda: (_ for _ in ()).throw(Exception("query failed")),
    )
    status, _counts = _compute_phylactery_status()
    assert status == "degraded"


def test_tick_skips_write_on_db_error(monkeypatch):
    """tick() must not raise when write_tick returns None (DB outage)."""
    monkeypatch.setattr(
        "src.heart.heartbeat.collect_body_metrics", lambda: {"disk_usage_pct": 5.0}
    )
    monkeypatch.setattr(
        "src.heart.heartbeat._compute_phylactery_status",
        lambda: ("ok", {}),
    )
    monkeypatch.setattr("src.heart.heartbeat.latest_tick", lambda _c: None)
    monkeypatch.setattr("src.heart.heartbeat._maybe_schedule_backup", lambda _p: None)
    monkeypatch.setattr("src.heart.heartbeat.write_tick", lambda *a, **kw: None)

    mock_conn = MagicMock()
    # Should not raise
    tick(mock_conn)


def test_tick_calls_write_tick_with_correct_fields(monkeypatch):
    captured: list[dict] = []

    def _fake_write(conn, *, pid, body_metrics, phylactery_status, signature):
        captured.append(
            {"pid": pid, "bm": body_metrics, "st": phylactery_status, "sig": signature}
        )
        return 99

    monkeypatch.setattr(
        "src.heart.heartbeat.collect_body_metrics",
        lambda: {"disk_usage_pct": 12.0},
    )
    monkeypatch.setattr(
        "src.heart.heartbeat._compute_phylactery_status",
        lambda: ("ok", {"memories": 5}),
    )
    monkeypatch.setattr("src.heart.heartbeat.latest_tick", lambda _c: None)
    monkeypatch.setattr("src.heart.heartbeat._maybe_schedule_backup", lambda _p: None)
    monkeypatch.setattr("src.heart.heartbeat.sign_tick_payload", lambda _b, _s: None)
    monkeypatch.setattr("src.heart.heartbeat.write_tick", _fake_write)

    mock_conn = MagicMock()
    tick(mock_conn)

    assert len(captured) == 1
    assert captured[0]["st"] == "ok"
    assert captured[0]["sig"] is None
    assert "phylactery_counts" in captured[0]["bm"]


def test_require_advisory_lock_exits_when_lock_held():
    conn = MagicMock()
    with patch("src.heart.heartbeat._acquire_advisory_lock", return_value=False):
        with patch("src.heart.heartbeat.sys.exit") as mock_exit:
            _require_advisory_lock(conn)
    conn.close.assert_called_once()
    mock_exit.assert_called_once_with(1)


def test_require_advisory_lock_ok_when_acquired():
    conn = MagicMock()
    with patch("src.heart.heartbeat._acquire_advisory_lock", return_value=True):
        assert _require_advisory_lock(conn) is True


def test_reconnect_loop_exits_when_lock_not_acquired(monkeypatch):
    monkeypatch.setattr("src.heart.heartbeat._shutdown_requested", False)
    monkeypatch.setattr("src.heart.heartbeat._sleep_interruptible", lambda _s: None)
    with patch("src.heart.heartbeat.psycopg.connect") as mock_connect:
        mock_connect.return_value = MagicMock()
        with patch("src.heart.heartbeat._acquire_advisory_lock", return_value=False):
            with patch("src.heart.heartbeat.sys.exit") as mock_exit:
                _reconnect_loop(1.0)
    mock_exit.assert_called_once_with(1)


def test_reconnect_loop_returns_conn_when_lock_acquired(monkeypatch):
    monkeypatch.setattr("src.heart.heartbeat._shutdown_requested", False)
    monkeypatch.setattr("src.heart.heartbeat._sleep_interruptible", lambda _s: None)
    conn = MagicMock()
    with patch("src.heart.heartbeat.psycopg.connect", return_value=conn):
        with patch("src.heart.heartbeat._acquire_advisory_lock", return_value=True):
            out = _reconnect_loop(1.0)
    assert out is conn
