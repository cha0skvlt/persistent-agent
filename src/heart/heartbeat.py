"""Heart daemon main loop.

Run as a standalone process via launchd:
    python -m src.heart.heartbeat

Loop: every HEART_INTERVAL_SEC seconds:
  1. Collect body_metrics (disk, battery, ollama, network).
  2. Determine phylactery_status by probing PG + counting phylactery tables.
  3. Write tick to vitals. If DB is down — log and continue, do NOT exit.
  4. Log WARNING if phylactery_status != 'ok'.

Advisory lock pg_try_advisory_lock(42) prevents duplicate heart instances.
Graceful shutdown on SIGTERM / SIGINT.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

import psycopg

from src.config import settings
from src.heart.alerts import emit_critical_vital_alerts
from src.heart.api_watch import api_watch_body_patch
from src.heart.body import collect_body_metrics, pg_alive, phylactery_counts
from src.heart.notify import notify_telegram
from src.heart.outreach import send_outreach_if_due
from src.heart.pulse import mark_due_intents
from src.heart.vitals import latest_tick, write_tick

_TICK_SIGN_CONTENT_VERSION = "v1"

_log = logging.getLogger(__name__)

_ADVISORY_LOCK_KEY = 42
_shutdown_requested = False
_HEART_START_UNIX = 0.0
_last_backup_unix = 0.0
# pg_dump may run for many minutes — it must not stall vitals/outreach/pulse,
# so tick() runs it on a worker thread and collects the result next tick.
_backup_thread: "threading.Thread | None" = None
_backup_result_iso: str | None = None

# Resolved once for tests (`monkeypatch` can replace this binding).
_PHYLACTERY_BACKUP_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "phylactery_backup.py"
)


def _pg_conninfo() -> str:
    url = settings.database_url
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


def _build_sys_logger() -> None:
    """Same log frame as API/REPL: ConsoleFormatter + LOG_LEVEL + quiet httpx."""
    from src.obs.logging_setup import configure_agent_logging

    configure_agent_logging(level=settings.log_level, stream=sys.stdout)


def _handle_signal(signum: int, frame: FrameType | None) -> None:
    # Flag only — no blocking I/O in a signal handler; notify happens in run().
    global _shutdown_requested
    _log.info("Heart received signal %s — shutting down gracefully.", signum)
    _shutdown_requested = True


def _compute_phylactery_status() -> tuple[str, dict]:
    """Return (status, counts_or_empty). No exception escapes."""
    if not pg_alive():
        return "fail", {}
    try:
        counts = phylactery_counts()
    except Exception as exc:
        _log.warning("phylactery_counts failed: %s", exc)
        return "degraded", {}
    if all(v > 0 for v in counts.values()):
        return "ok", counts
    # PG is alive but at least one table is empty → degraded
    empty = [k for k, v in counts.items() if v == 0]
    _log.warning("heart.phylactery.degraded empty_tables=%s", empty)
    return "degraded", counts


def _acquire_advisory_lock(conn: psycopg.Connection) -> bool:
    """Try pg_try_advisory_lock(42). Returns True if acquired (we are sole instance)."""
    row = conn.execute(
        "SELECT pg_try_advisory_lock(%s)", (_ADVISORY_LOCK_KEY,)
    ).fetchone()
    return bool(row and row[0])


def vitals_tick_sign_content(body_metrics: dict, phylactery_status: str) -> str:
    """Canonical signed payload for one vitals tick (write + verify must match)."""
    return (
        json.dumps(body_metrics, sort_keys=True)
        + phylactery_status
        + _TICK_SIGN_CONTENT_VERSION
    )


def sign_tick_payload(body_metrics: dict, phylactery_status: str) -> str | None:
    """Compute tick signature. Returns None if identity unavailable."""
    try:
        from src.identity import get_signer

        return get_signer().sign(
            vitals_tick_sign_content(body_metrics, phylactery_status)
        )
    except Exception:
        return None


def _require_advisory_lock(conn: psycopg.Connection) -> bool:
    """Acquire lock or exit — prevents duplicate heart after reconnect."""
    if _acquire_advisory_lock(conn):
        return True
    _log.error(
        '{"event":"heart.duplicate","msg":"Another heart instance holds the advisory lock. Exiting."}'
    )
    try:
        conn.close()
    except Exception:
        pass
    sys.exit(1)


def _maybe_schedule_backup(prev_backup_iso: str | None) -> str | None:
    """Run encrypted pg_dump when interval elapsed. Returns new ISO timestamp or None."""
    global _last_backup_unix

    interval = int(settings.heart_backup_interval_sec)
    if interval <= 0:
        return None

    now = time.time()
    interval_f = float(interval)

    # Cooldown after last attempt (ok or fail). Without this, missing pg_dump
    # under launchd PATH spam WARNING every HEART_INTERVAL_SEC forever.
    if _last_backup_unix > 0 and (now - _last_backup_unix) < interval_f:
        return None

    ref_ts: float | None = None
    if prev_backup_iso:
        try:
            iso = prev_backup_iso.replace("Z", "+00:00")
            ref_ts = datetime.fromisoformat(iso).timestamp()
        except ValueError:
            ref_ts = None

    if ref_ts is not None and (now - ref_ts) < interval_f:
        return None
    if (
        ref_ts is None
        and _last_backup_unix == 0
        and _HEART_START_UNIX > 0
        and (now - _HEART_START_UNIX) < interval_f
    ):
        return None

    script = _PHYLACTERY_BACKUP_SCRIPT
    if not script.is_file():
        _last_backup_unix = now
        _log.warning("phylactery_backup.py missing at %s", script)
        return None

    repo = script.parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    # launchd PATH is often /usr/bin:/bin — Homebrew pg_dump would not resolve.
    brew_bin = "/opt/homebrew/bin"
    path = env.get("PATH", "")
    if brew_bin not in path.split(":"):
        env["PATH"] = f"{brew_bin}:{path}" if path else brew_bin

    _last_backup_unix = now
    try:
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo),
            env=env,
            check=True,
            timeout=7200,
            capture_output=True,
            text=True,
        )
        iso = datetime.now(timezone.utc).isoformat()
        _log.info('{"event":"heart.backup.ok","at":"%s"}', iso)
        return iso
    except Exception as exc:
        err = getattr(exc, "stderr", None) or str(exc)
        _log.warning('{"event":"heart.backup.fail","detail":"%s"}', err)
        return None


def _collect_backup_result() -> str | None:
    """Pick up a finished background backup's timestamp (None while running/idle)."""
    global _backup_thread, _backup_result_iso
    if _backup_thread is not None and _backup_thread.is_alive():
        return None
    _backup_thread = None
    iso = _backup_result_iso
    _backup_result_iso = None
    return iso


def _spawn_backup(prev_backup_iso: str | None) -> None:
    """Run ``_maybe_schedule_backup`` on a daemon thread (one at a time)."""
    global _backup_thread
    if _backup_thread is not None and _backup_thread.is_alive():
        return
    fn = _maybe_schedule_backup  # bind now (tests monkeypatch the module attr)

    def _worker() -> None:
        global _backup_result_iso
        try:
            iso = fn(prev_backup_iso)
        except Exception as exc:
            _log.warning('{"event":"heart.backup.fail","detail":"%s"}', exc)
            return
        if iso:
            _backup_result_iso = iso

    t = threading.Thread(target=_worker, name="phylactery-backup", daemon=True)
    _backup_thread = t
    t.start()


def tick(conn: psycopg.Connection) -> None:
    """Execute one heartbeat cycle and write vitals row."""
    prev = latest_tick(conn)
    prev_bm: dict = {}
    if prev:
        raw = prev.get("body_metrics")
        if isinstance(raw, dict):
            prev_bm = raw
        elif isinstance(raw, str):
            try:
                prev_bm = json.loads(raw)
            except json.JSONDecodeError:
                prev_bm = {}
    prev_backup_iso = prev_bm.get("last_backup_at")
    if not isinstance(prev_backup_iso, str):
        prev_backup_iso = None

    pid = os.getpid()
    body = collect_body_metrics()
    body.update(api_watch_body_patch())
    status, counts = _compute_phylactery_status()
    if counts:
        body["phylactery_counts"] = counts

    backup_iso = _collect_backup_result()
    if backup_iso is None:
        _spawn_backup(prev_backup_iso)
    if backup_iso:
        body["last_backup_at"] = backup_iso
    elif prev_backup_iso:
        body["last_backup_at"] = prev_backup_iso

    emit_critical_vital_alerts(body)

    signature = sign_tick_payload(body, status)

    tick_id = write_tick(
        conn,
        pid=pid,
        body_metrics=body,
        phylactery_status=status,
        signature=signature,
    )
    if tick_id is None:
        _log.warning('{"event":"heart.tick.skipped","reason":"db_write_failed"}')
    else:
        _log.info(
            '{"event":"heart.tick","tick_id":%s,"status":"%s","pid":%s}',
            tick_id,
            status,
            pid,
        )

    if status != "ok":
        _log.warning(
            '{"event":"heart.phylactery.alert","status":"%s","body":%s}',
            status,
            body,
        )

    try:
        send_outreach_if_due(conn)
    except Exception as exc:
        _log.warning('{"event":"outreach.error","detail":"%s"}', exc)

    try:
        mark_due_intents(conn)
    except Exception as exc:
        _log.warning('{"event":"pulse.error","detail":"%s"}', exc)

    try:
        from src.obs.log_rotate import rotate_agent_logs_if_needed

        rotate_agent_logs_if_needed()
    except Exception as exc:
        _log.warning('{"event":"log.rotate_error","detail":"%s"}', exc)


def run() -> None:
    """Heart main loop. Exits only on SIGTERM or unrecoverable startup failure."""
    global _HEART_START_UNIX

    from src.obs.tracing import init_telemetry

    init_telemetry("agent-heart", settings)
    _build_sys_logger()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _HEART_START_UNIX = time.time()

    interval = int(settings.heart_interval_sec)
    _log.info(
        '{"event":"heart.start","pid":%s,"interval_sec":%s}', os.getpid(), interval
    )
    try:
        from src.tools.web_core import check_searxng_health

        check_searxng_health()
    except Exception as exc:
        _log.warning('{"event":"searxng.health_probe_failed","detail":"%s"}', exc)

    # Attempt initial DB connection with retries.
    conn: psycopg.Connection | None = None
    while not _shutdown_requested:
        try:
            conn = psycopg.connect(_pg_conninfo(), autocommit=True)
            break
        except Exception as exc:
            _log.warning(
                '{"event":"heart.tick.skipped","reason":"db_down","detail":"%s"}', exc
            )
            _sleep_interruptible(interval)

    if conn is None or _shutdown_requested:
        _log.info('{"event":"heart.stop","reason":"shutdown_before_lock"}')
        return

    # Singleton enforcement via advisory lock.
    try:
        _require_advisory_lock(conn)
    except Exception as exc:
        _log.error('{"event":"heart.lock_error","detail":"%s"}', exc)
        conn.close()
        sys.exit(1)

    _log.info('{"event":"heart.lock_acquired","advisory_key":%s}', _ADVISORY_LOCK_KEY)
    notify_telegram(
        f"🟢 heart started @ {socket.gethostname()}",
        alert_key="startup",
        respect_cooldown=False,
    )

    last_tick_wall = time.time()

    while not _shutdown_requested:
        now_wall = time.time()
        gap = now_wall - last_tick_wall
        if gap > 3 * interval:
            _log.info('{"event":"heart.wake_catchup","gap_sec":%d}', int(gap))

        try:
            # Re-connect if connection was lost.
            if conn.closed:
                conn = psycopg.connect(_pg_conninfo(), autocommit=True)
                _require_advisory_lock(conn)
            tick(conn)
        except psycopg.OperationalError as exc:
            _log.warning(
                '{"event":"heart.tick.skipped","reason":"db_down","detail":"%s"}', exc
            )
            try:
                conn.close()
            except Exception:
                pass
            conn = _reconnect_loop(interval)
            if conn is None:
                break
        except Exception as exc:
            _log.error('{"event":"heart.tick.error","detail":"%s"}', exc)

        last_tick_wall = time.time()
        _sleep_interruptible(interval)

    _log.info('{"event":"heart.stop","pid":%s}', os.getpid())
    notify_telegram(
        "🟡 heart stopping (SIGTERM)",
        alert_key="shutdown",
        respect_cooldown=False,
    )
    try:
        conn.close()
    except Exception:
        pass


def _sleep_interruptible(seconds: float) -> None:
    """Sleep in small slices so SIGTERM is handled promptly."""
    deadline = time.monotonic() + seconds
    while not _shutdown_requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))


def _reconnect_loop(interval: float) -> psycopg.Connection | None:
    """Keep trying to reconnect until success or shutdown signal."""
    while not _shutdown_requested:
        _sleep_interruptible(interval)
        if _shutdown_requested:
            return None
        try:
            conn = psycopg.connect(_pg_conninfo(), autocommit=True)
            if not _acquire_advisory_lock(conn):
                _log.error(
                    '{"event":"heart.duplicate","msg":"Advisory lock held on reconnect; exiting."}'
                )
                conn.close()
                sys.exit(1)
            _log.info('{"event":"heart.db_reconnected"}')
            return conn
        except Exception as exc:
            _log.warning(
                '{"event":"heart.tick.skipped","reason":"db_down","detail":"%s"}', exc
            )
    return None


if __name__ == "__main__":  # pragma: no cover
    run()
