"""System-level metrics with zero LLM/embedding dependencies.

All functions are pure OS/network checks. Called by heartbeat.py every tick.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import urllib.request
from urllib.error import URLError

import psycopg

from src.config import settings


def disk_usage_pct(path: str = "/") -> float:
    """Fraction 0–100 of used disk on *path*."""
    usage = shutil.disk_usage(path)
    if usage.total == 0:
        return 0.0
    return round(usage.used / usage.total * 100, 2)


def battery_pct() -> int | None:
    """macOS battery percentage via ``pmset -g batt``. Returns None on non-Mac or error."""
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=3)
        for line in out.splitlines():
            line = line.strip()
            if "%" in line:
                # e.g. "InternalBattery-0 (id=...)  72%; discharging; ..."
                pct_str = line.split("%")[0].rsplit(None, 1)[-1]
                return int(pct_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def ollama_running(base_url: str | None = None) -> bool:
    """True if Ollama HTTP API responds to GET /api/tags within 2 seconds."""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def network_up(host: str = "ollama.com") -> bool:
    """True if *host* resolves via DNS (basic network-up check)."""
    try:
        socket.gethostbyname(host)
        return True
    except OSError:
        return False


def _pg_conninfo() -> str:
    """Return plain psycopg conninfo string from settings (strip SQLAlchemy prefix)."""
    url = settings.database_url
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


def pg_alive() -> bool:
    """True if Postgres responds to SELECT 1 within 3 seconds."""
    try:
        with psycopg.connect(_pg_conninfo(), connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def phylactery_counts() -> dict[str, int]:
    """Row counts for the core phylactery tables.

    Returns a dict with keys: memories, entities, knowledge_graph, self_reflections.
    Raises on DB error — callers should catch and treat as degraded/fail.
    """
    tables = ("memories", "entities", "knowledge_graph", "self_reflections")
    counts: dict[str, int] = {}
    with psycopg.connect(_pg_conninfo(), connect_timeout=3) as conn:
        for tbl in tables:
            row = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()  # noqa: S608
            counts[tbl] = int(row[0]) if row else 0
    return counts


def collect_body_metrics() -> dict:
    """Assemble one JSONB body_metrics snapshot for a vitals tick."""
    metrics: dict = {
        "disk_usage_pct": disk_usage_pct(),
        "network_up": network_up(),
        "ollama_running": ollama_running(),
    }
    bat = battery_pct()
    if bat is not None:
        metrics["battery_pct"] = bat
    return metrics
