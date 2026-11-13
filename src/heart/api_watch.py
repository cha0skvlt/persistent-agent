"""API /health watchdog for heart."""

from __future__ import annotations

import logging
import subprocess
import time

import httpx

from src.config import Settings, get_settings
from src.heart.notify import notify_telegram

_log = logging.getLogger(__name__)

_fail_streak: int = 0
_emergency_active: bool = False
_last_restart_at: float = 0.0


def reset_api_watch_state() -> None:
    """Reset fail counter (pytest)."""
    global _fail_streak, _emergency_active, _last_restart_at
    _fail_streak = 0
    _emergency_active = False
    _last_restart_at = 0.0


def check_api_health(settings: Settings | None = None) -> tuple[bool, str]:
    """GET health URL; return (ok, detail)."""
    s = settings or get_settings()
    url = (s.heart_api_health_url or "").strip()
    if not url:
        return False, "empty_url"
    try:
        resp = httpx.get(url, timeout=5.0)
        if 200 <= resp.status_code < 300:
            return True, f"status={resp.status_code}"
        return False, f"status={resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def maybe_restart_api(settings: Settings | None = None) -> None:
    global _last_restart_at
    s = settings or get_settings()
    if not s.heart_api_auto_restart:
        return
    if time.time() - _last_restart_at < 60:
        _log.info(
            '{"event":"heart.api.restart_cooldown","seconds_since_last":%d}',
            int(time.time() - _last_restart_at),
        )
        return
    label = (s.heart_api_launchd_label or "").strip()
    if not label:
        _log.warning(
            '{"event":"heart.api.restart_skipped","reason":"no_launchd_label"}'
        )
        return
    try:
        subprocess.run(
            ["launchctl", "kickstart", "-k", label],
            check=True,
            timeout=30,
            capture_output=True,
            text=True,
        )
        _last_restart_at = time.time()
        _log.info('{"event":"heart.api.restart","label":"%s"}', label)
    except Exception as exc:
        err = getattr(exc, "stderr", None) or str(exc)
        _log.warning('{"event":"heart.api.restart_failed","detail":"%s"}', err)


def api_watch_body_patch(settings: Settings | None = None) -> dict:
    """Run one health probe; return keys to merge into ``body_metrics``."""
    global _fail_streak, _emergency_active
    s = settings or get_settings()
    if not s.heart_check_api_enabled:
        return {}

    ok, detail = check_api_health(s)
    if ok:
        _log.info('{"event":"api.health.ok","detail":"%s"}', detail)
        _fail_streak = 0
        _emergency_active = False
        return {"api_health_ok": True}

    _fail_streak += 1
    threshold = max(1, int(s.heart_api_fail_threshold))
    _log.warning(
        '{"event":"api.health.fail","streak":%s,"threshold":%s,"detail":"%s"}',
        _fail_streak,
        threshold,
        detail,
    )
    patch: dict = {
        "api_health_ok": False,
        "api_health_fail_streak": _fail_streak,
    }
    if _fail_streak < threshold:
        return patch

    patch["emergency_api_dead"] = True
    if not _emergency_active:
        _emergency_active = True
        maybe_restart_api(s)
        notify_telegram(
            f"🔴 emergency_api_dead ({detail}, streak={_fail_streak})",
            settings=s,
            alert_key="emergency_api_dead",
        )
    return patch


__all__ = [
    "api_watch_body_patch",
    "check_api_health",
    "maybe_restart_api",
    "reset_api_watch_state",
]
