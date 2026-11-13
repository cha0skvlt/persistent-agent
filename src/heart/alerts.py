"""Critical vitals → Telegram (rate-limited per alert key)."""

from __future__ import annotations

import time

from src.config import Settings, get_settings
from src.heart.notify import notify_telegram

_ollama_down_since: float | None = None


def reset_vital_alert_state() -> None:
    """Reset cross-tick trackers (pytest)."""
    global _ollama_down_since
    _ollama_down_since = None


def ollama_down_seconds(body_metrics: dict) -> float | None:
    """Seconds Ollama has been down continuously; ``None`` when up."""
    global _ollama_down_since
    if body_metrics.get("ollama_running"):
        _ollama_down_since = None
        return None
    now = time.monotonic()
    if _ollama_down_since is None:
        _ollama_down_since = now
    return now - _ollama_down_since


def critical_vital_alerts(
    body_metrics: dict,
    *,
    settings: Settings | None = None,
) -> list[tuple[str, str]]:
    """Return (alert_key, telegram_message) for active critical conditions."""
    s = settings or get_settings()
    out: list[tuple[str, str]] = []

    disk = body_metrics.get("disk_usage_pct")
    if disk is not None:
        try:
            if float(disk) >= float(s.heart_disk_alert_pct):
                out.append(
                    (
                        "emergency_disk",
                        f"🔴 emergency_disk disk_usage_pct={disk}",
                    )
                )
        except (TypeError, ValueError):
            pass

    bat = body_metrics.get("battery_pct")
    if bat is not None:
        try:
            if float(bat) < float(s.heart_battery_alert_pct):
                out.append(
                    (
                        "emergency_battery",
                        f"🔴 emergency_battery battery_pct={bat}",
                    )
                )
        except (TypeError, ValueError):
            pass

    down_sec = ollama_down_seconds(body_metrics)
    if down_sec is not None and down_sec >= float(s.heart_ollama_down_alert_sec):
        out.append(
            (
                "ollama_down",
                f"🔴 ollama_down for {int(down_sec)}s",
            )
        )

    return out


def emit_critical_vital_alerts(
    body_metrics: dict,
    *,
    settings: Settings | None = None,
) -> None:
    """Notify Telegram for each active critical vital (rate-limited per key)."""
    for key, message in critical_vital_alerts(body_metrics, settings=settings):
        notify_telegram(message, settings=settings, alert_key=key)


__all__ = [
    "critical_vital_alerts",
    "emit_critical_vital_alerts",
    "ollama_down_seconds",
    "reset_vital_alert_state",
]
