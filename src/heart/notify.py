"""Best-effort Telegram alerts from the heart daemon."""

from __future__ import annotations

import logging
import time

import httpx

from src.config import Settings, get_settings

_log = logging.getLogger(__name__)

_cooldown_at: dict[str, float] = {}


def reset_notify_cooldowns() -> None:
    """Clear rate-limit state (pytest)."""
    _cooldown_at.clear()


def _cooldown_allows(key: str, cooldown_sec: float) -> bool:
    now = time.monotonic()
    last = _cooldown_at.get(key)
    if last is not None and (now - last) < cooldown_sec:
        return False
    _cooldown_at[key] = now
    return True


def notify_telegram(
    message: str,
    settings: Settings | None = None,
    *,
    alert_key: str = "default",
    respect_cooldown: bool = True,
) -> bool:
    """Send a Telegram message. Never raises; returns True if HTTP post was attempted."""
    s = settings or get_settings()
    token = (s.telegram_bot_token or "").strip()
    chat_id = (s.heart_notify_chat_id or "").strip()
    if not token or not chat_id:
        _log.info(
            '{"event":"heart.notify.skipped","reason":"no_token_or_chat","key":"%s"}',
            alert_key,
        )
        return False
    cooldown = float(s.heart_notify_cooldown_sec)
    if respect_cooldown and cooldown > 0 and not _cooldown_allows(alert_key, cooldown):
        _log.info(
            '{"event":"heart.notify.skipped","reason":"cooldown","key":"%s"}',
            alert_key,
        )
        return False
    _log.info('{"event":"heart.notify.send","key":"%s"}', alert_key)
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=5.0,
        )
        return True
    except Exception as exc:
        _log.warning(
            '{"event":"heart.notify_failed","key":"%s","detail":"%s"}', alert_key, exc
        )
        return False


__all__ = ["notify_telegram", "reset_notify_cooldowns"]
