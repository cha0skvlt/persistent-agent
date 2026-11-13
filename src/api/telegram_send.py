"""Shared Telegram send path — HTML chunk + plain fallback.

Live (async ``TelegramClient``) and Heart outreach (sync) both use
``send_chunk_*`` so proactive messages render like live replies.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import httpx

from src.api.telegram_format import (
    plain_fallback_text,
    should_fallback_plain,
    simplify_thinking_html,
    split_telegram_plain_text,
    telegram_message_chunks,
)

_log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


_FLOOD_RETRY_CAP_SEC = 30.0


def post_send_message_sync(
    token: str, payload: dict[str, Any], *, timeout: float = 10.0
) -> tuple[bool, str]:
    """POST one sendMessage (sync). Respects 429 retry_after once. Returns ``(ok, error)``."""
    ok, desc, retry_after = _post_send_once(token, payload, timeout=timeout)
    if retry_after is not None:
        _log.warning(
            '{"event":"telegram.send.flood_wait","retry_after":%.1f}', retry_after
        )
        time.sleep(retry_after)
        ok, desc, _ = _post_send_once(token, payload, timeout=timeout)
    return ok, desc


def _post_send_once(
    token: str, payload: dict[str, Any], *, timeout: float
) -> tuple[bool, str, float | None]:
    resp = httpx.post(
        f"{_TELEGRAM_API}/bot{token}/sendMessage",
        json=payload,
        timeout=timeout,
    )
    try:
        data = resp.json()
    except ValueError:
        data = {"ok": False, "description": (getattr(resp, "text", "") or "")[:500]}
    if not isinstance(data, dict):
        data = {"ok": False, "description": str(data)}
    if resp.status_code >= 400:
        desc = str(data.get("description") or getattr(resp, "text", "")[:200])
        retry_after: float | None = None
        if resp.status_code == 429:
            params = data.get("parameters")
            raw = params.get("retry_after") if isinstance(params, dict) else None
            try:
                sec = float(raw)
                retry_after = min(sec, _FLOOD_RETRY_CAP_SEC) if sec > 0 else None
            except (TypeError, ValueError):
                retry_after = None
        return False, desc, retry_after
    return bool(data.get("ok")), str(data.get("description") or data), None


def send_chunk_sync(
    token: str,
    chat_id: str | int,
    html_text: str,
    plain_fallback: str,
    *,
    log_prefix: str = "telegram",
) -> bool:
    """One sendMessage — HTML first, simplify thinking HTML, then plain fallback."""
    html = (html_text or "").strip()
    plain = plain_fallback or ""
    if not html and not plain.strip():
        return True

    err = "empty html"
    if html:
        ok, err = post_send_message_sync(
            token, {"chat_id": chat_id, "text": html, "parse_mode": "HTML"}
        )
        if ok:
            return True
        # Nested <i> inside expandable blockquote sometimes trips parse_mode;
        # retry once without italic before abandoning the green window.
        simplified = simplify_thinking_html(html)
        if simplified and should_fallback_plain(err):
            ok, err2 = post_send_message_sync(
                token,
                {"chat_id": chat_id, "text": simplified, "parse_mode": "HTML"},
            )
            if ok:
                return True
            err = err2 or err

    if should_fallback_plain(err):
        _log.warning(
            '{"event":"%s.send.plain_fallback","detail":"%s"}',
            log_prefix,
            err[:120],
        )
        cleaned = plain_fallback_text(plain)
        sent_any = False
        for part in split_telegram_plain_text(cleaned):
            ok, _ = post_send_message_sync(token, {"chat_id": chat_id, "text": part})
            if ok:
                sent_any = True
            elif sent_any:
                # Partial delivery — do not report total failure (avoids redeliver).
                _log.warning(
                    '{"event":"%s.send.partial","detail":"plain chunk failed after success"}',
                    log_prefix,
                )
                return True
            else:
                return False
        return sent_any
    _log.warning('{"event":"%s.send.failed","detail":"%s"}', log_prefix, err[:200])
    return False


def send_body_sync(
    token: str,
    chat_id: str | int,
    body: str,
    footer: str = "",
    *,
    outreach: bool = False,
    log_prefix: str = "telegram",
) -> bool:
    """Chunk *body* (+ optional footer) and send each piece via ``send_chunk_sync``.

    Returns False when chunking yields nothing (e.g. outreach sanitize emptied
    the body) — never report success for a silent no-op send.
    If some chunks already reached Telegram, returns True on later failure so
    callers do not re-pend/redeliver the same outbox row.
    """
    try:
        chunks = telegram_message_chunks(body, footer, outreach=outreach)
        if not chunks:
            if (body or "").strip() or (footer or "").strip():
                _log.warning(
                    '{"event":"%s.send.skipped","reason":"empty_after_chunk"}',
                    log_prefix,
                )
            return False
        sent_any = False
        for html_text, plain_fallback in chunks:
            if send_chunk_sync(
                token, chat_id, html_text, plain_fallback, log_prefix=log_prefix
            ):
                sent_any = True
            elif sent_any:
                _log.warning(
                    '{"event":"%s.send.partial","detail":"chunk failed after success"}',
                    log_prefix,
                )
                return True
            else:
                return False
        return sent_any
    except Exception as exc:
        _log.warning('{"event":"%s.send.failed","detail":"%s"}', log_prefix, exc)
        return False


async def send_chunk_async(
    post: Callable[..., Any],
    chat_id: str | int,
    html_text: str,
    plain_fallback: str,
) -> None:
    """Async twin of ``send_chunk_sync`` — *post* is ``TelegramClient._post``."""
    html = (html_text or "").strip()
    plain = plain_fallback or ""
    if not html and not plain.strip():
        return

    err = "empty html"
    if html:
        data = await post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": html,
                "parse_mode": "HTML",
            },
        )
        if data.get("ok"):
            return
        err = str(data.get("description") or data)
        simplified = simplify_thinking_html(html)
        if simplified and should_fallback_plain(err):
            data = await post(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": simplified,
                    "parse_mode": "HTML",
                },
            )
            if data.get("ok"):
                return
            err = str(data.get("description") or data)

    if should_fallback_plain(err):
        _log.warning("Telegram HTML send failed (%s); plain fallback", err[:120])
        cleaned = plain_fallback_text(plain)
        for part in split_telegram_plain_text(cleaned):
            data = await post(
                "sendMessage",
                {"chat_id": chat_id, "text": part},
            )
            if not data.get("ok"):
                detail = str(data.get("description") or data)
                _log.warning("Telegram plain fallback failed: %s", detail)
                raise RuntimeError(f"Telegram send failed: {detail[:200]}")
        return
    _log.warning("Telegram sendMessage failed: %s", err)
    raise RuntimeError(f"Telegram send failed: {err[:200]}")


__all__ = [
    "post_send_message_sync",
    "send_body_sync",
    "send_chunk_async",
    "send_chunk_sync",
]
