"""Async Telegram Bot API client (httpx, no extra framework)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from src.api.telegram_format import (
    TELEGRAM_MAX_MESSAGE_LEN,
    should_fallback_plain,
    split_telegram_plain_text,
    telegram_message_chunks,
)

_log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"
_POLL_READ_BUFFER_SEC = 20.0

# Back-compat alias for tests.
_split_telegram_text = split_telegram_plain_text
_MAX_MESSAGE_LEN = TELEGRAM_MAX_MESSAGE_LEN


class TelegramClient:
    def __init__(self, token: str, *, timeout: float = 120.0) -> None:
        self._token = (token or "").strip()
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None
        # Filled from getMe at polling start; used to match /cmd@bot mentions.
        self.bot_username: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    def _url(self, method: str) -> str:
        return f"{_TELEGRAM_API}/bot{self._token}/{method}"

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def get_me(self) -> dict[str, Any]:
        http = await self._get_http()
        resp = await http.get(self._url("getMe"))
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getMe failed: {data!r}")
        return data["result"]

    async def get_updates(
        self,
        *,
        offset: int = 0,
        timeout: int = 30,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"offset": offset, "timeout": timeout}
        if allowed_updates:
            # Bot API expects a JSON-serialized array, not a bare form list.
            params["allowed_updates"] = json.dumps(list(allowed_updates))
        http = await self._get_http()
        poll_sec = max(0, int(timeout))
        read_timeout = max(self._timeout, float(poll_sec) + _POLL_READ_BUFFER_SEC)
        request_timeout = httpx.Timeout(
            connect=10.0,
            read=read_timeout,
            write=10.0,
            pool=10.0,
        )
        resp = await http.get(
            self._url("getUpdates"),
            params=params,
            timeout=request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {data!r}")
        result = data.get("result")
        return result if isinstance(result, list) else []

    async def set_my_commands(
        self,
        commands: list[dict[str, str]],
        *,
        scope: dict[str, Any] | None = None,
    ) -> bool:
        """Register the bot command menu (``setMyCommands``)."""
        payload: dict[str, Any] = {"commands": list(commands or [])}
        if scope:
            payload["scope"] = scope
        data = await self._post("setMyCommands", payload)
        return bool(data.get("ok"))

    async def send_chat_action(
        self, chat_id: int | str, action: str = "typing"
    ) -> None:
        await self._post(
            "sendChatAction",
            {"chat_id": chat_id, "action": action},
        )

    async def send_message(self, chat_id: int | str, text: str) -> None:
        body = (text or "").strip() or "(empty)"
        for html_text, plain_fallback in telegram_message_chunks(body):
            await self._send_chunk(chat_id, html_text, plain_fallback)

    async def send_message_with_footer(
        self,
        chat_id: int | str,
        body: str,
        footer: str,
    ) -> None:
        """Send reply as one or more ``sendMessage`` calls (4096 parsed chars each)."""
        text = (body or "").strip() or "(empty)"
        if not (footer or "").strip():
            await self.send_message(chat_id, text)
            return
        for html_text, plain_fallback in telegram_message_chunks(text, footer):
            await self._send_chunk(chat_id, html_text, plain_fallback)

    async def _send_chunk(
        self,
        chat_id: int | str,
        html_text: str,
        plain_fallback: str,
    ) -> None:
        """One Bot API ``sendMessage`` — shared Delivery path."""
        from src.api.telegram_send import send_chunk_async

        await send_chunk_async(self._post, chat_id, html_text, plain_fallback)

    async def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self._post_once(method, payload)
        retry_after = _flood_retry_after(data)
        if retry_after is not None:
            # Bot API flood control: wait parameters.retry_after, retry once.
            _log.warning(
                "Telegram 429 on %s — waiting %.1fs (flood control)",
                method,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            data = await self._post_once(method, payload)
        return data

    async def _post_once(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        http = await self._get_http()
        resp = await http.post(self._url(method), json=payload)
        try:
            data = resp.json()
        except ValueError:
            data = {"ok": False, "description": (resp.text or "")[:500]}
        if not isinstance(data, dict):
            data = {"ok": False, "description": str(data)}
        if resp.status_code >= 400:
            _log.warning(
                "Telegram HTTP %s on %s: %s",
                resp.status_code,
                method,
                data.get("description") or resp.text[:200],
            )
            data["ok"] = False
            data.setdefault("error_code", resp.status_code)
            return data
        if not data.get("ok"):
            _log.warning("Telegram %s failed: %s", method, data)
        return data


_FLOOD_RETRY_CAP_SEC = 30.0


def _flood_retry_after(data: dict[str, Any]) -> float | None:
    """Seconds to wait from a 429 response's ``parameters.retry_after``, capped."""
    if data.get("ok") or int(data.get("error_code") or 0) != 429:
        return None
    params = data.get("parameters")
    raw = params.get("retry_after") if isinstance(params, dict) else None
    try:
        sec = float(raw)
    except (TypeError, ValueError):
        return None
    if sec <= 0:
        return None
    return min(sec, _FLOOD_RETRY_CAP_SEC)


# Back-compat alias — shared predicate now lives in telegram_format.
_should_fallback_plain = should_fallback_plain
