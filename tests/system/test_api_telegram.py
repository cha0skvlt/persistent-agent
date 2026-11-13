"""Tests for Telegram bot helpers and client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent_runner import AgentTurnResult
from src.api import telegram_bot as tb
from src.api.telegram_client import TelegramClient, _split_telegram_text
from src.api.telegram_format import (
    TELEGRAM_MAX_MESSAGE_LEN,
    telegram_chunk_plain_coverage,
    telegram_message_chunks,
)
from src.config import settings


def _turn(
    reply: str = "reply",
    *,
    tools: tuple[str, ...] = (),
    ms: float = 100.0,
    summary_pending: bool = False,
):
    return AgentTurnResult(
        reply=reply,
        tools_used=tools,
        elapsed_ms=ms,
        node_timings={"agent_node": ms},
        thread_id="telegram_1",
        summary_pending=summary_pending,
    )


@pytest.mark.unit
def test_parse_allowed_chat_ids():
    assert tb.parse_allowed_chat_ids("") == set()
    assert tb.parse_allowed_chat_ids("1, 2, bad, 3") == {1, 2, 3}


@pytest.mark.unit
def test_chat_allowed_fail_closed_when_empty(monkeypatch):
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)
    assert tb.effective_allowed_chat_ids() == frozenset()
    assert tb.chat_allowed(99) is False
    assert tb.telegram_allowlist_configured() is False
    assert tb.telegram_polling_may_start(token_enabled=True) is False


@pytest.mark.unit
def test_handle_telegram_update_denied_empty_allowlist_logs(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    import logging

    caplog.set_level(logging.WARNING)
    client = MagicMock()
    client.send_message = AsyncMock()
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 131126905}, "text": "hello"}},
        )

    asyncio.run(_run())
    client.send_message.assert_not_called()
    assert any("telegram.chat_denied" in r.message for r in caplog.records)


@pytest.mark.unit
def test_assert_telegram_boot_allowlist_fails_without_owner_or_allowlist(
    monkeypatch,
):
    monkeypatch.setattr(settings, "telegram_bot_token", "test-token", raising=False)
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_ALLOWED_CHAT_IDS"):
        tb.assert_telegram_boot_allowlist()


@pytest.mark.unit
def test_assert_telegram_boot_allowlist_ok_with_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "test-token", raising=False)
    monkeypatch.setattr(
        settings, "telegram_allowed_chat_ids", [131126905], raising=False
    )
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)
    tb.assert_telegram_boot_allowlist()


@pytest.mark.unit
def test_chat_allowed_owner_fallback_when_allowlist_empty(monkeypatch):
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "42", raising=False)
    assert tb.chat_allowed(42) is True
    assert tb.chat_allowed(99) is False
    assert tb.telegram_polling_may_start(token_enabled=True) is True


@pytest.mark.unit
def test_chat_allowed(monkeypatch):
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", "42", raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)
    assert tb.chat_allowed(42) is True
    assert tb.chat_allowed(99) is False


@pytest.mark.unit
def test_extract_incoming_text():
    upd = {"message": {"chat": {"id": 7}, "text": " hello "}}
    assert tb.extract_incoming_text(upd) == (7, "hello")
    assert tb.extract_incoming_text({}) is None


@pytest.mark.unit
def test_telegram_message_chunks_long_thinking():
    think = "Q: plan.\n→ step " * 30
    body = f"<think>{think.strip()}</think>\n\nShort answer."
    footer = "**→ turn 9.0s**  run_selftest"
    chunks = telegram_message_chunks(body, footer)
    # Thinking bubble + reply bubble (never packed).
    assert len(chunks) == 2
    assert all(len(plain) <= 4096 for _html, plain in chunks)
    think_html, think_plain = chunks[0]
    main_html, main_plain = chunks[1]
    assert "<blockquote expandable>" in think_html
    assert "Short answer." not in think_html
    assert "Short answer." in main_html
    assert "Short answer." in main_plain
    rec_think, rec_main, rec_foot = telegram_chunk_plain_coverage(body, footer)
    assert rec_think == think.strip()
    assert rec_main == "Short answer."
    assert "run_selftest" in rec_foot


@pytest.mark.unit
def test_split_telegram_text():
    short = "hi"
    assert _split_telegram_text(short) == ["hi"]
    long = "a" * 5000
    parts = _split_telegram_text(long)
    assert len(parts) >= 2
    assert "".join(parts) == long
    assert all(len(p) <= TELEGRAM_MAX_MESSAGE_LEN for p in parts)


@pytest.mark.unit
def test_send_message_with_footer_empty():
    client = TelegramClient("token")
    client.send_message = AsyncMock()

    async def _run():
        await client.send_message_with_footer(1, "hello", "  ")

    asyncio.run(_run())
    client.send_message.assert_awaited_once_with(1, "hello")


@pytest.mark.unit
def test_send_message_with_footer_short():
    client = TelegramClient("token")

    async def _post(method, payload):
        client._last_payload = payload
        return {"ok": True}

    client._post = AsyncMock(side_effect=_post)

    async def _run():
        await client.send_message_with_footer(1, "hello", "**→ turn 1.0s**")

    asyncio.run(_run())
    payload = client._post.await_args.args[1]
    assert payload["parse_mode"] == "HTML"
    assert "<b>→ turn 1.0s</b>" in payload["text"]


@pytest.mark.unit
def test_send_message_html_fallback(monkeypatch):
    client = TelegramClient("token")
    calls: list[dict] = []

    async def _post(method, payload):
        calls.append(dict(payload))
        if len(calls) == 1:
            return {"ok": False, "description": "can't parse entities: bad html"}
        return {"ok": True}

    client._post = AsyncMock(side_effect=_post)

    async def _run():
        await client.send_message_with_footer(1, "**bold**", "→ turn 1.0s")

    asyncio.run(_run())
    assert len(calls) == 2
    assert calls[0]["parse_mode"] == "HTML"
    assert "parse_mode" not in calls[1]
    assert "**" not in calls[1]["text"]
    assert "bold" in calls[1]["text"]


@pytest.mark.unit
def test_send_message_plain_fallback_stops_on_failure():
    """Plain fallback must raise on ok:false — do not silently drop the reply."""
    client = TelegramClient("token")
    calls: list[dict] = []

    async def _post(method, payload):
        calls.append(dict(payload))
        return {"ok": False, "description": "can't parse entities: bad html"}

    client._post = AsyncMock(side_effect=_post)

    async def _run():
        with pytest.raises(RuntimeError, match="Telegram send failed"):
            await client.send_message(1, "**bold**")

    asyncio.run(_run())
    # HTML attempt + one plain attempt that failed — then raise.
    assert len(calls) == 2
    assert calls[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in calls[1]


@pytest.mark.unit
def test_send_message_with_footer_long():
    client = TelegramClient("token")
    sent: list[str] = []

    async def _post(method, payload):
        sent.append(payload["text"])
        return {"ok": True}

    client._post = AsyncMock(side_effect=_post)
    body = "line\n" * 3000
    footer = "**→ turn 9.0s**  search_web"

    async def _run():
        await client.send_message_with_footer(1, body, footer)

    asyncio.run(_run())
    chunks = telegram_message_chunks(body.strip(), footer)
    assert len(sent) == len(chunks)
    assert len(sent) >= 2
    assert "search_web" in sent[-1]
    _think, rec_main, rec_foot = telegram_chunk_plain_coverage(body.strip(), footer)
    assert rec_main == body.strip()
    assert "search_web" in rec_foot


@pytest.mark.unit
def test_handle_telegram_update_denies_unknown_chat(monkeypatch):
    client = MagicMock()
    client.send_message = AsyncMock()
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [42], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 99}, "text": "hello"}},
        )

    asyncio.run(_run())
    client.send_message.assert_not_called()


@pytest.mark.unit
def test_handle_telegram_update_start(monkeypatch):
    client = MagicMock()
    client.send_message = AsyncMock()
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [1], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 1}, "text": "/start"}},
        )

    asyncio.run(_run())
    client.send_message.assert_awaited_once()


@pytest.mark.unit
def test_handle_telegram_update_agent(monkeypatch):
    client = MagicMock()
    client.send_chat_action = AsyncMock()
    client.send_message_with_footer = AsyncMock()
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [5], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)
    monkeypatch.setattr(tb, "arun_agent_turn", AsyncMock(return_value=_turn("reply")))

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 5}, "text": "hello"}},
        )

    asyncio.run(_run())
    assert client.send_chat_action.await_count >= 1
    client.send_message_with_footer.assert_awaited_once()
    args = client.send_message_with_footer.await_args.args
    assert args[0] == 5
    assert args[1] == "reply"
    assert "→ turn" in args[2]


@pytest.mark.unit
def test_run_deferred_summary_skips_when_none(monkeypatch):
    client = MagicMock()
    client.send_message = AsyncMock()
    monkeypatch.setattr(
        "src.graph.session_summary_runner.run_session_summary_if_due",
        lambda **k: None,
    )

    async def _run():
        await tb._run_deferred_summary(client, 1, thread_id="telegram_1")

    asyncio.run(_run())
    client.send_message.assert_not_called()


@pytest.mark.unit
def test_run_deferred_summary_swallows_errors(monkeypatch):
    client = MagicMock()
    client.send_message = AsyncMock()

    async def _boom(**k):
        raise RuntimeError("fail")

    monkeypatch.setattr(tb.asyncio, "to_thread", _boom)

    async def _run():
        await tb._run_deferred_summary(client, 1, thread_id="telegram_1")

    asyncio.run(_run())
    client.send_message.assert_not_called()


@pytest.mark.unit
def test_run_deferred_summary_sends_timing(monkeypatch):
    client = MagicMock()
    client.send_message = AsyncMock()
    monkeypatch.setattr(
        "src.graph.session_summary_runner.run_session_summary_if_due",
        lambda **k: 65_000.0,
    )

    async def _run():
        await tb._run_deferred_summary(client, 1, thread_id="telegram_1")

    asyncio.run(_run())
    client.send_message.assert_awaited_once()
    assert "summarize_session" in client.send_message.await_args.args[1]


@pytest.mark.unit
def test_handle_telegram_update_schedules_summary(monkeypatch):
    client = MagicMock()
    client.send_chat_action = AsyncMock()
    client.send_message_with_footer = AsyncMock()
    client.send_message = AsyncMock()
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", [5], raising=False)
    monkeypatch.setattr(settings, "telegram_owner_chat_id", "", raising=False)
    monkeypatch.setattr(
        tb,
        "arun_agent_turn",
        AsyncMock(return_value=_turn("reply", summary_pending=True)),
    )
    monkeypatch.setattr(
        "src.graph.session_summary_runner.run_session_summary_if_due",
        lambda **k: 2_000.0,
    )
    created: list = []
    real_create_task = asyncio.create_task

    def _track(coro, **kwargs):
        task = real_create_task(coro, **kwargs)
        created.append(task)
        return task

    monkeypatch.setattr(tb.asyncio, "create_task", _track)

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 5}, "text": "hello"}},
        )
        for task in created:
            await task

    asyncio.run(_run())
    assert created
    client.send_message.assert_awaited()


@pytest.mark.unit
def test_telegram_client_get_me(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "result": {"username": "bot"}}

    class _Client:
        async def get(self, url, params=None, timeout=None):
            return _Resp()

        async def aclose(self):
            return None

    monkeypatch.setattr(
        "src.api.telegram_client.httpx.AsyncClient", lambda **k: _Client()
    )
    out = asyncio.run(TelegramClient("token").get_me())
    assert out["username"] == "bot"


@pytest.mark.unit
def test_typing_keepalive_refreshes(monkeypatch):
    client = MagicMock()
    client.send_chat_action = AsyncMock()
    client.send_message_with_footer = AsyncMock()
    monkeypatch.setattr(tb, "_typing_refresh_seconds", lambda: 0.05)

    async def slow_turn(*_a, **_k):
        await asyncio.sleep(0.15)
        return _turn("done")

    monkeypatch.setattr(tb, "arun_agent_turn", slow_turn)

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 1}, "text": "hello"}},
        )

    asyncio.run(_run())
    assert client.send_chat_action.await_count >= 2


@pytest.mark.unit
def test_extract_incoming_text_non_str():
    upd = {"message": {"chat": {"id": 7}, "text": 123}}
    assert tb.extract_incoming_text(upd) == (7, "123")


@pytest.mark.unit
def test_typing_keepalive_direct(monkeypatch):
    client = MagicMock()
    client.send_chat_action = AsyncMock(side_effect=[None, RuntimeError("x"), None])
    monkeypatch.setattr(tb, "_typing_refresh_seconds", lambda: 0.02)

    async def _run():
        async with tb.typing_keepalive(client, 1):
            await asyncio.sleep(0.05)

    asyncio.run(_run())
    assert client.send_chat_action.await_count >= 2


@pytest.mark.unit
def test_typing_keepalive_stops_cleanly(monkeypatch):
    client = MagicMock()
    client.send_chat_action = AsyncMock()
    monkeypatch.setattr(tb, "_typing_refresh_seconds", lambda: 10.0)

    async def _run():
        async with tb.typing_keepalive(client, 1):
            return

    asyncio.run(_run())
    assert client.send_chat_action.await_count >= 1


@pytest.mark.unit
def test_safe_handle_update_logs_exception(monkeypatch):
    client = MagicMock()

    async def boom(*_a, **_k):
        raise RuntimeError("fail")

    monkeypatch.setattr(tb, "handle_telegram_update", boom)

    async def _run():
        await tb._safe_handle_update(client, {"update_id": 1})

    asyncio.run(_run())


@pytest.mark.unit
def test_typing_refresh_seconds_fallback(monkeypatch):
    monkeypatch.setattr(settings, "telegram_typing_interval_sec", "bad", raising=False)
    assert tb._typing_refresh_seconds() == 4.0


@pytest.mark.unit
def test_typing_keepalive_send_failure(monkeypatch):
    client = MagicMock()
    client.send_chat_action = AsyncMock(side_effect=[None, RuntimeError("net")])
    client.send_message_with_footer = AsyncMock()
    monkeypatch.setattr(settings, "telegram_typing_interval_sec", 0.02, raising=False)

    async def slow_turn(*_a, **_k):
        await asyncio.sleep(0.05)
        return _turn("ok")

    monkeypatch.setattr(tb, "arun_agent_turn", slow_turn)

    async def _run():
        await tb.handle_telegram_update(
            client,
            {"message": {"chat": {"id": 1}, "text": "hello"}},
        )

    asyncio.run(_run())
    client.send_message_with_footer.assert_awaited_once()


@pytest.mark.unit
def test_telegram_polling_loop_disabled():
    client = TelegramClient("")
    stop = asyncio.Event()
    stop.set()
    asyncio.run(tb.telegram_polling_loop(client, stop_event=stop))


@pytest.mark.unit
def test_telegram_polling_loop_retries_on_read_timeout():
    import httpx

    client = MagicMock()
    client.enabled = True
    client.get_me = AsyncMock(return_value={"username": "g"})
    stop = asyncio.Event()
    calls = {"n": 0}

    async def get_updates(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("long-poll")
        stop.set()
        return []

    client.get_updates = AsyncMock(side_effect=get_updates)

    asyncio.run(tb.telegram_polling_loop(client, stop_event=stop))
    assert calls["n"] >= 2


@pytest.mark.unit
def test_telegram_polling_loop_backoff_on_transport_error(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch

    import httpx

    client = MagicMock()
    client.enabled = True
    client.get_me = AsyncMock(return_value={"username": "g"})
    stop = asyncio.Event()
    calls = {"n": 0}
    slept = {"n": 0}

    async def get_updates(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("network down")
        stop.set()
        return []

    async def fake_sleep(_sec: float) -> None:
        slept["n"] += 1

    client.get_updates = AsyncMock(side_effect=get_updates)

    async def _run():
        with patch("src.api.telegram_bot.asyncio.sleep", new=fake_sleep):
            await tb.telegram_polling_loop(client, stop_event=stop)

    asyncio.run(_run())
    assert slept["n"] == 1
    assert calls["n"] >= 2
