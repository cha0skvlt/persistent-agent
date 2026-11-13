"""Telegram operator commands — parse + dispatch."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent_mode import MODE_AWAKE, MODE_SLEEP
from src.api.telegram_commands import (
    BOT_COMMANDS,
    SLEEP_NOTICE,
    parse_command,
    try_handle_command,
)


def _run(coro):
    return asyncio.run(coro)


def test_parse_command_basic():
    p = parse_command("/make sleep")
    assert p is not None
    assert p.name == "make"
    assert p.args == "sleep"
    assert p.bot_mention == ""


def test_parse_command_with_bot_mention():
    p = parse_command("/start@MyBot extra")
    assert p is not None
    assert p.name == "start"
    assert p.bot_mention == "MyBot"
    assert p.args == "extra"


def test_parse_command_non_command():
    assert parse_command("hello") is None
    assert parse_command("") is None


def test_bot_commands_menu_entries():
    assert {c["command"] for c in BOT_COMMANDS} == {"make", "start"}


def test_try_handle_make_sleep(monkeypatch: pytest.MonkeyPatch):
    set_calls: list[tuple] = []
    monkeypatch.setattr(
        "src.api.telegram_commands.set_mode",
        lambda mode, *, source="": set_calls.append((mode, source)) or mode,
    )
    monkeypatch.setattr(
        "src.api.telegram_commands.unload_ollama_models",
        lambda: ["m1"],
    )
    monkeypatch.setattr(
        "src.api.telegram_commands.command_authorized",
        lambda _cid: True,
    )
    client = MagicMock()
    client.send_message = AsyncMock()
    assert _run(try_handle_command(client, 42, "/make sleep")) is True
    assert set_calls == [(MODE_SLEEP, "telegram:42")]
    text = client.send_message.await_args.args[1]
    assert text.startswith("Asleep:")
    assert "/make wake" in text


def test_try_handle_make_wake(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.api.telegram_commands.set_mode",
        lambda mode, *, source="": mode,
    )
    monkeypatch.setattr(
        "src.api.telegram_commands.command_authorized",
        lambda _cid: True,
    )
    monkeypatch.setattr(
        "src.api.telegram_commands.threading.Thread",
        lambda *a, **k: MagicMock(start=lambda: None),
    )
    client = MagicMock()
    client.send_message = AsyncMock()
    assert _run(try_handle_command(client, 1, "/make wake")) is True
    assert client.send_message.await_args.args[1].startswith("Awake.")


def test_try_handle_start_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.api.telegram_commands.get_mode", lambda: MODE_AWAKE)
    monkeypatch.setattr(
        "src.api.telegram_commands.speaker_label",
        lambda: "Agent",
    )
    client = MagicMock()
    client.send_message = AsyncMock()
    assert _run(try_handle_command(client, 1, "/start")) is True
    assert client.send_message.await_args.args[1].startswith(
        "**Agent online — awake.**"
    )


def test_try_handle_unknown_command(monkeypatch: pytest.MonkeyPatch):
    client = MagicMock()
    client.send_message = AsyncMock()
    assert _run(try_handle_command(client, 1, "/nope")) is True
    assert client.send_message.await_args.args[1].startswith("Unknown command.")


def test_try_handle_other_bot_mention_consumed_silently():
    client = MagicMock()
    client.send_message = AsyncMock()
    assert (
        _run(try_handle_command(client, 1, "/start@OtherBot", bot_username="OurBot"))
        is True
    )
    client.send_message.assert_not_awaited()


def test_make_sleep_unauthorized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.api.telegram_commands.command_authorized",
        lambda _cid: False,
    )
    client = MagicMock()
    client.send_message = AsyncMock()
    assert _run(try_handle_command(client, 99, "/make sleep")) is True
    assert client.send_message.await_args.args[1] == ("This command is owner-only.")


def test_sleep_notice_constant():
    assert SLEEP_NOTICE == "Asleep. /make wake — wake up."
