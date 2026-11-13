"""Tests for Typer CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from src.cli.app import (
    build_status_payload,
    cli,
    emit_json_or_text,
    resolve_chat_message,
    resolve_thread_id,
)
from src.config import settings

runner = CliRunner()


@pytest.mark.unit
def test_resolve_chat_message_from_arg():
    assert resolve_chat_message(" hello ") == "hello"


@pytest.mark.unit
def test_resolve_chat_message_from_stdin(monkeypatch):
    monkeypatch.setattr("src.cli.app.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("src.cli.app.sys.stdin.read", lambda: " piped\n")
    assert resolve_chat_message(None) == "piped"


@pytest.mark.unit
def test_resolve_chat_message_empty():
    import typer

    with pytest.raises(typer.BadParameter):
        resolve_chat_message("   ")


@pytest.mark.unit
def test_resolve_thread_id_default():
    assert resolve_thread_id(None, default="cli_session_v1") == "cli_session_v1"
    assert resolve_thread_id("custom", default="cli_session_v1") == "custom"


@pytest.mark.unit
def test_resolve_thread_id_empty():
    import typer

    with pytest.raises(typer.BadParameter):
        resolve_thread_id("  ", default="cli_session_v1")


@pytest.mark.unit
def test_build_status_payload(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)
    monkeypatch.setattr(settings, "telegram_polling", True, raising=False)
    payload = build_status_payload()
    assert payload["status"] == "ok"
    assert payload["telegram_polling"] is False


@pytest.mark.unit
def test_emit_json_or_text(capsys):
    emit_json_or_text({"status": "ok", "agent_id": "agent"}, json_out=True)
    out = capsys.readouterr().out.strip()
    assert json.loads(out)["status"] == "ok"

    emit_json_or_text({"status": "ok"}, json_out=False)
    text = capsys.readouterr().out
    assert "status=ok" in text


@pytest.mark.unit
def test_cli_help():
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.stdout


@pytest.mark.unit
def test_cli_chat(monkeypatch):
    from src.agent_runner import AgentTurnResult

    monkeypatch.setattr(
        "src.cli.app.run_agent_turn_detailed",
        lambda msg, *, thread_id, session_uuid=None: AgentTurnResult(
            reply=f"echo:{msg}",
            tools_used=(),
            elapsed_ms=1.0,
            thread_id=thread_id,
        ),
    )
    monkeypatch.setattr(
        "src.cli.app.spawn_session_summary_if_pending", lambda **k: None
    )
    result = runner.invoke(cli, ["chat", "hello"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "echo:hello"


@pytest.mark.unit
def test_cli_chat_json_and_options(monkeypatch):
    from src.agent_runner import AgentTurnResult

    monkeypatch.setattr(
        "src.cli.app.run_agent_turn_detailed",
        lambda msg, *, thread_id, session_uuid=None: AgentTurnResult(
            reply="reply",
            tools_used=(),
            elapsed_ms=1.0,
            thread_id=thread_id,
        ),
    )
    monkeypatch.setattr(
        "src.cli.app.spawn_session_summary_if_pending", lambda **k: None
    )
    result = runner.invoke(
        cli,
        [
            "chat",
            "hi",
            "--thread-id",
            "t1",
            "--session-uuid",
            "sid",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {"reply": "reply", "thread_id": "t1"}


@pytest.mark.unit
def test_cli_chat_stdin(monkeypatch):
    from src.agent_runner import AgentTurnResult

    monkeypatch.setattr(
        "src.cli.app.run_agent_turn_detailed",
        lambda msg, *, thread_id, session_uuid=None: AgentTurnResult(
            reply=msg,
            tools_used=(),
            elapsed_ms=1.0,
            thread_id=thread_id,
        ),
    )
    monkeypatch.setattr(
        "src.cli.app.spawn_session_summary_if_pending", lambda **k: None
    )
    result = runner.invoke(cli, ["chat"], input="stdin msg\n")
    assert result.exit_code == 0
    assert result.stdout.strip() == "stdin msg"


@pytest.mark.unit
def test_cli_chat_missing_message():
    result = runner.invoke(cli, ["chat"])
    assert result.exit_code != 0


@pytest.mark.unit
def test_cli_status(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "agent_id=" in result.stdout


@pytest.mark.unit
def test_cli_status_json(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)
    result = runner.invoke(cli, ["status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "ok"


@pytest.mark.unit
def test_cli_api_serve(monkeypatch):
    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(cli, ["api", "serve", "--host", "0.0.0.0", "--port", "9000"])
    assert result.exit_code == 0
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9000


@pytest.mark.unit
def test_cli_package_exports():
    from src.cli import cli as pkg_cli

    assert pkg_cli is cli


@pytest.mark.unit
def test_cli_main_entry(monkeypatch):
    called = {"n": 0}

    def fake_app():
        called["n"] += 1

    monkeypatch.setattr("src.cli.app.cli", fake_app)
    from src.cli.app import main

    main()
    assert called["n"] == 1


@pytest.mark.unit
def test_cli_module_main(monkeypatch):
    monkeypatch.setattr("src.cli.app.main", lambda: None)
    import runpy

    runpy.run_module("src.cli.__main__", run_name="__main__")
