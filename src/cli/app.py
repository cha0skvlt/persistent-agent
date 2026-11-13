"""Typer CLI: one-shot chat, status, API server."""

from __future__ import annotations

import json
import sys
from typing import Any

import typer

from src.agent_runner import run_agent_turn_detailed
from src.api.telegram_client import TelegramClient
from src.config import settings
from src.graph.reflect_runner import spawn_reflect_if_pending
from src.graph.session_summary_runner import spawn_session_summary_if_pending

cli = typer.Typer(
    name="agent",
    help="Agent Core CLI — same LangGraph agent as REPL, FastAPI, and Telegram.",
    no_args_is_help=True,
)

api_app = typer.Typer(help="FastAPI + optional Telegram polling.")
cli.add_typer(api_app, name="api")


def resolve_chat_message(message: str | None) -> str:
    """Read message from argument or stdin (pipe-friendly)."""
    raw = message
    if raw is None and not sys.stdin.isatty():
        raw = sys.stdin.read()
    text = (raw or "").strip()
    if not text:
        raise typer.BadParameter("message required (argument or stdin)")
    return text


def resolve_thread_id(thread_id: str | None, *, default: str) -> str:
    tid = (thread_id or default).strip()
    if not tid:
        raise typer.BadParameter("thread_id required")
    return tid


def build_status_payload() -> dict[str, Any]:
    client = TelegramClient(settings.telegram_bot_token)
    return {
        "status": "ok",
        "agent_id": settings.default_agent_id,
        "telegram_polling": bool(settings.telegram_polling and client.enabled),
        "cli_default_thread_id": settings.cli_default_thread_id,
        "api_default_thread_id": settings.api_default_thread_id,
    }


def emit_json_or_text(payload: dict[str, Any], *, json_out: bool) -> None:
    if json_out:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        for key, value in payload.items():
            typer.echo(f"{key}={value}")


@cli.command()
def chat(
    message: str | None = typer.Argument(
        None,
        help="User message (omit to read from stdin).",
    ),
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        "-t",
        help="LangGraph checkpoint thread (default: CLI_DEFAULT_THREAD_ID).",
    ),
    session_uuid: str | None = typer.Option(
        None,
        "--session-uuid",
        help="Optional session uuid for this turn.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help='Emit {"reply", "thread_id"} JSON.',
    ),
) -> None:
    """Run one agent turn and print the reply."""
    text = resolve_chat_message(message)
    tid = resolve_thread_id(thread_id, default=settings.cli_default_thread_id)
    turn = run_agent_turn_detailed(text, thread_id=tid, session_uuid=session_uuid)
    spawn_session_summary_if_pending(
        thread_id=turn.thread_id or tid,
        summary_pending=turn.summary_pending,
    )
    spawn_reflect_if_pending(
        thread_id=turn.thread_id or tid,
        reflect_pending=turn.reflect_pending,
    )
    if turn.role_review_pending:
        from src.graph.role_review_runner import spawn_role_review

        spawn_role_review(
            thread_id=turn.thread_id or tid,
            pending=True,
        )
    if json_out:
        typer.echo(
            json.dumps({"reply": turn.reply, "thread_id": tid}, ensure_ascii=False)
        )
    else:
        typer.echo(turn.reply)


@cli.command()
def status(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON status."),
) -> None:
    """Lightweight agent/API status (not full DB health — see make health)."""
    emit_json_or_text(build_status_payload(), json_out=json_out)


@api_app.command("serve")
def api_serve(
    host: str | None = typer.Option(None, "--host", help="Bind address."),
    port: int | None = typer.Option(None, "--port", help="Bind port."),
) -> None:
    """Start FastAPI + optional Telegram polling (same as make api)."""
    import uvicorn

    bind_host = (host or settings.api_host).strip() or "127.0.0.1"
    bind_port = port if port is not None else settings.api_port
    uvicorn.run(
        "src.api.app:app",
        host=bind_host,
        port=bind_port,
        factory=False,
    )


def main() -> None:
    from src.obs.tracing import init_telemetry

    init_telemetry("agent-cli", settings)
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
