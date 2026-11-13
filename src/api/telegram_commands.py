"""Operator bot commands over Telegram.

Dispatched before any agent turn — commands never enter the graph. Parsing
follows Bot API command semantics: ``/cmd[@botusername] [args]``; a command
addressed to a different bot (group chats) is consumed silently. The command
list is registered with Telegram via ``setMyCommands`` at polling start so
clients render the command menu.

Adding a command: append to ``BOT_COMMANDS`` (menu entry) and register an
async handler in ``_HANDLERS``.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.agent_mode import (
    MODE_AWAKE,
    MODE_SLEEP,
    get_mode,
    prewarm_chat_model,
    set_mode,
    unload_ollama_models,
)
from src.config import settings
from src.obs.logs import log_event
from src.speaker import speaker_label

_COMMAND_RE = re.compile(
    r"^/(?P<name>[A-Za-z0-9_]{1,32})"
    r"(?:@(?P<bot>[A-Za-z0-9_]{1,64}))?"
    r"(?:\s+(?P<args>.+))?$",
    re.DOTALL,
)

SLEEP_NOTICE = "Asleep. /make wake — wake up."

# Registered with setMyCommands (Telegram command menu).
BOT_COMMANDS: list[dict[str, str]] = [
    {"command": "make", "description": "sleep | wake — control the agent"},
    {"command": "start", "description": "Agent status"},
]


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    args: str
    bot_mention: str


def parse_command(text: str) -> ParsedCommand | None:
    """Parse ``/cmd[@bot] args`` from message text; None for non-commands."""
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    m = _COMMAND_RE.match(raw)
    if m is None:
        return None
    return ParsedCommand(
        name=m.group("name").lower(),
        args=(m.group("args") or "").strip(),
        bot_mention=(m.group("bot") or "").strip(),
    )


def command_authorized(chat_id: int) -> bool:
    """State-changing commands: owner chat only when an owner is configured."""
    raw = str(settings.telegram_owner_chat_id or "").strip()
    if not raw:
        # No owner configured — allowlist upstream already gates access.
        return True
    try:
        return int(chat_id) == int(raw)
    except (TypeError, ValueError):
        return False


async def _cmd_make(client: Any, chat_id: int, args: str) -> str:
    if not command_authorized(chat_id):
        return "This command is owner-only."
    sub = (args.split() or [""])[0].lower()
    if sub == "sleep":
        set_mode(MODE_SLEEP, source=f"telegram:{chat_id}")
        unloaded = await asyncio.to_thread(unload_ollama_models)
        ram = (
            f"models unloaded from RAM ({', '.join(unloaded)})"
            if unloaded
            else "models already unloaded"
        )
        return (
            f"Asleep: turns, background and outreach paused; {ram}. "
            "/make wake — wake up."
        )
    if sub == "wake":
        set_mode(MODE_AWAKE, source=f"telegram:{chat_id}")
        # Cold-load of a big model takes a while — warm it off-thread.
        threading.Thread(
            target=prewarm_chat_model, name="ollama-prewarm", daemon=True
        ).start()
        return "Awake. Warming the model — the first reply may be slow."
    return "Usage: /make sleep | /make wake"


async def _cmd_start(client: Any, chat_id: int, args: str) -> str:
    mode = get_mode()
    state = "asleep (/make wake)" if mode == MODE_SLEEP else "awake"
    return (
        f"**{speaker_label()} online — {state}.** "
        "One thread per chat — memory and checkpoint persist."
    )


_HANDLERS: dict[str, Callable[[Any, int, str], Awaitable[str]]] = {
    "make": _cmd_make,
    "start": _cmd_start,
}


async def try_handle_command(
    client: Any,
    chat_id: int,
    text: str,
    *,
    bot_username: str = "",
) -> bool:
    """Dispatch a command message. True → consumed, no agent turn."""
    parsed = parse_command(text)
    if parsed is None:
        return False
    if (
        parsed.bot_mention
        and bot_username
        and parsed.bot_mention.lower() != bot_username.lower()
    ):
        # Addressed to another bot — not ours, not an agent turn either.
        return True
    handler = _HANDLERS.get(parsed.name)
    log_event(
        "telegram.command",
        command=parsed.name,
        known=handler is not None,
        chat_id=chat_id,
    )
    if handler is None:
        await client.send_message(
            chat_id, "Unknown command. Available: /make sleep | /make wake"
        )
        return True
    reply = await handler(client, chat_id, parsed.args)
    if reply:
        await client.send_message(chat_id, reply)
    return True


__all__ = [
    "BOT_COMMANDS",
    "SLEEP_NOTICE",
    "ParsedCommand",
    "command_authorized",
    "parse_command",
    "try_handle_command",
]
