"""Telegram long-polling bot wired to the LangGraph agent."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx

from src.agent_mode import is_asleep
from src.agent_runner import AgentTurnResult, arun_agent_turn
from src.api.telegram_client import TelegramClient
from src.api.telegram_commands import (
    BOT_COMMANDS,
    SLEEP_NOTICE,
    try_handle_command,
)
from src.config import settings
from src.graph.reflect_runner import run_reflect_if_due
from src.sync_lock_prune import prune_unlocked_locks

_log = logging.getLogger(__name__)

# Telegram shows "typing" for ~5s; refresh before it expires.
_DEFAULT_TYPING_REFRESH_SEC = 4.0

_chat_locks: dict[int, asyncio.Lock] = {}

# Strong refs to fire-and-forget deferred tasks (event loop keeps only weak refs).
_deferred_tasks: set[asyncio.Task] = set()


def _spawn_deferred(coro, *, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _deferred_tasks.add(task)
    task.add_done_callback(_deferred_tasks.discard)


def parse_allowed_chat_ids(raw: str) -> set[int]:
    out: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            _log.warning("Ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry: %r", part)
    return out


def _configured_allowed_chat_ids() -> set[int]:
    """Explicit allowlist from ``TELEGRAM_ALLOWED_CHAT_IDS`` only."""
    raw = settings.telegram_allowed_chat_ids
    if isinstance(raw, str):
        return parse_allowed_chat_ids(raw)
    return {int(x) for x in (raw or [])}


def _owner_chat_id() -> int | None:
    raw = settings.telegram_owner_chat_id
    if raw == "" or raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def effective_allowed_chat_ids() -> frozenset[int]:
    """Fail-closed allowlist: configured ids plus optional owner fallback."""
    allowed = set(_configured_allowed_chat_ids())
    owner = _owner_chat_id()
    if owner is not None:
        allowed.add(owner)
    return frozenset(allowed)


def telegram_allowlist_configured() -> bool:
    """True when at least one chat id is allowed (required to start polling)."""
    return bool(effective_allowed_chat_ids())


def chat_allowed(chat_id: int) -> bool:
    """Fail-closed: unknown chat ids are denied when not on the effective allowlist."""
    return int(chat_id) in effective_allowed_chat_ids()


def telegram_polling_may_start(*, token_enabled: bool) -> bool:
    """Return False when bot token is set but no chat allowlist is configured."""
    if not token_enabled:
        return False
    if telegram_allowlist_configured():
        return True
    _log.error(_telegram_allowlist_boot_error())
    return False


def _telegram_allowlist_boot_error() -> str:
    return (
        "TELEGRAM_BOT_TOKEN is set but no chat allowlist is configured. "
        "Set TELEGRAM_ALLOWED_CHAT_IDS=<your chat id> and/or "
        "TELEGRAM_OWNER_CHAT_ID=<your chat id> in .env before deploy. "
        "Empty TELEGRAM_ALLOWED_CHAT_IDS denies ALL chats (fail-closed)."
    )


def assert_telegram_boot_allowlist() -> None:
    """Fail boot when a bot token is set but no effective chat allowlist exists."""
    if not (settings.telegram_bot_token or "").strip():
        return
    if telegram_allowlist_configured():
        return
    raise RuntimeError(_telegram_allowlist_boot_error())


def telegram_thread_id(chat_id: int | str) -> str:
    prefix = (settings.telegram_thread_prefix or "telegram").strip() or "telegram"
    return f"{prefix}_{chat_id}"


def session_uuid_for_chat(chat_id: int | str) -> str:
    return f"telegram-{chat_id}"


def _turn_footer(turn: AgentTurnResult) -> str:
    from src.obs.turn_footer import footer_from_turn

    return footer_from_turn(turn)


async def _run_deferred_summary(
    client: TelegramClient,
    chat_id: int,
    *,
    thread_id: str,
) -> None:
    from src.graph.session_summary_runner import run_session_summary_if_due
    from src.obs.turn_footer import format_summary_timing_line

    try:
        elapsed_ms = await asyncio.to_thread(
            run_session_summary_if_due,
            thread_id=thread_id,
        )
    except Exception:
        _log.exception("telegram.summary deferred failed thread_id=%s", thread_id)
        return
    if elapsed_ms is None or elapsed_ms <= 0:
        return
    line = format_summary_timing_line(elapsed_ms)
    await client.send_message(chat_id, line)


def _chat_lock(chat_id: int) -> asyncio.Lock:
    prune_unlocked_locks(_chat_locks, max_size=512)
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[chat_id] = lock
    return lock


def _typing_refresh_seconds() -> float:
    try:
        sec = float(settings.telegram_typing_interval_sec)
    except (TypeError, ValueError):
        sec = _DEFAULT_TYPING_REFRESH_SEC
    return max(0.5, min(sec, 4.5))


def extract_incoming_text(update: dict[str, Any]) -> tuple[int, str] | None:
    # Ignore edited_message — edits should not re-run a full agent turn.
    msg = update.get("message")
    if not isinstance(msg, dict):
        return None
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    text = msg.get("text") or msg.get("caption") or ""
    if not isinstance(text, str):
        text = str(text)
    return int(chat_id), text.strip()


@asynccontextmanager
async def typing_keepalive(client: TelegramClient, chat_id: int | str):
    """Refresh Telegram typing indicator until the agent turn completes."""
    stop = asyncio.Event()
    interval = _typing_refresh_seconds()

    async def _refresh() -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                try:
                    await client.send_chat_action(chat_id, "typing")
                except Exception:
                    _log.debug("telegram.typing refresh failed", exc_info=True)
                continue
            break

    await client.send_chat_action(chat_id, "typing")
    task = asyncio.create_task(_refresh(), name=f"telegram-typing-{chat_id}")
    try:
        yield
    finally:
        stop.set()
        await task


async def _handle_telegram_message(
    client: TelegramClient,
    *,
    chat_id: int,
    text: str,
) -> None:
    if not text:
        return

    # Operator commands run before any agent turn (and while asleep).
    if await try_handle_command(
        client, chat_id, text, bot_username=client.bot_username
    ):
        return

    if is_asleep():
        await client.send_message(chat_id, SLEEP_NOTICE)
        return

    thread_id = telegram_thread_id(chat_id)
    async with typing_keepalive(client, chat_id):
        turn = await arun_agent_turn(
            text,
            thread_id=thread_id,
            session_uuid=session_uuid_for_chat(chat_id),
        )
    try:
        await client.send_message_with_footer(
            chat_id,
            turn.reply,
            _turn_footer(turn),
        )
    except Exception:
        _log.exception("telegram.reply failed chat_id=%s", chat_id)
        # Propagate so _safe_handle_update can surface an error bubble;
        # do not schedule deferred work after a failed delivery.
        raise
    if turn.summary_pending and turn.thread_id:
        _spawn_deferred(
            _run_deferred_summary(
                client,
                chat_id,
                thread_id=turn.thread_id,
            ),
            name=f"telegram-summary-{chat_id}",
        )
    if turn.reflect_pending and turn.thread_id:
        _spawn_deferred(
            asyncio.to_thread(run_reflect_if_due, thread_id=turn.thread_id),
            name=f"telegram-reflect-{chat_id}",
        )
    if turn.role_review_pending and turn.thread_id:
        from src.graph.role_review_runner import run_role_review_for_thread

        _spawn_deferred(
            asyncio.to_thread(run_role_review_for_thread, thread_id=turn.thread_id),
            name=f"telegram-role-review-{chat_id}",
        )


async def handle_telegram_update(
    client: TelegramClient, update: dict[str, Any]
) -> None:
    parsed = extract_incoming_text(update)
    if parsed is None:
        return
    chat_id, text = parsed

    if not chat_allowed(chat_id):
        _log.warning(
            "telegram.chat_denied chat_id=%s (not in TELEGRAM_ALLOWED_CHAT_IDS / owner)",
            chat_id,
        )
        return

    async with _chat_lock(chat_id):
        await _handle_telegram_message(client, chat_id=chat_id, text=text)


async def _safe_handle_update(client: TelegramClient, update: dict[str, Any]) -> None:
    chat_id: int | None = None
    try:
        await handle_telegram_update(client, update)
    except Exception as exc:
        _log.exception(
            "telegram.handle failed update_id=%s",
            update.get("update_id"),
        )
        parsed = extract_incoming_text(update)
        if parsed is None:
            return
        chat_id, _ = parsed
        from src.agent_errors import format_agent_turn_error

        try:
            await client.send_message(
                chat_id,
                format_agent_turn_error(exc, channel="telegram"),
            )
        except Exception:
            _log.exception("telegram.error_reply failed chat_id=%s", chat_id)


async def _wait_get_me(client: TelegramClient, *, stop_event: asyncio.Event) -> bool:
    """Retry getMe until success or stop. Returns False if stopped without success."""
    delay = 1.0
    while not stop_event.is_set():
        try:
            me = await client.get_me()
            client.bot_username = str(me.get("username") or "")
            _log.info(
                "telegram.polling start username=%s",
                me.get("username", "?"),
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("telegram.getMe failed: %s — retry in %.1fs", exc, delay)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                return False
            except asyncio.TimeoutError:
                delay = min(30.0, delay * 1.5)
    return False


async def telegram_polling_loop(
    client: TelegramClient,
    *,
    stop_event: asyncio.Event,
) -> None:
    """Long-poll Telegram and dispatch messages to the agent."""
    if not client.enabled:
        _log.info("telegram.polling skipped (no token)")
        return

    if not await _wait_get_me(client, stop_event=stop_event):
        _log.info("telegram.polling stop (getMe never succeeded)")
        return

    # Register the command menu once per start (clients cache the list).
    try:
        if await client.set_my_commands(BOT_COMMANDS):
            _log.info("telegram.commands registered n=%d", len(BOT_COMMANDS))
        else:
            _log.warning("telegram.commands registration refused by API")
    except Exception as exc:
        _log.warning("telegram.commands registration failed: %s", exc)

    offset = 0
    while not stop_event.is_set():
        try:
            updates = await client.get_updates(
                offset=offset,
                timeout=settings.telegram_poll_timeout,
                allowed_updates=["message"],
            )
            for update in updates:
                uid = int(update.get("update_id", 0))
                # Handle before ack so a crash mid-turn is redelivered.
                await _safe_handle_update(client, update)
                offset = max(offset, uid + 1)
        except asyncio.CancelledError:
            break
        except httpx.TimeoutException:
            # Long-poll read timeout or stalled connection — retry immediately.
            continue
        except httpx.HTTPError as exc:
            _log.warning("telegram.poll transport error: %s", exc)
            await asyncio.sleep(3.0)
        except Exception:
            _log.exception("telegram.poll error")
            await asyncio.sleep(3.0)

    _log.info("telegram.polling stop")
