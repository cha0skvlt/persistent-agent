"""FastAPI app: health, REST chat, Telegram long-polling."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.agent_runner import arun_agent_turn, get_graph_app
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.api.telegram_bot import telegram_polling_loop, telegram_polling_may_start
from src.api.telegram_client import TelegramClient
from src.boot_checks import verify_boot_dependencies_with_retry
from src.config import settings
from src.graph.session_summary_runner import run_session_summary_if_due
from src.obs.tracing import init_telemetry

_fastapi_instrumented = False

_log = logging.getLogger(__name__)

_stop_event: asyncio.Event | None = None
_polling_task: asyncio.Task | None = None
_reverie_task: asyncio.Task | None = None
_telegram_client: TelegramClient | None = None

# Strong refs to fire-and-forget deferred tasks (event loop keeps only weak refs).
_deferred_tasks: set[asyncio.Task] = set()


def _spawn_deferred(coro, *, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _deferred_tasks.add(task)
    task.add_done_callback(_deferred_tasks.discard)


def _host_is_loopback(host: str) -> bool:
    h = (host or "").strip().lower()
    if h in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def assert_api_bind_auth() -> None:
    """Fail boot when the API binds beyond loopback without an API key."""
    if _host_is_loopback(settings.api_host):
        return
    if (settings.api_key or "").strip():
        return
    raise RuntimeError(
        f"API_HOST={settings.api_host} is not loopback and API_KEY is empty — "
        "POST /chat would be open to the network. Set API_KEY in .env."
    )


def _configure_api_logging() -> None:
    from src.obs.logging_setup import configure_agent_logging

    configure_agent_logging(
        level=settings.log_level or "INFO",
        clear_handlers=True,
    )


def _verify_api_key(authorization: str | None = Header(default=None)) -> None:
    required = (settings.api_key or "").strip()
    if not required:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token.encode(), required.encode()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


async def _reverie_loop(stop: asyncio.Event) -> None:
    """Background loop: pick up due intents fast; solitary gates its own interval.

    Sleeps only REVERIE_LOOP_MIN_SEC — due intents run within one loop tick.
    Solitary pacing is enforced in-process by run_solitary_pass_if_due
    (skip_interval), so a short loop does not speed up reflection."""
    while not stop.is_set():
        try:
            await asyncio.sleep(float(settings.reverie_loop_min_sec))
        except asyncio.CancelledError:
            return
        if stop.is_set():
            return
        try:
            from src.agent_mode import is_asleep

            if is_asleep():
                continue
            from src.reverie.intent_pass import run_due_intent_if_any

            ran = await asyncio.to_thread(run_due_intent_if_any)
            if not ran:
                from src.reverie.solitary import run_solitary_pass_if_due

                await asyncio.to_thread(run_solitary_pass_if_due)
        except Exception as exc:
            _log.warning("reverie loop error (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stop_event, _polling_task, _reverie_task, _telegram_client, _fastapi_instrumented
    init_telemetry("agent-api", settings)
    if not _fastapi_instrumented:
        FastAPIInstrumentor.instrument_app(app, excluded_urls="/health")
        _fastapi_instrumented = True
    _configure_api_logging()
    _log.info("api.start host=%s port=%s", settings.api_host, settings.api_port)
    try:
        verify_boot_dependencies_with_retry()
        assert_api_bind_auth()
    except Exception as e:
        _log.error("api.boot failed: %s", e)
        raise
    get_graph_app()
    try:
        from src.tools.web_core import check_searxng_health

        check_searxng_health()
    except Exception as exc:
        _log.warning("searxng.health probe failed: %s", exc)

    _stop_event = asyncio.Event()
    _telegram_client = TelegramClient(settings.telegram_bot_token)
    polling_ok = telegram_polling_may_start(token_enabled=_telegram_client.enabled)
    if settings.telegram_polling and polling_ok:
        assert _stop_event is not None and _telegram_client is not None

        async def _polling_supervisor() -> None:
            """Restart the poll loop if it exits unexpectedly while API is up."""
            while _stop_event is not None and not _stop_event.is_set():
                try:
                    await telegram_polling_loop(
                        _telegram_client, stop_event=_stop_event
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _log.exception("telegram.polling crashed — restarting")
                if _stop_event is None or _stop_event.is_set():
                    break
                await asyncio.sleep(3.0)

        _polling_task = asyncio.create_task(
            _polling_supervisor(),
            name="telegram-polling",
        )
    else:
        _log.info(
            "telegram.polling disabled polling=%s token_set=%s allowlist_ok=%s",
            settings.telegram_polling,
            _telegram_client.enabled,
            polling_ok if _telegram_client.enabled else False,
        )

    # Pulse (intent) and solitary share one loop; either flag is enough to start.
    if settings.solitary_enabled or settings.pulse_enabled:
        _reverie_task = asyncio.create_task(
            _reverie_loop(_stop_event), name="reverie-loop"
        )

    yield

    if _stop_event is not None:
        _stop_event.set()
    if _reverie_task is not None:
        _reverie_task.cancel()
        try:
            await _reverie_task
        except asyncio.CancelledError:
            pass
        _reverie_task = None
    if _polling_task is not None:
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass
        _polling_task = None
    if _telegram_client is not None:
        await _telegram_client.close()
        _telegram_client = None
    _stop_event = None
    _log.info("api.stop")


app = FastAPI(title="Agent Core API", lifespan=lifespan)


def _db_alive() -> bool:
    from src.db.pool import get_pool

    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:
        _log.warning("health.db check failed: %s", exc)
        return False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    db_ok = await asyncio.to_thread(_db_alive)
    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database unavailable",
        )
    tg_enabled = (
        _telegram_client.enabled
        if _telegram_client is not None
        else bool((settings.telegram_bot_token or "").strip())
    )
    polling_alive = bool(
        _polling_task is not None and not _polling_task.done() and tg_enabled
    )
    return HealthResponse(
        status="ok",
        agent_id=settings.default_agent_id,
        telegram_polling=polling_alive,
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_verify_api_key)])
async def chat(body: ChatRequest) -> ChatResponse:
    thread_id = (body.thread_id or settings.api_default_thread_id).strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    turn = await arun_agent_turn(body.message, thread_id=thread_id)
    if turn.summary_pending and turn.thread_id:
        _spawn_deferred(
            asyncio.to_thread(run_session_summary_if_due, thread_id=turn.thread_id),
            name=f"api-summary-{thread_id}",
        )
    if turn.reflect_pending and turn.thread_id:
        from src.graph.reflect_runner import run_reflect_if_due

        _spawn_deferred(
            asyncio.to_thread(run_reflect_if_due, thread_id=turn.thread_id),
            name=f"api-reflect-{thread_id}",
        )
    if turn.role_review_pending and turn.thread_id:
        from src.graph.role_review_runner import run_role_review_for_thread

        _spawn_deferred(
            asyncio.to_thread(run_role_review_for_thread, thread_id=turn.thread_id),
            name=f"api-role-review-{thread_id}",
        )
    return ChatResponse(reply=turn.reply, thread_id=thread_id)


def create_app() -> FastAPI:
    """Factory for tests."""
    return app


__all__ = ["app", "create_app", "lifespan"]
