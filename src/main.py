import logging
import os
import platform
import sys
import threading
import time
import uuid
import warnings
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text

from src.agent_runner import get_graph_app, run_agent_turn_detailed
from src.boot_checks import verify_boot_dependencies
from src.config import settings
from src.db.pool import close_pool
from src.memory import get_default_memory, reset_memory_singletons
from src.obs.console_format import ConsoleFormatter
from src.obs.logging_setup import AgentStreamHandler, configure_agent_logging
from src.obs.logs import normalize_repl_user_text

# Quiet urllib3 SSL noise.
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

_AGENT_HISTORY = os.path.expanduser("~/.agent_history")

# Set in main() before REPL (no graph / tracing / logging side effects at import).
app: Any = None
PROCESS_SESSION_UUID: str = ""
_console: Console | None = None
_user_prompt: PromptSession | None = None

logger = logging.getLogger(__name__)
log = logging.getLogger("agent")


_thinking_indicator: "ThinkingIndicator | None" = None
BOOT_LEVEL = 25
logging.addLevelName(BOOT_LEVEL, "BOOT")


class ThinkingIndicator:
    """Small stderr spinner shown while LangGraph is running."""

    _frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _grey = "\x1b[90m"
    _reset = "\x1b[0m"

    def __init__(self, *, enabled: bool, stream: Any = sys.stderr) -> None:
        self.enabled = enabled
        self.stream = stream
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="agent-thinking", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None
        self.clear()

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.stream.write("\r\033[2K")
            self.stream.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.25):
            elapsed = int(time.monotonic() - self._started_at)
            line = self._render(elapsed=elapsed, tick=i)
            i += 1
            with self._lock:
                self.stream.write(f"\r\033[2K{line}")
                self.stream.flush()

    @classmethod
    def _render(cls, *, elapsed: int, tick: int) -> str:
        frame = cls._frames[tick % len(cls._frames)]
        dots = "." * (tick % 4)
        return f"{cls._grey}{frame} Agent is thinking{dots} {elapsed}s{cls._reset}"


# True after REPL prints streamed agent text with print(..., end=""). Cleared when we flush a
# full stdout line or when _StdoutSyncStreamHandler emits stderr after syncing stdout.
_repl_stdout_pending: list[bool] = [False]


class _StdoutSyncStreamHandler(AgentStreamHandler):
    """Finish an incomplete stdout line before stderr so logs never glue to streamed agent text."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if _thinking_indicator is not None:
                _thinking_indicator.clear()
            if _repl_stdout_pending[0]:
                sys.stdout.write("\n")
                sys.stdout.flush()
                _repl_stdout_pending[0] = False
            super().emit(record)
        except (ValueError, OSError):
            # pytest / process shutdown may close stderr while handlers still flush
            return
        except Exception:
            self.handleError(record)


def _read_user_line() -> str:
    if _user_prompt is not None:
        return normalize_repl_user_text(_user_prompt.prompt())
    return normalize_repl_user_text(input("[User]: "))


def _is_exit_command(line: str) -> bool:
    """psql-style \\q plus common REPL exits (whole line, trimmed)."""
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    return low in ("\\q", "\\quit", "exit", "quit")


def _shutdown_session(turns: list[tuple[str, str]]) -> None:
    """Flush stdio and close DB pool on REPL exit."""
    del turns  # transcript lives in Postgres checkpoint / conversations snapshot
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover
        pass
    try:
        close_pool()
        reset_memory_singletons()
    except Exception as e:
        log.warning("DB pool close failed: %s", e)


def _boot_box_lines(hostname: str) -> list[str]:
    width = 42
    title = f"  AGENT CORE @ {hostname}  "
    if len(title) > width:
        title = title[: width - 1] + " "
    return [
        "╔" + "═" * width + "╗",
        "║" + f"{title:<{width}}" + "║",
        "╚" + "═" * width + "╝",
    ]


def _boot_row(label: str, value: str, extra: str = "") -> str:
    suffix = f"     {extra}" if extra else ""
    return f"{label:<13} {value}{suffix}"


def _write_boot_header(hostname: str, stream: Any = sys.stderr) -> None:
    use_color = bool(getattr(stream, "isatty", lambda: False)())
    for line in _boot_box_lines(hostname):
        if use_color:
            stream.write(f"\x1b[90m\x1b[2m{line}\x1b[0m\n")
        else:
            stream.write(line + "\n")
    stream.flush()


def _log_boot_block() -> None:
    boot_log = logging.getLogger("agent.boot")
    _write_boot_header(platform.node())
    reflection = "on" if settings.reflection else "off"
    boot_log.log(BOOT_LEVEL, _boot_row("agent", settings.default_agent_id))
    boot_log.log(
        BOOT_LEVEL,
        _boot_row(
            "checkpointer",
            settings.checkpointer,
            f"reflection={reflection}/{settings.reflect_every_n}",
        ),
    )
    boot_log.log(
        BOOT_LEVEL,
        _boot_row(
            "checkpoint",
            f"max_msgs={settings.checkpoint_max_messages}",
            f"collapse_pairs={settings.tool_collapse_keep_recent_pairs}",
        ),
    )
    boot_log.log(
        BOOT_LEVEL,
        _boot_row(
            "graph",
            f"recursion={settings.recursion_limit}",
            f"tool_rounds≤{settings.max_tool_rounds_per_turn}",
        ),
    )
    boot_log.log(
        BOOT_LEVEL,
        _boot_row(
            "db",
            settings.database_name,
            f"pool={settings.db_pool_min}..{settings.db_pool_max}",
        ),
    )
    boot_log.log(BOOT_LEVEL, _boot_row("model", settings.reasoning_model))
    boot_log.log(BOOT_LEVEL, _boot_row("embed", settings.embed_model))
    from src.llm_factory import fixed_sampling_summary

    samp = fixed_sampling_summary()
    boot_log.log(
        BOOT_LEVEL,
        _boot_row(
            "sampling",
            f"temp={samp['temperature']}",
            f"top_p={samp['top_p']}  repeat_penalty={samp['repeat_penalty']}",
        ),
    )
    boot_log.log(BOOT_LEVEL, "[ SYSTEM ONLINE ]")


def main() -> None:
    global app, PROCESS_SESSION_UUID, _console, _user_prompt
    global _thinking_indicator

    from src.obs.tracing import init_telemetry

    init_telemetry("agent-cli", settings)

    _log_mode = (settings.log_level or "INFO").strip().upper()

    _h = _StdoutSyncStreamHandler(stream=sys.stderr)
    _h.setFormatter(ConsoleFormatter())
    configure_agent_logging(
        level=_log_mode,
        clear_handlers=True,
        handler=_h,
    )

    try:
        verify_boot_dependencies()
        app = get_graph_app()
    except Exception as e:
        logger.error("boot                  %s", e)
        raise SystemExit(1) from e

    PROCESS_SESSION_UUID = str(uuid.uuid4())
    if _log_mode != "OFF":
        _log_boot_block()

    tty = bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )
    _thinking_indicator = ThinkingIndicator(enabled=tty, stream=sys.stderr)
    _console = Console(force_terminal=tty, soft_wrap=True, highlight=False)
    if tty:
        # multiline=False + PT input: bracketed paste (e.g. \x1b[200~...\x1b[201~) is applied as one
        # buffer update on supported terminals. Newlines are then folded to spaces in normalize_repl_user_text.
        _user_prompt = PromptSession(
            message=HTML('<style fg="cyan">[User]</style> '),
            history=FileHistory(_AGENT_HISTORY),
            multiline=False,
        )
    else:
        _user_prompt = None

    assert _console is not None

    transcript: list[tuple[str, str]] = []
    snapshot_buffer: list[dict[str, str]] = []
    repl_turn = 0

    try:
        while True:
            try:
                _console.print()
                stripped = _read_user_line()
                if not stripped:
                    continue
                if _is_exit_command(stripped):
                    log.info("Shutting down.")
                    break

                repl_thread_id = (settings.langgraph_thread_id or "").strip()
                agent_text = ""
                assert _thinking_indicator is not None
                _thinking_indicator.start()
                try:
                    turn = run_agent_turn_detailed(
                        stripped,
                        thread_id=repl_thread_id,
                        session_uuid=PROCESS_SESSION_UUID,
                    )
                finally:
                    _thinking_indicator.stop()
                agent_text = turn.reply

                if agent_text:
                    if tty:
                        _console.print()
                        _console.print(
                            Text.assemble(
                                ("[Agent] ", "bold magenta"),
                                (agent_text, "default"),
                            )
                        )
                    else:
                        print(f"\n[Agent] {agent_text}", flush=True)
                    _repl_stdout_pending[0] = False

                transcript.append((stripped, agent_text))
                repl_turn += 1
                snapshot_buffer.extend(
                    [
                        {"type": "human", "content": stripped},
                        {"type": "ai", "content": agent_text},
                    ]
                )
                every_n = max(1, int(settings.conversation_snapshot_every_n))
                if repl_turn % every_n == 0:
                    try:
                        get_default_memory().save_conversation_snapshot(
                            agent_id=settings.default_agent_id,
                            messages=list(snapshot_buffer),
                            summary=None,
                            merge=True,
                        )
                        snapshot_buffer.clear()
                    except Exception as e:
                        log.warning("Failed to persist conversation snapshot: %s", e)

                from src.graph.reflect_runner import spawn_reflect_if_pending
                from src.graph.role_review_runner import spawn_role_review
                from src.graph.session_summary_runner import (
                    spawn_session_summary_if_pending,
                )

                tid = turn.thread_id or repl_thread_id
                spawn_session_summary_if_pending(
                    thread_id=tid, summary_pending=turn.summary_pending
                )
                spawn_reflect_if_pending(
                    thread_id=tid, reflect_pending=turn.reflect_pending
                )
                spawn_role_review(thread_id=tid, pending=turn.role_review_pending)

            except (EOFError, KeyboardInterrupt):
                log.info("Shutting down.")
                break
            except Exception as e:
                log.exception("Runtime error: %s", e)
    finally:
        if snapshot_buffer:
            try:
                get_default_memory().save_conversation_snapshot(
                    agent_id=settings.default_agent_id,
                    messages=list(snapshot_buffer),
                    summary=None,
                    merge=True,
                )
            except Exception as e:
                log.warning("Failed to flush conversation snapshot: %s", e)
        _shutdown_session(transcript)


if __name__ == "__main__":  # pragma: no cover
    main()
