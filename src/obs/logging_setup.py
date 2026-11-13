"""Shared root logging: ConsoleFormatter for REPL, API daemon, and ``make log``."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from src.obs.console_format import SUPPRESS, ConsoleFormatter
from src.obs.logs import setup_agent_event_logger

_NOISE_WARNING = (
    "httpx",
    "httpcore",
    "urllib3",
    "ollama",
    "uvicorn.access",
    "keyring",
    "keyring.backend",
)
# OTLP exporters spam WARNING/ERROR when otelcol is down — poison ``make log``.
_NOISE_CRITICAL = (
    "opentelemetry.exporter.otlp",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.sdk._logs.export",
    "opentelemetry.sdk.metrics.export",
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.http",
)


class AgentStreamHandler(logging.StreamHandler):
    """Stream handler that skips empty ``ConsoleFormatter`` folds (``SUPPRESS``)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if msg == SUPPRESS or msg == "":
                return
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except (ValueError, OSError):
            return
        except Exception:
            self.handleError(record)


def apply_log_level(root: logging.Logger, level_name: str) -> None:
    level = (level_name or "INFO").strip().upper()
    if level == "OFF":
        root.setLevel(logging.ERROR)
    elif level == "DEBUG":
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)


def quiet_noise_loggers() -> None:
    for name in _NOISE_WARNING:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _NOISE_CRITICAL:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def configure_agent_logging(
    *,
    level: str = "INFO",
    stream: TextIO | None = None,
    clear_handlers: bool = True,
    handler: logging.Handler | None = None,
) -> logging.Handler:
    """Install pretty ``agent.event`` console logging on the root logger.

    ``clear_handlers=True`` (API / REPL) replaces uvicorn/default handlers so
    ``make log`` / launchd tails see dialogue-folded lines instead of raw JSON.
    """
    root = logging.getLogger()
    if clear_handlers:
        for existing in list(root.handlers):
            root.removeHandler(existing)
            try:
                existing.close()
            except Exception:
                pass

    apply_log_level(root, level)
    out = handler
    if out is None:
        out = AgentStreamHandler(stream if stream is not None else sys.stderr)
        out.setFormatter(ConsoleFormatter())
    elif out.formatter is None:
        out.setFormatter(ConsoleFormatter())
    root.addHandler(out)
    quiet_noise_loggers()
    setup_agent_event_logger()
    return out
