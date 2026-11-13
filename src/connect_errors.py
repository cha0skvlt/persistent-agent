"""Transient network / Ollama connection errors (shared by LLM + embed paths)."""

from __future__ import annotations

from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


def is_transient_connection_error(exc: BaseException) -> bool:
    """True for unreachable/refused Ollama or other HTTP connect failures."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if httpx is not None and isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
        ),
    ):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return is_transient_connection_error(cause)
    s = str(exc).casefold()
    return any(
        needle in s
        for needle in (
            "connection refused",
            "connection reset",
            "connection attempts failed",
            "failed to connect",
            "failed to establish",
            "unreachable",
            "timed out",
            "timeout",
            "name or service not known",
            "connect call failed",
        )
    )


def ollama_unreachable_user_message(*, settings: Any, channel: str | None) -> str:
    """Short user-visible hint when a turn failed before reaching the model."""
    url = str(getattr(settings, "ollama_base_url", "") or "http://127.0.0.1:11434")
    ch = (channel or "").strip().lower()
    if ch == "telegram":
        return (
            "**Ollama is unreachable** — the turn never reached the model.\n\n"
            f"URL: `{url}`\n"
            f"Check that `ollama serve` is alive (Activity Monitor / `curl {url.rstrip('/')}/api/tags`). "
            "Restart Ollama and send the message again."
        )
    return (
        f"Ollama unavailable at {url}. " "Start `ollama serve` and retry the message."
    )


__all__ = ["is_transient_connection_error", "ollama_unreachable_user_message"]
