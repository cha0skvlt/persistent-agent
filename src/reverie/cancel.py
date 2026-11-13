"""Cooperative cancel for reverie LLM work when a live turn starts."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_active_token: ReverieCancelToken | None = None


class ReverieCancelledError(RuntimeError):
    """A live turn preempted the running background reverie graph pass."""


class ReverieCancelToken:
    """Thread-safe flag checked before each reverie LLM call."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()


def begin_reverie_cycle() -> ReverieCancelToken:
    """Register a new cancel token for the active reverie cycle."""
    global _active_token
    token = ReverieCancelToken()
    with _lock:
        _active_token = token
    return token


def end_reverie_cycle(token: ReverieCancelToken) -> None:
    """Clear the active token when reverie cycle finishes."""
    global _active_token
    with _lock:
        if _active_token is token:
            _active_token = None


def request_reverie_cancel() -> None:
    """Live-turn priority: stop reverie before the next LLM invocation."""
    with _lock:
        if _active_token is not None:
            _active_token.cancel()


def is_reverie_cancel_requested() -> bool:
    """True when the active reverie cycle has been asked to stop."""
    with _lock:
        return _active_token is not None and _active_token.is_set()


def reset_reverie_cancel_state() -> None:
    """Drop active token (pytest)."""
    global _active_token
    with _lock:
        _active_token = None


__all__ = [
    "ReverieCancelToken",
    "ReverieCancelledError",
    "begin_reverie_cycle",
    "end_reverie_cycle",
    "is_reverie_cancel_requested",
    "request_reverie_cancel",
    "reset_reverie_cancel_state",
]
