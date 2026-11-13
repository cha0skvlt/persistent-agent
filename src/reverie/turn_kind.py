"""TurnKind — single API for live / solitary / intent turn typing."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from enum import Enum
from typing import Iterator


class TurnKind(str, Enum):
    LIVE = "live"
    SOLITARY = "solitary"
    INTENT = "intent"


_turn_kind: contextvars.ContextVar[TurnKind] = contextvars.ContextVar(
    "turn_kind",
    default=TurnKind.LIVE,
)


def current_turn_kind() -> TurnKind:
    return _turn_kind.get()


def is_background_turn() -> bool:
    return current_turn_kind() is not TurnKind.LIVE


def is_intent_turn() -> bool:
    return current_turn_kind() is TurnKind.INTENT


def is_solitary_turn() -> bool:
    return current_turn_kind() is TurnKind.SOLITARY


@contextmanager
def turn_kind(kind: TurnKind) -> Iterator[TurnKind]:
    """Set TurnKind for the duration of a graph pass."""
    token = _turn_kind.set(kind)
    try:
        yield kind
    finally:
        _turn_kind.reset(token)


# --- Backward-compat shims (old contextvar API) ---


def is_reverie_background_pass() -> bool:
    return is_background_turn()


def is_intent_background_pass() -> bool:
    return is_intent_turn()


class reverie_background_pass:
    """Compat: marks a background solitary (or generic reverie) pass."""

    def __enter__(self) -> reverie_background_pass:
        # If already INTENT, keep INTENT (nested with intent_background_pass).
        current = current_turn_kind()
        kind = TurnKind.INTENT if current is TurnKind.INTENT else TurnKind.SOLITARY
        self._token = _turn_kind.set(kind)
        return self

    def __exit__(self, *_exc: object) -> None:
        _turn_kind.reset(self._token)


class intent_background_pass:
    """Compat: marks a scheduled-intent graph pass."""

    def __enter__(self) -> intent_background_pass:
        self._token = _turn_kind.set(TurnKind.INTENT)
        return self

    def __exit__(self, *_exc: object) -> None:
        _turn_kind.reset(self._token)


__all__ = [
    "TurnKind",
    "current_turn_kind",
    "is_background_turn",
    "is_intent_turn",
    "is_solitary_turn",
    "turn_kind",
    "is_reverie_background_pass",
    "is_intent_background_pass",
    "reverie_background_pass",
    "intent_background_pass",
]
