"""Solitary attention clock — wall-clock budget bound to attention_reserve."""

from __future__ import annotations

import contextvars
import time

_start: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "solitary_attention_start", default=None
)

# Base wall-clock for one solitary essay pass (seconds).
_BASE_WALL_SEC = 120.0
# attention_reserve=1.0 shaves this many seconds off the budget.
_RESERVE_SHAVE_SEC = 90.0
_MIN_WALL_SEC = 30.0


def begin_attention_clock() -> contextvars.Token:
    return _start.set(time.perf_counter())


def end_attention_clock(token: contextvars.Token) -> None:
    _start.reset(token)


def solitary_wall_budget_sec(knobs: dict[str, float] | None = None) -> float:
    """Higher attention_reserve → shorter solitary wall-clock (live owns channel)."""
    data = knobs
    if data is None:
        try:
            from src.policy_knobs import effective_knobs

            data = effective_knobs()
        except Exception:
            data = {}
    reserve = float(data.get("attention_reserve", 0.3))
    return max(_MIN_WALL_SEC, _BASE_WALL_SEC - reserve * _RESERVE_SHAVE_SEC)


def solitary_wall_clock_exceeded(knobs: dict[str, float] | None = None) -> bool:
    started = _start.get()
    if started is None:
        # Solitary pass without a clock = fail-closed (no unbounded tool slack).
        try:
            from src.reverie.turn_kind import is_solitary_turn

            return bool(is_solitary_turn())
        except Exception:
            return True
    return (time.perf_counter() - started) >= solitary_wall_budget_sec(knobs)


def solitary_used_attention_cost(
    *,
    tools_used: list[str] | tuple[str, ...] | None,
    elapsed_ms: float,
    knobs: dict[str, float] | None = None,
) -> bool:
    """True when the pass spent scarce attention (multi-tool or long wall-clock)."""
    names = [n for n in (tools_used or ()) if n]
    if len(names) >= 2:
        return True
    budget_ms = solitary_wall_budget_sec(knobs) * 1000.0
    return float(elapsed_ms or 0.0) >= budget_ms * 0.5


__all__ = [
    "begin_attention_clock",
    "end_attention_clock",
    "solitary_used_attention_cost",
    "solitary_wall_budget_sec",
    "solitary_wall_clock_exceeded",
]
