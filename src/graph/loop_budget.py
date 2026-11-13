"""When may the agent keep looping — measured by progress, not by length.

A counter is the wrong brake. "search → read → act → verify → act again" is
ordinary work whose length is unknown in advance, and cutting it at a fixed
round number is what forced the USER to keep saying "keep going". So the
round cap is demoted to a runaway backstop and the real stop condition becomes
*stalling*: the turn ends when it stops producing anything new.

Three stop signals, in the order they are cheap to check:

* ``wall_clock`` — a live turn may not run longer than the USER will wait.
* ``max_rounds`` — the existing per-turn cap, now a far backstop.
* ``stall`` — the loop is spinning: a tool batch it already ran is coming
  round again, or several consecutive rounds returned nothing usable.

The decision is mechanical: tools stay bound or they do not. Nothing is said
to the model about it — the graph does not narrate its own brakes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import settings
from src.graph.tool_round_limit import tool_round_cap_reached
from src.state.agent_state import AgentState

# Tool output that means "this round bought nothing".
#
# Matched as a PREFIX, never as a substring: a web result about "error handling"
# or "empty tables" is useful material, not a failed round. Only the structured
# failure strings the tools actually emit count.
_FRUITLESS_PREFIXES = (
    "browse_error:",
    "lesson_error:",
    "lesson_empty:",
    "deixis_error:",
    "list_failed:",
    "cancel_failed:",
    "distill_rejected:",
    "not scheduled:",
    "not sent:",
    "not_found",
    "empty_content",
    "no valid ids",
    "no results",
    "web search unavailable",
    "http 4",
    "http 5",
    "[]",
)


@dataclass(frozen=True)
class LoopDecision:
    """Whether tools stay bound this pass, and why not (for logs only)."""

    allow_tools: bool
    stop_reason: str | None = None


def round_is_fruitless(results: list[str]) -> bool:
    """True when nothing in this round's output looks like usable material."""
    if not results:
        return True
    for text in results:
        body = (text or "").strip().lower()
        if not body:
            continue
        if any(body.startswith(prefix) for prefix in _FRUITLESS_PREFIXES):
            continue
        return False
    return True


def _stall_reason(ledger: list[dict]) -> str | None:
    if not ledger:
        return None

    repeat_cap = max(2, int(settings.loop_stall_repeat_threshold))
    seen: dict[str, int] = {}
    for entry in ledger:
        sig = str(entry.get("sig") or "")
        if not sig:
            continue
        seen[sig] = seen.get(sig, 0) + 1
        if seen[sig] >= repeat_cap:
            # Anywhere in the turn, not just back-to-back: A→B→A→B never
            # trips an adjacent-only check and loops forever.
            return "stall_repeat"

    fruitless_cap = max(1, int(settings.loop_stall_fruitless_rounds))
    streak = 0
    for entry in reversed(ledger):
        if entry.get("fruitless"):
            streak += 1
            if streak >= fruitless_cap:
                return "stall_fruitless"
        else:
            break
    return None


def loop_decision(state: AgentState, *, is_background: bool = False) -> LoopDecision:
    """Decide whether the tool loop may continue for another round."""
    if tool_round_cap_reached(state):
        return LoopDecision(allow_tools=False, stop_reason="max_rounds")

    limit = float(settings.loop_wall_clock_seconds or 0)
    started = state.get("turn_started_at")
    if started:
        elapsed = max(0.0, time.monotonic() - float(started))
        # Background passes keep their own clock (solitary wall-clock guard).
        if not is_background and limit > 0 and elapsed >= limit:
            return LoopDecision(allow_tools=False, stop_reason="wall_clock")

    ledger = list(state.get("tool_ledger") or [])
    stalled = _stall_reason(ledger)
    if stalled:
        return LoopDecision(allow_tools=False, stop_reason=stalled)

    return LoopDecision(allow_tools=True)


__all__ = [
    "LoopDecision",
    "loop_decision",
    "round_is_fruitless",
]
