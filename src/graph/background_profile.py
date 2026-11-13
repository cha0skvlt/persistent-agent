"""Light background graph profile — fewer SYS/gates for solitary/intent."""

from __future__ import annotations

from src.state.agent_state import AgentState


def is_light_background_profile(state: AgentState) -> bool:
    """True for background turns (solitary/intent; old diary_source rows included)."""
    from src.reverie.turn_kind import is_background_turn

    if is_background_turn():
        return True
    src = str(state.get("diary_source") or "").strip()
    return src in {"solitary", "intent", "vigil"}


__all__ = ["is_light_background_profile"]
