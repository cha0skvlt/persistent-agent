"""Per-turn cap on agent→tools loops."""

from __future__ import annotations

import logging
from typing import Any

from src import config
from src.state.agent_state import AgentState

_log = logging.getLogger(__name__)


def tool_round_cap_reached(state: AgentState) -> bool:
    cap = int(config.settings.max_tool_rounds_per_turn)
    if cap <= 0:
        return False
    # Background passes (solitary/intent) need slack: memory → web →
    # message_user, plus a failed message_user attempt + corrected retry
    # already exceed a live-turn cap of 2.
    from src.reverie.background import is_reverie_background_pass

    if is_reverie_background_pass():
        try:
            from src.reverie.attention import solitary_wall_clock_exceeded
            from src.reverie.turn_kind import is_solitary_turn

            if is_solitary_turn() and solitary_wall_clock_exceeded():
                return True
        except Exception as exc:
            _log.warning("solitary wall-clock check failed: %s", exc)
        bg = int(config.settings.reverie_max_tool_rounds)
        if bg > 0:
            cap = bg
        try:
            from src.policy_knobs import (
                curiosity_extra_tool_rounds,
                essay_extra_tool_rounds,
            )

            # Essay slack subsumes curiosity extra — do not stack +1/+2.
            essay = int(essay_extra_tool_rounds())
            if essay > 0:
                cap += essay
            else:
                cap += int(curiosity_extra_tool_rounds())
        except Exception as exc:
            _log.warning("curiosity/essay tool rounds failed: %s", exc)
    else:
        try:
            from src.graph.turn_router import TurnPlan

            plan = TurnPlan.from_dict(state.get("turn_plan"))
            # Web multi-hop: +1 on web_intent; full browse bonus only after browse_web.
            if plan is not None and plan.web_intent:
                from langchain_core.messages import HumanMessage, ToolMessage

                cap += 1
                used_browse = False
                for m in reversed(list(state.get("messages") or [])):
                    if isinstance(m, HumanMessage):
                        break
                    if (
                        isinstance(m, ToolMessage)
                        and str(getattr(m, "name", "") or "") == "browse_web"
                    ):
                        used_browse = True
                        break
                if used_browse:
                    bonus = int(
                        getattr(config.settings, "web_browse_extra_tool_rounds", 3) or 0
                    )
                    if bonus > 0:
                        cap += max(0, bonus - 1)
        except Exception as exc:
            _log.warning("web tool-cap bind failed: %s", exc)
    return int(state.get("tool_rounds", 0) or 0) >= cap


def _round_ledger_entry(state: AgentState, messages: list[Any]) -> dict[str, Any]:
    """One compact record of what this round ran and whether it bought anything."""
    from src.graph.loop_budget import round_is_fruitless

    names: list[str] = []
    results: list[str] = []
    for m in messages or []:
        name = str(getattr(m, "name", "") or "")
        if name:
            names.append(name)
        content = getattr(m, "content", "")
        results.append(content if isinstance(content, str) else str(content))

    # Signature keys the batch by tool names + a digest of the arguments the
    # agent chose, so "same call again" is detectable anywhere in the turn.
    args_digest = ""
    last = (state.get("messages") or [])[-1:] if state.get("messages") else []
    for m in last:
        calls = getattr(m, "tool_calls", None) or []
        parts = []
        for tc in calls:
            if isinstance(tc, dict):
                parts.append(f"{tc.get('name')}:{tc.get('args')}")
        args_digest = "|".join(sorted(parts))
    sig = ",".join(sorted(names)) + "#" + str(hash(args_digest) & 0xFFFFFFF)

    return {
        "round": int(state.get("tool_rounds", 0) or 0) + 1,
        "tools": names,
        "sig": sig,
        "fruitless": round_is_fruitless(results),
        "preview": (results[0][:160] if results else ""),
    }


def merge_tool_round_increment(
    state: AgentState, out: dict[str, Any] | Any
) -> dict[str, Any]:
    rounds = int(state.get("tool_rounds", 0) or 0) + 1
    produced = out.get("messages") if isinstance(out, dict) else None
    ledger = list(state.get("tool_ledger") or [])
    ledger.append(_round_ledger_entry(state, list(produced or [])))
    # Bounded: the ledger is a memory aid, not a transcript.
    ledger = ledger[-40:]
    if isinstance(out, dict):
        return {**out, "tool_rounds": rounds, "tool_ledger": ledger}
    return {"tool_rounds": rounds, "tool_ledger": ledger}
