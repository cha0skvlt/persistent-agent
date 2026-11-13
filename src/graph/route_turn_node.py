"""Graph node: work out what this turn needs retrieved (no verdict on the reply)."""

from __future__ import annotations

import time

from src.graph.turn_router import plan_turn_from_state
from src.obs.logs import log_event
from src.obs.spans import traced_node
from src.state.agent_state import AgentState


@traced_node("route_turn")
def route_turn_node(state: AgentState) -> dict:
    log_event("node.start", node="route_turn")
    t0 = time.perf_counter()
    try:
        plan = plan_turn_from_state(list(state.get("messages") or []))
        log_event(
            "turn.route",
            web=plan.web_intent,
            web_fast=plan.web_fast_path,
        )
        return {"turn_plan": plan.to_dict()}
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_event("node.timing", node="route_turn", elapsed_ms=elapsed_ms)


__all__ = ["route_turn_node"]
