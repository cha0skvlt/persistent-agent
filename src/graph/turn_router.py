"""Retrieval routing for one USER turn — what to fetch, never what to say.

The plan decides which lanes ``retrieve_memory`` reads and whether the web slice is
prefetched. Nothing here reaches the prompt as instruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import BaseMessage

from src.config import settings
from src.tools.web_intent import (
    is_web_search_intent_with_context,
    prior_human_texts,
    resolve_page_url,
    suggest_web_search_query,
    wants_url_fetch,
)
from src.utils import last_real_human_text


@dataclass(frozen=True)
class TurnPlan:
    """What this turn needs retrieved — immutable, log-visible."""

    user_text: str
    web_intent: bool
    web_search_query: str | None
    web_fetch_url: str | None
    web_fetch_wanted: bool = False

    @property
    def web_fast_path(self) -> bool:
        """Skip embed retrieve only when skip-memory is on. Prefetch keeps MEM."""
        if not self.web_intent:
            return False
        return bool(settings.web_skip_memory_on_intent)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TurnPlan | None:
        if not data or not isinstance(data, dict):
            return None
        try:
            return cls(
                user_text=str(data.get("user_text") or ""),
                web_intent=bool(data.get("web_intent")),
                web_search_query=(
                    str(data["web_search_query"])
                    if data.get("web_search_query")
                    else None
                ),
                web_fetch_url=(
                    str(data["web_fetch_url"]) if data.get("web_fetch_url") else None
                ),
                web_fetch_wanted=bool(data.get("web_fetch_wanted")),
            )
        except (TypeError, ValueError, KeyError):
            return None


def plan_turn(
    user_text: str,
    messages: list[BaseMessage] | list | None,
) -> TurnPlan:
    """Classify the latest real USER message into a single ``TurnPlan``."""
    text = (user_text or "").strip()
    msgs = list(messages or [])
    prior = prior_human_texts(msgs, exclude_last=True)

    web = is_web_search_intent_with_context(text, msgs)

    search_q: str | None = None
    fetch_url: str | None = None
    fetch_wanted = False
    if web:
        fetch_wanted = wants_url_fetch(text, prior)
        if fetch_wanted:
            fetch_url = resolve_page_url(text, prior)
        else:
            q = suggest_web_search_query(text)
            search_q = q if q else None

    return TurnPlan(
        user_text=text,
        web_intent=web,
        web_search_query=search_q,
        web_fetch_url=fetch_url,
        web_fetch_wanted=fetch_wanted,
    )


def plan_turn_from_state(messages: list) -> TurnPlan:
    """Plan from checkpoint messages (fallback when state lacks ``turn_plan``)."""
    text = last_real_human_text(messages)
    return plan_turn(text, messages)


__all__ = [
    "TurnPlan",
    "plan_turn",
    "plan_turn_from_state",
]
