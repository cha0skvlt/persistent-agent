"""Promote spoken prose out of reasoning into ``content`` — nothing else.

Reasoning models often burn the generation on thinking and leave ``content``
empty while the actual answer sits in the reasoning field. That is a transport
bug, not a behaviour to correct: the fix is to move the text she already said,
never to send her an instruction about how to answer.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from src.agent_reply import ai_message_visible_prose
from src.obs.logs import log_event
from src.reply.visible import is_empty_claim
from src.utils import get_message_text

# Tool results that mean "the store has nothing". When the model says nothing
# after one of these, the turn would end with an operator error for no reason.
_EMPTY_RESULT_MARKERS = (
    "no triples matched",
    "no entity found",
    "no results",
    "not_found",
    "(empty)",
    "[]",
    "∅",
)


def synthesize_from_tool_result(messages: list) -> str | None:
    """Speak for the model when every tool in the turn returned nothing."""
    bodies = [
        str(m.content or "").strip().lower()
        for m in (messages or [])
        if isinstance(m, ToolMessage)
    ]
    bodies = [b for b in bodies if b]
    if not bodies:
        return None
    if not all(
        any(marker in body for marker in _EMPTY_RESULT_MARKERS) for body in bodies
    ):
        return None
    return "Nothing found. The store has no matching rows for that query."


def materialize_spoken_reply(response: AIMessage) -> AIMessage:
    """Write promoted speech into ``content`` when thinking holds the answer."""
    prose = (ai_message_visible_prose(response) or "").strip()
    if not prose or is_empty_claim(prose):
        return response
    raw = (get_message_text(response) or "").strip()
    if raw == prose:
        return response
    if raw and not is_empty_claim(raw) and not _content_is_think_only(raw):
        # Already has distinct spoken content in the message body.
        return response

    kwargs = dict(getattr(response, "additional_kwargs", None) or {})
    # Keep native reasoning; replace empty/claim/think-only content with prose.
    log_event(
        "empty_reply.materialize",
        prose_chars=len(prose),
        prior_content_chars=len(raw),
    )
    return AIMessage(
        content=prose,
        additional_kwargs=kwargs,
        response_metadata=dict(getattr(response, "response_metadata", None) or {}),
        tool_calls=list(getattr(response, "tool_calls", None) or []),
        id=getattr(response, "id", None),
    )


def _content_is_think_only(raw: str) -> bool:
    from src.reply.visible import split_redacted_thinking

    think, main, _trunc = split_redacted_thinking(raw)
    return bool(think.strip()) and not (main or "").strip()


__all__ = ["materialize_spoken_reply", "synthesize_from_tool_result"]
