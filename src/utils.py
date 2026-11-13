"""Small shared helpers (message → text, self-referential query detection)."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# Self-referential user queries (identity / about me) — language-fixed phrases.
# Anchor-specific "who is X" is built in is_self_referential_query from surfaces.
_SYNTHETIC_HUMAN_PREFIXES = ("[Task:", "[Thread:", "[Deixis:")


def human_message_text(msg: HumanMessage) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content).strip()


def is_synthetic_human_text(text: str) -> bool:
    """True for graph re-anchors (``[Task:]``, ``[Thread:]``) — not real USER turns."""
    t = (text or "").strip()
    return any(t.startswith(p) for p in _SYNTHETIC_HUMAN_PREFIXES)


def last_human_text(messages: Sequence[Any]) -> str:
    """Plain text of the last ``HumanMessage`` in ``messages`` (REPL / routing)."""
    last_human = next(
        (m for m in reversed(list(messages)) if isinstance(m, HumanMessage)), None
    )
    if last_human is None:
        return ""
    return human_message_text(last_human)


def last_real_human_text(messages: Sequence[Any]) -> str:
    """Last USER turn, skipping synthetic ``[Task:]`` / ``[Thread:]`` re-anchors."""
    for m in reversed(list(messages)):
        if not isinstance(m, HumanMessage):
            continue
        text = human_message_text(m)
        if is_synthetic_human_text(text):
            continue
        return text
    return ""


def last_real_human_index(messages: Sequence[Any]) -> int | None:
    """Index of the latest real USER ``HumanMessage``, or ``None``."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, HumanMessage):
            continue
        if is_synthetic_human_text(human_message_text(m)):
            continue
        return i
    return None


def _body_from_content(msg: BaseMessage) -> str:
    text = getattr(msg, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    content: Any = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content) if content else ""


def get_message_text(
    msg: BaseMessage,
    *,
    style: Literal["plain", "transcript"] = "plain",
) -> str:
    """Extract human-visible text from a LangChain message.

    ``plain`` — agent-visible text only (non-``AIMessage`` → ``""``), for streaming / reflection.
    ``transcript`` — one log line per message with role prefix (summary LLM input).
    """
    if style == "transcript":
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            body = _body_from_content(msg)
            return f"Tool ({name}): {body}"
        if isinstance(msg, HumanMessage):
            return f"USER: {_body_from_content(msg)}"
        if isinstance(msg, AIMessage):
            from src.speaker import speaker_label

            return f"{speaker_label()}: {_body_from_content(msg)}"
        return f"{type(msg).__name__}: {_body_from_content(msg)}"

    if not isinstance(msg, AIMessage):
        return ""
    return _body_from_content(msg)


__all__ = [
    "get_message_text",
    "human_message_text",
    "is_synthetic_human_text",
    "last_human_text",
    "last_real_human_index",
    "last_real_human_text",
]
