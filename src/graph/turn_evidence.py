"""This-turn tool evidence for background write filters.

This is a **lexical prefilter**, not a provenance proof. Token and id overlap
between a write and this turn's evidence is a cheap necessary condition: three
shared words can be coincidence, and a UUID can appear in unrelated text.
Passing the filter means "not obviously invented", never "attributed".
"""

from __future__ import annotations

import contextvars
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

_evidence: contextvars.ContextVar[str] = contextvars.ContextVar(
    "turn_evidence_text", default=""
)

_TOKEN_RE = re.compile(r"[0-9a-z]{3,}", re.IGNORECASE)
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_USER_PREFIX_RE = re.compile(r"^USER:\s*", re.IGNORECASE)
# Tool row that already carried operator speech — optional [id=…] prefix.
_USER_LINE_RE = re.compile(r"(?im)^(?:\[id=[^\]]+\]\s*)?USER:\s*.+")
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "was",
        "are",
        "not",
        "agent",
    }
)


def reset_turn_evidence() -> None:
    _evidence.set("")


def current_evidence() -> str:
    return _evidence.get() or ""


def tool_text_for_evidence(content: Any) -> str:
    """Normalize ToolMessage content to newline facts.

    ``search_memory`` / ``search_by_time`` / ``search_diary`` return a list;
    ToolNode stores ``json.dumps(list)``. Line-anchor filters must see the
    inner ``[id=…] USER:`` rows, not the JSON wrapper.
    """
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            chunk = tool_text_for_evidence(item)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    if not isinstance(content, str):
        content = str(content)
    text = content.strip()
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return text
        if isinstance(parsed, list):
            return tool_text_for_evidence(parsed)
    return text


def note_tool_text(text: Any) -> None:
    chunk = tool_text_for_evidence(text)
    if not chunk:
        return
    prev = current_evidence()
    _evidence.set((prev + "\n" + chunk).strip() if prev else chunk)


def seed_from_messages(messages: list[Any] | None) -> None:
    """Load this-turn Human as ``USER:`` plus ToolMessage bodies after it."""
    from src.utils import human_message_text, is_synthetic_human_text

    human_row = ""
    tool_parts: list[str] = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            text = (human_message_text(m) or "").strip()
            if text and not is_synthetic_human_text(text):
                human_row = f"USER: {text}"
                tool_parts = []
            continue
        if isinstance(m, ToolMessage):
            text = tool_text_for_evidence(getattr(m, "content", ""))
            if text:
                tool_parts.append(text)
    reset_turn_evidence()
    parts = ([human_row] if human_row else []) + tool_parts
    if parts:
        _evidence.set("\n".join(parts))


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _TOKEN_RE.findall((text or "").casefold()):
        if tok not in _STOP:
            out.add(tok)
    return out


def user_prefixed_evidence(evidence: Any) -> str:
    """USER: tool rows only. A USER: write must overlap these, not the web blob."""
    text = tool_text_for_evidence(evidence)
    lines = [
        line.strip() for line in text.splitlines() if _USER_LINE_RE.match(line.strip())
    ]
    return "\n".join(lines)


def lexical_evidence_prefilter(write: str, evidence: str | None = None) -> bool:
    """Lexical/id overlap prefilter. Heuristic, not provenance.

    A write passes when it shares at least three tokens with this turn's
    evidence (or two when the write is short), or shares a UUID. That is a
    necessary condition, not a proof: it filters obvious invention, it does not
    establish that the fact follows from the source.

    ``USER:``-labeled writes overlap against ``USER:`` tool rows only, so a
    claim about the operator cannot borrow a web page.
    """
    text = (write or "").strip()
    src = tool_text_for_evidence(
        evidence if evidence is not None else current_evidence()
    )
    if _USER_PREFIX_RE.match(text):
        src = user_prefixed_evidence(src)
    ev = src.strip()
    if not text or not ev:
        return False
    ids_w = {u.casefold() for u in _UUID_RE.findall(text)}
    ids_e = {u.casefold() for u in _UUID_RE.findall(ev)}
    if ids_w and ids_w & ids_e:
        return True
    tw, te = _tokens(text), _tokens(ev)
    shared = tw & te
    if len(shared) >= 3:
        return True
    if len(tw) <= 6 and len(shared) >= 2:
        return True
    return False


# Backwards-compatible name. Prefer ``lexical_evidence_prefilter`` in new code.
write_has_turn_evidence = lexical_evidence_prefilter


def live_extract_evidence(user_text: str, messages: list[Any] | None = None) -> str:
    """Human this turn as a USER: row plus this-turn tool bodies. Not agent_text."""
    human = (user_text or "").strip()
    parts: list[str] = [f"USER: {human}"] if human else []
    tool_parts: list[str] = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            tool_parts = []
            continue
        if isinstance(m, ToolMessage):
            text = tool_text_for_evidence(getattr(m, "content", ""))
            if text:
                tool_parts.append(text)
    parts.extend(tool_parts)
    return "\n".join(parts)


def invented_user_prefix(write: str, evidence: str | None = None) -> bool:
    """True when write is USER:-labeled but no tool row this turn carried USER:."""
    text = (write or "").strip()
    if not _USER_PREFIX_RE.match(text):
        return False
    ev = evidence if evidence is not None else current_evidence()
    return not bool(user_prefixed_evidence(ev))


__all__ = [
    "current_evidence",
    "invented_user_prefix",
    "lexical_evidence_prefilter",
    "live_extract_evidence",
    "note_tool_text",
    "reset_turn_evidence",
    "seed_from_messages",
    "tool_text_for_evidence",
    "user_prefixed_evidence",
    "write_has_turn_evidence",
]
