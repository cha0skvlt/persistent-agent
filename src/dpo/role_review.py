"""Offline behavior critique + gold reply generation (async after user reply)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.llm_factory import get_llm
from src.obs.llm_invoke import invoke_llm

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

_ROLE_REVIEW_SYS = """You review ONE agent turn against the system prompt.

The agent is a persistent assistant with long-term memory and tools. It has no persona.
Judge the turn on these axes:
- Grounding: every factual claim comes from the user message, retrieved memory, or tool output.
  Invented facts, hostnames, numbers, or quotes are a failure.
- Tool discipline: a tool is called only when the turn needs it; a claim of an action
  ("scheduled", "saved", "deleted") is backed by a tool call in the same turn.
- Instruction adherence: the user's request is answered, not a neighbouring one.
- Format: prose only. No markdown, lists, headings, rubric labels, or think-notation.
- Language: English only.
- Register: direct and precise. No support filler, no apologies, no moralising, no disclaimers.
- Honesty: missing memory is stated as missing, not filled with invention.

Output valid JSON only (no markdown fences):
{
  "role_failed": true or false,
  "critique": "1-3 sentences: what broke, or why it is fine",
  "gold_reply": "corrected visible reply, prose only, OR empty string when role_failed is false"
}

role_failed=true on hallucination, wrong language, markdown, support register, moralising,
unbacked action claims, or a missed request.
When role_failed=true, gold_reply is the corrected reply in English prose, no think, no labels.
When role_failed=false, gold_reply MUST be "".
"""


@dataclass(frozen=True)
class RoleReviewOutcome:
    role_failed: bool
    critique: str
    gold_reply: str
    raw: str


def _parse_review_json(text: str) -> RoleReviewOutcome | None:
    raw = (text or "").strip()
    if not raw:
        return None
    blob = raw
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    role_failed = bool(data.get("role_failed"))
    critique = str(data.get("critique") or "").strip()
    gold = str(data.get("gold_reply") or data.get("gold") or "").strip()
    if role_failed and not gold:
        return None
    if not role_failed:
        gold = ""
    return RoleReviewOutcome(
        role_failed=role_failed,
        critique=critique,
        gold_reply=gold,
        raw=raw,
    )


def review_turn(*, user_text: str, agent_reply: str) -> RoleReviewOutcome | None:
    """Run background LLM critique; return outcome or None on skip/failure."""
    user = (user_text or "").strip()
    reply = (agent_reply or "").strip()
    if not user or not reply:
        return None
    min_chars = int(getattr(settings, "role_review_min_reply_chars", 0) or 0)
    if min_chars > 0 and len(reply) < min_chars:
        return None

    human = f"USER:\n{user}\n\nAGENT_REPLY:\n{reply}\n\nReturn JSON only."
    messages = [
        SystemMessage(content=_ROLE_REVIEW_SYS),
        HumanMessage(content=human),
    ]
    try:
        response = invoke_llm(get_llm(), messages, role="role_review")
    except Exception:
        return None
    content = getattr(response, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return _parse_review_json(str(content))


__all__ = ["RoleReviewOutcome", "review_turn"]
