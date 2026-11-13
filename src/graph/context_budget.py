"""Agent prompt token budgeting — coarse estimates, SYS block caps, overflow sacrifice."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.config import settings
from src.obs.logs import log_event
from src.utils import get_message_text, human_message_text, is_synthetic_human_text

TRUNCATION_MARKER = "[…truncated]"
TRUNCATION_SUFFIX = f"\n{TRUNCATION_MARKER}"

SysCategory = Literal[
    "core",
    "summary",
    "mem",
    "web",
    "pulse",
    "prepared",
    "diary",
    "mind",
]

# Sacrifice first what we can lose; keep LIVE WEB EVIDENCE longest.
# diary → mind → summary → mem → pulse → prepared → web
_SACRIFICE_RANK: dict[SysCategory, int | None] = {
    "core": None,
    "diary": 0,
    "mind": 1,
    "summary": 2,
    "mem": 3,
    "pulse": 4,
    "prepared": 5,
    "web": 6,
}

_PROMPT_RESERVE_TOKENS = 1024
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def count_cyrillic_chars(text: str) -> int:
    """Count Cyrillic letters for script-weighted token estimates."""
    return len(_CYRILLIC_RE.findall(text or ""))


def estimate_tokens(text: str) -> int:
    """Script-weighted token estimate (Latin vs Cyrillic chars-per-token).

    Cyrillic is denser for Qwen (~2–2.5 chars/token); a flat ``len//3`` under-counts
    Russian sessions and misses ``context.overflow_risk``. Coefficients come from
    Settings (``TOKEN_EST_CPT_*``); defaults are conservative (slightly over-estimate).
    """
    raw = text or ""
    if not raw:
        return 0
    cyr = count_cyrillic_chars(raw)
    non_cyr = len(raw) - cyr
    cpt_latin = float(settings.token_est_cpt_latin) or 3.4
    cpt_cyr = float(settings.token_est_cpt_cyrillic) or 2.0
    est = non_cyr / cpt_latin + cyr / cpt_cyr
    return int(est) + 1


def estimate_token_script_split(text: str) -> tuple[int, int]:
    """Return ``(est_tokens_latin, est_tokens_cyrillic)`` contributions for logging."""
    raw = text or ""
    if not raw:
        return 0, 0
    cyr = count_cyrillic_chars(raw)
    non_cyr = len(raw) - cyr
    cpt_latin = float(settings.token_est_cpt_latin) or 3.4
    cpt_cyr = float(settings.token_est_cpt_cyrillic) or 2.0
    return int(non_cyr / cpt_latin), int(cyr / cpt_cyr)


def estimate_prompt_tokens(text: str) -> int:
    """Alias for :func:`estimate_tokens` (historical name used by callers/tests)."""
    return estimate_tokens(text)


def truncate_tail(text: str, max_chars: int) -> str:
    """Trim from the end of a block, preserving a truncation marker."""
    if max_chars <= 0:
        return TRUNCATION_MARKER
    body = text or ""
    if len(body) <= max_chars:
        return body
    suffix = TRUNCATION_SUFFIX
    keep = max_chars - len(suffix)
    if keep <= 0:
        return TRUNCATION_MARKER
    return body[:keep].rstrip() + suffix


def prompt_token_budget() -> int:
    """Input token budget before generation (ctx − predict − reserve)."""
    return max(
        0,
        int(settings.llm_num_ctx)
        - int(settings.llm_num_predict)
        - _PROMPT_RESERVE_TOKENS,
    )


@dataclass
class SysBlock:
    category: SysCategory
    text: str

    @property
    def sacrifice_rank(self) -> int | None:
        return _SACRIFICE_RANK[self.category]


@dataclass(frozen=True)
class ContextBreakdown:
    system: int
    mem: int
    web: int
    prepared: int
    summary: int
    diary: int
    messages: int

    @property
    def total_chars(self) -> int:
        return (
            self.system
            + self.mem
            + self.web
            + self.prepared
            + self.summary
            + self.diary
            + self.messages
        )

    def as_token_fields(self) -> dict[str, int]:
        return {
            "system_chars": self.system,
            "mem_chars": self.mem,
            "web_chars": self.web,
            "prepared_chars": self.prepared,
            "summary_chars": self.summary,
            "diary_chars": self.diary,
            "messages_chars": self.messages,
        }


def _message_chars(messages: list[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        total += len(get_message_text(msg))
    return total


def _combined_prompt_text(
    blocks: list[SysBlock],
    messages: list[BaseMessage],
) -> str:
    sys_text = blocks_to_system_text(blocks)
    msg_text = "".join(get_message_text(m) for m in messages)
    return sys_text + msg_text


def _total_estimated_tokens(
    blocks: list[SysBlock],
    messages: list[BaseMessage],
) -> int:
    return estimate_tokens(_combined_prompt_text(blocks, messages))


def compute_context_breakdown(
    blocks: list[SysBlock],
    messages: list[BaseMessage],
) -> ContextBreakdown:
    system = mem = web = prepared = summary = diary = 0
    for block in blocks:
        n = len(block.text or "")
        if block.category == "mem":
            mem += n
        elif block.category == "web":
            web += n
        elif block.category == "prepared":
            prepared += n
        elif block.category == "summary":
            summary += n
        elif block.category == "diary":
            diary += n
        else:
            system += n
    return ContextBreakdown(
        system=system,
        mem=mem,
        web=web,
        prepared=prepared,
        summary=summary,
        diary=diary,
        messages=_message_chars(messages),
    )


def apply_sys_char_budgets(blocks: list[SysBlock]) -> list[SysBlock]:
    """Per-block char caps (truncate tail); protected categories unchanged."""
    mem_cap = int(settings.sys_mem_max_chars)
    summary_cap = int(settings.sys_summary_max_chars)
    web_cap = int(settings.web_evidence_max_chars) + 700  # header + prefetch labels
    pulse_cap = int(settings.pulse_sys_max_chars)
    mind_cap = int(settings.sys_mind_max_chars)
    diary_cap = int(settings.sys_diary_max_chars)

    out: list[SysBlock] = []
    for block in blocks:
        text = block.text
        if block.category == "mem" and mem_cap > 0:
            text = truncate_tail(text, mem_cap)
        elif block.category == "summary" and summary_cap > 0:
            text = truncate_tail(text, summary_cap)
        elif block.category == "web" and web_cap > 0:
            text = truncate_tail(text, web_cap)
        elif block.category == "pulse" and pulse_cap > 0:
            text = truncate_tail(text, pulse_cap)
        elif block.category == "mind" and mind_cap > 0:
            text = truncate_tail(text, mind_cap)
        elif block.category == "diary" and diary_cap > 0:
            text = truncate_tail(text, diary_cap)
        out.append(SysBlock(category=block.category, text=text))
    return out


def _protected_tail_start(messages: list[BaseMessage]) -> int:
    """Index of first protected synthetic tail HumanMessage (deixis / thread / task)."""
    start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and is_synthetic_human_text(
            human_message_text(msg)
        ):
            start = i
        else:
            break
    return start


def _trim_messages_tail(
    messages: list[BaseMessage],
    *,
    chars_target: int,
) -> list[BaseMessage]:
    """Drop oldest non-system, non-protected-tail messages until under char target."""
    if not messages or chars_target <= 0:
        return messages

    sys_msgs: list[BaseMessage] = []
    body = list(messages)
    if body and isinstance(body[0], SystemMessage):
        sys_msgs = [body[0]]
        body = body[1:]

    tail_start = _protected_tail_start(body)
    droppable = body[:tail_start]
    protected = body[tail_start:]

    while droppable and _message_chars(sys_msgs + droppable + protected) > chars_target:
        droppable = droppable[1:]
        # Dropping an AIMessage(tool_calls) must take its ToolMessages with it —
        # an orphaned tool result at the window head breaks the chat template.
        while droppable and isinstance(droppable[0], ToolMessage):
            droppable = droppable[1:]
        if not droppable:
            break

    return sys_msgs + droppable + protected


def _trim_sacrifice_block(block: SysBlock, *, fraction: float = 0.25) -> SysBlock:
    text = block.text or ""
    if not text:
        return block
    new_len = max(0, int(len(text) * (1.0 - fraction)))
    if new_len >= len(text):
        return SysBlock(category=block.category, text="")
    return SysBlock(category=block.category, text=truncate_tail(text, new_len))


def enforce_prompt_token_budget(
    blocks: list[SysBlock],
    messages: list[BaseMessage],
) -> tuple[list[SysBlock], list[BaseMessage]]:
    """Log overflow risk and trim in sacrifice order; core SYS blocks stay intact."""
    budget = prompt_token_budget()
    blocks = apply_sys_char_budgets(blocks)

    tokens = _total_estimated_tokens(blocks, messages)
    if tokens <= budget:
        return blocks, messages

    breakdown = compute_context_breakdown(blocks, messages)
    est_latin, est_cyr = estimate_token_script_split(
        _combined_prompt_text(blocks, messages)
    )
    log_event(
        "context.overflow_risk",
        estimated_tokens=tokens,
        token_budget=budget,
        est_tokens_latin=est_latin,
        est_tokens_cyrillic=est_cyr,
        **breakdown.as_token_fields(),
    )

    working_blocks = list(blocks)
    working_messages = list(messages)
    # Token→char fallback for message trim: Latin CPT (same order of magnitude as old *3).
    cpt_latin = float(settings.token_est_cpt_latin) or 3.4

    for _ in range(32):
        if _total_estimated_tokens(working_blocks, working_messages) <= budget:
            break

        sacrificed = False
        rank_groups: dict[int, list[int]] = {}
        for idx, block in enumerate(working_blocks):
            rank = block.sacrifice_rank
            if rank is None or not (block.text or "").strip():
                continue
            rank_groups.setdefault(rank, []).append(idx)

        for rank in sorted(rank_groups):
            for idx in reversed(rank_groups[rank]):
                before = working_blocks[idx].text
                working_blocks[idx] = _trim_sacrifice_block(working_blocks[idx])
                if working_blocks[idx].text != before:
                    sacrificed = True
                    break
            if sacrificed:
                break

        if not sacrificed:
            prev_tokens = _total_estimated_tokens(working_blocks, working_messages)
            char_budget = int(budget * cpt_latin)
            working_messages = _trim_messages_tail(
                working_messages, chars_target=char_budget
            )
            if _total_estimated_tokens(working_blocks, working_messages) >= prev_tokens:
                break

    return working_blocks, working_messages


def blocks_to_system_text(blocks: list[SysBlock]) -> str:
    parts = [b.text.strip() for b in blocks if (b.text or "").strip()]
    return "\n\n".join(parts)


def inject_system_message(
    blocks: list[SysBlock],
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Prepend a single SystemMessage built from blocks (drops prior system)."""
    body = [m for m in messages if not isinstance(m, SystemMessage)]
    sys_text = blocks_to_system_text(blocks)
    if not sys_text:
        return list(body)
    return [SystemMessage(content=sys_text)] + body


def core_system_text(blocks: list[SysBlock]) -> str:
    """Protected persona / identity blocks — must survive overflow trimming."""
    return "\n\n".join(
        b.text.strip()
        for b in blocks
        if b.category == "core" and (b.text or "").strip()
    )


__all__ = [
    "ContextBreakdown",
    "SysBlock",
    "SysCategory",
    "TRUNCATION_MARKER",
    "TRUNCATION_SUFFIX",
    "apply_sys_char_budgets",
    "blocks_to_system_text",
    "compute_context_breakdown",
    "core_system_text",
    "count_cyrillic_chars",
    "enforce_prompt_token_budget",
    "estimate_prompt_tokens",
    "estimate_token_script_split",
    "estimate_tokens",
    "inject_system_message",
    "prompt_token_budget",
    "truncate_tail",
]
