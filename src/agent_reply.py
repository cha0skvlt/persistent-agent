"""Extract user-facing agent text (REPL + HTTP/Telegram)."""

from __future__ import annotations

import json
import re
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.agent_errors import format_empty_turn_fallback
from src.obs.logs import log_event
from src.obs.metrics import record_reply_extraction_fallback
from src.reply.visible import (
    promote_speech_from_thinking,
    split_redacted_thinking,
    strip_leaked_tool_markup,
    strip_redacted_thinking,
)
from src.utils import get_message_text, last_real_human_index

_HALLUCINATED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "save_memory",
        "save_fact_to_graph",
        "add_triple",
        "search_memory",
        "search_graph",
        "find_related",
        "query_graph",
        "get_current_time",
    }
)

_HALLUCINATED_TOOL_PREFIX_RE = re.compile(
    r'^\s*\{"name"\s*:\s*"([^"]+)"',
)
_NATIVE_THINKING_KEYS = ("reasoning_content", "reasoning", "thinking")
_EMPTY_CONTENT_CHARS = 20
_THINKING_BLOCK_RE = re.compile(
    r"<(?:think(?:ing)?|thought)\b[^>]*>[\s\S]*?</(?:think(?:ing)?|thought)\s*>",
    re.IGNORECASE,
)
_CONTENT_BLOCK_RE = re.compile(
    r"<content\b[^>]*>([\s\S]*?)</content\s*>",
    re.IGNORECASE,
)
_ORPHAN_CLOSE_START_RE = re.compile(
    r"^\s*</(?:think(?:ing)?|thought|content)\s*>\s*",
    re.IGNORECASE,
)
_CLOSE_TAG_RE = re.compile(r"</(?:think(?:ing)?|thought|content)\s*>", re.IGNORECASE)
_ORPHAN_TAG_RE = re.compile(
    r"</?(?:think(?:ing)?|thought|content|redacted_thinking)\b[^>]*>",
    re.IGNORECASE,
)


def is_hallucinated_tool_call(text: str) -> bool:
    """Return True when *text* is a bare JSON tool-call blob, not a real reply."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            name = data.get("name")
            has_params = "parameters" in data or "args" in data
            if isinstance(name, str) and (
                has_params or name in _HALLUCINATED_TOOL_NAMES
            ):
                return True
    except (json.JSONDecodeError, ValueError):
        pass

    m = _HALLUCINATED_TOOL_PREFIX_RE.match(stripped)
    if m:
        name = m.group(1)
        if name in _HALLUCINATED_TOOL_NAMES:
            return True
        if re.search(r'"parameters"\s*:', stripped[:300]):
            return True

    return False


def is_tool_markup_dump(text: str) -> bool:
    """True when content is a Qwen/JSON tool dump with no spoken prose left."""
    from src.reply.visible import split_redacted_thinking, strip_leaked_tool_markup

    stripped = (text or "").strip()
    if not stripped:
        return False
    if is_hallucinated_tool_call(stripped):
        return True
    low = stripped.casefold()
    if "<tool_call" not in low and "<function=" not in low:
        return False
    leftover = strip_leaked_tool_markup(stripped).strip()
    if not leftover:
        return True
    # ``<think>…</think>`` + tool XML still counts as dump (no spoken main).
    _think, main, _trunc = split_redacted_thinking(leftover)
    return not (main or "").strip()


def strip_agent_speaker_prefix(text: str) -> str:
    """Remove leaked wrapper text such as ``{anchor} answers:`` from visible prose."""
    from src.speaker import speaker_prefix_pattern

    pattern = speaker_prefix_pattern()
    out = text or ""
    while True:
        cleaned = pattern.sub("", out, count=1)
        if cleaned == out:
            return cleaned.strip()
        out = cleaned


def _native_thinking(msg: AIMessage) -> str:
    data = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(data, dict):
        for key in _NATIVE_THINKING_KEYS:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _log_extraction_fallback(
    reason: str,
    *,
    raw_content: str,
    raw_reasoning: str,
) -> None:
    log_event(
        "reply.extraction_fallback",
        reason=reason,
        length_raw_content=len(raw_content or ""),
        length_raw_reasoning=len(raw_reasoning or ""),
    )
    record_reply_extraction_fallback(reason)


def _extract_after_last_close_tag(text: str) -> str:
    raw = text or ""
    matches = list(_CLOSE_TAG_RE.finditer(raw))
    if not matches:
        return raw.strip()
    return raw[matches[-1].end() :].strip()


def _strip_inline_thinking_blocks(text: str) -> str:
    """Remove legacy think/thought blocks; unwrap ``<content>`` to inner prose."""
    out = text or ""
    while True:
        cleaned = _THINKING_BLOCK_RE.sub("", out, count=1)
        if cleaned == out:
            break
        out = cleaned
    out = _CONTENT_BLOCK_RE.sub(r"\1", out)
    out = _ORPHAN_TAG_RE.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def resolve_visible_reply_text(
    raw_content: str,
    raw_reasoning: str = "",
) -> tuple[str, str | None]:
    """Pick visible prose from Ollama ``content`` / ``reasoning_content`` (tag surgery only)."""
    content = raw_content or ""
    reasoning = (raw_reasoning or "").strip()
    content_stripped = content.strip()
    fallback: str | None = None

    if len(content_stripped) < _EMPTY_CONTENT_CHARS and reasoning:
        # Legacy: the whole turn arrived inline in one field as
        # ``<think>…</think>answer`` — recover the answer after the close tag.
        if _CLOSE_TAG_RE.search(reasoning):
            visible = strip_agent_speaker_prefix(
                _extract_after_last_close_tag(reasoning)
            )
            return visible, "empty_content"
        # Native Ollama reasoning lives in its own field with the tags already
        # stripped. The answer is ``content``; never promote the raw thought to
        # visible or it leaks into the reply instead of folding into <think>.
        return strip_agent_speaker_prefix(_strip_inline_thinking_blocks(content)), None

    visible = content

    if _ORPHAN_CLOSE_START_RE.match(visible.lstrip()):
        visible = _ORPHAN_CLOSE_START_RE.sub("", visible.lstrip(), count=1).strip()
        fallback = "orphan_close_tag"

    stripped = _strip_inline_thinking_blocks(visible)
    if stripped != visible.strip():
        visible = stripped
    else:
        visible = stripped

    visible = strip_agent_speaker_prefix(visible.strip())
    return visible, fallback


def _split_ai_message(msg: AIMessage) -> tuple[str, str, bool]:
    from src.reply.visible import recover_think_main_split

    raw = get_message_text(msg)
    native_think = _native_thinking(msg)
    visible, fallback = resolve_visible_reply_text(raw, native_think)
    if fallback:
        _log_extraction_fallback(
            fallback,
            raw_content=raw,
            raw_reasoning=native_think,
        )

    # Always run the shared think/reply boundary (tags + planning recovery).
    # Native reasoning is merged with any planning that leaked into content.
    tag_think, main, truncated = split_redacted_thinking(raw)
    if not tag_think.strip() and not truncated:
        # Untagged content: prefer recovered split over raw visible.
        recovered = recover_think_main_split(visible or raw)
        if recovered:
            tag_think, main = recovered
            truncated = False
        else:
            main = strip_agent_speaker_prefix(visible or main)
    elif tag_think.strip() and not main.strip():
        main = strip_agent_speaker_prefix(visible) if visible else main
    else:
        main = strip_agent_speaker_prefix(visible or main)

    # Tool XML/JSON in content is not spoken prose (Telegram strips it later —
    # treating it as visible here skips empty-reply recovery and leaves think-only).
    main = strip_leaked_tool_markup(main) if main else main

    # Tool-dump content + planning think must not become the Telegram reply
    # (promote_from_thinking would leak the English plan as "spoken" prose).
    tool_dump = is_tool_markup_dump(raw)

    if native_think:
        # Prefer native field as think; if content still holds planning+speech,
        # keep recovered main and prepend native to think.
        think = native_think
        if tag_think.strip() and tag_think.strip() not in native_think:
            think = f"{native_think}\n\n{tag_think.strip()}"
        if tool_dump:
            if not (main or "").strip() and len(native_think) >= 80:
                _log_extraction_fallback(
                    "reasoning_unpromoted",
                    raw_content=raw,
                    raw_reasoning=native_think,
                )
            return think, main, False
        promoted = promote_speech_from_thinking(think, main)
        if promoted:
            think, main = promoted
            _log_extraction_fallback(
                "promote_from_thinking",
                raw_content=raw,
                raw_reasoning=native_think,
            )
        elif not (main or "").strip() and len(native_think) >= 80:
            # Answer was generated into reasoning but not recovered — surface the loss.
            _log_extraction_fallback(
                "reasoning_unpromoted",
                raw_content=raw,
                raw_reasoning=native_think,
            )
        return think, main, False

    if not tool_dump:
        promoted = promote_speech_from_thinking(tag_think, main)
        if promoted:
            tag_think, main = promoted
            _log_extraction_fallback(
                "promote_from_thinking",
                raw_content=raw,
                raw_reasoning=tag_think,
            )
    return tag_think, main, truncated


def _ai_message_reply_chunk(msg: AIMessage) -> str:
    think, main, truncated = _split_ai_message(msg)
    if think.strip():
        return _compose_with_thinking(think, main)
    raw = get_message_text(msg)
    if truncated:
        return strip_leaked_tool_markup(raw).strip()
    return main.strip() or strip_leaked_tool_markup(
        strip_agent_speaker_prefix(raw.strip())
    )


def ai_message_visible_prose(msg: AIMessage) -> str:
    """User-visible prose on an AIMessage (no think tags, no tool-call markup).

    Uses the same think/main split as extraction — including
    ``promote_speech_from_thinking`` — so empty-claim recovery does not
    overwrite a spoken cut already sitting in thinking.
    """
    raw = get_message_text(msg).strip()
    native_think = _native_thinking(msg)
    if not raw and not native_think:
        return ""
    if raw and is_tool_markup_dump(raw):
        return ""
    _think, main, _truncated = _split_ai_message(msg)
    visible = (main or "").strip()
    if not visible:
        visible, fallback = resolve_visible_reply_text(raw, native_think)
        if fallback:
            _log_extraction_fallback(
                fallback,
                raw_content=raw,
                raw_reasoning=native_think,
            )
    return strip_leaked_tool_markup(strip_agent_speaker_prefix(visible)).strip()


def _compose_with_thinking(think: str, main: str) -> str:
    think = (think or "").strip()
    main = (main or "").strip()
    if not think:
        return main
    if main:
        return f"<think>{think}</think>\n\n{main}"
    return f"<think>{think}</think>"


def _merge_tool_pass_thinking(pre_tool_raw: str, post_tool_raw: str) -> str:
    """Keep first-pass think + final answer across a tool round."""
    pre_think, pre_main, _ = split_redacted_thinking(pre_tool_raw)
    post_think, post_main, truncated = split_redacted_thinking(post_tool_raw)
    think = pre_think.strip() or post_think.strip()
    main = post_main.strip() or pre_main.strip()
    if truncated and think and not main:
        return _compose_with_thinking(think, "")
    return (
        _compose_with_thinking(think, main)
        or post_tool_raw.strip()
        or pre_tool_raw.strip()
    )


def extract_agent_reply_text(
    messages: Sequence[BaseMessage],
    *,
    preserve_thinking: bool = True,
) -> str:
    """Pick the canonical user-visible reply from checkpoint messages after invoke."""
    text, _, _, _ = _extract_agent_reply_text_with_meta(
        messages, preserve_thinking=preserve_thinking
    )
    return text


def _extract_agent_reply_text_with_meta(
    messages: Sequence[BaseMessage],
    *,
    preserve_thinking: bool = True,
) -> tuple[str, bool, str, str]:
    """Return ``(reply_text, empty_after_extraction, last_raw_content, last_raw_reasoning)``."""
    if not messages:
        return "", False, "", ""

    last_human_idx = last_real_human_index(messages)
    start = (last_human_idx + 1) if last_human_idx is not None else 0

    pre_tool_parts: list[str] = []
    post_tool_candidates: list[tuple[bool, str]] = []
    saw_tool = False
    saw_extractable_ai = False
    last_raw_content = ""
    last_raw_reasoning = ""
    for msg in messages[start:]:
        if isinstance(msg, ToolMessage):
            saw_tool = True
            continue
        if not isinstance(msg, AIMessage):
            continue
        raw = get_message_text(msg)
        native = _native_thinking(msg)
        if raw and is_hallucinated_tool_call(raw):
            continue
        # Pure leaked tool XML/JSON (no prose) is not a reply candidate.
        if raw.strip() and not strip_leaked_tool_markup(raw).strip() and not native:
            saw_extractable_ai = True
            last_raw_content = raw
            last_raw_reasoning = native
            continue
        if raw.strip() or native:
            saw_extractable_ai = True
            last_raw_content = raw
            last_raw_reasoning = native
        chunk = _ai_message_reply_chunk(msg)
        if not chunk:
            continue
        has_tool_calls = bool(getattr(msg, "tool_calls", None))
        if saw_tool:
            post_tool_candidates.append((has_tool_calls, chunk))
        else:
            pre_tool_parts.append(chunk)

    agent_text = ""
    if post_tool_candidates:
        for has_tool_calls, chunk in reversed(post_tool_candidates):
            if not has_tool_calls:
                agent_text = chunk
                break
        else:
            agent_text = post_tool_candidates[-1][1]
        if preserve_thinking and pre_tool_parts:
            agent_text = _merge_tool_pass_thinking("".join(pre_tool_parts), agent_text)
    else:
        agent_text = "".join(pre_tool_parts).strip()

    if preserve_thinking:
        think, main, truncated = split_redacted_thinking(agent_text)
        main = strip_leaked_tool_markup(main)
        if think.strip():
            agent_text = _compose_with_thinking(think, main)
        elif truncated:
            agent_text = strip_leaked_tool_markup(agent_text).strip()
        else:
            agent_text = strip_agent_speaker_prefix(main.strip() or agent_text.strip())
    else:
        agent_text = strip_agent_speaker_prefix(
            strip_leaked_tool_markup(strip_redacted_thinking(agent_text))
        )

    if agent_text and is_hallucinated_tool_call(agent_text):
        return "", bool(saw_extractable_ai), last_raw_content, last_raw_reasoning

    agent_text = agent_text.strip()
    visible_only = strip_agent_speaker_prefix(
        strip_leaked_tool_markup(strip_redacted_thinking(agent_text))
    ).strip()
    if not visible_only:
        return "", bool(saw_extractable_ai), last_raw_content, last_raw_reasoning
    from src.reply.visible import is_empty_claim

    # Void denials are not spoken answers — force empty-turn fallback.
    if is_empty_claim(visible_only):
        return "", bool(saw_extractable_ai), last_raw_content, last_raw_reasoning
    return agent_text, False, last_raw_content, last_raw_reasoning


def resolve_agent_reply_text_with_meta(
    messages: Sequence[BaseMessage],
    *,
    user_message: str = "",
    preserve_thinking: bool = True,
    channel: str | None = None,
) -> tuple[str, bool]:
    """Extract user-visible reply; return ``(text, is_synthetic_fallback)``.

    ``is_synthetic_fallback`` is True for ``format_empty_turn_fallback`` —
    never genuine model prose.
    """
    reply, empty_after, last_raw, last_reasoning = _extract_agent_reply_text_with_meta(
        messages, preserve_thinking=preserve_thinking
    )
    if reply:
        return reply, False
    if empty_after:
        _log_extraction_fallback(
            "empty_after_extraction",
            raw_content=last_raw,
            raw_reasoning=last_reasoning,
        )
    elif not messages:
        return "", False
    else:
        _log_extraction_fallback(
            "no_visible_prose",
            raw_content=last_raw,
            raw_reasoning=last_reasoning,
        )
    return format_empty_turn_fallback(messages, channel=channel), True


def resolve_agent_reply_text(
    messages: Sequence[BaseMessage],
    *,
    user_message: str = "",
    preserve_thinking: bool = True,
    channel: str | None = None,
) -> str:
    """Extract user-visible reply; fall back when the model left no visible prose."""
    reply, _ = resolve_agent_reply_text_with_meta(
        messages,
        user_message=user_message,
        preserve_thinking=preserve_thinking,
        channel=channel,
    )
    return reply


def ollama_response_to_ai_message(payload: dict) -> AIMessage:
    """Build ``AIMessage`` from a ChatOllama-style response dict (tests / fixtures)."""
    content = payload.get("content")
    if content is None:
        content = ""
    kwargs = dict(payload.get("additional_kwargs") or {})
    for key in _NATIVE_THINKING_KEYS:
        if key in payload and key not in kwargs:
            val = payload.get(key)
            if isinstance(val, str):
                kwargs[key] = val
    return AIMessage(content=str(content), additional_kwargs=kwargs)
