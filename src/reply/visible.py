"""VisibleReply boundary: think/main split and outreach sanitize.

Architectural — not Telegram-specific. Live, API, outbox, and diary all use
the same cut so planning never leaks as spoken prose.
"""

from __future__ import annotations

import re

_THINKING_OPEN_RE = re.compile(r"^<think(?:ing)?>", re.IGNORECASE)
_THINKING_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_THINKING_ANY_TAG_RE = re.compile(
    r"</?(?:think(?:ing)?|thought|content|redacted_thinking)\b[^>]*>",
    re.IGNORECASE,
)
# Model sometimes invents XML scaffold: <thought>…</thought><content>…</content>
_THOUGHT_BLOCK_RE = re.compile(
    r"(?is)<thought\b[^>]*>(.*?)</thought\s*>",
)
_CONTENT_BLOCK_RE = re.compile(
    r"(?is)<content\b[^>]*>(.*?)</content\s*>",
)

# Meta-planning that models dump into content without a closing </think>
# (or without tags at all). Used to recover the think/reply boundary.
_PLANNING_CUE_RE = re.compile(
    r"(?im)^(?:\s*)(?:"
    r"the user (?:asks|is asking|said|wants|wrote|is|questions)|"
    r"the (?:draft|reply|response|answer|message)\b|"
    r"i (?:need|want|should|have|had|will|'ll|can|could|must) (?:to |going to )?\w|"
    r"i'?m (?:going|gonna|about) (?:to )?\w|"
    r"i'?ll\b|"
    r"let me\b|"
    r"let'?s\b|"
    r"this (?:explains|is better|works|fits|means|feels|version|draft|avoids|keeps|removes|cuts|drops|is cleaner|is stronger|is more)|"
    r"that (?:explains|works|fits|means)|"
    r"one (?:minor |small |last )?(?:adjustment|tweak|change|note)|"
    r"make it (?:more|slightly|less|it|sound)|"
    r"key points to hit|"
    r"plan\s*:|"
    r"drafting(?:\s+thoughts)?|"
    r"draft(?:ing)?\s*:|"
    r"refining the voice|"
    r"(?:final|last|double)(?:\s|-)?check|"
    r"final(?:izing|ize|\s+version|\s+answer|\s+polish)?\b|"
    # Draft-cycle editorial markers (model narrates its own revisions).
    r"even (?:shorter|better|sharper|simpler|punchier|tighter)|"
    r"(?:shorter|sharper|tighter|punchier|cleaner) and (?:sharper|shorter|tighter|punchier|cleaner|better)|"
    r"(?:last |final )?polish\b|polishing\b|"
    r"maybe\s*:|"
    r"wait[,.]|"
    r"hmm\b|"
    r"actually,\s|"
    r"on second thought|"
    r"it'?s (?:a bit |too |quite |still )?(?:repetitive|redundant|better|cleaner|stronger|long|much|vulgar)|"
    r"too (?:long|much|vulgar|repetitive|formal|casual)\b|"
    r"let'?s combine|"
    r"self-correction|"
    r"text structure\s*:|"
    r"looking at (?:the mind hub|the timeline|my)|"
    r"checking (?:my|the)\b|"
    r"okay,?\s+(?:so|let|i)\b|"
    r"so,?\s+i\b|"
    r"turn contract\s*:|"
    r"constraints?\s*:|"
    r"check constraints|"
    r"execute tool|"
    r"no markdown|"
    r"no [\"']?i am ai|"
    r"final polish"
    r")"
)
_SPEECH_START_RE = re.compile(r"(?m)^(?:" r"[«\"'(\[]*" r"[A-Z]" r"[^\n]{15,}" r")$")

# Model sometimes dumps XML / Qwen-style tool calls into visible content.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call\b[^>]*>[\s\S]*?</tool_call\s*>",
    re.IGNORECASE,
)
_TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call\b[^>]*>[\s\S]*$", re.IGNORECASE)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function\s*=\s*[^>]+>[\s\S]*?</function\s*>",
    re.IGNORECASE,
)
_FUNCTION_UNCLOSED_RE = re.compile(r"<function\s*=\s*[^>]+>[\s\S]*$", re.IGNORECASE)
_PARAMETER_TAG_RE = re.compile(
    r"</?parameter\s*=\s*[^>]+>",
    re.IGNORECASE,
)
_INVOKE_TOOL_RE = re.compile(
    r"<invoke\b[^>]*>[\s\S]*?</invoke\s*>",
    re.IGNORECASE,
)
# Bare JSON tool-call blobs that slipped past extraction.
_JSON_TOOL_BLOB_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"(?:parameters|args)\"\s*:\s*\{[\s\S]*?\}\s*\}",
)
# Bare ``name(args)`` calls: a model without native tool calling writes the call
# as prose. Only known tool names are stripped, so ordinary prose survives.
_TOOL_NAMES_ALT = (
    r"search_memory|save_memory|memory_replace|forget_memory|forget_memories|"
    r"search_diary|diary_replace|forget_diary|"
    r"search_graph|query_graph|find_related|"
    r"search_by_time|search_web|fetch_url|browse_web|"
    r"schedule_intent|list_intents|cancel_intent|"
    r"run_selftest|message_user|record_lesson|search_lessons"
)
_BARE_TOOL_CALL_RE = re.compile(rf"(?m)^\s*(?:{_TOOL_NAMES_ALT})\s*\([^\n]*\)\s*$")
# The model also echoes results as prose: ``search_memory` -> `[]``.
_BARE_TOOL_RESULT_RE = re.compile(
    rf"(?m)^\s*`?(?:{_TOOL_NAMES_ALT})`?\s*(?:->|→|:)\s*`?[^\n]*$"
)
# And sometimes just the name on its own line, backticked or bare.
_BARE_TOOL_NAME_LINE_RE = re.compile(rf"(?m)^\s*`?(?:{_TOOL_NAMES_ALT})`?\s*$")
# The model also labels the raw result: ``tool_output: []``.
_TOOL_OUTPUT_LABEL_RE = re.compile(r"(?im)^\s*tool_output\s*:\s*.*$")
# And narrates the call: ``Tool: `save_memory` `` / ``Args: `text="…"` ``.
_TOOL_LABEL_RE = re.compile(
    rf"(?im)^\s*(?:"
    rf"tool\s*:\s*`?(?:{_TOOL_NAMES_ALT})`?"
    rf"|args\s*:\s*`[^`]*`"
    rf"|text\s*:\s*`[^`]*`"
    rf")\s*$"
)
# ``save_memory USER: …`` — a tool name followed by a labeled payload, no parens.
_BARE_TOOL_NAME_PREFIX_RE = re.compile(
    rf"(?m)^\s*`?(?:{_TOOL_NAMES_ALT})`?\s+(?:USER|AGENT):.*$"
)
# ``Reply: "…"`` — keep the text, drop the label.
_REPLY_LABEL_RE = re.compile(r"(?im)^\s*reply\s*:\s*")
# Internal scaffolding that must never reach USER (line-start or mid-line leak).
_TURN_CONTRACT_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:Turn contract\s*:)[^\n]*",
)
_TURN_CONTRACT_INLINE_RE = re.compile(
    r"(?is)\b(?:Turn contract\s*:)[^\n]*",
)
_MOTIVE_LABEL_RE = re.compile(
    r"(?im)(?:^|\n)\s*motive\s*=\s*\w+[^\n]*",
)


def strip_think_tag_leaks(text: str) -> str:
    """Remove orphan think/thought/content markup tags."""
    out = _THINKING_ANY_TAG_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _split_thought_content_scaffold(text: str) -> tuple[str, str, bool] | None:
    """Handle invented ``<thought>`` / ``<content>`` XML scaffold.

    Returns ``(think, main, truncated)`` when those tags are present; else None.
    """
    raw = (text or "").lstrip()
    low = raw.lower()
    if "<thought" not in low and "<content" not in low:
        return None
    think_m = _THOUGHT_BLOCK_RE.search(raw)
    content_m = _CONTENT_BLOCK_RE.search(raw)
    if not think_m and not content_m:
        return "", strip_think_tag_leaks(raw), False

    think = (think_m.group(1) if think_m else "").strip()
    if content_m:
        main = strip_think_tag_leaks(content_m.group(1) or "")
        return think, main, False

    assert think_m is not None
    after = strip_think_tag_leaks(raw[think_m.end() :])
    if after.strip():
        return think, after, False
    recovered = recover_think_main_split(think)
    if recovered:
        return recovered[0], recovered[1], False
    return think, "", False


def _has_planning_signal(text: str) -> bool:
    return bool(_PLANNING_CUE_RE.search(text or ""))


_EMPTY_CLAIM_RE = re.compile(
    r"(?is)^\s*(?:(?:well|ok|okay|so|hmm)[,!.\s]+)*(?:"
    r"nothing\b|"
    r"nothing there\b|"
    r"nothing found\b|"
    r"i (?:found|see) nothing\b|"
    r"there is nothing\b|"
    r"no (?:data|records|notes|information|facts|memories|mentions)\s+(?:about|on)\b"
    r")"
)

# Spoken cut buried under English process labels (leaked into thinking/content).
_FINAL_POLISH_BODY_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:\d{1,2}\.\s*)?\*{0,2}Final\s+Polish[^*\n]*\*{0,2}:?\s*\n+"
    r"(.+?)"
    r"(?=\n\s*(?:\d{1,2}\.\s*)?\*{0,2}Check\s+Constraints|\n\s*\*\s*No\s|\Z)"
)


def is_empty_claim(text: str) -> bool:
    """True when visible prose is a false emptiness denial (not a real answer)."""
    raw = (text or "").strip()
    if not raw:
        return False
    if not _EMPTY_CLAIM_RE.search(raw):
        return False
    if len(raw) <= 280:
        return True
    # Longer waffle that still opens as a void denial with almost no residue.
    residue = _EMPTY_CLAIM_RE.sub("", raw, count=1).strip(" .,!—-–…")
    return len(residue) < 40


def salvage_final_polish_prose(text: str) -> str | None:
    """Pull spoken prose out of a Final Polish process section."""
    m = _FINAL_POLISH_BODY_RE.search(text or "")
    if not m:
        return None
    body = (m.group(1) or "").strip()
    if len(body) < 40:
        return None
    # Drop trailing checklist crumbs.
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            if lines:
                lines.append("")
            continue
        if _PLANNING_CUE_RE.match(s) or re.match(r"^\*\s*No\b", s, re.I):
            break
        if re.match(r"^\d{1,2}\.\s*\*\*", s):
            break
        lines.append(s)
    out = "\n".join(lines).strip()
    return out if len(out) >= 40 else None


def trailing_spoken_from_reasoning(think: str) -> str | None:
    """Recover spoken prose parked in native reasoning when ``content`` is empty.

    Reasoning models often finish the answer inside ``thinking`` /
    ``reasoning_content`` and leave ``content`` blank. That text is not
    "empty LLM output" — it is lost if we only accept Final Polish labels.
    """
    raw = (think or "").strip()
    if len(raw) < 24:
        return None
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not parts:
        return None

    i = len(parts)
    while i > 0 and _is_speech_paragraph(parts[i - 1]):
        i -= 1
    if i < len(parts):
        tail = "\n\n".join(parts[i:]).strip()
        if len(tail) >= 24 and not is_empty_claim(tail):
            return tail

    # Whole block is speech (model dumped the reply into thinking).
    if i == 0:
        tail = "\n\n".join(parts).strip()
        if (
            len(tail) >= 40
            and _planning_line_density(tail) <= 0.25
            and not is_empty_claim(tail)
        ):
            return tail

    # Single dense paragraph without blank-line splits.
    if (
        len(parts) == 1
        and _is_speech_paragraph(parts[0])
        and not is_empty_claim(parts[0])
    ):
        return parts[0]
    return None


def promote_speech_from_thinking(think: str, main: str) -> tuple[str, str] | None:
    """When thinking holds the real answer and main is empty/claim/scaffold, promote it."""
    think_t = (think or "").strip()
    main_t = (main or "").strip()
    if not think_t:
        return None
    main_bad = (
        not main_t
        or is_empty_claim(main_t)
        or (
            _has_planning_signal(main_t)
            and not _is_speech_paragraph(main_t.split("\n\n", 1)[0])
        )
    )
    if not main_bad:
        return None

    polished = salvage_final_polish_prose(think_t)
    if polished:
        # Keep think without the promoted body when possible.
        head = think_t
        idx = think_t.find(polished)
        if idx > 0:
            head = think_t[:idx].rstrip()
        return head, polished
    recovered = recover_think_main_split(think_t)
    if recovered and recovered[1].strip():
        return recovered[0], recovered[1]
    # Native Ollama: answer often lives only in reasoning_content with no tags.
    trailing = trailing_spoken_from_reasoning(think_t)
    if trailing:
        head = think_t
        idx = think_t.rfind(trailing)
        if idx > 0:
            head = think_t[:idx].rstrip()
        elif trailing == think_t:
            head = ""
        return head, trailing
    return None


# Planning that can appear anywhere inside a paragraph, not only at its start.
# A paragraph carrying any of these is a plan, never a spoken reply.
_PLANNING_INLINE_RE = re.compile(
    r"(?i)(?:"
    r"\bi (?:need|want|should|will|'ll|can|could|must|am going|'m going) to\b|"
    r"\bi (?:will|should|need|must)\b|"
    r"\bi'll\b|\bi'm going to\b|"
    r"\blet me\b|\blet's\b|"
    r"\bthe user (?:asks|is asking|said|wants|wrote|questions)\b|"
    r"\bbased on the prompt\b|\bthe prompt (?:says|explicitly|includes|instructs)\b|"
    r"\bthen i will\b|\bso i will\b|\bnow i will\b|"
    r"\bwait,|\bactually,|\bon second thought\b|"
    r"\bresponse should\b|\bi will (?:mention|verify|also|keep|draft|synthesize)\b|"
    r"\blet's (?:do that|check|perform|run)\b|"
    r"\bi should (?:read|check|search|verify|confirm|mention)\b"
    r")"
)


def _is_speech_paragraph(para: str) -> bool:
    """True when a paragraph looks like spoken reply, not meta-planning.

    Deliberately strict: a false negative costs one speak pass, a false positive
    shows the operator the model's plan as if it were the answer.
    """
    p = (para or "").strip()
    if len(p) < 20:
        return False
    if _PLANNING_CUE_RE.match(p):
        return False
    if _PLANNING_INLINE_RE.search(p):
        return False
    if _SPEECH_START_RE.match(p):
        return True
    return False


def _planning_line_density(text: str) -> float:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _PLANNING_CUE_RE.match(ln))
    return hits / len(lines)


def _split_on_speech_suffix(parts: list[str]) -> tuple[str, str] | None:
    """Earliest cut where the entire remainder is speech → longest final reply."""
    for i in range(1, len(parts)):
        tail_parts = parts[i:]
        if not all(_is_speech_paragraph(p) for p in tail_parts):
            continue
        head = "\n\n".join(parts[:i]).strip()
        tail = "\n\n".join(tail_parts).strip()
        if len(head) < 40 or len(tail) < 20:
            continue
        if not _has_planning_signal(head):
            continue
        if _planning_line_density(tail) > 0.25:
            continue
        return head, tail
    return None


def recover_think_main_split(text: str) -> tuple[str, str] | None:
    """Recover ``(think, main)`` when the model mixed planning into the reply.

    Planning cues in the prefix → take the longest trailing speech suffix.
    """
    raw = (text or "").strip()
    if len(raw) < 80:
        return None

    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(parts) < 2:
        return None

    # No planning evidence anywhere → never reclassify prose as think.
    if not _has_planning_signal(raw):
        return None

    return _split_on_speech_suffix(parts)


def strip_leaked_tool_markup(text: str) -> str:
    """Remove hallucinated tool-call XML/JSON from user-visible prose.

    Unchanged text is returned as-is (no ``.strip()``) so chunk coverage stays
    lossless for clean prose.
    """
    out = text or ""
    if not out:
        return ""
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", out)
    cleaned = _TOOL_CALL_UNCLOSED_RE.sub("", cleaned)
    cleaned = _FUNCTION_BLOCK_RE.sub("", cleaned)
    cleaned = _FUNCTION_UNCLOSED_RE.sub("", cleaned)
    cleaned = _INVOKE_TOOL_RE.sub("", cleaned)
    cleaned = _PARAMETER_TAG_RE.sub("", cleaned)
    cleaned = _JSON_TOOL_BLOB_RE.sub("", cleaned)
    cleaned = _BARE_TOOL_CALL_RE.sub("", cleaned)
    cleaned = _BARE_TOOL_RESULT_RE.sub("", cleaned)
    cleaned = _BARE_TOOL_NAME_LINE_RE.sub("", cleaned)
    cleaned = _TOOL_OUTPUT_LABEL_RE.sub("", cleaned)
    cleaned = _TOOL_LABEL_RE.sub("", cleaned)
    cleaned = _BARE_TOOL_NAME_PREFIX_RE.sub("", cleaned)
    cleaned = _REPLY_LABEL_RE.sub("", cleaned)
    cleaned = _TURN_CONTRACT_RE.sub("\n", cleaned)
    cleaned = _TURN_CONTRACT_INLINE_RE.sub("", cleaned)
    cleaned = _MOTIVE_LABEL_RE.sub("\n", cleaned)
    # Leftover bare tags if the model truncated mid-call.
    cleaned = re.sub(
        r"</?(?:tool_call|function|parameter|invoke)\b[^>]*>",
        "",
        cleaned,
        flags=re.I,
    )
    if cleaned == out:
        return out
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def sanitize_outreach_text(text: str) -> str:
    """Proactive body: visible prose only — no think, no tool dumps.

    Returns empty string when nothing user-facing remains (caller should
    suppress send/queue).
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    visible = strip_redacted_thinking(raw)
    return strip_leaked_tool_markup(visible)


def split_redacted_thinking(text: str) -> tuple[str, str, bool]:
    """Split Qwen3 ``<think>`` prefix from the user-visible reply.

    Returns ``(think, main, truncated)``. ``truncated`` is True when the open tag
    has no closing tag *and* no recoverable reply boundary (runaway thinking).
    Also peels invented ``<thought>`` / ``<content>`` XML scaffold.
    """
    scaffold = _split_thought_content_scaffold(text)
    if scaffold is not None:
        return scaffold

    raw = (text or "").lstrip()
    open_m = _THINKING_OPEN_RE.match(raw)
    if open_m:
        rest = raw[open_m.end() :]
        close_m = _THINKING_CLOSE_RE.search(rest)
        if close_m:
            think = rest[: close_m.start()]
            main = strip_think_tag_leaks(rest[close_m.end() :])
            if not main.strip():
                recovered = recover_think_main_split(think)
                if recovered:
                    return recovered[0], recovered[1], False
            elif _has_planning_signal(main) and not _is_speech_paragraph(
                main.split("\n\n", 1)[0]
            ):
                # Close tag too early / planning continued into "main".
                recovered = recover_think_main_split(f"{think}\n\n{main}")
                if recovered:
                    return recovered[0], recovered[1], False
            return think, main, False
        recovered = recover_think_main_split(rest)
        if recovered:
            return recovered[0], recovered[1], False
        return rest, "", True

    # Model sometimes emits closing tag without opening (template-implicit think).
    close_m = _THINKING_CLOSE_RE.search(raw)
    if close_m:
        think = raw[: close_m.start()]
        main = strip_think_tag_leaks(raw[close_m.end() :])
        if not main.strip():
            recovered = recover_think_main_split(think)
            if recovered:
                return recovered[0], recovered[1], False
        return think, main, False

    if _THINKING_ANY_TAG_RE.search(raw):
        return "", strip_think_tag_leaks(raw), False

    # No tags — still recover when content is clearly planning + spoken reply.
    recovered = recover_think_main_split(raw)
    if recovered:
        return recovered[0], recovered[1], False
    return "", raw, False


def strip_redacted_thinking(text: str) -> str:
    """Drop Qwen3 ``<think>`` blocks; keep final reply only.

    For model context / history — not display (use ``split_redacted_thinking``).
    """
    think, main, truncated = split_redacted_thinking(text)
    if truncated:
        return ""
    if main.strip():
        return main.strip()
    if think.strip():
        return ""
    return strip_think_tag_leaks(text or "")


__all__ = [
    "is_empty_claim",
    "promote_speech_from_thinking",
    "recover_think_main_split",
    "salvage_final_polish_prose",
    "sanitize_outreach_text",
    "split_redacted_thinking",
    "strip_leaked_tool_markup",
    "strip_redacted_thinking",
    "strip_think_tag_leaks",
    "trailing_spoken_from_reasoning",
]
