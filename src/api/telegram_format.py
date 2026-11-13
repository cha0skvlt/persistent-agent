"""Lightweight Markdown → Telegram HTML (parse_mode=HTML).

VisibleReply think/main cut lives in ``src.reply`` — this module owns
Telegram transport only (HTML, chunking, plain fallback).
"""

from __future__ import annotations

import html
import re

from src.reply.visible import (
    recover_think_main_split,
    sanitize_outreach_text,
    split_redacted_thinking,
    strip_leaked_tool_markup,
    strip_redacted_thinking,
    strip_think_tag_leaks,
)

_FENCE_RE = re.compile(r"```(?:[^\n]*\n)?([\s\S]*?)```")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC_RE = re.compile(
    r"(?<!\*)\*([^*\n]+?)\*(?!\*)|(?<![A-Za-z0-9_])_([^_\n]+?)_(?![A-Za-z0-9_])"
)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^(\s*)\d+[.)]\s+", re.MULTILINE)
_HRULE_RE = re.compile(r"^\s*([-*_]){3,}\s*$", re.MULTILINE)

TELEGRAM_MAX_MESSAGE_LEN = 4096
# Display cap for thinking in one bubble (UTF-16 units). Leave headroom under
# 4096 so expandable blockquote HTML never trips "message is too long".
TELEGRAM_THINK_DISPLAY_MAX = 3500
# Leftover HTML / think markup that must never appear as visible plain text.
_LEAKED_TAG_RE = re.compile(
    r"</?(?:think(?:ing)?|thought|content|redacted_thinking|blockquote|i|em|b|strong|u|s|pre|code|a|tg-spoiler)"
    r"(?:\s[^>]*)?>",
    re.IGNORECASE,
)


def sanitize_leaked_markdown(text: str) -> str:
    """Strip leaked structure markers toward spoken prose (format contract).

    Headers / list markers become plain lines — do not invent emoji, bold, or ▸.
    Lightweight ``**bold**`` / code left for runtime chrome (footers) and rare
    explicit USER-requested structure that already reached the channel.
    """
    raw = (text or "").replace("\r\n", "\n")
    if not raw.strip():
        return raw

    raw = _HRULE_RE.sub("", raw)

    def _header(match: re.Match[str]) -> str:
        title = match.group(2).strip()
        title = re.sub(r"^\*\*(.+)\*\*$", r"\1", title)
        title = re.sub(r"^__(.+?)__$", r"\1", title)
        return title

    raw = _HEADER_RE.sub(_header, raw)
    raw = _BULLET_RE.sub(r"\1", raw)
    raw = _NUMBERED_RE.sub(r"\1", raw)
    return raw


def _strip_think_tags_if_leaked(text: str) -> str:
    """Strip orphan think/thought/content tags only when present — lossless for clean prose."""
    raw = text or ""
    if not raw:
        return raw
    low = raw.lower()
    if not (
        "<think" in low
        or "</think" in low
        or "<thought" in low
        or "</thought" in low
        or "<content" in low
        or "</content" in low
        or "redacted_thinking" in low
    ):
        return raw
    # Prefer full scaffold peel (keeps <content> body) over bare tag strip.
    think, main, _trunc = split_redacted_thinking(raw)
    if main.strip():
        return strip_think_tag_leaks(main)
    if think.strip() and not main.strip():
        return ""
    return strip_think_tag_leaks(raw)


def escape_telegram_html(text: str) -> str:
    return html.escape(text or "", quote=False)


def telegram_utf16_len(text: str) -> int:
    """Telegram 4096 limit is UTF-16 code units after entity parsing."""
    return len((text or "").encode("utf-16-le")) // 2


def truncate_telegram_utf16(text: str, limit: int) -> str:
    """Truncate *text* so its UTF-16 length is ≤ *limit* (no mid-surrogate cut)."""
    raw = text or ""
    if limit <= 0:
        return ""
    if telegram_utf16_len(raw) <= limit:
        return raw
    # Walk codepoints; each non-BMP char costs 2 UTF-16 units.
    out: list[str] = []
    used = 0
    for ch in raw:
        cost = 2 if ord(ch) > 0xFFFF else 1
        if used + cost > limit:
            break
        out.append(ch)
        used += cost
    return "".join(out)


def format_thinking_block(
    think: str, *, truncated: bool = False, italic: bool = True
) -> str:
    """Collapsible blockquote for model thinking (Telegram HTML).

    Bot API: ``<blockquote expandable>…</blockquote>`` → expandable_blockquote
    entity (collapsed by default — the green “thinking” window). Inner text is
    always escaped so ``<``/``>``/``&`` from the model never break parse_mode.
    """
    escaped = escape_telegram_html(think)
    inner = f"<i>{escaped}</i>" if italic else escaped
    block = f"<blockquote expandable>{inner}</blockquote>"
    if truncated:
        note = escape_telegram_html("(Reply truncated — thinking cap reached.)")
        return f"{block}\n<i>{note}</i>" if italic else f"{block}\n{note}"
    return block


def simplify_thinking_html(html_text: str) -> str | None:
    """Drop nested ``<i>`` inside expandable blockquote (parse-retry helper)."""
    raw = (html_text or "").strip()
    if "<blockquote expandable>" not in raw:
        return None
    simplified = re.sub(
        r"(<blockquote expandable>)\s*<i>([\s\S]*?)</i>\s*(</blockquote>)",
        r"\1\2\3",
        raw,
        count=1,
        flags=re.IGNORECASE,
    )
    if simplified == raw:
        return None
    return simplified


def format_footer_for_telegram(footer: str) -> str:
    """Turn log line — bold timing via lightweight markup."""
    clean = (footer or "").strip()
    if not clean:
        return ""
    return format_for_telegram(clean)


def format_for_telegram(text: str) -> str:
    """Convert agent markup to Telegram HTML.

      Supported input (Telegram-native dialect):
      - ``**bold**`` / ``__bold__``
      - ``*italic*`` / ``_italic_``
      - `` `inline` `` monospace
      - `` ``` `` fenced blocks → ``<pre>`` (copy-paste friendly)
    - ``[label](url)`` links
    """
    raw = text or ""
    if not raw.strip():
        return escape_telegram_html(raw)

    raw = sanitize_leaked_markdown(raw)

    blocks: list[str] = []
    inlines: list[str] = []

    def _stash_block(match: re.Match[str]) -> str:
        blocks.append(match.group(1).rstrip("\n"))
        return f"\x00BLOCK{len(blocks) - 1}\x00"

    def _stash_inline(match: re.Match[str]) -> str:
        inlines.append(match.group(1))
        return f"\x00INLINE{len(inlines) - 1}\x00"

    work = _FENCE_RE.sub(_stash_block, raw)
    work = _INLINE_CODE_RE.sub(_stash_inline, work)
    work = escape_telegram_html(work)
    work = _apply_bold(work)
    work = _apply_italic(work)
    work = _apply_links(work)

    for idx, code in enumerate(inlines):
        work = work.replace(
            f"\x00INLINE{idx}\x00",
            f"<code>{escape_telegram_html(code)}</code>",
        )
    for idx, block in enumerate(blocks):
        work = work.replace(
            f"\x00BLOCK{idx}\x00",
            f"<pre>{escape_telegram_html(block)}</pre>",
        )
    return work


def split_telegram_plain_text(
    text: str, *, limit: int = TELEGRAM_MAX_MESSAGE_LEN
) -> list[str]:
    """Split plain text into chunks ≤ *limit* UTF-16 units (Telegram's metric).

    Prefer newline breaks; hard-cut on UTF-16 budget when no newline fits.
    """
    if not text:
        return []
    if telegram_utf16_len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        if telegram_utf16_len(rest) <= limit:
            parts.append(rest)
            break
        # Prefer a newline within the UTF-16 budget.
        window = truncate_telegram_utf16(rest, limit)
        cut = window.rfind("\n")
        if cut <= 0:
            cut = len(window)
            if cut <= 0:
                # Single non-BMP char edge: take one codepoint.
                cut = 1
        parts.append(rest[:cut])
        rest = rest[cut:]
    return parts


def split_main_plain_with_footer(
    main: str,
    footer: str,
    *,
    limit: int = TELEGRAM_MAX_MESSAGE_LEN,
    first_chunk_overhead: int = 0,
) -> list[str]:
    """Split visible main prose; reserve UTF-16 budget for footer on the last chunk.

    *first_chunk_overhead* reduces the budget on the first chunk only.
    """
    if not main:
        return []
    footer_s = (footer or "").strip()
    suffix_len = telegram_utf16_len(f"\n\n{footer_s}") if footer_s else 0
    if suffix_len >= limit:
        return split_telegram_plain_text(main, limit=limit)
    overhead = max(0, first_chunk_overhead)
    first_limit = max(1, limit - overhead)
    if telegram_utf16_len(main) + suffix_len <= first_limit:
        return [main]
    parts: list[str] = []
    rest = main
    chunk_idx = 0
    while rest:
        chunk_limit = first_limit if chunk_idx == 0 else limit
        last_max = chunk_limit - suffix_len if suffix_len < chunk_limit else chunk_limit
        rest_len = telegram_utf16_len(rest)
        if rest_len <= last_max:
            parts.append(rest)
            break
        # Always cut so a later piece + footer can fit: full chunk when rest is
        # longer than chunk_limit; otherwise leave last_max room for the footer.
        cut_budget = chunk_limit if rest_len > chunk_limit else last_max
        window = truncate_telegram_utf16(rest, cut_budget)
        cut = window.rfind("\n")
        if cut <= 0:
            cut = len(window)
            if cut <= 0:
                cut = 1
        parts.append(rest[:cut])
        rest = rest[cut:]
        chunk_idx += 1
    return parts


def _think_plain_overhead(think: str, *, has_main: bool) -> int:
    """UTF-16 units thinking consumes on the first chunk."""
    if not (think or "").strip():
        return 0
    if has_main:
        return telegram_utf16_len(think) + 2
    return telegram_utf16_len(think)


def _main_parts_for_telegram(
    main: str,
    footer: str,
    *,
    think_overhead: int,
    limit: int = TELEGRAM_MAX_MESSAGE_LEN,
) -> list[str]:
    """Split main prose; first chunk budget reduced by *think_overhead* (UTF-16)."""
    footer_s = (footer or "").strip()
    suffix_len = telegram_utf16_len(f"\n\n{footer_s}") if footer_s else 0
    footer_inline = bool(footer_s and suffix_len < limit)
    if footer_inline:
        return split_main_plain_with_footer(
            main,
            footer_s,
            limit=limit,
            first_chunk_overhead=think_overhead,
        )
    first_limit = max(1, limit - think_overhead)
    if telegram_utf16_len(main) <= first_limit:
        return [main]
    first_parts = split_telegram_plain_text(main, limit=first_limit)
    first = first_parts[0]
    rest = main[len(first) :]
    return [first] + (split_telegram_plain_text(rest, limit=limit) if rest else [])


def _plain_parsed_len(text: str) -> int:
    """Visible character count in UTF-16 units (Telegram's 4096 metric)."""
    return telegram_utf16_len(text or "")


def _append_footer_chunks(
    chunks: list[tuple[str, str]], footer: str, *, limit: int
) -> None:
    for part in split_telegram_plain_text(footer, limit=limit):
        chunks.append(_compose_body_chunk("", part))


def _compose_body_chunk(
    think: str,
    main: str,
    footer: str = "",
    *,
    truncated: bool = False,
) -> tuple[str, str]:
    """HTML + parsed plain for one chunk (thinking only — never mixed with main)."""
    think_raw = think or ""
    main_s = main or ""
    footer_s = (footer or "").strip()
    html_parts: list[str] = []
    plain_parts: list[str] = []

    if think_raw.strip():
        html_parts.append(format_thinking_block(think_raw, truncated=truncated))
        plain_parts.append(think_raw)
    elif truncated and not main_s.strip():
        html_parts.append(format_thinking_block("", truncated=True))
        plain_parts.append("(Reply truncated — thinking cap reached.)")

    if main_s.strip():
        html_parts.append(format_for_telegram(main_s))
        plain_parts.append(main_s)

    body_html = "\n\n".join(html_parts) if html_parts else ""
    body_plain = "\n\n".join(plain_parts) if plain_parts else ""

    if footer_s:
        foot_html = format_footer_for_telegram(footer_s)
        html_out = f"{body_html}\n\n{foot_html}" if body_html else foot_html
        plain_out = f"{body_plain}\n\n{footer_s}" if body_plain else footer_s
        return html_out, plain_out

    if body_html:
        return body_html, body_plain
    return escape_telegram_html(body_plain), body_plain


def _compose_main_chunk(main: str, footer: str) -> tuple[str, str]:
    """HTML + parsed plain for one main-only chunk (optionally with footer)."""
    return _compose_body_chunk("", main, footer)


def _append_thinking_chunks(
    chunks: list[tuple[str, str]],
    think: str,
    *,
    truncated: bool,
    limit: int,
) -> None:
    """Emit thinking as its own bubble(s) — never packed with the spoken reply.

    Mirrors web/Cursor UX: reasoning is a separate collapsed panel; the answer
    is a separate message. Packing think+main caused HTML 400 "too long" →
    plain fallback that dumped reasoning outside the green block.
    """
    raw = (think or "").strip()
    if not raw:
        if truncated:
            chunks.append(_compose_body_chunk("", "", truncated=True))
        return

    display_cap = min(limit, TELEGRAM_THINK_DISPLAY_MAX)
    shown = truncate_telegram_utf16(raw, display_cap)
    was_capped = shown != raw or truncated
    # One bubble for capped think (common); split only if somehow still huge.
    for i, part in enumerate(split_telegram_plain_text(shown, limit=limit)):
        trunc_flag = was_capped and i == 0
        chunks.append(_compose_body_chunk(part, "", truncated=trunc_flag))


def telegram_message_chunks(
    body: str,
    footer: str = "",
    *,
    limit: int = TELEGRAM_MAX_MESSAGE_LEN,
    outreach: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(html, plain_after_parsing)`` — one ``sendMessage`` per chunk.

    Thinking is **always** its own message(s) with ``<blockquote expandable>``
    (Telegram expandable_blockquote). Main prose + footer follow in later
    messages — same separation as Cursor/web reasoning panels.

    When *outreach* is True, strip ``<think>`` and leaked tool markup entirely —
    proactive posts are finished letters, not live debug bubbles.
    """
    raw_body = body or ""
    if outreach:
        cleaned = sanitize_outreach_text(raw_body)
        if not cleaned:
            return []
        think, main, truncated = "", cleaned, False
    else:
        think, main, truncated = split_redacted_thinking(raw_body)
        main = strip_leaked_tool_markup(main)
        think = strip_leaked_tool_markup(think) if think else think
        # Orphan ``<think>`` / close tags must never reach USER as visible text.
        main = _strip_think_tags_if_leaked(main)
        think = _strip_think_tags_if_leaked(think) if think else think
    # Size chunks against the SANITIZED main (headers/list markers stripped to
    # prose before HTML build — Telegram 4096 counts visible UTF-16 units).
    main = sanitize_leaked_markdown(main)
    footer_s = (footer or "").strip()
    suffix_len = telegram_utf16_len(f"\n\n{footer_s}") if footer_s else 0
    footer_inline = bool(footer_s and suffix_len < limit)
    chunks: list[tuple[str, str]] = []

    main_text = main
    has_main = bool(main_text.strip())
    leading_think = think if (think or "").strip() else ""

    if leading_think or (truncated and not has_main):
        _append_thinking_chunks(chunks, leading_think, truncated=truncated, limit=limit)

    if has_main:
        main_parts = _main_parts_for_telegram(
            main_text,
            footer_s,
            think_overhead=0,
            limit=limit,
        )
        for i, part in enumerate(main_parts):
            foot = footer_s if footer_inline and i == len(main_parts) - 1 else ""
            html, plain = _compose_body_chunk("", part, foot)
            chunks.append((html, plain))
        if footer_s and not footer_inline:
            _append_footer_chunks(chunks, footer_s, limit=limit)
    elif footer_s and chunks:
        # Thinking-only turn: footer as its own trailing bubble.
        _append_footer_chunks(chunks, footer_s, limit=limit)
    elif not chunks:
        # Never re-emit raw body after outreach sanitize emptied it.
        fallback = (main_text or raw_body).strip() or "(empty)"
        fallback = _strip_think_tags_if_leaked(fallback)
        if footer_inline:
            html, plain = _compose_main_chunk(fallback, footer_s)
            chunks.append((html, plain))
        else:
            html, plain = _compose_main_chunk(fallback, "")
            chunks.append((html, plain))
            if footer_s:
                _append_footer_chunks(chunks, footer_s, limit=limit)

    for _html, plain in chunks:
        if _plain_parsed_len(plain) > limit:
            raise ValueError(
                f"Telegram chunk exceeds {limit} parsed chars "
                f"(utf16={_plain_parsed_len(plain)})"
            )
    return chunks


def plain_fallback_text(text: str) -> str:
    """Strip lightweight markdown / leaked tags when HTML ``parse_mode`` is rejected."""
    raw = sanitize_leaked_markdown(text or "")
    if not raw.strip():
        return raw

    def _drop_bold(match: re.Match[str]) -> str:
        return match.group(1) or match.group(2) or ""

    def _drop_italic(match: re.Match[str]) -> str:
        return match.group(1) or match.group(2) or ""

    raw = _BOLD_RE.sub(_drop_bold, raw)
    raw = _ITALIC_RE.sub(_drop_italic, raw)
    raw = _INLINE_CODE_RE.sub(lambda m: m.group(1), raw)
    raw = strip_think_tag_leaks(raw)
    raw = _LEAKED_TAG_RE.sub("", raw)
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def should_fallback_plain(err: str) -> bool:
    """True when a Telegram HTML send failed for a reason a plain retry can fix."""
    low = (err or "").lower()
    return (
        "can't parse" in low
        or "parse entities" in low
        or "message is too long" in low
        or "message_too_long" in low
    )


def telegram_chunk_plain_coverage(
    body: str,
    footer: str = "",
    *,
    max_len: int = TELEGRAM_MAX_MESSAGE_LEN,
) -> tuple[str, str, str]:
    """Return ``(recovered_think, recovered_main, footer)`` — verifies lossless split."""
    think, main, truncated = split_redacted_thinking(body or "")
    # Mirror the chunker: sizing/splitting happens on sanitized main; think alone.
    main = sanitize_leaked_markdown(main)
    main = _strip_think_tags_if_leaked(main)
    think = _strip_think_tags_if_leaked(think) if think else think
    footer_s = (footer or "").strip()
    suffix_len = telegram_utf16_len(f"\n\n{footer_s}") if footer_s else 0
    footer_inline = bool(footer_s and suffix_len < max_len)
    chunks = telegram_message_chunks(body, footer, limit=max_len)

    has_main = bool(main.strip())
    display_cap = min(max_len, TELEGRAM_THINK_DISPLAY_MAX)
    think_shown = truncate_telegram_utf16(think, display_cap) if think.strip() else ""
    think_truncated = bool(think.strip()) and (
        think_shown != think.strip() or truncated
    )

    main_parts = (
        _main_parts_for_telegram(main, footer_s, think_overhead=0, limit=max_len)
        if has_main
        else []
    )

    idx = 0
    recovered_think = ""
    recovered_main: list[str] = []
    recovered_footer = ""

    if think_shown or (truncated and not has_main):
        think_parts = (
            split_telegram_plain_text(think_shown, limit=max_len)
            if think_shown
            else [""]
        )
        for i, part in enumerate(think_parts):
            _html, plain = chunks[idx]
            assert len(plain) <= max_len or telegram_utf16_len(plain) <= max_len
            if i == 0:
                recovered_think = part
            else:
                recovered_think += part
            idx += 1
        if think_truncated and think.strip():
            # Coverage only sees the displayed prefix when capped.
            recovered_think = think_shown

    if main_parts:
        for i, part in enumerate(main_parts):
            _html, plain = chunks[idx]
            foot = footer_s if footer_inline and i == len(main_parts) - 1 else ""
            expected = f"{part}\n\n{foot}" if foot else part
            assert plain == expected, (plain[:80], expected[:80])
            recovered_main.append(part)
            if foot:
                recovered_footer = foot
            idx += 1
    elif footer_s and idx < len(chunks):
        _html, plain = chunks[idx]
        if plain == footer_s:
            recovered_footer = footer_s

    return (recovered_think, "".join(recovered_main), recovered_footer)


def compose_telegram_message(body: str, footer: str = "") -> tuple[str, str]:
    """Return ``(html, plain_fallback)`` for one Telegram message chunk."""
    think, main, truncated = split_redacted_thinking(body)
    # Prefer thinking-only compose when present (matches send path separation).
    if think.strip() and not main.strip():
        html, plain = _compose_body_chunk(think, "", footer, truncated=truncated)
        return html, plain
    if think.strip():
        # Combined helper for tests: thinking block + main in one HTML string.
        html, plain = _compose_body_chunk(think, main, footer, truncated=truncated)
        return html, plain
    html, plain = _compose_body_chunk("", main, footer, truncated=truncated)
    return html, plain


def _apply_bold(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        inner = match.group(1) or match.group(2) or ""
        return f"<b>{inner}</b>"

    return _BOLD_RE.sub(_repl, text)


def _apply_italic(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        inner = match.group(1) or match.group(2) or ""
        return f"<i>{inner}</i>"

    return _ITALIC_RE.sub(_repl, text)


def _apply_links(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        label = match.group(1)
        # ``text`` is already HTML-escaped; re-escaping href doubles ``&`` → ``&amp;amp;``.
        href = match.group(2).strip()
        return f'<a href="{href}">{label}</a>'

    return _LINK_RE.sub(_repl, text)


# Shim re-exports — prefer ``src.reply`` for new call sites.
__all__ = [
    "TELEGRAM_MAX_MESSAGE_LEN",
    "TELEGRAM_THINK_DISPLAY_MAX",
    "compose_telegram_message",
    "escape_telegram_html",
    "format_footer_for_telegram",
    "format_for_telegram",
    "format_thinking_block",
    "plain_fallback_text",
    "recover_think_main_split",
    "sanitize_leaked_markdown",
    "sanitize_outreach_text",
    "should_fallback_plain",
    "simplify_thinking_html",
    "split_main_plain_with_footer",
    "split_redacted_thinking",
    "split_telegram_plain_text",
    "strip_leaked_tool_markup",
    "strip_redacted_thinking",
    "strip_think_tag_leaks",
    "telegram_chunk_plain_coverage",
    "telegram_message_chunks",
    "telegram_utf16_len",
    "truncate_telegram_utf16",
]
