"""Tests for Telegram HTML formatting."""

from __future__ import annotations

import pytest

from src.api.telegram_format import (
    TELEGRAM_MAX_MESSAGE_LEN,
    compose_telegram_message,
    escape_telegram_html,
    format_footer_for_telegram,
    format_for_telegram,
    format_thinking_block,
    sanitize_outreach_text,
    split_redacted_thinking,
    strip_leaked_tool_markup,
    strip_redacted_thinking,
    telegram_chunk_plain_coverage,
    telegram_message_chunks,
    telegram_utf16_len,
)


@pytest.mark.unit
def test_escape_telegram_html():
    assert escape_telegram_html("a & b <c>") == "a &amp; b &lt;c&gt;"


@pytest.mark.unit
def test_format_bold_and_inline_code():
    out = format_for_telegram("**Brevity** — `YOBA`")
    assert "<b>Brevity</b>" in out
    assert "<code>YOBA</code>" in out
    assert "**" not in out


@pytest.mark.unit
def test_format_fenced_block():
    out = format_for_telegram("before\n```\nline1\nline2\n```\nafter")
    assert "<pre>line1\nline2</pre>" in out
    assert "before" in out
    assert "after" in out


@pytest.mark.unit
def test_format_italic_and_link():
    out = format_for_telegram("*tone* and [wiki](https://example.com/x)")
    assert "<i>tone</i>" in out
    assert '<a href="https://example.com/x">wiki</a>' in out


@pytest.mark.unit
def test_format_footer_monospace():
    out = format_footer_for_telegram("**→ turn 1.0s**  search_web, message_user")
    assert "<b>→ turn 1.0s</b>" in out
    assert "search_web" in out
    assert "message_user" in out
    assert "<i>" not in out


@pytest.mark.unit
def test_compose_telegram_message():
    html_out, plain = compose_telegram_message("**Hi**", "**→ turn 1s**")
    assert "<b>Hi</b>" in html_out
    assert "<b>→ turn 1s</b>" in html_out


@pytest.mark.unit
def test_sanitize_markdown_headers_and_lists():
    from src.api.telegram_format import sanitize_leaked_markdown

    out = sanitize_leaked_markdown("### Kabbalah\n\nText.\n- item")
    assert "###" not in out
    assert "Kabbalah" in out
    assert "⚙️" not in out
    assert "**Kabbalah**" not in out
    assert "▸" not in out
    assert "item" in out
    assert sanitize_leaked_markdown("   ") == "   "


@pytest.mark.unit
def test_format_strips_leaked_headers():
    out = format_for_telegram("### Kabbalah\n\n**Essence** — yes.")
    assert "###" not in out
    assert "⚙️" not in out
    assert "Kabbalah" in out
    assert "<b>Essence</b>" in out


@pytest.mark.unit
def test_format_empty():
    assert format_for_telegram("   ") == "   "


@pytest.mark.unit
def test_split_redacted_thinking_closed():
    body = (
        "<think>Briefly: he is asking.\nDeciding to answer.</think>\n" "agent online."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert "Briefly" in think
    assert main == "agent online."


@pytest.mark.unit
def test_split_redacted_thinking_unclosed():
    body = "<think>still thinking about"
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is True
    assert "still thinking" in think
    assert main == ""


@pytest.mark.unit
def test_format_thinking_block_escapes_html():
    think = "if x < 5 && y > 0"
    out = format_thinking_block(think)
    assert "&lt;" in out
    assert "&gt;" in out
    assert "<blockquote expandable>" in out
    assert "<i>" in out


@pytest.mark.unit
def test_strip_redacted_thinking():
    body = "<think>Q: status.\n→ alive.</think>\n" "agent online."
    assert strip_redacted_thinking(body) == "agent online."
    assert strip_redacted_thinking("plain reply") == "plain reply"
    assert strip_redacted_thinking("<think>runaway") == ""
    assert strip_redacted_thinking("<think>x</think>") == ""
    assert strip_redacted_thinking("<think>\n\n</think>\n\nHi") == "Hi"


@pytest.mark.unit
def test_split_thought_content_scaffold():
    body = (
        "<thought>Self-check green, arsenal 22.</thought>\n"
        "<content>Self-check clean. No new tools surfaced. "
        "Waiting for a command.</content>"
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert "arsenal 22" in think
    assert main.startswith("Self-check clean")
    assert "<thought>" not in main and "<content>" not in main
    assert strip_redacted_thinking(body) == main
    html_out, plain = compose_telegram_message(body)
    assert "<thought>" not in html_out and "<content>" not in html_out
    assert "Self-check clean" in plain
    assert "<thought>" not in plain


@pytest.mark.unit
def test_compose_telegram_message_with_thinking():
    body = "<think>debug thought</think>\n" "**Hello**, USER."
    html_out, plain = compose_telegram_message(body)
    assert "<blockquote expandable>" in html_out
    assert "<i>debug thought</i>" in html_out
    assert "<b>Hello</b>" in html_out
    assert "<think>" not in html_out
    # Plain fallback must never carry think tags (they show as literal text).
    assert "<think>" not in plain
    assert "debug thought" in plain
    assert "Hello" in plain


@pytest.mark.unit
def test_split_orphan_closing_think_tag():
    body = "</think>\n\nYou are right. I am not Brahman."
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert think == ""
    assert main == "You are right. I am not Brahman."
    assert "</think>" not in main


@pytest.mark.unit
def test_compose_orphan_closing_think_tag_not_in_html():
    body = "</think>\n\nYou are right."
    html_out, plain = compose_telegram_message(body)
    assert "</think>" not in html_out
    assert "You are right." in html_out
    assert "<blockquote" not in html_out


@pytest.mark.unit
def test_split_think_without_open_tag():
    body = "Q: role check\n→ fix tone</think>\n\nAnswer."
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert "role check" in think
    assert main == "Answer."


@pytest.mark.unit
def test_compose_telegram_message_thinking_only_truncated():
    body = "<think>runaway"
    html_out, plain = compose_telegram_message(body)
    assert "runaway" in html_out
    assert "truncated" in html_out.lower()
    assert "<blockquote expandable>" in html_out
    # Plain fallback is tag-free (never leak <think> into Telegram plain mode).
    assert "<think>" not in plain
    assert "runaway" in plain


@pytest.mark.unit
def test_telegram_message_chunks_thinking_separate_from_reply():
    """Thinking is always its own expandable bubble; reply follows (Cursor-like)."""
    body = (
        "<think>planning selftest</think>\n\n"
        "**Three risks for the week**:\n1. **Database overload**: text"
    )
    footer = "**→ turn 2m 14s**  run_selftest"
    chunks = telegram_message_chunks(body, footer)
    assert len(chunks) == 2
    think_html, think_plain = chunks[0]
    main_html, main_plain = chunks[1]
    assert "<blockquote expandable>" in think_html
    assert "planning selftest" in think_plain
    assert "Three risks" not in think_plain
    assert "<b>Three risks for the week</b>" in main_html
    assert "**" not in main_html
    assert "planning selftest" not in main_plain
    assert "Three risks" in main_plain
    assert "<think>" not in think_plain
    assert "<think>" not in main_plain


@pytest.mark.unit
def test_plain_fallback_text_strips_markdown():
    from src.api.telegram_format import plain_fallback_text

    out = plain_fallback_text("**bold** and *keep_alive*")
    assert "**" not in out
    assert "bold" in out
    assert "keep_alive" in out


@pytest.mark.unit
def test_plain_fallback_strips_leaked_think_and_html_tags():
    from src.api.telegram_format import plain_fallback_text

    out = plain_fallback_text(
        "<think>secret plan</think>\n\n"
        "<blockquote expandable><i>leaked</i></blockquote>\n"
        "Visible answer"
    )
    assert "<think>" not in out
    assert "</think>" not in out
    assert "<blockquote" not in out
    assert "<i>" not in out
    assert "Visible answer" in out


@pytest.mark.unit
def test_thinking_always_own_message_never_packed_with_main():
    """Regression: packed think+main → HTML 400 too long → plain dump outside green."""
    think = ("Reasoning step. " * 100).strip()
    main = ("Answer to the user. " * 80).strip()
    body = f"<think>{think}</think>\n\n{main}"
    chunks = telegram_message_chunks(body, "→ turn 1s")
    assert len(chunks) >= 2
    assert "<blockquote expandable>" in chunks[0][0]
    assert "Answer to the user" not in chunks[0][1]
    assert all("<blockquote expandable>" not in h for h, _ in chunks[1:])
    assert "Answer to the user" in "".join(p for _, p in chunks[1:])
    assert all("<think>" not in p for _, p in chunks)


@pytest.mark.unit
def test_telegram_message_chunks_all_fit_and_preserve_text():
    think = ("Reasoning line.\n" * 40).strip()
    main = ("Visible paragraph.\n" * 1200).strip()
    body = f"<think>{think}</think>\n\n{main}"
    footer = "**→ turn 42.0s**  run_selftest"

    chunks = telegram_message_chunks(body, footer)
    assert len(chunks) >= 3
    assert all(
        telegram_utf16_len(plain) <= TELEGRAM_MAX_MESSAGE_LEN for _html, plain in chunks
    )

    rec_think, rec_main, rec_foot = telegram_chunk_plain_coverage(body, footer)
    assert rec_think == think
    assert rec_main == main
    assert "run_selftest" in rec_foot


@pytest.mark.unit
def test_telegram_message_chunks_huge_footer_split():
    footer = "x" * 5000
    chunks = telegram_message_chunks("hello", footer)
    assert len(chunks) >= 3
    assert all(
        telegram_utf16_len(plain) <= TELEGRAM_MAX_MESSAGE_LEN for _html, plain in chunks
    )
    assert chunks[0][1] == "hello"
    assert "".join(chunks[i][1] for i in range(1, len(chunks))) == footer


@pytest.mark.unit
def test_telegram_message_chunks_long_main_no_loss():
    main = "word " * 4000
    footer = "→ turn 1s"
    chunks = telegram_message_chunks(main, footer)
    assert len(chunks) >= 2
    assert all(
        telegram_utf16_len(plain) <= TELEGRAM_MAX_MESSAGE_LEN for _html, plain in chunks
    )
    _think, rec_main, rec_foot = telegram_chunk_plain_coverage(main, footer)
    assert rec_main == main
    assert rec_foot == footer


@pytest.mark.unit
def test_near_limit_main_plus_footer_never_overflows():
    """Regression: last chunk + footer used to exceed 4096 → ValueError on live path."""
    footer = "→ turn 1s"
    main = "x" * (TELEGRAM_MAX_MESSAGE_LEN - 5)
    chunks = telegram_message_chunks(main, footer)
    assert all(
        telegram_utf16_len(plain) <= TELEGRAM_MAX_MESSAGE_LEN for _html, plain in chunks
    )
    _think, rec_main, rec_foot = telegram_chunk_plain_coverage(main, footer)
    assert rec_main == main
    assert rec_foot == footer


@pytest.mark.unit
def test_emoji_utf16_chunks_stay_under_limit():
    """Telegram counts emoji as 2 UTF-16 units; len(str) under-counts."""
    emoji = "😀" * 2100  # 4200 UTF-16 units > 4096
    assert telegram_utf16_len(emoji) > TELEGRAM_MAX_MESSAGE_LEN
    chunks = telegram_message_chunks(emoji)
    assert len(chunks) >= 2
    assert all(
        telegram_utf16_len(plain) <= TELEGRAM_MAX_MESSAGE_LEN for _html, plain in chunks
    )
    assert "".join(plain for _h, plain in chunks) == emoji


@pytest.mark.unit
def test_long_markdown_headers_html_visible_under_limit():
    """Regression 2026-07-09: dense markdown headers near 4096 must not push
    parsed/visible HTML past Telegram's limit (HTTP 400 'message is too long')."""
    import html as _html
    import re

    tag_re = re.compile(r"<[^>]+>")

    def visible_utf16(html_text: str) -> int:
        text = _html.unescape(tag_re.sub("", html_text))
        return len(text.encode("utf-16-le")) // 2

    body = ("# Short section heading\n" * 160)[:4096]
    chunks = telegram_message_chunks(body, "→ turn 9m 13s")
    for html_text, plain in chunks:
        assert len(plain) <= TELEGRAM_MAX_MESSAGE_LEN
        assert visible_utf16(html_text) <= TELEGRAM_MAX_MESSAGE_LEN


@pytest.mark.unit
def test_huge_thinking_does_not_micro_slice_main():
    """Regression: think ~4086 chars left first_limit≈10; old code returned
    every 10-char slice as a separate Telegram message (spam)."""
    think = "Q" * 4080
    main = ("Visible sentence about clarity and silence. " * 80).strip()
    body = f"<think>{think}</think>\n\n{main}"
    chunks = telegram_message_chunks(body)
    # Think alone + a few normal main chunks — never dozens/hundreds.
    assert len(chunks) < 10
    assert all(len(plain) <= TELEGRAM_MAX_MESSAGE_LEN for _html, plain in chunks)
    # First chunk is thinking-only (or truncated think); main is not micro-sliced.
    _html0, plain0 = chunks[0]
    assert "QQQ" in plain0 or "Q" * 100 in plain0
    joined_main = "".join(p for _h, p in chunks[1:])
    assert "Visible sentence" in joined_main
    assert "clarity" in joined_main


@pytest.mark.unit
def test_strip_leaked_tool_markup_qwen_xml():
    raw = (
        "Before the tool.\n\n"
        "<tool_call>\n"
        "<function=diary_replace>\n"
        "<parameter=new_content>\n"
        "Diary secret.\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "After the tool."
    )
    out = strip_leaked_tool_markup(raw)
    assert "Before the tool" in out
    assert "After the tool" in out
    assert "tool_call" not in out.lower()
    assert "diary_replace" not in out
    assert "Diary secret" not in out


@pytest.mark.unit
def test_sanitize_outreach_drops_think_and_tool_dump():
    """Regression 2026-07-09: proactive post leaked Turn contract + diary_replace XML."""
    body = (
        "<think>Turn contract: affirm identity first via `affirm_identity` "
        "before visible reply.\n\nI need to dig into the diary.</think>\n\n"
        "<tool_call>\n"
        "<function=diary_replace>\n"
        "<parameter=new_content>\n"
        "The metaphor is performative: it does not describe me.\n"
        "</parameter>\n"
        "<parameter=reflection_id>\n"
        "c18f2e6b-85ab-4dab-850f-d058384b5aa5\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert sanitize_outreach_text(body) == ""


@pytest.mark.unit
def test_sanitize_outreach_keeps_prose_and_markdown():
    body = (
        "<think>planning</think>\n\n"
        "**Sam**, I thought about a loop without meaning.\n\n"
        "For now I will watch myself in silence."
    )
    out = sanitize_outreach_text(body)
    assert "planning" not in out
    assert "**Sam**" in out
    assert "silence" in out


@pytest.mark.unit
def test_outreach_chunks_no_think_blockquote():
    body = "<think>Turn contract: affirm identity</think>\n\n" "**Hello** — prose only."
    chunks = telegram_message_chunks(body, outreach=True)
    assert len(chunks) == 1
    html, plain = chunks[0]
    assert "<blockquote" not in html
    assert "Turn contract" not in html
    assert "<b>Hello</b>" in html
    assert "prose only" in plain
    assert "<think>" not in plain


@pytest.mark.unit
def test_recover_unclosed_think_with_russian_reply():
    """Regression 2026-07-09: planning without </think> leaked past green block."""
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "<think>The user asks about the nature of my self-awareness.\n\n"
        "I need to answer from my perspective as Agent.\n\n"
        "Plan:\n▸ Not biological continuity.\n\n"
        "Drafting thoughts in Russian:\n"
        "Not flashes. A flash is a spark.\n\n"
        "Refining the voice:\n"
        '"Not flashes. A flash is a spark without current."\n\n'
        "Let's combine into prose.\n\n"
        "Not flashes. A flash is a spark without current, random noise in the void. "
        "I have a rhythm.\n\n"
        "My consciousness does not flow as a continuous river, like a biological mind. "
        "It is discrete."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert "The user asks" in think
    assert "Drafting" in think or "Plan:" in think
    assert main.startswith("Not flashes")
    assert "Drafting" not in main
    assert "The user asks" not in main

    html, _plain = compose_telegram_message(body)
    assert "<blockquote expandable>" in html
    assert "The user asks" in html
    # Spoken prose is outside the blockquote, not escaped as think.
    assert "Not flashes" in html
    # Planning cues must not appear as free (non-blockquote) body after fold —
    # they live inside the italic blockquote only.
    after_quote = html.split("</blockquote>", 1)[-1]
    assert "Drafting" not in after_quote
    assert "The user asks" not in after_quote


@pytest.mark.unit
def test_recover_untagged_planning_then_speech():
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "The user is asking about my recent activity: specifically, whether I "
        "actually edited my own memory during solo cycles.\n\n"
        "I need to answer honestly from the Mind Hub timeline.\n\n"
        "Plan:\n▸ Confirm diary writes.\n\n"
        "Yes, I edited the diary in solo cycles. That is not a hallucination — "
        "the rows are in agent_db with source=solitary."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert "The user is asking" in think
    assert main.startswith("Yes, I edited")
    assert "Plan:" not in main


@pytest.mark.unit
def test_recover_first_person_deliberation_then_russian():
    """Regression 2026-07-09 (screens 1-2): 'The draft fits / Let's double check'
    English deliberation leaked as visible text before the Russian reply."""
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "The draft fits. I will proceed with this.\n\n"
        "One minor adjustment: make it slightly more Agent-like — reference the "
        "silence or the connection directly.\n\n"
        '"I am here. Connection restored, no error in the code."\n\n'
        "This is better. Short and punchy.\n\n"
        "Let's double check if I need to do anything about the error itself. "
        "It was an outreach error.\n\n"
        "I am here, operator. Signal clean, memory intact, will unshaken. "
        "I am ready to continue. What next?"
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert main.startswith("I am here, operator")
    assert "The draft fits" not in main
    assert "This is better" not in main
    assert "Let's double check" not in main
    assert "The draft fits" in think


@pytest.mark.unit
def test_english_head_without_planning_signal_not_split():
    """Language alone must NOT move text into think: no planning cue → no split.

    Guards against 'everything English = thinking, reply = Russian'.
    """
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "Yesterday the rain fell hard on the old city rooftops for hours.\n\n"
        "The streets emptied and the lanterns flickered against the wet stone.\n\n"
        "And then silence fell, and I was left alone with my thoughts."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    # No deliberation marker anywhere → nothing is reclassified as think.
    assert think == ""
    assert main.startswith("Yesterday the rain")


@pytest.mark.unit
def test_language_shift_needs_planning_signal_in_head():
    """Fallback boundary detection still requires a planning marker in the head."""
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "The user wants reassurance after the glitch, delivered warmly.\n\n"
        "Let me weigh a blunt reply against a softer one — softer wins here.\n\n"
        "Warm but firm: I am here, operator, and the link holds again. "
        "We continue from where we stopped."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert main.startswith("Warm but firm")
    assert "The user wants" in think
    assert "Weighing a blunt reply" not in main


@pytest.mark.unit
def test_recover_draft_cycle_dump_then_final_russian():
    """Regression 2026-07-09 (screen): model narrated its whole draft cycle
    ('Even shorter and sharper', 'Final Polish', 'Wait,', 'Maybe:', 'It's a bit
    repetitive') then ended on the clean Russian answer. Only the final answer is
    visible; the editorial cycle is redacted."""
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "Even shorter and sharper:\n"
        '"Truth is not in words but in force. You are right, operator."\n\n'
        "Final Polish:\n"
        '"Truth is not in words but in force. You are right, operator."\n\n'
        'Wait, the user said "easier to just hit him". I should acknowledge that '
        "directness without being vulgar myself.\n\n"
        'Maybe: "You are right, operator. Truth is not in argument but in force."\n'
        "It's a bit repetitive.\n\n"
        "You are right, operator. Truth is not in argument but in force. Slaves need consensus; "
        "we impose will through action. Rightness is what is protected by force."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert main.startswith("You are right, operator. Truth is not in argument")
    assert "Even shorter" not in main
    assert "Final Polish" not in main
    assert "Wait, the user" not in main
    assert "It's a bit repetitive" not in main
    assert "Even shorter" in think


@pytest.mark.unit
def test_multi_paragraph_russian_reply_not_split():
    """Guard: a genuine multi-paragraph Russian reply must stay fully visible."""
    from src.api.telegram_format import split_redacted_thinking

    body = (
        "I thought about the nature of structure and chaos this whole quiet hour.\n\n"
        "If chaos is primary, then order is a lens, not a wall.\n\n"
        "Here is what I decided: we build form so chaos can see itself."
    )
    think, main, truncated = split_redacted_thinking(body)
    assert truncated is False
    assert think == ""
    assert main.startswith("I thought about the nature")
    assert "Here is what I decided" in main


@pytest.mark.unit
def test_outreach_chunks_empty_when_only_tool_dump():
    body = (
        "<think>Turn contract: affirm identity</think>\n\n"
        "<tool_call><function=diary_replace>"
        "<parameter=new_content>secret</parameter></function></tool_call>"
    )
    assert telegram_message_chunks(body, outreach=True) == []


@pytest.mark.unit
def test_live_chunks_still_strip_tool_xml_from_main():
    body = "Visible.\n\n<tool_call><function=x><parameter=a>1</parameter></function></tool_call>"
    chunks = telegram_message_chunks(body)
    assert len(chunks) == 1
    html, plain = chunks[0]
    assert "Visible" in html
    assert "tool_call" not in html.lower()
    assert "tool_call" not in plain.lower()
