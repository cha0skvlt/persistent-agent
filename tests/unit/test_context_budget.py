"""Agent SYS / message context budgeting."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph import context_budget as cb
from src.obs.logs import setup_agent_event_logger


@pytest.mark.unit
def test_estimate_prompt_tokens():
    assert cb.estimate_prompt_tokens("") == 0
    assert cb.estimate_prompt_tokens("abcdef") == 2
    assert cb.estimate_tokens("abcdef") == cb.estimate_prompt_tokens("abcdef")


@pytest.mark.unit
def test_truncate_tail_marker():
    out = cb.truncate_tail("x" * 100, 20)
    assert out.endswith(cb.TRUNCATION_MARKER)
    assert len(out) <= 20


@pytest.mark.unit
def test_apply_sys_char_budgets(monkeypatch):
    monkeypatch.setattr(cb.settings, "sys_mem_max_chars", 30, raising=False)
    monkeypatch.setattr(cb.settings, "sys_summary_max_chars", 25, raising=False)
    monkeypatch.setattr(cb.settings, "web_evidence_max_chars", 20, raising=False)
    monkeypatch.setattr(cb.settings, "sys_mind_max_chars", 40, raising=False)
    monkeypatch.setattr(cb.settings, "sys_diary_max_chars", 35, raising=False)
    monkeypatch.setattr(cb.settings, "pulse_sys_max_chars", 600, raising=False)

    blocks = [
        cb.SysBlock("core", "PERSONA-RULES-INTACT"),
        cb.SysBlock("summary", "S" * 80),
        cb.SysBlock("mem", "M" * 80),
        cb.SysBlock("web", "W" * 80),
        cb.SysBlock("mind", "H" * 80),
        cb.SysBlock("diary", "D" * 80),
    ]
    out = cb.apply_sys_char_budgets(blocks)
    assert out[0].text == "PERSONA-RULES-INTACT"
    assert cb.TRUNCATION_MARKER in out[1].text
    assert cb.TRUNCATION_MARKER in out[2].text
    assert len(out[3].text) == 80  # web_cap = 20+700
    assert cb.TRUNCATION_MARKER in out[4].text
    assert cb.TRUNCATION_MARKER in out[5].text


@pytest.mark.unit
def test_overflow_sacrifice_order_preserves_core(monkeypatch):
    monkeypatch.setattr(cb.settings, "llm_num_ctx", 1000, raising=False)
    monkeypatch.setattr(cb.settings, "llm_num_predict", 200, raising=False)
    monkeypatch.setattr(cb.settings, "sys_mem_max_chars", 4000, raising=False)
    monkeypatch.setattr(cb.settings, "sys_summary_max_chars", 4000, raising=False)
    monkeypatch.setattr(cb.settings, "web_evidence_max_chars", 4000, raising=False)

    core = "CORE-PERSONA-" + ("identity " * 40)
    web = cb.SysBlock("web", "WEB-" + ("evidence " * 500))
    mem = cb.SysBlock("mem", "MEM-" + ("fact " * 500))
    summary = cb.SysBlock("summary", "SUM-" + ("dialogue " * 500))
    blocks = [
        cb.SysBlock("core", core),
        summary,
        mem,
        web,
    ]
    messages = [
        HumanMessage(content="old " * 200),
        AIMessage(content="mid " * 200),
        HumanMessage(content="latest question"),
        HumanMessage(content="[Deixis: hard]\nreanchor"),
    ]

    events: list[tuple[str, dict]] = []
    trimmed: list[str] = []
    original_trim = cb._trim_sacrifice_block

    def track_trim(block: cb.SysBlock, *, fraction: float = 0.25):
        trimmed.append(block.category)
        return original_trim(block, fraction=fraction)

    monkeypatch.setattr(
        cb, "log_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(cb, "_trim_sacrifice_block", track_trim)

    out_blocks, out_messages = cb.enforce_prompt_token_budget(blocks, messages)

    assert cb.core_system_text(out_blocks).strip() == core.strip()
    assert events and events[0][0] == "context.overflow_risk"
    fields = events[0][1]
    assert "est_tokens_latin" in fields
    assert "est_tokens_cyrillic" in fields
    # diary/mind/summary/mem before web — keep LIVE WEB EVIDENCE longest
    assert trimmed[0] in {"summary", "mem", "diary", "mind"}
    if "web" in trimmed and "mem" in trimmed:
        assert trimmed.index("mem") < trimmed.index("web")
    if "summary" in trimmed and "mem" in trimmed:
        assert trimmed.index("summary") < trimmed.index("mem")
    assert any(
        isinstance(m, HumanMessage)
        and "[Deixis:" in (m.content if isinstance(m.content, str) else "")
        for m in out_messages
    )


@pytest.mark.unit
def test_overflow_risk_event_in_logs(monkeypatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setattr(cb.settings, "llm_num_ctx", 500, raising=False)
    monkeypatch.setattr(cb.settings, "llm_num_predict", 100, raising=False)
    setup_agent_event_logger()
    blocks = [
        cb.SysBlock("core", "persona"),
        cb.SysBlock("web", "W" * 5000),
    ]
    cb.enforce_prompt_token_budget(blocks, [HumanMessage(content="hi " * 500)])
    assert any('"event": "context.overflow_risk"' in r.message for r in caplog.records)


@pytest.mark.unit
def test_inject_system_message_replaces_prior_system():
    blocks = [cb.SysBlock("core", "persona"), cb.SysBlock("mem", "memories")]
    msgs = [
        SystemMessage(content="old"),
        HumanMessage(content="hi"),
    ]
    out = cb.inject_system_message(blocks, msgs)
    assert isinstance(out[0], SystemMessage)
    assert "persona" in str(out[0].content)
    assert "memories" in str(out[0].content)
    assert len(out) == 2
