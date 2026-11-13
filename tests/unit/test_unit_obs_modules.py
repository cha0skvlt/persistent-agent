"""Unit tests for src.obs.* (console, logs, log_format, debug_context, tool_logging)."""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.obs.console_format import SUPPRESS, ConsoleFormatter
from src.obs.debug_context import _msg_role, _msg_text, emit_llm_context
from src.obs.log_format import (
    clip_one_line,
    clip_words,
    fmt_kv,
    fmt_preview,
    fmt_score,
    fmt_size_chars,
)
from src.obs.logs import (
    format_elapsed_ms,
    is_debug_enabled,
    log_debug_event,
    log_event,
    normalize_repl_user_text,
    setup_agent_event_logger,
)
from src.obs.tool_logging import (
    _find_tool_call_args,
    _query_preview,
    invoke_tool_node_logged,
    log_tool_results,
    tool_result_is_empty,
)


def _fmt_record(
    payload: dict,
    *,
    level: int = logging.INFO,
    use_color: bool = False,
) -> str:
    f = ConsoleFormatter()
    f._use_color = use_color
    record = logging.LogRecord(
        name="agent.event",
        level=level,
        pathname="",
        lineno=0,
        msg=json.dumps(payload),
        args=(),
        exc_info=None,
    )
    return f.format(record)


@pytest.mark.unit
def test_log_format_clip_and_fmt_helpers():
    assert clip_one_line("  a\nb  ") == "a b"
    assert clip_words("") == "∅"
    assert fmt_preview("") == "∅"
    assert "say" in fmt_preview('say "hi"')
    assert fmt_preview('say "hi"').startswith("'say ")
    assert fmt_score("bad") == "bad"
    assert fmt_score(0.25) == "0.250"
    assert fmt_size_chars("x") == "x"
    assert fmt_size_chars(999) == "999c"
    assert fmt_size_chars(1500) == "1.5k"
    assert fmt_size_chars(12000) == "12k"


@pytest.mark.unit
def test_logs_encode_fallback_and_helpers(caplog):
    with caplog.at_level(logging.INFO, logger="agent.event"):
        log_event("ok.payload", x=1)

    from src.obs import logs as logs_mod

    class Bad:
        def __str__(self):
            raise TypeError("no")

    line = logs_mod._encode_event("x", {"bad": Bad()})
    doc = json.loads(line)
    assert doc["error"] == "non_serializable"

    _log = logging.getLogger("agent.event")

    assert format_elapsed_ms("nope") == "nope"
    assert format_elapsed_ms(-100) == "0s"
    assert format_elapsed_ms(65_000) == "1m 5s"
    assert format_elapsed_ms(400) == "0s"
    assert format_elapsed_ms(500) == "0.5s"
    assert format_elapsed_ms(59_500) == "59.5s"
    assert format_elapsed_ms(59_900) == "1m 0s"

    assert normalize_repl_user_text("a\r\nb\n\nc") == "a b c"

    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        root.addHandler(h)
        try:
            setup_agent_event_logger()
        finally:
            root.removeHandler(h)
    else:
        setup_agent_event_logger()

    _log.setLevel(logging.WARNING)
    assert is_debug_enabled() is False
    log_debug_event("noop", x=1)  # gated off
    _log.setLevel(logging.DEBUG)


@pytest.mark.unit
def test_console_formatter_levels_and_bad_json():
    fmt = ConsoleFormatter()
    fmt._use_color = False
    r = logging.LogRecord(
        "agent.event",
        21,
        "",
        0,
        "plain",
        (),
        None,
    )
    r.levelname = "BOOT"
    assert "| BOOT  |" in fmt.format(r)

    r.levelno = logging.ERROR
    r.levelname = "ERROR"
    r.msg = "e"
    assert "| ERROR |" in fmt.format(r)

    r.levelno = 25
    r.levelname = "CUSTOM"
    r.msg = "c"
    out = fmt.format(r)
    assert "CUSTOM" in out

    r.msg = "{not-json"
    assert fmt.format(r).endswith("{not-json")

    r2 = logging.LogRecord(
        "agent.event",
        logging.INFO,
        "",
        0,
        '{"borked": }',
        (),
        None,
    )
    assert '{"borked": }' in fmt.format(r2)


@pytest.mark.unit
@patch.object(sys.stderr, "isatty", return_value=True)
def test_console_formatter_color_paths(_isatty):
    fmt = ConsoleFormatter()
    r = logging.LogRecord(
        "agent.event",
        logging.WARNING,
        "",
        0,
        "watch out",
        (),
        None,
    )
    r.levelname = "WARNING"
    assert "\x1b[" in fmt.format(r)

    fmt2 = ConsoleFormatter()
    r = logging.LogRecord(
        "agent.event",
        logging.DEBUG,
        "",
        0,
        json.dumps({"event": "state.snapshot", "keys": "k"}),
        (),
        None,
    )
    out = fmt2.format(r)
    assert "\x1b[" in out

    r.levelno = logging.INFO
    r.levelname = "INFO"
    r.msg = "hello"
    out2 = fmt2.format(r)
    assert "\x1b[" in out2


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload,expect_substr",
    [
        ({"event": 123}, "{"),  # ev not str -> pretty None; raw JSON in line
        (
            {"event": "mem.search", "hits": 2, "top_score": 0.5},
            "search",
        ),
        (
            {
                "event": "mem.dedup",
                "preview": "x",
                "reason": "r",
                "old_id": "abc",
                "similarity": 1.0,
            },
            "dedup",
        ),
        ({"event": "mem.save", "preview": "saved"}, "save="),
        ({"event": "mem.replace", "preview": "rep"}, "replace="),
        (
            {
                "event": "retrieve.result",
                "kg": 1,
                "mem": 2,
                "self_reflections": 4,
                "emergency": True,
            },
            "emergency",
        ),
        ({"event": "retrieve.rerank_weights", "rerank_weights": "a=1"}, "rerank="),
        ({"event": "retrieve.skip", "reason": "x"}, "skip="),
        (
            {"event": "retrieve.anchor", "entity": "one two three four", "kg": 1},
            "anchor=",
        ),
        ({"event": "tool.call", "name": "t"}, "tool="),
        ({"event": "htc.recover", "tool": "invoke", "path": "/p"}, "htc→"),
        ({"event": "htc.recover", "tool": "invoke"}, "htc→invoke"),
        ({"event": "reflect.run", "turn_count": 1, "reflect_every_n": 2}, "turn="),
        (
            {
                "event": "reflect.skip",
                "reason": "r",
                "turn_count": 1,
                "reflect_every_n": 2,
            },
            "reflect",
        ),
        ({"event": "reflect.skipped", "reason": "r"}, "reflect"),
        ({"event": "reflect.blocked", "reason": "b", "preview": "p"}, "reflect"),
        ({"event": "reflect.self", "preview": "me"}, "reflect"),
        (
            {
                "event": "reflect.extract",
                "facts": 1,
            },
            "facts=1",
        ),
        (
            {
                "event": "llm.invoke",
                "path": "p",
                "msgs": 1,
                "sys_chars": 2,
                "tools": "t",
            },
            "llm",
        ),
        # llm.sys / llm.ctx_msg no longer have dedicated console pretty handlers
        # (full dumps go to agent-context.log); unknown-event fallback keeps keys.
        ({"event": "llm.sys", "path": "p", "text": "long " * 20}, "llm sys"),
        (
            {
                "event": "llm.ctx_msg",
                "role": "user",
                "idx": 0,
                "text": "hi",
                "tool_calls": "a,b",
                "tool_name": "tn",
            },
            "llm ctx_msg",
        ),
        (
            {
                "event": "tool.result",
                "name": "search_memory",
                "empty": True,
                "chars": 0,
                "query": "q",
            },
            "tool=search_memory → ∅",
        ),
        (
            {
                "event": "tool.result",
                "name": "forget_memory",
                "empty": False,
                "chars": 9,
                "preview": "forgotten",
            },
            "tool=forget_memory → forgotten",
        ),
        (
            {"event": "llm.response", "path": "p", "tool_calls": "x", "text_chars": 10},
            "reply=",
        ),
        ({"event": "llm.reflect_raw", "path": "p", "text": "raw"}, "reflect"),
        ({"event": "retrieve.score", "tag": "t", "preview": "pv", "score": 0.5}, "mem"),
        (
            {"event": "reflect.pair", "user_chars": 1, "asst_chars": 2},
            "reflect",
        ),
        (
            {
                "event": "reflect.triple_blocked",
                "subject": "s s s s",
                "predicate": "p p p p",
                "object": "o o o o",
                "reason": "r",
            },
            "reflect",
        ),
        ({"event": "htc.detail", "raw": "detail text"}, "htc"),
        ({"event": "custom.other", "x": 1}, "custom other"),
        ({"event": "empty.extra"}, "empty extra"),
        (
            {"event": "turn.complete", "elapsed_ms": 1500, "tools": ["search_web"]},
            "→ turn",
        ),
    ],
)
def test_console_formatter_event_branches(payload, expect_substr):
    out = _fmt_record(payload, level=logging.DEBUG)
    if expect_substr is SUPPRESS:
        assert out == SUPPRESS
    else:
        assert expect_substr in out


@pytest.mark.unit
def test_console_formatter_turn_complete_no_tools():
    fmt = ConsoleFormatter()
    fmt._use_color = False
    out = fmt.format(
        logging.LogRecord(
            "agent.event",
            logging.INFO,
            "",
            0,
            json.dumps({"event": "turn.complete", "elapsed_ms": 800}),
            (),
            None,
        )
    )
    assert "→ turn" in out
    assert "INFO" in out


@pytest.mark.unit
def test_console_formatter_folded_node_timeline():
    fmt = ConsoleFormatter()
    fmt._use_color = False

    def _record(payload, level=logging.INFO):
        return logging.LogRecord(
            "agent.event",
            level,
            "",
            0,
            json.dumps(payload),
            (),
            None,
        )

    assert fmt.format(_record({"event": "node.start", "node": "n1"})) == SUPPRESS
    mid = fmt.format(_record({"event": "tool.call", "name": "x"}))
    assert mid == SUPPRESS
    end = fmt.format(
        _record({"event": "node.timing", "node": "n1", "elapsed_ms": 1000})
    )
    assert "→ n1" in end
    assert "tool=x" in end


@pytest.mark.unit
def test_console_formatter_tool_result_empty_and_preview():
    empty = _fmt_record(
        {"event": "tool.result", "name": "n", "empty": True, "chars": 0, "query": "q"},
        level=logging.INFO,
    )
    assert "tool=n → ∅" in empty
    filled = _fmt_record(
        {
            "event": "tool.result",
            "name": "n",
            "empty": False,
            "chars": 5,
            "preview": "hello",
        },
        level=logging.INFO,
    )
    assert "tool=n → hello" in filled
    assert "(5c)" in filled


@pytest.mark.unit
def test_debug_context_helpers_and_emit(monkeypatch):
    assert _msg_role(HumanMessage(content="h")) == "user"
    assert _msg_role(AIMessage(content="a")) == "agent"
    assert _msg_role(ToolMessage(content="t", tool_call_id="1")) == "tool"
    assert _msg_role(SystemMessage(content="s")) == "sys"
    assert _msg_role(type("Weird", (), {"content": [888]})()) == "weird"

    class BlockList:
        content = [{"text": "x"}, "y", {"nope": 1}]

    assert _msg_text(BlockList()) == "xy"
    assert _msg_text(type("N", (), {"content": 3})()) == "3"

    main_calls: list[tuple[str, dict]] = []
    ctx_calls: list[tuple[str, dict]] = []

    def _cap_main(event, **kw):
        main_calls.append((event, kw))

    def _cap_ctx(event, **kw):
        ctx_calls.append((event, kw))

    monkeypatch.setattr("src.obs.debug_context.log_debug_event", _cap_main)
    monkeypatch.setattr("src.obs.debug_context.context_debug_event", _cap_ctx)
    emit_llm_context(
        "path",
        "sys",
        [
            HumanMessage(content="h"),
            ToolMessage(content="{}", tool_call_id="t1", name="tn"),
        ],
        "tools",
    )
    assert any(c[0] == "llm.invoke" for c in main_calls)
    assert any(c[0] == "llm.sys" for c in ctx_calls)
    assert any(
        c[0] == "llm.ctx_msg" and c[1].get("tool_name") == "tn" for c in ctx_calls
    )
    assert not any(c[0] == "llm.ctx_msg" for c in main_calls)


@pytest.mark.unit
def test_tool_logging_branches(caplog):
    assert _find_tool_call_args([], "") == {}
    assert _find_tool_call_args([], "x") == {}
    ai_bad = AIMessage(content="")
    ai_bad.tool_calls = [object()]  # type: ignore[attr-defined]
    assert _find_tool_call_args([ai_bad], "1") == {}
    ai_ok = AIMessage(
        content="",
        tool_calls=[{"id": "z", "name": "n", "args": {"query": "qq"}}],
    )
    assert _find_tool_call_args([ai_ok], "z") == {"query": "qq"}
    ai_non_dict_args = AIMessage(
        content="",
        tool_calls=[{"id": "a", "name": "n", "args": {}}],
    )
    ai_non_dict_args.tool_calls[0]["args"] = "bad"  # type: ignore[index]
    assert _find_tool_call_args([ai_non_dict_args], "a") == {}
    assert _query_preview({}) == ""
    assert _query_preview({"subject": "sub"}) == "sub"
    assert "other" in _query_preview({"other": 1})

    caplog.set_level(logging.INFO, logger="agent.event")
    log_tool_results([], [ToolMessage(content="", tool_call_id="1")])
    assert any("tool.result" in r.getMessage() for r in caplog.records)

    assert tool_result_is_empty("  ∅  ") is True
    log_tool_results([], [AIMessage(content="")])  # no crash — skips non-ToolMessage

    log_tool_results(
        [],
        [ToolMessage(content="has body", tool_call_id="t", name="n")],
    )
    filled = [
        json.loads(r.getMessage())
        for r in caplog.records
        if "tool.result" in r.getMessage()
    ]
    assert any(d.get("empty") is False and d.get("preview") for d in filled)


@pytest.mark.unit
def test_console_fold_without_active_node():
    fmt = ConsoleFormatter()
    fmt._use_color = False
    r = logging.LogRecord(
        "agent.event",
        logging.INFO,
        "",
        0,
        json.dumps({"event": "mem.search", "hits": 3, "top_score": 0.1}),
        (),
        None,
    )
    out = fmt.format(r)
    assert "search" in out
    assert "·" in out


@pytest.mark.unit
def test_log_format_fmt_kv_skips_empty():
    assert fmt_kv(("a", 1), ("b", "")) == "a=1"


@pytest.mark.unit
def test_setup_agent_event_logger_no_root_handlers():
    import logging

    from src.obs.logs import setup_agent_event_logger

    root = logging.getLogger()
    old = root.handlers[:]
    for h in old:
        root.removeHandler(h)
    try:
        setup_agent_event_logger()
    finally:
        for h in old:
            root.addHandler(h)


def test_setup_agent_event_logger_propagate_true_with_handler():
    """agent.event must propagate to root so file/console get events when OTLP is down."""
    import logging

    from src.obs.logs import setup_agent_event_logger

    log = logging.getLogger("agent.event")
    log.handlers.clear()
    log.propagate = False
    log.addHandler(logging.NullHandler())
    setup_agent_event_logger()
    assert log.propagate is True
    log.handlers.clear()
    log.propagate = True


def test_logs_debug_emit_and_setup_no_handlers(caplog):
    caplog.set_level(logging.DEBUG, logger="agent.event")
    log_debug_event("unit.test", k=1)
    assert caplog.records

    root = logging.getLogger()
    old = root.handlers[:]
    for h in old:
        root.removeHandler(h)
    try:
        setup_agent_event_logger()
    finally:
        for h in old:
            root.addHandler(h)


@pytest.mark.unit
def test_tool_logging_empty_detection_and_find_skip_non_ai(caplog):
    assert tool_result_is_empty(None) is True
    assert tool_result_is_empty([]) is True
    assert _find_tool_call_args(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "n", "args": {"q": "v"}}],
            ),
            HumanMessage(content="x"),
        ],
        "1",
    ) == {"q": "v"}

    caplog.set_level(logging.INFO, logger="agent.event")
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "t", "id": "tc1", "args": {"query": "q"}}],
    )
    tool = ToolMessage(content="", name="t", tool_call_id="tc1")
    log_tool_results([ai], [tool])
    assert any("tool.result" in r.getMessage() for r in caplog.records)
    data = json.loads(
        next(r.getMessage() for r in caplog.records if "tool.result" in r.getMessage())
    )
    assert data["empty"] is True
    assert data["query"] == "q"


@pytest.mark.unit
def test_tool_result_nonempty_and_invoke_logged(monkeypatch):
    assert tool_result_is_empty({"a": 1}) is False

    class TN:
        def invoke(self, state, config):
            return {"messages": [ToolMessage(content="ok", tool_call_id="1")]}

    monkeypatch.setattr("src.obs.tool_logging.log_tool_results", lambda *a, **k: None)
    out = invoke_tool_node_logged(TN(), {"messages": []}, {})
    assert out["messages"]
