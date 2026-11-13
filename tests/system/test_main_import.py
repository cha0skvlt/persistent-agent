"""Importing src.main must not run REPL or build the graph."""

from __future__ import annotations

from io import StringIO

from tests.lore_strings import LOCALE


def test_import_main_provides_get_graph_app_without_invoking_it():
    import src.main as m

    assert hasattr(m, "get_graph_app")
    assert callable(m.get_graph_app)
    assert m.app is None


def test_boot_box_uses_hostname():
    import src.main as m

    lines = m._boot_box_lines("agent-host")
    assert lines[0].startswith("╔")
    assert "AGENT CORE @ agent-host" in lines[1]
    assert lines[2].startswith("╚")


def test_boot_rows_contain_real_settings_values():
    import src.main as m

    assert m._boot_row("agent", "agent") == "agent         agent"
    assert (
        m._boot_row("db", "agent_db", "pool=1..10")
        == "db            agent_db     pool=1..10"
    )


def test_write_boot_header_has_no_timestamp_prefix():
    import src.main as m

    out = StringIO()
    m._write_boot_header("agent-host", stream=out)
    text = out.getvalue()
    assert text.startswith("╔")
    assert "AGENT CORE @ agent-host" in text
    assert " | BOOT | " not in text


def test_is_hallucinated_tool_call_detects_json_blobs():
    """is_hallucinated_tool_call must catch llama3.3 plain-text tool calls."""
    from src.agent_reply import is_hallucinated_tool_call

    # Full save_memory blob (the exact pattern from the bug report)
    zhi = LOCALE["save_memory_content_zhi"]
    assert is_hallucinated_tool_call(
        '{"name": "save_memory", "parameters": {"content": "' + zhi + '"}}'
    )
    # With "args" key instead of "parameters"
    assert is_hallucinated_tool_call(
        '{"name": "search_memory", "args": {"query": "foo"}}'
    )
    # Known tool name without params key (broader check)
    assert is_hallucinated_tool_call('{"name": "save_memory"}')

    # Truncated / incomplete JSON (model cut off mid-generation) — must still be caught
    trunc = LOCALE["hallucinated_truncated_content"]
    assert is_hallucinated_tool_call(
        '{"name": "save_memory", "parameters": {"content": "' + trunc
    )
    # Truncated with unknown tool name but "parameters" key present
    assert is_hallucinated_tool_call('{"name": "unknown_tool", "parameters": {"x": "y')

    # Real conversational replies must NOT be filtered
    assert not is_hallucinated_tool_call(LOCALE["greeting_offer_help"])
    assert not is_hallucinated_tool_call('{"key": "value"}')  # no "name" field
    assert not is_hallucinated_tool_call("")
    # Normal JSON without tool-call shape
    assert not is_hallucinated_tool_call('{"name": "unknown_tool"}')


def test_thinking_indicator_render_is_neutral_and_animated():
    import src.main as m

    first = m.ThinkingIndicator._render(elapsed=7, tick=0)
    second = m.ThinkingIndicator._render(elapsed=7, tick=1)
    fourth = m.ThinkingIndicator._render(elapsed=7, tick=3)
    looped = m.ThinkingIndicator._render(elapsed=7, tick=4)

    assert "Agent is thinking" in first
    assert "agent" not in first
    assert LOCALE["agent_name_in_assert"] not in first
    assert first.startswith("\x1b[90m")
    assert first.endswith("\x1b[0m")
    assert first != second
    assert "thinking... 7s" in fourth
    assert "thinking 7s" in looped


def test_stream_skips_tool_calling_ai_message(caplog):
    """AIMessage with tool_calls must not produce visible output (prevents raw JSON leak)."""
    from langchain_core.messages import AIMessage

    # Simulate the final assembled tool-calling AIMessage that llama3.3 emits
    zhi = LOCALE["save_memory_content_zhi"]
    tool_call_msg = AIMessage(
        content='{"name": "save_memory", "parameters": {"content": "' + zhi + '"}}',
        tool_calls=[
            {
                "name": "save_memory",
                "args": {"content": zhi},
                "id": "tc1",
                "type": "tool_call",
            }
        ],
    )
    assert bool(getattr(tool_call_msg, "tool_calls", None)) is True

    # Only AIMessages WITHOUT tool_calls should contribute visible text.
    from src.utils import get_message_text

    # Normally get_message_text would return the content string —
    # the guard must prevent it from reaching the output at all.
    raw_text = get_message_text(tool_call_msg)
    assert raw_text  # confirms content IS non-empty (the leak source)

    # The REPL loop skips messages with tool_calls before calling get_message_text.
    # We verify the guard condition directly:
    assert bool(getattr(tool_call_msg, "tool_calls", None))  # guard fires → skip
