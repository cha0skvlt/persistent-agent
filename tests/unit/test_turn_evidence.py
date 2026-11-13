"""Lexical/id reject-filter — necessary, not AIS."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage

from src.graph.turn_evidence import (
    current_evidence,
    invented_user_prefix,
    live_extract_evidence,
    note_tool_text,
    reset_turn_evidence,
    seed_from_messages,
    tool_text_for_evidence,
    write_has_turn_evidence,
)
from src.reverie.distill import gate_background_provenance, gate_solitary_memory_write

_USER_HIT = "[id=aaaaaaaa-1111-2222-3333-444444444444] USER: drinks dark roast"
_WEB = "LIVE WEB: copper crashed after the press drop tonight near 100k"


def setup_function() -> None:
    reset_turn_evidence()


def _toolnode_list(*rows: str) -> str:
    """ToolNode serializes list tool returns with json.dumps(ensure_ascii=False)."""
    return json.dumps(list(rows), ensure_ascii=False)


def test_overlap_requires_shared_tokens_or_id():
    ev = "[id=aaaaaaaa-1111-2222-3333-444444444444] copper closed near 100k after fetch"
    assert write_has_turn_evidence("copper closed near 100k on the quoted page", ev)
    assert write_has_turn_evidence("remember aaaaaaaaa-1111-2222-3333-444444444444", ev)
    assert not write_has_turn_evidence("USER: I secretly own pigeons downtown", ev)


def test_invented_user_prefix_needs_user_row_in_evidence():
    ev = "page text: copper closed near 100k after the press drop"
    assert invented_user_prefix("USER: copper closed near 100k", ev) is True
    assert (
        invented_user_prefix("USER: drinks dark roast in the morning", _USER_HIT)
        is False
    )


def test_gate_rejects_without_evidence():
    out = gate_background_provenance("AGENT: the dollar rate today is around 90")
    assert out is not None
    assert "no_evidence" in out


def test_gate_allows_after_tool_note():
    note_tool_text(
        "search hit: the dollar rate today is around 90 according to the page"
    )
    fact = "AGENT: the dollar rate today is around 90 according to the search."
    assert gate_solitary_memory_write(fact) is None


def test_seed_from_messages_keeps_human_user_row():
    seed_from_messages(
        [
            HumanMessage(content="prefers dark roast in the morning"),
            ToolMessage(content=_WEB, tool_call_id="1", name="fetch_url"),
        ]
    )
    ev = current_evidence()
    assert ev.startswith("USER: prefers dark roast in the morning")
    assert write_has_turn_evidence("USER: prefers dark roast in the morning")
    assert gate_background_provenance("USER: prefers dark roast in the morning") is None
    assert (
        gate_background_provenance("USER: I secretly own pigeons downtown") is not None
    )


def test_seed_from_messages_uses_this_turn_tools():
    seed_from_messages(
        [
            HumanMessage(content="go"),
            ToolMessage(
                content=_USER_HIT,
                tool_call_id="1",
                name="search_memory",
            ),
        ]
    )
    assert gate_background_provenance("USER: drinks dark roast every morning") is None


def test_user_write_does_not_borrow_web_tokens():
    note_tool_text(_USER_HIT)
    note_tool_text(_WEB)
    invented = "USER: copper crashed after the press drop tonight"
    out = gate_background_provenance(invented)
    assert out is not None
    assert "no_evidence" in out
    assert "invented USER" not in out
    assert gate_background_provenance("USER: drinks dark roast every morning") is None


def test_gal_write_may_use_full_tool_blob():
    note_tool_text(_USER_HIT)
    note_tool_text(_WEB)
    assert (
        gate_background_provenance("AGENT: copper crashed after the press drop tonight")
        is None
    )


def test_toolnode_json_list_allows_user_quote():
    """search_memory returns a list; ToolNode stores JSON — quote must still pass."""
    raw = _toolnode_list(_USER_HIT)
    assert raw.startswith("[")
    assert "USER:" in raw
    assert tool_text_for_evidence(raw).splitlines() == [_USER_HIT]
    note_tool_text(raw)
    assert "USER: drinks dark roast" in current_evidence()
    assert invented_user_prefix("USER: drinks dark roast") is False
    assert gate_background_provenance("USER: drinks dark roast every morning") is None


def test_toolnode_json_list_user_cannot_borrow_web():
    note_tool_text(_toolnode_list(_USER_HIT))
    note_tool_text(_WEB)
    out = gate_background_provenance(
        "USER: copper crashed after the press drop tonight"
    )
    assert out is not None
    assert "no_evidence" in out
    assert "invented USER" not in out


def test_toolnode_json_list_without_user_row_is_invented():
    note_tool_text(
        _toolnode_list(
            "[id=aaaaaaaa-1111-2222-3333-444444444444] AGENT: lives near the river"
        )
    )
    out = gate_background_provenance("USER: drinks dark roast every morning")
    assert out is not None
    assert "invented USER" in out


def test_empty_json_list_is_not_evidence():
    assert tool_text_for_evidence("[]") == ""
    note_tool_text(_toolnode_list())
    assert current_evidence() == ""
    out = gate_background_provenance("USER: drinks dark roast")
    assert out is not None
    assert "no_evidence" in out


def test_pretty_json_list_flattens():
    raw = json.dumps([_USER_HIT], ensure_ascii=False, indent=2)
    assert tool_text_for_evidence(raw) == _USER_HIT


def test_native_list_content_flattens():
    note_tool_text(
        [
            _USER_HIT,
            "[id=bbbbbbbb-1111-2222-3333-444444444444] AGENT: lives near the river",
        ]
    )
    assert gate_background_provenance("USER: drinks dark roast every morning") is None
    assert gate_background_provenance("AGENT: lives near the river in Lisbon") is None


def test_seed_from_messages_flattens_toolnode_json():
    seed_from_messages(
        [
            HumanMessage(content="go"),
            ToolMessage(
                content=_toolnode_list(_USER_HIT),
                tool_call_id="1",
                name="search_memory",
            ),
            ToolMessage(content=_WEB, tool_call_id="2", name="fetch_url"),
        ]
    )
    assert gate_background_provenance("USER: drinks dark roast every morning") is None
    assert (
        gate_background_provenance("USER: copper crashed after the press drop tonight")
        is not None
    )


def test_invoke_tool_node_logged_notes_json_list():
    from src.obs.tool_logging import invoke_tool_node_logged

    payload = _toolnode_list(_USER_HIT)

    class _Node:
        def invoke(self, _state, _config):
            return {
                "messages": [
                    ToolMessage(
                        content=payload,
                        tool_call_id="1",
                        name="search_memory",
                    )
                ]
            }

    invoke_tool_node_logged(_Node(), {"messages": []}, {})
    assert gate_background_provenance("USER: drinks dark roast every morning") is None


def test_live_extract_evidence_human_as_user_row():
    ev = live_extract_evidence("drinks dark roast every morning")
    assert ev.startswith("USER:")
    assert write_has_turn_evidence("USER: drinks dark roast every morning", ev)
    assert not write_has_turn_evidence("USER: I secretly own pigeons downtown", ev)


def test_live_extract_evidence_includes_this_turn_tools():
    ev = live_extract_evidence(
        "look this up",
        [
            HumanMessage(content="look this up"),
            ToolMessage(
                content=_toolnode_list(_USER_HIT),
                tool_call_id="1",
                name="search_memory",
            ),
        ],
    )
    assert write_has_turn_evidence("USER: drinks dark roast every morning", ev)
    assert not write_has_turn_evidence(
        "USER: copper crashed after the press drop tonight", ev
    )
