"""Full branch coverage for hallucinated tool-call recovery and arg validation."""

from __future__ import annotations

import json

import pytest

from src.graph import htc_recovery as htc

ALLOWED = frozenset(
    {
        "search_memory",
        "save_memory",
        "save_fact_to_graph",
        "add_triple",
        "search_graph",
        "find_related",
        "query_graph",
    }
)


@pytest.mark.unit
def test_recover_strips_and_rejects_non_json_prefix():
    assert htc.recover_hallucinated_tool_call("  ", allowed_tool_names=ALLOWED) is None
    assert (
        htc.recover_hallucinated_tool_call("not json", allowed_tool_names=ALLOWED)
        is None
    )


@pytest.mark.unit
def test_recover_valid_dict_parameters_and_args_aliases():
    raw = json.dumps(
        {"name": "search_memory", "parameters": {"query": "some topic here"}}
    )
    out = htc.recover_hallucinated_tool_call(raw, allowed_tool_names=ALLOWED)
    assert out == {"name": "search_memory", "args": {"query": "some topic here"}}

    raw2 = json.dumps({"name": "save_memory", "args": {"content": "hello world"}})
    out2 = htc.recover_hallucinated_tool_call(raw2, allowed_tool_names=ALLOWED)
    assert out2 == {"name": "save_memory", "args": {"content": "hello world"}}


@pytest.mark.unit
def test_recover_dict_non_dict_args_becomes_empty():
    raw = json.dumps({"name": "search_memory", "parameters": "nope"})
    out = htc.recover_hallucinated_tool_call(raw, allowed_tool_names=ALLOWED)
    assert out == {"name": "search_memory", "args": {}}


@pytest.mark.unit
def test_recover_dict_name_not_allowed():
    raw = json.dumps({"name": "unknown_tool", "args": {}})
    assert htc.recover_hallucinated_tool_call(raw, allowed_tool_names=ALLOWED) is None


@pytest.mark.unit
def test_recover_prefix_fallback_when_json_invalid():
    # Valid fragment for prefix RE; json.loads fails so we parse query via regex.
    blob = '{"name":"search_memory","query":"my search text"'
    out = htc.recover_hallucinated_tool_call(blob, allowed_tool_names=ALLOWED)
    assert out == {"name": "search_memory", "args": {"query": "my search text"}}


@pytest.mark.unit
def test_recover_invalid_json_prefix_save_memory_long_content():
    # Prefix RE only needs opening fragment; add invalid tail so json.loads fails
    blob2 = '{"name":"save_memory","content":"' + ("y" * 25) + '" trailing'
    out = htc.recover_hallucinated_tool_call(blob2, allowed_tool_names=ALLOWED)
    assert out == {"name": "save_memory", "args": {"content": "y" * 25}}


@pytest.mark.unit
def test_recover_prefix_subject_key():
    text = '{"name":"find_related","subject":"entity name here"'  # invalid JSON
    out = htc.recover_hallucinated_tool_call(text, allowed_tool_names=ALLOWED)
    assert out == {"name": "find_related", "args": {"subject": "entity name here"}}


@pytest.mark.unit
def test_recover_prefix_fact_key_infers_name():
    text = '{"name":"query_graph","fact":"abc longer than three"'  # invalid JSON
    out = htc.recover_hallucinated_tool_call(text, allowed_tool_names=ALLOWED)
    assert out == {"name": "query_graph", "args": {"fact": "abc longer than three"}}


@pytest.mark.unit
def test_recover_prefix_name_not_allowed():
    text = '{"name":"bad_tool","query":"something longer"}'
    assert htc.recover_hallucinated_tool_call(text, allowed_tool_names=ALLOWED) is None


@pytest.mark.unit
def test_recover_prefix_query_too_short_no_match():
    text = '{"name":"search_memory","query":"ab"'  # invalid JSON; query too short for second RE
    assert htc.recover_hallucinated_tool_call(text, allowed_tool_names=ALLOWED) is None


@pytest.mark.unit
def test_validate_rejects_non_dict_args():
    assert htc.validate_recovered_tool_args("search_memory", []) is None


@pytest.mark.unit
def test_validate_search_memory_paths():
    assert htc.validate_recovered_tool_args("search_memory", {"query": ""}) is None
    assert htc.validate_recovered_tool_args("search_memory", {"query": "short"}) == {
        "query": "short"
    }
    assert htc.validate_recovered_tool_args(
        "search_memory", {"query": "valid enough phrase"}
    ) == {"query": "valid enough phrase"}


@pytest.mark.unit
def test_validate_save_memory():
    assert htc.validate_recovered_tool_args("save_memory", {"content": ""}) is None
    assert htc.validate_recovered_tool_args(
        "save_memory", {"content": "  real content  "}
    ) == {"content": "real content"}
    long_c = "c" * 5000
    assert htc.validate_recovered_tool_args("save_memory", {"content": long_c}) == {
        "content": "c" * 4000
    }


@pytest.mark.unit
def test_validate_save_fact_to_graph_object_str():
    assert htc.validate_recovered_tool_args(
        "save_fact_to_graph",
        {"subject": "s", "predicate": "p", "object_str": "o"},
    ) == {"subject": "s", "predicate": "p", "object": "o"}
    assert (
        htc.validate_recovered_tool_args(
            "save_fact_to_graph", {"subject": "", "predicate": "p", "object": "o"}
        )
        is None
    )


@pytest.mark.unit
def test_validate_add_triple_object_alias():
    assert htc.validate_recovered_tool_args(
        "add_triple",
        {"subject": "a", "predicate": "b", "object": "c"},
    ) == {"subject": "a", "predicate": "b", "object_entity": "c"}


@pytest.mark.unit
def test_validate_add_triple_rejects_empty_object_entity():
    assert (
        htc.validate_recovered_tool_args(
            "add_triple",
            {"subject": "a", "predicate": "b", "object_entity": ""},
        )
        is None
    )


@pytest.mark.unit
def test_validate_search_graph():
    assert htc.validate_recovered_tool_args("search_graph", {"query": ""}) is None
    assert htc.validate_recovered_tool_args("search_graph", {"query": "x"}) == {
        "query": "x"
    }
    assert htc.validate_recovered_tool_args(
        "search_graph", {"query": "long enough two words"}
    ) == {"query": "long enough two words"}


@pytest.mark.unit
def test_validate_find_related_depth_and_predicates():
    assert htc.validate_recovered_tool_args("find_related", {"name": ""}) is None
    assert htc.validate_recovered_tool_args("find_related", {"name": "n"}) == {
        "name": "n"
    }
    assert htc.validate_recovered_tool_args(
        "find_related", {"name": "n", "depth": "2"}
    ) == {"name": "n", "depth": 2}
    assert (
        htc.validate_recovered_tool_args("find_related", {"name": "n", "depth": 0})
        is None
    )
    assert (
        htc.validate_recovered_tool_args("find_related", {"name": "n", "depth": 4})
        is None
    )
    assert (
        htc.validate_recovered_tool_args("find_related", {"name": "n", "depth": "nope"})
        is None
    )
    assert htc.validate_recovered_tool_args(
        "find_related",
        {"name": "n", "predicates": ["owns", " ", "runs"]},
    ) == {"name": "n", "predicates": ["owns", "runs"]}
    assert (
        htc.validate_recovered_tool_args(
            "find_related", {"name": "n", "predicates": "bad"}
        )
        is None
    )


@pytest.mark.unit
def test_validate_query_graph_types_and_minimum():
    assert (
        htc.validate_recovered_tool_args(
            "query_graph", {"subject": 1, "predicate": None, "object_str": None}
        )
        is None
    )
    assert (
        htc.validate_recovered_tool_args(
            "query_graph",
            {"subject": None, "predicate": 99, "object_str": None},
        )
        is None
    )
    assert (
        htc.validate_recovered_tool_args(
            "query_graph",
            {"subject": "a", "predicate": "b", "object_str": object()},
        )
        is None
    )
    assert (
        htc.validate_recovered_tool_args(
            "query_graph",
            {"subject": None, "predicate": None, "object_str": None},
        )
        is None
    )
    assert htc.validate_recovered_tool_args(
        "query_graph",
        {"subject": "  x  ", "predicate": None, "object_str": None},
    ) == {"subject": "x", "predicate": None, "object_str": None}


@pytest.mark.unit
def test_validate_recovered_web_tools():
    assert htc.validate_recovered_tool_args("search_web", {"query": "ab"}) == {
        "query": "ab"
    }
    assert htc.validate_recovered_tool_args("search_web", {"query": "x"}) is None
    assert htc.validate_recovered_tool_args(
        "fetch_url", {"url": "https://example.com"}
    ) == {"url": "https://example.com"}
    assert htc.validate_recovered_tool_args("fetch_url", {"url": "  "}) is None


@pytest.mark.unit
def test_validate_unknown_tool():
    assert htc.validate_recovered_tool_args("unknown_x", {}) is None


@pytest.mark.unit
def test_schedule_intent_htc_arguments_and_validate():
    allowed = frozenset({"schedule_intent"})
    recovered = htc.recover_hallucinated_tool_call(
        '{"name":"schedule_intent","arguments":{"text":"write a test","when":"in 3 min"}}',
        allowed_tool_names=allowed,
    )
    assert recovered == {
        "name": "schedule_intent",
        "args": {"text": "write a test", "when": "in 3 min"},
    }
    assert htc.validate_recovered_tool_args("schedule_intent", recovered["args"]) == {
        "text": "write a test",
        "when": "in 3 min",
    }
    assert htc.validate_recovered_tool_args("schedule_intent", {"text": "x"}) is None


@pytest.mark.unit
def test_try_materialize_htc_tool_call():
    from langchain_core.messages import AIMessage

    allowed = frozenset({"schedule_intent"})
    msg = AIMessage(
        content='{"name":"schedule_intent","arguments":{"text":"t","when":"in 1 min"}}'
    )
    out = htc.try_materialize_htc_tool_call(msg, allowed_tool_names=allowed)
    assert out is not None
    assert out.tool_calls[0]["name"] == "schedule_intent"
    assert out.tool_calls[0]["args"] == {"text": "t", "when": "in 1 min"}
    assert str(out.tool_calls[0]["id"]).startswith("htc_")
    assert htc.try_materialize_htc_tool_call(out, allowed_tool_names=allowed) is None


@pytest.mark.unit
def test_recover_qwen_xml_message_user_incident_sample():
    """2026-07-11: model dumped tool XML in content → Telegram never got the report."""
    allowed = frozenset({"message_user", "run_selftest"})
    # Exact shape from mind_events agent_out on intent-06c02e48 (think + Qwen XML).
    blob = (
        "<think>\nSelftest done; report to USER via message_user.\n</think>\n"
        "<tool_call>\n"
        "<function=message_user>\n"
        "<parameter=text>\n"
        "Intent completed on time. Selftest: ok.\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    recovered = htc.recover_hallucinated_tool_call(blob, allowed_tool_names=allowed)
    assert recovered is not None
    assert recovered["name"] == "message_user"
    assert "Intent completed on time" in recovered["args"]["text"]
    assert htc.validate_recovered_tool_args("message_user", recovered["args"]) == {
        "text": "Intent completed on time. Selftest: ok."
    }

    from langchain_core.messages import AIMessage

    out = htc.try_materialize_htc_tool_call(
        AIMessage(content=blob), allowed_tool_names=allowed
    )
    assert out is not None
    assert out.tool_calls[0]["name"] == "message_user"
    assert out.content == ""


@pytest.mark.unit
def test_recover_xml_rejects_unknown_tool():
    blob = "<function=not_a_tool><parameter=text>hi</parameter>"
    assert (
        htc.recover_xml_hallucinated_tool_call(
            blob, allowed_tool_names=frozenset({"message_user"})
        )
        is None
    )


@pytest.mark.unit
def test_materialize_qwen_xml_batch_cancel_and_reschedule():
    """2026-07-19: second pass dumped cancel+schedule XML; must recover both calls."""
    from langchain_core.messages import AIMessage

    allowed = frozenset({"cancel_intent", "schedule_intent", "list_intents"})
    blob = (
        "<tool_call>\n"
        "<function=cancel_intent>\n"
        "<parameter=intent_id>\n"
        "072aa731\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=schedule_intent>\n"
        "<parameter=text>\n"
        "test passed\n"
        "</parameter>\n"
        "<parameter=when>\n"
        "in 5 minutes\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    items = htc.recover_xml_hallucinated_tool_calls(blob, allowed_tool_names=allowed)
    assert [i["name"] for i in items] == ["cancel_intent", "schedule_intent"]
    assert items[0]["args"] == {"intent_id": "072aa731"}
    assert items[1]["args"] == {"text": "test passed", "when": "in 5 minutes"}

    out = htc.try_materialize_htc_tool_call(
        AIMessage(content=blob), allowed_tool_names=allowed
    )
    assert out is not None
    assert out.content == ""
    assert [tc["name"] for tc in out.tool_calls] == [
        "cancel_intent",
        "schedule_intent",
    ]
    assert out.tool_calls[0]["args"] == {"intent_id": "072aa731"}
    assert out.tool_calls[1]["args"] == {
        "text": "test passed",
        "when": "in 5 minutes",
    }


@pytest.mark.unit
def test_agent_node_htc_skipped_when_tools_capped(monkeypatch):
    """Capped pass (use_tools=False): do not materialize HTC — avoids tool loop."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    import src.graph.workflow as w

    blob = (
        "<tool_call>\n"
        "<function=cancel_intent>\n"
        "<parameter=intent_id>\n072aa731\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n"
        "<function=schedule_intent>\n"
        "<parameter=text>\ntest passed\n</parameter>\n"
        "<parameter=when>\nin 5 minutes\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    class _LLM:
        def invoke(self, _msgs, **_kwargs):
            return AIMessage(
                content=blob,
                additional_kwargs={
                    "reasoning_content": "The user wants me to test scheduling tools."
                },
            )

    from tests.support.llm_mocks import patch_agent_llm

    patch_agent_llm(monkeypatch, _LLM())
    monkeypatch.setattr(w.settings, "max_tool_rounds_per_turn", 1, raising=False)

    state = {
        "messages": [
            HumanMessage(content="schedule test"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "schedule_intent",
                        "id": "c1",
                        "args": {"text": "x", "when": "in 10 minutes"},
                    },
                    {"name": "list_intents", "id": "c2", "args": {}},
                ],
            ),
            ToolMessage(
                content="Scheduled: x", tool_call_id="c1", name="schedule_intent"
            ),
            ToolMessage(
                content="072aa731 — pending", tool_call_id="c2", name="list_intents"
            ),
        ],
        "retrieved_memories": [],
        "current_time": "2026-07-19T13:47:00",
        "session_summary": "",
        "tool_rounds": 1,
    }
    out = w.agent_node(state)  # type: ignore[arg-type]
    ai = out["messages"][0]
    assert isinstance(ai, AIMessage)
    assert not (getattr(ai, "tool_calls", None) or [])


@pytest.mark.unit
def test_recover_json_decode_error_triggers_prefix():
    assert (
        htc.recover_hallucinated_tool_call("{garbage", allowed_tool_names=ALLOWED)
        is None
    )


@pytest.mark.unit
def test_recover_json_list_skips_dict_branch():
    assert htc.recover_hallucinated_tool_call("[1]", allowed_tool_names=ALLOWED) is None


@pytest.mark.unit
def test_validate_search_memory_truncates_query():
    long_q = "hello world " + "x" * 3000
    assert htc.validate_recovered_tool_args("search_memory", {"query": long_q}) == {
        "query": long_q[:2000]
    }


@pytest.mark.unit
def test_validate_non_string_fields_rejected():
    assert htc.validate_recovered_tool_args("search_memory", {"query": 123}) is None
    assert (
        htc.validate_recovered_tool_args(
            "save_fact_to_graph",
            {"subject": "s", "predicate": "p", "object": 1},
        )
        is None
    )
