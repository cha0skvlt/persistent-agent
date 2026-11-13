from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.config import settings
from src.graph import summary_node as sn
from src.state.agent_state import _messages_reducer


class _StubLLM:
    __slots__ = ()

    def invoke(self, _msgs, **_kwargs):
        return SimpleNamespace(content="")


_STUB_LLM = _StubLLM()


def test_is_first_turn_after_summary():
    assert sn.is_first_turn_after_summary({"turn_count": 21, "last_summary_turn": 20})
    assert not sn.is_first_turn_after_summary(
        {"turn_count": 22, "last_summary_turn": 20}
    )
    assert not sn.is_first_turn_after_summary(
        {"turn_count": 21, "last_summary_turn": 0}
    )


def test_session_summary_sys_block_post_summary_guard():
    # First turn after compress: omit the summary wall entirely so the model
    # does not treat it as the user prompt.
    block = sn.session_summary_sys_block(
        {
            "session_summary": "User lives in Lisbon.",
            "turn_count": 21,
            "last_summary_turn": 20,
        }
    )
    assert block is None


def test_session_summary_sys_block_normal_turn():
    block = sn.session_summary_sys_block(
        {
            "session_summary": "Prior facts.",
            "turn_count": 25,
            "last_summary_turn": 20,
        }
    )
    assert block is not None
    assert "just refreshed" not in block


def test_transcript_for_summary_strips_thinking():
    ai = AIMessage(content=("<think>Q: status.</think>\n" "Alive."))
    line = sn._transcript_for_summary(ai)
    assert line == "Agent: Alive."
    assert "redacted_thinking" not in line


def test_effective_session_offset_restart_uses_full_message_count():
    """On uuid mismatch / missing anchor, offset is len(msgs), not len-1."""
    assert (
        sn._effective_session_offset(
            {
                "session_uuid": "new",
                "session_uuid_anchor": "old",
                "session_message_offset": 0,
            },
            5,
        )
        == 5
    )
    assert (
        sn._effective_session_offset(
            {
                "session_uuid": "new",
                "session_uuid_anchor": "",
                "session_message_offset": 0,
            },
            3,
        )
        == 3
    )


def test_summarize_session_fires_at_turn_twenty_when_every_n_twenty(monkeypatch):
    """At turn_count 20 and SUMMARY_EVERY_N=20, node runs."""
    monkeypatch.setattr(settings, "summary_every_n", 20, raising=False)
    monkeypatch.setattr(settings, "summary_max_chars", 1500, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 30, raising=False)

    monkeypatch.setattr(
        sn,
        "_incremental_summarize",
        lambda prior_summary, messages, max_chars, llm: "AT20",
        raising=True,
    )

    base = {"messages": [HumanMessage(content="x")] * 19, "turn_count": 19}
    assert sn.summarize_session(base, {}, llm=_STUB_LLM) == {}  # type: ignore[arg-type]
    out = sn.summarize_session(  # type: ignore[arg-type]
        {
            "messages": [HumanMessage(content=f"m{i}", id=f"m{i}") for i in range(20)],
            "turn_count": 20,
        },
        {},
        llm=_STUB_LLM,
    )
    assert out.get("session_summary") == "AT20"
    assert out.get("last_summary_turn") == 20
    if out.get("messages"):
        base = [HumanMessage(content=f"m{i}", id=f"m{i}") for i in range(20)]
        trimmed = list(_messages_reducer(base, out["messages"]))
        assert len(trimmed) <= int(settings.context_window_messages)


def test_summarize_session_noop_when_not_enough(monkeypatch):
    monkeypatch.setattr(settings, "summary_every_n", 20, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 1, raising=False)
    state = {"messages": [HumanMessage(content="a")] * 39, "turn_count": 19}
    out = sn.summarize_session(state, {}, llm=_STUB_LLM)  # type: ignore[arg-type]
    assert out == {}


def test_summarize_session_caps_chunk_sent_to_compress(monkeypatch):
    monkeypatch.setattr(settings, "summary_every_n", 2, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 1, raising=False)
    monkeypatch.setattr(settings, "summary_llm_max_messages", 2, raising=False)

    seen: dict[str, int] = {}

    def capture(prior_summary, messages, max_chars, llm):
        seen["len"] = len(messages)
        return "OK"

    monkeypatch.setattr(sn, "_incremental_summarize", capture, raising=True)

    state = {
        "messages": [HumanMessage(content=f"x{i}") for i in range(6)],
        "turn_count": 6,
    }
    sn.summarize_session(state, {}, llm=_STUB_LLM)  # type: ignore[arg-type]
    assert seen["len"] == 2


def test_summarize_session_prefix_scoped_to_session_after_offset(monkeypatch):
    monkeypatch.setattr(settings, "summary_every_n", 2, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 1, raising=False)
    monkeypatch.setattr(settings, "summary_llm_max_messages", 50, raising=False)

    captured: dict[str, list] = {}

    def capture(prior_summary, messages, max_chars, llm):
        captured["messages"] = list(messages)
        return "SUM"

    monkeypatch.setattr(sn, "_incremental_summarize", capture, raising=True)

    old = [HumanMessage(content="checkpoint_old")]
    session = [HumanMessage(content=f"s{i}") for i in range(4)]
    state = {
        "messages": old + session,
        "session_message_offset": 1,
        "session_uuid": "u1",
        "session_uuid_anchor": "u1",
        "turn_count": 4,
    }
    sn.summarize_session(state, {}, llm=_STUB_LLM)  # type: ignore[arg-type]

    contents = [getattr(m, "content", "") for m in captured["messages"]]
    assert "checkpoint_old" not in contents
    assert "s0" in contents


def test_summarize_session_noop_when_session_uuid_mismatches_anchor(monkeypatch):
    """Stale checkpoint offset must not expand the window before update_time runs."""
    monkeypatch.setattr(settings, "summary_every_n", 2, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 1, raising=False)

    called: list[bool] = []

    def capture(prior_summary, messages, max_chars, llm):
        called.append(True)
        return "X"

    monkeypatch.setattr(sn, "_incremental_summarize", capture, raising=True)

    state = {
        "messages": [HumanMessage(content=str(i)) for i in range(10)],
        "session_message_offset": 0,
        "session_uuid": "new-run",
        "session_uuid_anchor": "old-run",
        "turn_count": 10,
    }
    assert sn.summarize_session(state, {}, llm=_STUB_LLM) == {}  # type: ignore[arg-type]
    assert called == []


def test_summarize_session_node_uses_llm(monkeypatch):
    """summarize_session_node wires the shared chat LLM."""
    import src.graph.summary_node as sn_mod
    import src.llm_factory as lf

    chosen: list[str] = []

    def _track_llm():
        chosen.append("llm")
        return _STUB_LLM

    monkeypatch.setattr(lf, "get_llm", _track_llm, raising=True)
    monkeypatch.setattr(
        sn,
        "_incremental_summarize",
        lambda prior_summary, messages, max_chars, llm: "SUM",
        raising=True,
    )
    monkeypatch.setattr(settings, "summary_every_n", 2, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 2, raising=False)

    state = {
        "messages": [HumanMessage(content="a"), HumanMessage(content="b")],
        "turn_count": 2,
    }
    out = sn_mod.summarize_session_node(state, {})  # type: ignore[arg-type]
    assert out.get("session_summary") == "SUM"
    assert chosen == ["llm"]


def test_summarize_session_cap_defaults_when_env_zero(monkeypatch):
    monkeypatch.setattr(settings, "summary_every_n", 2, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 1, raising=False)
    monkeypatch.setattr(settings, "summary_llm_max_messages", 0, raising=False)

    seen: dict[str, int] = {}

    def capture(prior_summary, messages, max_chars, llm):
        seen["len"] = len(messages)
        return "OK"

    monkeypatch.setattr(sn, "_incremental_summarize", capture, raising=True)

    state = {"messages": [HumanMessage(content="x")] * 200, "turn_count": 200}
    sn.summarize_session(state, {}, llm=_STUB_LLM)  # type: ignore[arg-type]
    assert seen["len"] == 80


def test_session_summary_due_skips_already_summarized_turn(monkeypatch):
    monkeypatch.setattr(settings, "summary_every_n", 8, raising=False)
    monkeypatch.setattr(settings, "context_window_messages", 12, raising=False)
    state = {
        "turn_count": 8,
        "last_summary_turn": 8,
        "messages": [
            HumanMessage(content="one"),
            HumanMessage(content="two"),
        ],
        "session_message_offset": 0,
    }
    assert sn.session_summary_due(state) is False
