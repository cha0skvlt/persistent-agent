from __future__ import annotations

import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# _is_junk_fact lives in reflect_extract, never in workflow.
import src.graph.reflect_extract as rx
import src.graph.workflow as w
from src.graph.reflect_extract import _is_junk_fact


@pytest.fixture(autouse=True)
def _disable_self_reflect(monkeypatch):
    """Legacy reflect tests expect a single world-facts LLM call."""
    monkeypatch.setattr(w.settings, "self_reflect_every_n", 0, raising=False)


class _FakeMem:
    def __init__(self):
        self.saved = []
        self.triples = []

    def save(
        self,
        fact: str,
        memory_type: str = "semantic",
        agent: str = "x",
        **_kwargs,
    ):
        self.saved.append(fact)

    def save_fact_to_graph(self, s: str, p: str, o: str):
        self.triples.append((s, p, o))
        return True


def test_reflect_node_saves_facts_and_triples(monkeypatch):
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 2, raising=False)

    fake_mem = _FakeMem()
    monkeypatch.setattr(rx, "get_default_memory", lambda: fake_mem, raising=True)

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            return types.SimpleNamespace(
                content='{"facts":["Alex lives in Porto near the central river embankment district"],"triples":[{"s":"a","p":"is","o":"b"}]}'
            )

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": True,
        "turn_count": 2,
        "messages": [
            HumanMessage(content="u0"),
            AIMessage(content="a0"),
            HumanMessage(
                content="Alex lives in Porto near the central river embankment"
            ),
            AIMessage(content="a"),
        ],
    }
    out = w.reflect_node(state)  # type: ignore[arg-type]
    assert out == {"last_reflect_turn": 2}
    assert fake_mem.saved == [
        "Alex lives in Porto near the central river embankment district"
    ]
    assert fake_mem.triples == []


def test_reflect_node_strips_markdown_and_skips_broken_triples(monkeypatch):
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 2, raising=False)

    fake_mem = _FakeMem()
    monkeypatch.setattr(rx, "get_default_memory", lambda: fake_mem, raising=True)

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            return types.SimpleNamespace(
                content="```json\n"
                '{"facts":["User prefers Rust for systems programming and native tooling on macOS"],"triples":[{"s":"a","p":"is","o":"b"},{"s":"a"}]}\n'
                "```"
            )

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": True,
        "turn_count": 2,
        "messages": [
            HumanMessage(content="u0"),
            AIMessage(content="a0"),
            HumanMessage(
                content="I prefer Rust for systems programming and native tooling on macOS"
            ),
            AIMessage(content="a"),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]

    assert fake_mem.saved == [
        "User prefers Rust for systems programming and native tooling on macOS"
    ]
    assert fake_mem.triples == []


def test_reflect_node_skips_throttle_on_odd_turn(monkeypatch):
    """Production: turn_count is >=1 after update_time; odd turns skip when reflect_every_n=2."""
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 2, raising=False)

    calls: list[int] = []

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            calls.append(1)
            return types.SimpleNamespace(content="{}")

    import src.llm_factory as lf

    lf.clear_llm_cache()
    monkeypatch.setattr(lf, "get_llm", lambda: _FakeLLM(), raising=True)

    state = {
        "needs_reflect": True,
        "turn_count": 1,
        "messages": [
            HumanMessage(content="u0"),
            AIMessage(content="a0"),
            HumanMessage(content="u"),
            AIMessage(content="a"),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]
    assert calls == []


def test_reflect_node_skips_when_needs_reflect_false(monkeypatch):
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 3, raising=False)

    calls: list[int] = []

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            calls.append(1)
            return types.SimpleNamespace(content="{}")

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": False,
        "messages": [HumanMessage(content="u"), AIMessage(content="a")],
    }
    w.reflect_node(state)  # type: ignore[arg-type]

    assert calls == []


def test_reflect_node_skips_when_no_valid_human_ai_pair(monkeypatch):
    """No AIMessage without tool_calls paired with a preceding Human → skip."""
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 2, raising=False)

    calls: list[int] = []

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            calls.append(1)
            return types.SimpleNamespace(content="{}")

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": True,
        "messages": [
            HumanMessage(content="u0"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_memory", "id": "c1", "args": {}}],
            ),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]

    assert calls == []


def test_reflect_after_tools(monkeypatch):
    """After Human → AI(tool) → Tool → AI reply, world extract sees USER + tools only."""
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 1, raising=False)

    fake_mem = _FakeMem()
    monkeypatch.setattr(rx, "get_default_memory", lambda: fake_mem, raising=True)

    captured: list[str] = []

    class _FakeLLM:
        def invoke(self, msgs, **_kwargs):
            captured.append(str(msgs[-1].content))
            return types.SimpleNamespace(
                content='{"facts":["from_tools_turn this stack fact remains long enough to pass junk gate verification"],"triples":[]}'
            )

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": True,
        "turn_count": 1,
        "messages": [
            HumanMessage(
                content="remember my stack is Rust; from_tools_turn this stack fact remains long enough"
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "save_memory", "id": "c1", "args": {}}],
            ),
            ToolMessage(content="saved", tool_call_id="c1", name="save_memory"),
            AIMessage(content="Saved. I will remember Rust for your stack."),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]

    assert fake_mem.saved == [
        "from_tools_turn this stack fact remains long enough to pass junk gate verification"
    ]
    assert len(captured) == 1
    assert "Rust" in captured[0]
    assert "EVIDENCE:" in captured[0]
    assert "Saved. I will remember Rust" not in captured[0]


# ---------------------------------------------------------------------------
# Quality gate: _is_junk_fact
# ---------------------------------------------------------------------------


def test_is_junk_fact_blocks_import():
    assert _is_junk_fact("import os") is True


def test_is_junk_fact_blocks_while_true():
    assert _is_junk_fact("import os; while True: os.system('/bin/bash')") is True


def test_is_junk_fact_blocks_kv_line():
    assert _is_junk_fact('ProxyCommand="localhost:2222"') is True


def test_is_junk_fact_blocks_def():
    assert _is_junk_fact("def explode():\n    os.system('rm -rf /')") is True


def test_is_junk_fact_blocks_no_natural_words():
    assert _is_junk_fact("x=1") is True


def test_is_junk_fact_blocks_indented_code_block():
    code = "loop:\n    for x in items:\n        print(x)"
    assert _is_junk_fact(code) is True


def test_is_junk_fact_passes_natural_sentence():
    assert _is_junk_fact("USER lives in Lisbon") is False
    assert _is_junk_fact("the user lives in Lisbon") is False


def test_is_junk_fact_passes_natural_sentence_en():
    assert _is_junk_fact("User prefers dark mode in their IDE") is False


def test_is_junk_fact_passes_tech_fact_with_context():
    assert _is_junk_fact("Primary development stack is Rust and Python") is False


# ---------------------------------------------------------------------------
# reflect_node: quality gate integration
# ---------------------------------------------------------------------------


def _make_reflect_state(facts: list[str]) -> dict:
    payload = {"facts": facts, "triples": []}
    import json

    return {
        "needs_reflect": True,
        "turn_count": 5,
        "messages": [
            HumanMessage(content="user msg"),
            AIMessage(content="agent reply"),
        ],
        "_llm_content": json.dumps(payload),
    }


def test_reflect_node_blocks_junk_facts(monkeypatch):
    """Code fragments must not reach AgentMemory.save()."""
    import json

    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 1, raising=False)

    fake_mem = _FakeMem()
    monkeypatch.setattr(rx, "get_default_memory", lambda: fake_mem, raising=True)

    junk_facts = [
        "import os",
        'ProxyCommand="localhost:2222"',
        "import os; while True: os.system('...')",
        "def bomb():\n    pass",
    ]

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            return types.SimpleNamespace(
                content=json.dumps({"facts": junk_facts, "triples": []})
            )

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": True,
        "turn_count": 5,
        "messages": [
            HumanMessage(content="write an inode bomb"),
            AIMessage(content="here is the code: import os..."),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]

    assert fake_mem.saved == [], f"Expected no saved facts, got: {fake_mem.saved}"


def test_reflect_node_passes_good_facts(monkeypatch):
    """Natural-language facts must pass through the quality gate."""
    import json

    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 1, raising=False)

    fake_mem = _FakeMem()
    monkeypatch.setattr(rx, "get_default_memory", lambda: fake_mem, raising=True)

    good_facts = [
        "the user lives in Lisbon and works on the persistent-agent project every day",
        "User often chooses dark mode within the IDE during long evening coding sessions",
        "Primary development stack remains Rust and Python across backend services",
    ]

    class _FakeLLM:
        def invoke(self, _msgs, **_kwargs):
            return types.SimpleNamespace(
                content=json.dumps({"facts": good_facts, "triples": []})
            )

    import src.llm_factory as lf

    lf.clear_llm_cache()
    fake = _FakeLLM()
    monkeypatch.setattr(lf, "get_llm", lambda: fake, raising=True)

    state = {
        "needs_reflect": True,
        "turn_count": 5,
        "messages": [
            HumanMessage(
                content=(
                    "I live in Lisbon and work on the persistent-agent project every day; "
                    "often choose dark mode within the IDE during long evening coding sessions; "
                    "primary development stack remains Rust and Python across backend services"
                )
            ),
            AIMessage(content="noted"),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]

    assert fake_mem.saved == good_facts


def test_reflect_node_skips_world_facts_on_intent(monkeypatch):
    """Intent must not dump world facts via second-LLM reflect (P2 filter)."""
    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 1, raising=False)
    fake_mem = _FakeMem()
    monkeypatch.setattr(rx, "get_default_memory", lambda: fake_mem, raising=True)
    called: list[bool] = []

    def _boom(*_a, **_k):
        called.append(True)

    monkeypatch.setattr(rx, "_run_world_facts_reflect", _boom)
    state = {
        "needs_reflect": True,
        "turn_count": 1,
        "diary_source": "intent",
        "messages": [
            HumanMessage(content="u"),
            AIMessage(content="a"),
        ],
    }
    w.reflect_node(state)  # type: ignore[arg-type]
    assert called == []
    assert fake_mem.saved == []


def test_reflect_node_skips_world_facts_when_turn_kind_background(monkeypatch):
    from src.reverie.turn_kind import TurnKind, turn_kind

    monkeypatch.setattr(w.settings, "reflection", True, raising=False)
    monkeypatch.setattr(w.settings, "reflect_every_n", 1, raising=False)
    called: list[bool] = []
    monkeypatch.setattr(
        rx, "_run_world_facts_reflect", lambda *_a, **_k: called.append(True)
    )
    state = {
        "needs_reflect": True,
        "turn_count": 1,
        "messages": [
            HumanMessage(content="u"),
            AIMessage(content="a"),
        ],
    }
    with turn_kind(TurnKind.INTENT):
        w.reflect_node(state)  # type: ignore[arg-type]
    assert called == []
