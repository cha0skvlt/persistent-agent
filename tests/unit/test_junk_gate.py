"""junk_gate and KG curation unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.junk_gate import is_junk
from src.kg.predicates import ALLOWED_PREDICATES, is_allowed


@pytest.mark.unit
def test_is_junk_rejects_greeting_and_short():
    assert is_junk("Shalom agent!", user_text="Shalom", agent_text="Hello!")
    assert is_junk("hi there")
    assert not is_junk(
        "USER believes time is linear yet illusory and that matters to him"
    )
    assert not is_junk(
        "the user thinks time is linear yet illusory and that matters to him"
    )


@pytest.mark.unit
def test_predicate_whitelist_contains_seed_predicates():
    for pred in ("alias_of", "lives_in", "runs_on", "themes", "works_as"):
        assert is_allowed(pred)
    assert not is_allowed("uses")
    assert "is" in ALLOWED_PREDICATES


@pytest.mark.unit
def test_run_world_facts_reflect_code_fragment_blocked(monkeypatch):
    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    fact = (
        "import os and subprocess modules for remote deployment automation "
        "scripts on production servers"
    )
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: f'{{"facts": ["{fact}"], "triples": []}}',
    )
    mem = MagicMock()
    events: list[str] = []

    def capture(event: str, **fields):
        if event == "reflect.blocked":
            events.append(fields.get("reason", ""))

    monkeypatch.setattr(rx, "log_event", capture)
    rx._run_world_facts_reflect("u", "a", mem, turn_count=1)
    assert "code_fragment" in events
    mem.save.assert_not_called()


@pytest.mark.unit
def test_run_world_facts_reflect_routes_fact_to_memory(monkeypatch):
    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    user = "Mars has several small moons around the equatorial plane"
    fact = "Mars has several small moons orbiting around its equatorial plane always"
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: f'{{"facts": ["{fact}"], "triples": []}}',
    )
    mem = MagicMock()
    rx._run_world_facts_reflect(user, "a", mem, turn_count=1)
    mem.save.assert_called_once()
    mem.save_fact_to_graph.assert_not_called()


@pytest.mark.unit
def test_live_reflect_skips_fact_invented_from_agent_text(monkeypatch):
    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    fact = "Mars has several small moons orbiting around its equatorial plane always"
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: f'{{"facts": ["{fact}"], "triples": []}}',
    )
    mem = MagicMock()
    rx._run_world_facts_reflect("hello there how are you today", "a", mem, turn_count=1)
    mem.save.assert_not_called()


@pytest.mark.unit
def test_live_reflect_saves_user_quote_from_human(monkeypatch):
    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    user = "I drink green tea every morning before work near the river"
    fact = "USER: drinks green tea every morning before work"
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: f'{{"facts": ["{fact}"], "triples": []}}',
    )
    mem = MagicMock()
    rx._run_world_facts_reflect(user, "noted", mem, turn_count=1)
    mem.save.assert_called_once()


@pytest.mark.unit
def test_live_reflect_user_prefix_cannot_borrow_web_only(monkeypatch):
    from langchain_core.messages import HumanMessage, ToolMessage

    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    user = "search the web for ollama news today"
    fact = "USER: copper crashed after the press drop tonight"
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: f'{{"facts": ["{fact}"], "triples": []}}',
    )
    mem = MagicMock()
    msgs = [
        HumanMessage(content=user),
        ToolMessage(
            content=("LIVE WEB: copper crashed after the press drop tonight near 100k"),
            tool_call_id="1",
            name="search_web",
        ),
    ]
    rx._run_world_facts_reflect(user, "done", mem, turn_count=1, messages=msgs)
    mem.save.assert_not_called()


@pytest.mark.unit
def test_run_world_facts_reflect_junk_gate_blocked(monkeypatch):
    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: '{"facts": ["hello this is a long fact that should still be rejected by junk gate"], "triples": []}',
    )
    mem = MagicMock()
    events: list[str] = []

    def capture(event: str, **fields):
        if event == "reflect.blocked":
            events.append(fields.get("reason", ""))

    monkeypatch.setattr(rx, "log_event", capture)
    rx._run_world_facts_reflect("u", "a", mem, turn_count=1)
    assert "junk_gate" in events
    mem.save.assert_not_called()


@pytest.mark.unit
def test_run_world_facts_reflect_ignores_triples(monkeypatch):
    import src.graph.reflect_extract as rx

    monkeypatch.setattr(rx.settings, "reflect_every_n", 1)
    monkeypatch.setattr(
        rx,
        "_invoke_reflect_llm",
        lambda *a, **k: '{"facts": [], "triples": [{"s":"mars","p":"has","o":"moon"}]}',
    )
    mem = MagicMock()
    events: list[str] = []

    def capture(event: str, **fields):
        if event == "reflect.blocked":
            events.append(fields.get("reason", ""))

    monkeypatch.setattr(rx, "log_event", capture)
    rx._run_world_facts_reflect("u", "a", mem, turn_count=1)
    assert events == []
    mem.save.assert_not_called()
    mem.save_fact_to_graph.assert_not_called()


@pytest.mark.unit
def test_is_junk_punctuation_question_exclamation_and_literal_match():
    base = "This prose fact has enough length and alphabetic tokens inside"
    assert is_junk(f"{base}?")
    assert is_junk(f"{base}!")
    assert is_junk("123456789012345678901234567890123456789012345678")
    assert is_junk(base, user_text=base, agent_text="other")
    assert is_junk(base, user_text="other", agent_text=base)


@pytest.mark.unit
def test_is_junk_no_four_letter_noun():
    assert is_junk("aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn ooo ppp")
