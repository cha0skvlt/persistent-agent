"""Tool-level memory write path: save → search → replace."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.tools.memory_tools import memory_replace, save_memory, search_memory

_MID = "550e8400-e29b-41d4-a716-446655440000"


def _seed_live(text: str) -> None:
    from langchain_core.messages import HumanMessage

    from src.graph.turn_evidence import seed_from_messages

    seed_from_messages([HumanMessage(content=text)])


def _mock_pool():
    return MagicMock(
        connection=lambda: MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
    )


def test_save_memory_returns_id(monkeypatch):
    mem = MagicMock()
    mem.search.return_value = []
    mem.save.return_value = (_MID, "saved")
    monkeypatch.setattr("src.tools.memory_tools._get_mem", lambda: mem, raising=True)
    _seed_live("USER lives in Lisbon")
    out = save_memory.invoke({"content": "USER lives in Lisbon"})
    assert out == f"saved id={_MID}"


def test_save_memory_empty_content():
    assert save_memory.invoke({"content": "   "}) == "empty_content"


def test_save_memory_live_rejects_ungrounded(monkeypatch):
    from src.graph.turn_evidence import reset_turn_evidence

    mem = MagicMock()
    mem.save.side_effect = AssertionError("save must not run")
    monkeypatch.setattr("src.tools.memory_tools._get_mem", lambda: mem, raising=True)
    reset_turn_evidence()
    out = save_memory.invoke({"content": "USER: I secretly own pigeons downtown"})
    assert "distill_rejected" in str(out)
    mem.save.assert_not_called()


def test_save_memory_returns_dedup_id(monkeypatch):
    mem = MagicMock()
    mem.search.return_value = []
    mem.save.return_value = (_MID, "dedup")
    monkeypatch.setattr("src.tools.memory_tools._get_mem", lambda: mem, raising=True)
    _seed_live("USER lives in Lisbon")
    out = save_memory.invoke({"content": "USER lives in Lisbon"})
    assert out == f"unchanged id={_MID} (near-duplicate)"


def test_save_memory_applies_contradiction_gate(monkeypatch):
    mem = MagicMock()
    mem.search.return_value = [
        (_MID, "USER lives in Lisbon", 0.95),
    ]
    mem.scale_importance.return_value = True
    mem.save.return_value = ("new-id", "saved")
    monkeypatch.setattr("src.tools.memory_tools._get_mem", lambda: mem, raising=True)
    monkeypatch.setattr(
        "src.tools.memory_tools.settings.reflect_consistency_gate_enabled",
        True,
    )
    monkeypatch.setattr(
        "src.graph.reflect_consistency.settings.memory_contradiction_policy",
        "log",
    )
    monkeypatch.setattr(
        "src.graph.reflect_consistency.settings.memory_similarity_floor",
        0.3,
    )
    monkeypatch.setattr("src.policy_knobs.apply_outcome_safe", lambda *a, **k: None)
    _seed_live("USER does not live in Lisbon")
    out = save_memory.invoke({"content": "USER does not live in Lisbon"})
    assert out == "saved (contradiction policy applied)"
    mem.scale_importance.assert_called_once()
    mem.save.assert_called_once()


def test_save_search_replace_tool_roundtrip(monkeypatch):
    store: dict[str, str] = {}

    def fake_save(content, memory_type="semantic", agent=None, **_kwargs):
        store[_MID] = content
        return _MID, "saved"

    def fake_search(query, limit=3, agent=None, **kwargs):
        q = (query or "").lower()
        hits = [(mid, body, 0.91) for mid, body in store.items() if q in body.lower()]
        return hits[:limit]

    def fake_replace(memory_id, new_content, agent=None):
        if memory_id in store:
            store[memory_id] = new_content
            return True
        return False

    monkeypatch.setattr(
        "src.tools.memory_tools._get_mem",
        lambda: MagicMock(save=fake_save, search=fake_search, replace=fake_replace),
        raising=True,
    )
    monkeypatch.setattr(
        "src.tools.memory_tools.memory_curation_status",
        lambda conn, **kw: "editable",
        raising=True,
    )
    monkeypatch.setattr(
        "src.tools.memory_tools.get_pool",
        _mock_pool,
        raising=True,
    )

    _seed_live("USER lives in Porto")
    saved = save_memory.invoke({"content": "USER lives in Porto"})
    assert saved == f"saved id={_MID}"

    hits = search_memory.invoke({"query": "Porto"})
    assert hits == [f"[id={_MID}] {store[_MID]}"]

    _seed_live("USER lives in Lisbon")
    replaced = memory_replace.invoke(
        {"memory_id": _MID, "new_content": "USER lives in Lisbon"}
    )
    assert replaced == "replaced"
    assert "Lisbon" in store[_MID]

    hits2 = search_memory.invoke({"query": "Lisbon"})
    assert hits2 == [f"[id={_MID}] {store[_MID]}"]
