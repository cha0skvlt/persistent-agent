"""Memory / self-reflection dedup logic (cosine + fuzzy)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.memory as mem_mod
from src.memory import AgentMemory, SelfReflection
from tests.support.helpers import mock_pool_with_cursor


@pytest.mark.unit
@pytest.mark.parametrize(
    "new_text,existing,sim,expected",
    [
        ("hello world today", "hello world yesterday", 0.95, "cosine"),
        (
            "agent role agent role agent",
            "agent — role of agent, role of agent",
            0.88,
            "fuzzy",
        ),
        ("completely different", "unrelated topic here", 0.5, None),
    ],
)
def test_memory_dedup_reason(new_text, existing, sim, expected, monkeypatch):
    monkeypatch.setattr(mem_mod.settings, "memory_dedup_cosine", 0.92)
    monkeypatch.setattr(mem_mod.settings, "memory_dedup_fuzzy_cosine", 0.85)
    monkeypatch.setattr(mem_mod.settings, "memory_dedup_text_overlap", 0.8)
    reason = mem_mod._memory_dedup_reason(
        new_text,
        existing,
        sim,
        cosine_thr=0.92,
        fuzzy_thr=0.85,
    )
    assert reason == expected


@pytest.mark.unit
def test_self_reflection_save_fuzzy_dedup_skips_insert(monkeypatch):
    monkeypatch.setattr(mem_mod.settings, "memory_dedup_cosine", 0.99)
    monkeypatch.setattr(mem_mod.settings, "memory_dedup_fuzzy_cosine", 0.85)
    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "agent")

    cur = MagicMock()
    cur.fetchall.return_value = [
        (1, "I noticed the void is older than me.", 0.5, 0.87),
    ]
    pool = mock_pool_with_cursor(cur)
    sr = SelfReflection(pool=pool)
    monkeypatch.setattr(sr, "get_vector", lambda text, **_: [0.1] * 768)
    monkeypatch.setattr(
        "src.identity.get_signer",
        lambda: MagicMock(sign=lambda _c: None),
    )

    assert sr.save("I noticed the void is older than myself.") is False
    sql_text = " ".join(str(c.args[0]) for c in cur.execute.call_args_list if c.args)
    assert "INSERT INTO self_reflections" not in sql_text


@pytest.mark.unit
def test_memory_save_dedup_failure_raises(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(mem_mod.settings, "memory_dedup_cosine", 0.92)
    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "agent")

    cur = MagicMock()
    cur.execute.side_effect = RuntimeError("dedup sql broken")
    pool = mock_pool_with_cursor(cur)
    m = AgentMemory(pool=pool)
    monkeypatch.setattr(m, "get_vector", lambda text, **_: [0.1] * 768)

    with pytest.raises(RuntimeError, match="dedup sql broken"):
        m.save("hello world", memory_type="semantic")
