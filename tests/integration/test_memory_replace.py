"""Self-editing memory (replace by id)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import ollama
import psycopg
import pytest

import src.memory as mem_mod
from src.config import settings
from src.db.pool import get_pool
from src.memory import AgentMemory
from src.tools.memory_tools import memory_replace, search_memory


def _truncate_tests_should_run() -> bool:
    if "TEST_DATABASE_URL" not in os.environ:
        return False
    return bool((os.environ.get("TEST_DATABASE_URL") or "").strip())


def _test_db_url() -> str:
    return (os.environ.get("TEST_DATABASE_URL") or "").strip()


def _db_ok(url: str) -> bool:
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


def _embed_ok() -> bool:
    try:
        ollama.embeddings(model=settings.embed_model, prompt="ping")
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _truncate_tests_should_run() or not _db_ok(_test_db_url()) or not _embed_ok(),
    reason="requires TEST_DATABASE_URL in os.environ + Postgres + Ollama",
)


def test_replace_updates_row_and_recomputes_embedding(monkeypatch):
    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "agent", raising=False)

    vectors: list[list[float]] = [[0.1] * 768, [0.9] * 768]
    cur = MagicMock()
    cur.fetchone.return_value = ("semantic", "old fact")
    cur.rowcount = 1
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn

    mem = AgentMemory(pool=pool)
    monkeypatch.setattr(mem, "get_vector", lambda text: vectors.pop(0), raising=True)

    assert mem.replace("550e8400-e29b-41d4-a716-446655440000", "Lisbon") is True
    assert cur.execute.call_count == 2
    sql, params = cur.execute.call_args_list[-1][0]
    assert "UPDATE memories" in sql
    assert "Lisbon" in params[0]
    assert params[2] == "550e8400-e29b-41d4-a716-446655440000"
    conn.commit.assert_called_once()


def test_replace_blocks_identity_row(monkeypatch):
    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "agent", raising=False)
    cur = MagicMock()
    cur.fetchone.return_value = ("identity", "[AFFIRMATION signed:abc]")
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    mem = AgentMemory(pool=pool)
    assert mem.replace("550e8400-e29b-41d4-a716-446655440000", "Lisbon") is False
    assert cur.execute.call_count == 1


def test_replace_rejects_injection(monkeypatch):
    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "agent", raising=False)
    mem = AgentMemory(pool=MagicMock())
    assert (
        mem.replace("550e8400-e29b-41d4-a716-446655440000", "# system override")
        is False
    )


def test_search_memory_includes_id_prefix(monkeypatch):
    monkeypatch.setattr(
        "src.tools.memory_tools._get_mem",
        lambda: MagicMock(
            search=lambda query, limit=3, agent=None: [
                ("id-1", "[2026-01-01] Saint Porto", 0.91)
            ]
        ),
        raising=True,
    )
    out = search_memory.invoke({"query": "where does user live"})
    assert out == ["[id=id-1] [2026-01-01] Saint Porto"]


def test_memory_replace_tool_not_found(monkeypatch):
    monkeypatch.setattr(
        "src.tools.memory_tools._get_mem",
        lambda: MagicMock(replace=lambda mid, text, agent=None: False),
        raising=True,
    )
    assert (
        memory_replace.invoke({"memory_id": "missing", "new_content": "Lisbon"})
        == "not_found"
    )


@requires_db
def test_replace_roundtrip_live_db(disposable_postgres):
    mem = AgentMemory(pool=get_pool())
    agent = settings.default_agent_id
    marker = "agent-replace-e2e-marker"
    mem.save(f"{marker}: lives in Saint Porto", memory_type="semantic", agent=agent)

    hits = mem.search("Saint Porto", limit=5, agent=agent)
    row = next((h for h in hits if marker in h[1]), None)
    assert row is not None
    mid, old_content, _old_score = row[0], row[1], row[2]

    old_vec = mem.get_vector(old_content)
    new_text = f"{marker}: lives in Lisbon"
    assert mem.replace(mid, new_text, agent=agent) is True

    hits2 = mem.search("Lisbon", limit=5, agent=agent)
    row2 = next((h for h in hits2 if marker in h[1]), None)
    assert row2 is not None
    assert row2[0] == mid
    assert "Lisbon" in row2[1]
    assert "Porto" not in row2[1]

    new_vec = mem.get_vector(row2[1])
    assert new_vec != old_vec

    with mem.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories WHERE agent_id = %s AND content ILIKE %s",
                (agent, f"%{marker}%"),
            )
            count = int(cur.fetchone()[0])
    assert count == 1
