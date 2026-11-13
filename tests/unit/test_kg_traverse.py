from __future__ import annotations

import os

import ollama
import psycopg
import pytest

from src.config import settings
from src.db.pool import get_pool
from src.kg.entities import resolve_entity
from src.memory import AgentMemory


def _deterministic_embedding(text: str) -> list[float]:
    """Stable 768-dim vector from string (fewer short-text collisions than raw Ollama embeddings)."""
    import hashlib

    seed = hashlib.sha256((text or "").strip().encode()).digest()
    out: list[float] = []
    n = 0
    while len(out) < 768:
        block = hashlib.sha256(seed + n.to_bytes(4, "big")).digest()
        for b in block:
            out.append((b / 127.5) - 1.0)
            if len(out) >= 768:
                return out
        n += 1
    return out


def _truncate_tests_should_run() -> bool:
    """TRUNCATE tests only when TEST_DATABASE_URL is explicitly in the environment."""
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
    reason="requires TEST_DATABASE_URL in os.environ + Postgres + Ollama (isolated TRUNCATE tests)",
)


@pytest.fixture
def deterministic_embeddings(monkeypatch):
    def _fake_get_vector(self, text: str) -> list[float]:
        return _deterministic_embedding(text)

    monkeypatch.setattr(AgentMemory, "get_vector", _fake_get_vector)


@requires_db
def test_traverse_graph_two_hops_includes_ram(
    disposable_postgres, deterministic_embeddings
):
    mem = AgentMemory(pool=get_pool())
    with mem.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE knowledge_graph, entities RESTART IDENTITY CASCADE")
        conn.commit()

    mem.save_fact_to_graph("Alice", "owns", "M5 Max")
    mem.save_fact_to_graph("M5 Max", "has", "128GB RAM")
    mem.save_fact_to_graph("M5 Max", "uses", "PostgreSQL")
    mem.save_fact_to_graph("PostgreSQL", "version", "17")
    mem.save_fact_to_graph("Alice", "likes", "tea")

    alice_id = resolve_entity("Alice", mem=mem)
    rows = mem.traverse_graph(alice_id, depth=2)
    ram_edges = [r for r in rows if r["object"] == "128GB RAM"]
    assert ram_edges
    assert all(r["hop"] == 2 for r in ram_edges)

    filtered = mem.traverse_graph(alice_id, depth=2, predicates=["owns"])
    assert filtered
    assert all(r["predicate"] == "owns" for r in filtered)
    assert not any(r["object"] == "128GB RAM" for r in filtered)

    multi = mem.traverse_graph(alice_id, depth=2, predicates=["owns", "has"])
    assert any(r["object"] == "128GB RAM" for r in multi)
    assert all(r["predicate"] in ("owns", "has") for r in multi)

    shallow = mem.traverse_graph(alice_id, depth=1)
    assert not any(r["object"] == "128GB RAM" for r in shallow)


@requires_db
def test_query_graph_ilike_filters(disposable_postgres, deterministic_embeddings):
    mem = AgentMemory(pool=get_pool())
    with mem.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE knowledge_graph, entities RESTART IDENTITY CASCADE")
        conn.commit()

    mem.save_fact_to_graph("Alice", "owns", "M5 Max")
    mem.save_fact_to_graph("M5 Max", "has", "128GB RAM")

    rows = mem.query_graph(subject="Alice", predicate="owns")
    assert rows
    assert any(r[2] == "M5 Max" for r in rows)

    rows_o = mem.query_graph(object_="RAM")
    assert any("128GB" in r[2] for r in rows_o)
