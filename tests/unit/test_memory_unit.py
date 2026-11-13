"""AgentMemory helpers without Postgres / Ollama (mocked pool + embeddings)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.memory as mem_mod
from src.memory import AgentMemory
from tests.support.helpers import mock_pool_with_cursor


def test_vector_param_format():
    assert mem_mod._vector_param([1.0, 2.0]) == "[1.0,2.0]"


def test_vector_param_rejects_empty():
    with pytest.raises(ValueError):
        mem_mod._vector_param([])


def test_embed_text_truncates_to_max_input_chars(monkeypatch):
    import src.embeddings as emb_mod

    calls: list[str] = []

    def fake_embed(provider, text):
        calls.append(text)
        return [0.1] * 768

    monkeypatch.setattr(emb_mod, "_ollama_embed", fake_embed)
    monkeypatch.setattr(emb_mod, "_MAX_INPUT_CHARS", 512)
    emb_mod.embed_text("a" * 2000)
    assert len(calls) == 1
    assert len(calls[0]) == 512


def test_embed_text_interactive_uses_lower_retry_budget(monkeypatch):
    """REPL hot path: fewer attempts and shorter sleep cap than background embed."""
    import src.embeddings as emb_mod

    attempts: list[int] = []
    sleeps: list[float] = []

    def fake_embed(provider, text):
        attempts.append(1)
        raise RuntimeError("embed down")

    monkeypatch.setattr(emb_mod, "_ollama_embed", fake_embed)
    monkeypatch.setattr(emb_mod, "_MAX_RETRIES", 5)
    monkeypatch.setattr(emb_mod, "_RETRY_SLEEP_CAP", 2.0)
    monkeypatch.setattr(emb_mod, "_INTERACTIVE_MAX_RETRIES", 2)
    monkeypatch.setattr(emb_mod, "_INTERACTIVE_RETRY_SLEEP_CAP", 0.5)
    monkeypatch.setattr(emb_mod, "_RETRY_SLEEP_BASE", 0.2)
    with patch.object(emb_mod.time, "sleep", side_effect=lambda s: sleeps.append(s)):
        with pytest.raises(emb_mod.EmbeddingError, match="embed down"):
            emb_mod.embed_text("query", interactive=True)
    assert len(attempts) == 2
    assert all((s <= 0.5 for s in sleeps))


def test_embed_text_halves_prompt_on_context_length_error(monkeypatch):
    """Must not burn 5x backoff on 500 context overflow — shrink and retry immediately."""
    import src.embeddings as emb_mod

    lengths: list[int] = []

    def fake_embed(provider, text):
        lengths.append(len(text))
        if len(lengths) == 1:
            raise RuntimeError(
                "the input length exceeds the context length (status code: 500)"
            )
        return [0.1] * 768

    monkeypatch.setattr(emb_mod, "_ollama_embed", fake_embed)
    monkeypatch.setattr(emb_mod, "_MAX_INPUT_CHARS", 800)
    emb_mod.embed_text("b" * 800)
    assert lengths == [800, 400]


def test_input_exceeded_detects_ollama_message():
    import src.embeddings as emb_mod

    assert emb_mod._input_exceeded(
        RuntimeError("the input length exceeds the context length (status code: 500)")
    )
    assert not emb_mod._input_exceeded(RuntimeError("connection refused"))


def test_traverse_graph_batch_zero_max_edges_skips_traverse():
    pool = MagicMock()
    m = AgentMemory(pool=pool)
    from uuid import uuid4

    eid = uuid4()
    out = m.traverse_graph_batch([eid], max_edges=0)
    assert out == {eid: []}
    pool.connection.assert_not_called()


def test_traverse_graph_batch_empty_ids_returns_empty():
    m = AgentMemory(pool=MagicMock())
    assert m.traverse_graph_batch([]) == {}


def test_traverse_graph_batch_default_max_edges(monkeypatch):
    from uuid import uuid4

    cur = MagicMock()
    cur.fetchall.return_value = [("S", "p", "O", 1)]
    pool = mock_pool_with_cursor(cur)
    m = AgentMemory(pool=pool)
    id1, id2 = uuid4(), uuid4()
    out = m.traverse_graph_batch([id1, id2], depth=1)
    assert set(out) == {id1, id2}
    assert all(isinstance(v, list) for v in out.values())


def test_entities_matching_tokens_bigram_branch(monkeypatch):
    from uuid import uuid4

    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "a")
    cur = MagicMock()
    cur.fetchall.return_value = [(str(uuid4()), "foo bar")]
    pool = mock_pool_with_cursor(cur)
    m = AgentMemory(pool=pool)
    hits = m.entities_matching_tokens(["fo", "ba"], limit=3)
    assert len(hits) == 1
    sql = str(cur.execute.call_args.args[0])
    assert "unnest" in sql or "ANY" in sql


def test_entities_matching_tokens_single_token_branch(monkeypatch):
    from uuid import uuid4

    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "a")
    cur = MagicMock()
    cur.fetchall.return_value = [(str(uuid4()), "anchor")]
    pool = mock_pool_with_cursor(cur)
    m = AgentMemory(pool=pool)
    hits = m.entities_matching_tokens(["anchor"], limit=3)
    assert len(hits) == 1
    sql = str(cur.execute.call_args.args[0])
    assert "unnest" in sql


def test_traverse_graph_default_max_edges(monkeypatch):
    from uuid import uuid4

    monkeypatch.setattr(mem_mod.settings, "default_agent_id", "a")
    cur = MagicMock()
    cur.fetchall.return_value = [("S", "p", "O", 1)]
    pool = mock_pool_with_cursor(cur)
    m = AgentMemory(pool=pool)
    rows = m.traverse_graph(uuid4(), depth=1)
    assert rows == [{"subject": "S", "predicate": "p", "object": "O", "hop": 1}]
