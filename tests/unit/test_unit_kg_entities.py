"""Unit tests for src.kg.entities (resolution, aliases, embeddings)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.config import settings
from src.kg import entities as ent


class _FakeCursor:
    def __init__(self, rows: list):
        self._rows = list(rows)

    def execute(self, *_a, **_k):
        return self

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None


class _Ctx:
    def __init__(self, cur: _FakeCursor):
        self._cur = cur

    def __enter__(self) -> _FakeCursor:
        return self._cur

    def __exit__(self, *a):
        return False


class _FakeConn:
    """Minimal connection: ``with conn.cursor() as cur`` only."""

    def __init__(self, fetch_sequence: list):
        self._cur = _FakeCursor(list(fetch_sequence))

    def cursor(self):
        return _Ctx(self._cur)

    def transaction(self):
        return nullcontext()


class MagicMem:
    def __init__(self, vec: list[float]):
        self.vec = vec
        self.calls = 0
        self.pool = MagicMock()

    def get_vector(self, _text: str) -> list[float]:
        self.calls += 1
        return self.vec


@pytest.mark.unit
def test_identity_flags_and_surfaces(monkeypatch):
    assert ent.must_resolve_canonical_only("  ") is False
    assert ent.must_resolve_canonical_only("agent") is True
    assert ent.must_not_attach_as_alias("agent") is True

    assert ent.is_agent_persona_surface("agent persona") is True
    assert ent.is_agent_persona_surface("x") is False
    assert ent.is_agent_persona_surface("") is False
    assert ent.is_agent_persona_surface("self persona") is True
    assert ent.is_agent_persona_surface("agent") is False

    assert ent.requires_distinct_entity("USER") is False
    assert ent.requires_distinct_entity("black light") is True
    assert ent.requires_distinct_entity("agent") is False
    assert ent.requires_distinct_entity("") is False
    assert ent.requires_distinct_entity("random world entity") is False

    assert ent.is_assistant_surface("assistant") is True
    assert ent.is_agent_persona_surface("bot persona helper") is False


@pytest.mark.unit
def test_triple_hash_and_vector_literal():
    a, b = uuid4(), uuid4()
    h = ent.triple_hash_from_ids(a, "knows", b)
    assert len(h) == 32
    with pytest.raises(ValueError):
        ent._pg_vector_literal([])
    assert ent._pg_vector_literal([0.5, 1.0]).startswith("[0.5,")


@pytest.mark.unit
def test_embed_for_entity_cache():
    mem = MagicMem([0.1, 0.2])
    cache: dict = {}
    v1 = ent._embed_for_entity(mem, "Alpha", embed_cache=cache)
    v2 = ent._embed_for_entity(mem, "alpha", embed_cache=cache)
    assert v1 is v2 is mem.vec
    assert mem.calls == 1
    with pytest.raises(ValueError):
        ent._embed_for_entity(mem, "   ", embed_cache=None)


@pytest.mark.unit
def test_maybe_attach_alias(monkeypatch):
    calls: list = []

    def fake_append(conn, eid, alias, agent):
        calls.append((alias, eid, agent))

    monkeypatch.setattr(ent, "_append_entity_alias", fake_append)
    ent._maybe_attach_entity_alias(None, uuid4(), "short", "agent")
    assert calls
    calls.clear()
    ent._maybe_attach_entity_alias(None, uuid4(), "x" * 100, "agent")
    ent._maybe_attach_entity_alias(None, uuid4(), "agent", "agent")
    assert calls == []


@pytest.mark.unit
def test_lookup_entity_by_name_branches():
    assert ent._lookup_entity_by_name("  ") is None
    uid = uuid4()
    assert ent._lookup_entity_by_name("Only", conn=_FakeConn([(uid,)])) == uid
    uid2 = uuid4()
    assert (
        ent._lookup_entity_by_name(
            "alias", conn=_FakeConn([None, (uid2,)]), canonical_only=False
        )
        == uid2
    )
    assert (
        ent._lookup_entity_by_name("alias", conn=_FakeConn([None]), canonical_only=True)
        is None
    )
    assert ent._lookup_entity_by_name("missing", conn=_FakeConn([None, None])) is None


@pytest.mark.unit
def test_lookup_entity_by_name_via_pool(monkeypatch):
    uid = uuid4()
    conn = _FakeConn([(uid,)])

    @contextmanager
    def connection():
        yield conn

    mem = MagicMock()
    mem.pool.connection = connection
    monkeypatch.setattr("src.memory.get_default_memory", lambda: mem)
    assert ent._lookup_entity_by_name("Only") == uid


@pytest.mark.unit
def test_lookup_kg_entity_dispatches(monkeypatch):
    eid = uuid4()
    seen: list[tuple[str, bool | None]] = []

    def fake_lookup(name, **kw):
        seen.append((name, kw.get("canonical_only")))
        return eid

    monkeypatch.setattr(ent, "_lookup_entity_by_name", fake_lookup)
    monkeypatch.setattr(ent, "is_assistant_surface", lambda s: False)
    monkeypatch.setattr(ent, "is_agent_persona_surface", lambda s: False)
    monkeypatch.setattr(ent, "requires_distinct_entity", lambda s: False)
    assert ent.lookup_kg_entity("world") == eid
    assert seen[-1] == ("world", False)

    monkeypatch.setattr(ent, "is_assistant_surface", lambda s: True)
    seen.clear()
    ent.lookup_kg_entity("assistant")
    assert seen[-1] == ("Agent", True)

    assert ent.lookup_kg_entity("  ") is None


@pytest.mark.unit
def test_resolve_kg_entity_dispatches(monkeypatch):
    seen: list = []

    def fake(name, **kw):
        seen.append((name, kw.get("canonical_only")))
        return uuid4()

    monkeypatch.setattr(ent, "resolve_entity", fake)
    monkeypatch.setattr(ent, "is_assistant_surface", lambda s: True)
    monkeypatch.setattr(ent, "is_agent_persona_surface", lambda s: False)
    ent.resolve_kg_entity("assistant")
    assert seen[0][0] == "Agent"

    monkeypatch.setattr(ent, "is_assistant_surface", lambda s: False)
    monkeypatch.setattr(ent, "is_agent_persona_surface", lambda s: True)
    seen.clear()
    ent.resolve_kg_entity("agent")
    assert seen[0][0] == "Agent"

    monkeypatch.setattr(ent, "is_agent_persona_surface", lambda s: False)
    monkeypatch.setattr(ent, "requires_distinct_entity", lambda s: True)
    seen.clear()
    ent.resolve_kg_entity("unique")
    assert seen[0] == ("unique", True)


@pytest.mark.unit
def test_resolve_kg_entity_empty_name():
    with pytest.raises(ValueError):
        ent.resolve_kg_entity(" ")


@pytest.mark.unit
def test_resolve_entity_empty_name():
    mem = MagicMem([0.1])
    with pytest.raises(ValueError):
        ent.resolve_entity("   ", mem=mem, conn=_FakeConn([]))


@pytest.mark.unit
def test_resolve_entity_canonical_hit(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    out = ent.resolve_entity("Only", mem=MagicMem([0.1]), conn=_FakeConn([(uid,)]))
    assert out == uid


@pytest.mark.unit
def test_resolve_entity_alias_hit(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    out = ent.resolve_entity(
        "alias",
        mem=MagicMem([0.1]),
        conn=_FakeConn(
            [
                None,
                (uid,),
            ]
        ),
    )
    assert out == uid


@pytest.mark.unit
def test_resolve_entity_embed_merge(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    monkeypatch.setattr(ent, "must_not_attach_as_alias", lambda s: True)
    monkeypatch.setattr(settings, "kg_entity_resolve_threshold", 0.9)
    out = ent.resolve_entity(
        "surface",
        mem=MagicMem([0.2, 0.3]),
        conn=_FakeConn([None, None, (1,), (uid, 0.95)]),
        agent="ag",
    )
    assert out == uid


@pytest.mark.unit
def test_resolve_entity_insert_returning(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    monkeypatch.setattr(settings, "kg_entity_resolve_threshold", 0.99)
    out = ent.resolve_entity(
        "fresh",
        mem=MagicMem([0.2]),
        conn=_FakeConn([None, None, None, (uid,)]),
        agent="ag",
    )
    assert out == uid


@pytest.mark.unit
def test_resolve_entity_insert_conflict_then_select(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    monkeypatch.setattr(settings, "kg_entity_resolve_threshold", 0.99)
    out = ent.resolve_entity(
        "race",
        mem=MagicMem([0.2]),
        conn=_FakeConn([None, None, None, None, (uid,)]),
        agent="ag",
    )
    assert out == uid


@pytest.mark.unit
def test_resolve_entity_runtime_error_when_still_missing(monkeypatch):
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    monkeypatch.setattr(settings, "kg_entity_resolve_threshold", 0.99)
    with pytest.raises(RuntimeError):
        ent.resolve_entity(
            "ghost",
            mem=MagicMem([0.2]),
            conn=_FakeConn([None, None, None, None, None]),
            agent="ag",
        )


@pytest.mark.unit
def test_resolve_entity_pool_transaction(monkeypatch):
    uid = uuid4()
    rows = [None, None, None, (uid,)]
    outer_cur = _FakeCursor(list(rows))

    class InnerConn:
        def cursor(self):
            return _Ctx(outer_cur)

        def transaction(self):
            return nullcontext()

    class Pool:
        @contextmanager
        def connection(self):
            yield InnerConn()

    mem = MagicMem([0.3])
    mem.pool = Pool()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda s: False)
    monkeypatch.setattr(settings, "kg_entity_resolve_threshold", 0.99)
    out = ent.resolve_entity("poolent", mem=mem, conn=None, agent="ag")
    assert out == uid


@pytest.mark.unit
def test_resolve_entity_skips_embed_when_canonical_only(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr(ent, "must_resolve_canonical_only", lambda raw: True)
    monkeypatch.setattr(
        ent,
        "_embed_for_entity",
        MagicMock(side_effect=AssertionError("embed should not run")),
    )
    out = ent.resolve_entity(
        "agent",
        mem=MagicMem([0.1]),
        conn=_FakeConn([(uid,)]),
        agent="ag",
    )
    assert out == uid


@pytest.mark.unit
def test_append_entity_alias_executes():
    uid = uuid4()
    ops: list = []

    class Cur:
        def execute(self, sql, params=None):
            ops.append((sql, params))
            return self

    class Conn:
        def cursor(self):
            return _Ctx(Cur())

    ent._append_entity_alias(Conn(), uid, "newalias", "agent")
    assert ops and "UPDATE entities" in ops[0][0]
