"""seed-self idempotency."""

from __future__ import annotations

from unittest.mock import MagicMock

import scripts.seed_self as seed_self_mod
from src.config import settings
from src.memory import SelfReflection

SEED_CONTENT = seed_self_mod.SEED_CONTENT
seed_self = seed_self_mod.seed_self


def test_has_exact_true_when_row_exists():
    cur = MagicMock()
    cur.fetchone.return_value = (1,)
    cur_ctx = MagicMock()
    cur_ctx.__enter__ = MagicMock(return_value=cur)
    cur_ctx.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cur_ctx
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = conn_ctx

    sr = SelfReflection(pool=pool)
    assert sr.has_exact(SEED_CONTENT) is True


def test_seed_self_skips_when_exact_row_exists(monkeypatch):
    sr = SelfReflection(pool=MagicMock())
    monkeypatch.setattr(seed_self_mod, "ensure_agent_entity", lambda: None)
    monkeypatch.setattr(seed_self_mod, "SelfReflection", lambda: sr)
    monkeypatch.setattr(sr, "has_exact", lambda _c, agent=None: True)
    saves: list[str] = []
    monkeypatch.setattr(sr, "save", lambda *a, **k: saves.append("x") or True)

    seed_self()

    assert saves == []


def test_seed_self_saves_when_absent(monkeypatch):
    sr = SelfReflection(pool=MagicMock())
    monkeypatch.setattr(seed_self_mod, "ensure_agent_entity", lambda: None)
    monkeypatch.setattr(seed_self_mod, "SelfReflection", lambda: sr)
    monkeypatch.setattr(sr, "has_exact", lambda _c, agent=None: False)
    calls: list[tuple] = []

    def fake_save(text, *, importance=0.5, agent=None):
        calls.append((text, importance, agent))
        return True

    monkeypatch.setattr(sr, "save", fake_save)

    seed_self()

    assert calls == [(SEED_CONTENT, 1.0, settings.default_agent_id)]
