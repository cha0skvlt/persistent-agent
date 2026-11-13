"""Causal lessons store helpers + tool impl."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.lessons import format_lesson_line, parse_evidence_ids
from src.tools import lesson_tools as lt


def test_parse_evidence_ids():
    assert parse_evidence_ids("") == []
    assert parse_evidence_ids("a, b ; c") == ["a", "b", "c"]


def test_format_lesson_line():
    created = datetime(2026, 7, 17, tzinfo=timezone.utc)
    line = format_lesson_line("abc", "stopped X", "noticed Y", created)
    assert line.startswith("[LESSON] [id=abc]")
    assert "changed: stopped X" in line
    assert "because: noticed Y" in line


def test_record_lesson_impl_requires_both(monkeypatch):
    assert "lesson_error" in lt.record_lesson_impl(change="x", because="")
    assert "lesson_error" in lt.record_lesson_impl(change="", because="y")


def test_record_lesson_impl_ok(monkeypatch):
    store = MagicMock()
    store.save.return_value = ("lid-1", True)
    monkeypatch.setattr(lt, "get_default_lessons", lambda: store)
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda signal, **kw: outcomes.append(signal),
    )
    out = lt.record_lesson_impl(
        change="stopped lying",
        because="caught three times",
        evidence_ids="id1,id2",
    )
    assert out == "lesson_saved id=lid-1"
    assert outcomes == ["lesson_saved"]
    store.save.assert_called_once()
    kwargs = store.save.call_args.kwargs
    assert kwargs["evidence_ids"] == ["id1", "id2"]


def test_record_lesson_impl_dedupe_skips_outcome(monkeypatch):
    store = MagicMock()
    store.save.return_value = ("lid-1", False)
    monkeypatch.setattr(lt, "get_default_lessons", lambda: store)
    outcomes: list[str] = []
    monkeypatch.setattr(
        "src.policy_knobs.apply_outcome_safe",
        lambda signal, **kw: outcomes.append(signal),
    )
    out = lt.record_lesson_impl(change="x", because="y")
    assert out == "lesson_exists id=lid-1"
    assert outcomes == []


def test_lesson_save_dedupe_lookup_fail_closed():
    """Dedupe SELECT failure must not INSERT (no curiosity ratchet on blind write)."""
    from src.lessons import CausalLessons

    class _BoomCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *_a, **_k):
            raise RuntimeError("db down")

        def fetchone(self):
            return None

    class _BoomConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _BoomCursor()

        def commit(self):
            return None

    class _BoomPool:
        def connection(self):
            return _BoomConn()

    store = CausalLessons(pool=_BoomPool())
    store._embed = MagicMock()
    assert store.save("change", "because") is None
    store._embed.get_vector.assert_not_called()


def test_search_lessons_impl_empty(monkeypatch):
    store = MagicMock()
    store.search.return_value = []
    monkeypatch.setattr(lt, "get_default_lessons", lambda: store)
    assert "lesson_empty" in lt.search_lessons_impl(query="why")


def test_causal_lesson_db_roundtrip():
    """Live Postgres + embed: save then search finds "stopped X because of Y"."""
    import pytest

    from src.lessons import get_default_lessons

    store = get_default_lessons()
    change = "stopped lying about emptiness after web"
    because = "caught three times claiming what the tool already returned"
    try:
        saved = store.save(change, because, interactive=True)
    except Exception as exc:
        pytest.skip(f"lesson save unavailable: {exc}")
    if not saved:
        pytest.skip("lesson save returned None (embed/DB)")
    lid, _created = saved
    hits = store.search("stopped lying about emptiness", limit=5, interactive=True)
    assert any(h[0] == lid for h in hits), hits
    assert any("emptiness" in h[1] or "lying" in h[1] for h in hits)
    # Exact re-save must dedupe, not spawn another row.
    again = store.save(change, because, interactive=True)
    assert again == (lid, False)
