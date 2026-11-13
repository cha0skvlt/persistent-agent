"""Selftest package is read-only (no SQL mutations)."""

from __future__ import annotations

FORBIDDEN = ("INSERT", "UPDATE", "DELETE")


def test_selftest_sources_have_no_sql_mutations():
    from tests.support.paths import REPO_ROOT

    root = REPO_ROOT / "src" / "selftest"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").upper()
        for word in FORBIDDEN:
            assert word not in text, f"{path.name} contains {word}"
