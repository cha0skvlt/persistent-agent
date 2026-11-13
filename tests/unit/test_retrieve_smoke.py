"""Frozen retrieve smoke — regression only, not a TREC bake-off."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.eval.retrieve_smoke import evaluate_fixture, load_fixture, rank_query

_REPO = Path(__file__).resolve().parents[2]
_SMOKE = _REPO / "tests" / "fixtures" / "retrieve_smoke"


def test_retrieve_smoke_jsonl_not_gitignored():
    for name in ("corpus.jsonl", "queries.jsonl"):
        path = _SMOKE / name
        assert path.is_file()
        proc = subprocess.run(
            ["git", "check-ignore", "--", str(path.relative_to(_REPO))],
            cwd=_REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 128:  # not a git repository yet
            pytest.skip("not a git repository")
        # 1 = not ignored (trackable). -v would match the exception line and exit 0.
        assert proc.returncode == 1, proc.stdout + proc.stderr


def test_retrieve_smoke_metrics_in_range():
    metrics = evaluate_fixture()
    assert metrics["n"] == 24.0
    for key in ("ndcg@10", "recall@3", "recall@10", "mrr"):
        assert 0.0 <= metrics[key] <= 1.0
    # Easy semantic queries must still surface the gold in top-3.
    assert metrics["recall@3"] >= 0.7
    assert metrics["ndcg@10"] >= 0.7


def test_retrieve_smoke_known_query_ranks_gold():
    fixture = load_fixture()
    ranked = rank_query(fixture, "q08")
    assert ranked[0] == "d08"
