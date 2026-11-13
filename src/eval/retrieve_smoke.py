"""Frozen retrieve smoke — nDCG@10 / recall@k / MRR. Not a TREC comparison."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.graph.memory_distrust import apply_score_distrust
from src.graph.retrieval_node import _rank_memories_by_query

_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "retrieve_smoke"
)


def _dcg(gains: list[float], k: int) -> float:
    s = 0.0
    for i, g in enumerate(gains[:k]):
        s += (float(g)) / math.log2(i + 2)
    return s


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gains = [float(qrels.get(did, 0)) for did in ranked_ids[:k]]
    ideal = sorted((float(v) for v in qrels.values()), reverse=True)
    denom = _dcg(ideal, k)
    if denom <= 0.0:
        return 0.0
    return _dcg(gains, k) / denom


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gold = {did for did, rel in qrels.items() if rel > 0}
    if not gold:
        return 0.0
    hit = gold.intersection(ranked_ids[:k])
    return len(hit) / len(gold)


def mrr(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    gold = {did for did, rel in qrels.items() if rel > 0}
    for i, did in enumerate(ranked_ids, start=1):
        if did in gold:
            return 1.0 / i
    return 0.0


def load_fixture(root: Path | None = None) -> dict[str, Any]:
    base = root or _DEFAULT_FIXTURE
    corpus: dict[str, str] = {}
    with (base / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[str(row["_id"])] = str(row["text"])
    queries: dict[str, str] = {}
    with (base / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            queries[str(row["_id"])] = str(row["text"])
    qrels: dict[str, dict[str, int]] = {}
    with (base / "qrels.tsv").open(encoding="utf-8") as fh:
        header = fh.readline()
        if "query" not in header.lower() and "\t" in header:
            fh.seek(0)
        for raw in fh:
            parts = raw.strip().split("\t")
            if len(parts) < 3 or parts[0] in {"query-id", "qid"}:
                continue
            qid, did, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = rel
    scores = json.loads((base / "scores.json").read_text(encoding="utf-8"))
    meta_raw = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "scores": scores,
        "meta": meta_raw,
    }


def _meta_for_distrust(
    meta_raw: dict[str, Any], *, now: datetime
) -> dict[str, tuple[str, datetime | None]]:
    out: dict[str, tuple[str, datetime | None]] = {}
    for did, row in meta_raw.items():
        mtype = str(row.get("memory_type") or "semantic")
        age = float(row.get("age_days") or 0.0)
        created = now - timedelta(days=age) if age else now
        out[str(did)] = (mtype, created)
    return out


def rank_query(
    fixture: dict[str, Any],
    query_id: str,
    *,
    similarity_floor: float = 0.3,
    unverified_factor: float = 0.35,
    half_life_days: float = 180.0,
    now: datetime | None = None,
) -> list[str]:
    ref = now or datetime.now(timezone.utc)
    scores = fixture["scores"].get(query_id) or {}
    rows: list[tuple] = []
    for did, sim in scores.items():
        text = fixture["corpus"].get(did, "")
        rows.append((did, text, float(sim), None, 0.5))
    ranked = _rank_memories_by_query(rows, similarity_floor=similarity_floor)
    meta = _meta_for_distrust(fixture["meta"], now=ref)
    adjusted, _ = apply_score_distrust(
        ranked,
        meta,
        unverified_factor=unverified_factor,
        half_life_days=half_life_days,
        now=ref,
    )
    return [mid for mid, _c, _s, _q in adjusted]


def evaluate_fixture(
    root: Path | None = None,
    *,
    similarity_floor: float = 0.3,
) -> dict[str, float]:
    fixture = load_fixture(root)
    ndcgs: list[float] = []
    r3: list[float] = []
    r10: list[float] = []
    mrrs: list[float] = []
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for qid in fixture["queries"]:
        rel = fixture["qrels"].get(qid) or {}
        ranked = rank_query(fixture, qid, similarity_floor=similarity_floor, now=now)
        ndcgs.append(ndcg_at_k(ranked, rel, 10))
        r3.append(recall_at_k(ranked, rel, 3))
        r10.append(recall_at_k(ranked, rel, 10))
        mrrs.append(mrr(ranked, rel))
    n = max(1, len(ndcgs))
    return {
        "n": float(len(ndcgs)),
        "ndcg@10": sum(ndcgs) / n,
        "recall@3": sum(r3) / n,
        "recall@10": sum(r10) / n,
        "mrr": sum(mrrs) / n,
    }


__all__ = [
    "evaluate_fixture",
    "load_fixture",
    "mrr",
    "ndcg_at_k",
    "rank_query",
    "recall_at_k",
]
