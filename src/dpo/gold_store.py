"""Persist DPO gold pairs and optional conversation snapshot markers."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import settings
from src.memory import get_default_memory
from src.obs.logs import log_event
from src.obs.metrics import record_dpo_gold_insert_failure

_log = logging.getLogger(__name__)
_event_log = logging.getLogger("agent.event")


def insert_gold_pair(
    *,
    user_text: str,
    rejected_reply: str,
    chosen_reply: str,
    critique: str,
    thread_id: str = "",
    model: str = "",
    agent_id: str | None = None,
) -> str | None:
    """Insert one gold pair; return row id or None on failure (never raises)."""
    agent = (agent_id or settings.default_agent_id or "agent").strip() or "agent"
    user_text = (user_text or "").strip()
    rejected_reply = (rejected_reply or "").strip()
    chosen_reply = (chosen_reply or "").strip()
    if not user_text or not rejected_reply or not chosen_reply:
        return None

    tid = (thread_id or "").strip()
    try:
        mem = get_default_memory()
        row_id: str | None = None
        with mem.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dpo_gold_pairs (
                      agent_id, thread_id, user_text, rejected_reply,
                      chosen_reply, critique, role_failed, model
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, true, %s)
                    RETURNING id::text
                    """,
                    (
                        agent,
                        tid,
                        user_text,
                        rejected_reply,
                        chosen_reply,
                        (critique or "").strip(),
                        (model or settings.reasoning_model or "").strip(),
                    ),
                )
                fetched = cur.fetchone()
                row_id = str(fetched[0]) if fetched else None
            conn.commit()

        _append_gold_marker(
            agent_id=agent,
            user_text=user_text,
            rejected_reply=rejected_reply,
            chosen_reply=chosen_reply,
            critique=critique,
            thread_id=tid,
            pair_id=row_id,
        )
        return row_id
    except Exception as exc:
        err = str(exc)
        payload = {
            "event": "dpo.gold_insert_failed",
            "ts_ms": int(time.time() * 1000),
            "error": err,
            "thread_id": tid,
            "agent_id": agent,
        }
        try:
            encoded = json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = json.dumps(
                {
                    "event": "dpo.gold_insert_failed",
                    "ts_ms": payload["ts_ms"],
                    "error": err,
                },
                ensure_ascii=False,
            )
        _event_log.error(encoded)
        _log.error("dpo.gold_insert_failed error=%s thread_id=%s", err, tid)
        record_dpo_gold_insert_failure()
        return None


def _append_gold_marker(
    *,
    agent_id: str,
    user_text: str,
    rejected_reply: str,
    chosen_reply: str,
    critique: str,
    thread_id: str,
    pair_id: str | None,
) -> None:
    marker: dict[str, Any] = {
        "type": "dpo_gold",
        "user": user_text,
        "rejected": rejected_reply,
        "chosen": chosen_reply,
        "critique": (critique or "").strip(),
        "thread_id": (thread_id or "").strip(),
        "pair_id": pair_id,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        get_default_memory().save_conversation_snapshot(
            agent_id,
            [marker],
            merge=True,
        )
    except Exception as exc:
        log_event(
            "dpo.marker_failed",
            err=str(exc),
            pair_id=pair_id,
            thread_id=(thread_id or "").strip(),
        )
        _log.warning(
            "dpo gold marker snapshot failed pair_id=%s thread_id=%s err=%s",
            pair_id,
            (thread_id or "").strip(),
            exc,
        )


def list_gold_pairs(
    *, since_days: int = 7, agent_id: str | None = None
) -> list[dict[str, Any]]:
    """Return gold rows created within ``since_days`` (newest first)."""
    agent = (agent_id or settings.default_agent_id or "agent").strip() or "agent"
    since = datetime.now(timezone.utc) - timedelta(days=max(1, int(since_days)))
    mem = get_default_memory()
    with mem.pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, thread_id, user_text, rejected_reply,
                       chosen_reply, critique, model, created_at
                FROM dpo_gold_pairs
                WHERE agent_id = %s AND created_at >= %s
                ORDER BY created_at DESC
                """,
                (agent, since),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for row in rows or []:
        out.append(
            {
                "id": row[0],
                "thread_id": row[1],
                "user_text": row[2],
                "rejected_reply": row[3],
                "chosen_reply": row[4],
                "critique": row[5],
                "model": row[6],
                "created_at": row[7].isoformat() if row[7] else "",
            }
        )
    return out


def export_dpo_jsonl(*, since_days: int = 7, agent_id: str | None = None) -> str:
    """Build JSONL for preference tuning (one JSON object per line)."""
    lines: list[str] = []
    for row in list_gold_pairs(since_days=since_days, agent_id=agent_id):
        lines.append(
            json.dumps(
                {
                    "prompt": row["user_text"],
                    "chosen": row["chosen_reply"],
                    "rejected": row["rejected_reply"],
                    "critique": row["critique"],
                    "meta": {
                        "id": row["id"],
                        "thread_id": row["thread_id"],
                        "model": row["model"],
                        "created_at": row["created_at"],
                    },
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


__all__ = ["export_dpo_jsonl", "insert_gold_pair", "list_gold_pairs"]
