"""Background reverie prompts — Mind Hub aware, never contradict live dialogue."""

from __future__ import annotations

import re
import time

from src.config import settings
from src.db.pool import get_pool
from src.reverie.idle import minutes_since_last_user_turn
from src.system_prompt_text import load_solitary_prompt

_SPEAKER_PREFIX_RE = re.compile(r"^(?:USER|AGENT)\s*:\s*", re.IGNORECASE)

# (lane_id, band, instruction). band = P1|P2|P3 for seed framing.
_TOPIC_LANES: tuple[tuple[str, str, str], ...] = (
    (
        "discover_world",
        "P1",
        "P1 discover lane — one external topic (world, place, craft, science, culture). "
        "Start with search_web; fetch_url or browse_web open if a page/thread is worth it; "
        "read the full page text from tool output; a new fact gets save_memory after tool output.",
    ),
    (
        "discover_deep",
        "P1",
        "P1 discover lane — search_web for a lead, then fetch_url or browse_web open; "
        "read the full page text from tool output. "
        "If you keep a fact, save_memory in this pass.",
    ),
    (
        "discover_craft",
        "P1",
        "P1 discover lane — practice/tech (tools, code, methods). Web first; "
        "a durable fact gets save_memory.",
    ),
    (
        "hygiene",
        "P2",
        "P2 hygiene lane — sanitize: search_memory / search_by_time, take ids, "
        "forget_memory or memory_replace the false/stale row, save_memory with the correct slice. "
        "Not a 'who am I' loop.",
    ),
    (
        "intents",
        "P3",
        "P3 rhythm lane — list_intents is mandatory in this pass (do not claim an empty queue "
        "without the tool). schedule_intent / cancel_intent if there is a near-term continuation "
        "(time: minutes/hours, today, at most tomorrow).",
    ),
)


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _seed_text(content: str, *, limit: int = 120) -> str:
    """Short outward-facing spark from a memory/diary row (strip speaker prefix)."""
    raw = " ".join((content or "").split())
    raw = _SPEAKER_PREFIX_RE.sub("", raw).strip()
    return _clip(raw, limit)


# 8-bucket: P1 heavy, P2 twice, P3 once.
# indices: 0 world, 1 deep, 2 craft, 3 hygiene, 4 intents
_LANE_CYCLE: tuple[int, ...] = (0, 1, 3, 2, 0, 3, 1, 4)

# Last lane chosen for this process (solitary close-loop recovery).
_last_lane_id: str | None = None


def _hygiene_pressure() -> bool:
    """True while contradiction pile-up debt remains (cleared by hygiene_done)."""
    try:
        from src.policy_knobs import hygiene_debt

        return hygiene_debt() >= 1.0
    except Exception:
        # Fail closed — prefer hygiene lane when debt is unreadable.
        return True


def last_solitary_lane_id() -> str | None:
    return _last_lane_id


def _topic_lane(*, now: float | None = None) -> tuple[str, str, str]:
    """Rotate priority lanes (~5-minute buckets). Returns (lane_id, band, text)."""
    global _last_lane_id
    if _hygiene_pressure():
        lane = _TOPIC_LANES[3]
    else:
        t = time.time() if now is None else float(now)
        slot = int(t // 300)
        lane = _TOPIC_LANES[_LANE_CYCLE[slot % len(_LANE_CYCLE)]]
    _last_lane_id = lane[0]
    return lane


def _seed_lines(rows: list[tuple], *, prefix: str) -> list[str]:
    out: list[str] = []
    for row in rows:
        if not row or not row[1]:
            continue
        seed = _seed_text(str(row[1]))
        if len(seed) < 8:
            continue
        out.append(f"- [{prefix}{row[0]}] {seed}")
    return out


def _anchor_block(*, diary_k: int = 3, memory_k: int = 3) -> str:
    """Curiosity seeds — sparks for P1 or id hints for P2 (lane-aware framing)."""
    aid = settings.default_agent_id
    diary: list[str] = []
    mem: list[str] = []
    try:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, content
                    FROM self_reflections
                    WHERE agent_id = %s
                      AND source NOT IN ('outreach', 'solitary')
                      AND char_length(content) >= 20
                    ORDER BY random()
                    LIMIT %s
                    """,
                    (aid, max(0, int(diary_k))),
                )
                diary = _seed_lines(list(cur.fetchall()), prefix="id=")
                cur.execute(
                    """
                    SELECT id::text, content
                    FROM memories
                    WHERE agent_id = %s
                      AND char_length(content) >= 20
                    ORDER BY
                      CASE
                        WHEN content ~* '^\\s*USER\\s*:' THEN 0
                        WHEN content ~* '^\\s*AGENT\\s*:' THEN 2
                        ELSE 1
                      END,
                      random()
                    LIMIT %s
                    """,
                    (aid, max(0, int(memory_k))),
                )
                mem = _seed_lines(list(cur.fetchall()), prefix="MEM id=")
    except Exception:
        diary, mem = [], []

    lane_id, band, lane_text = _topic_lane()
    if band == "P2":
        spark = (
            "Seeds below are hygiene candidates (ids). "
            "Look them up with search_memory / search_by_time, fix or forget by id. "
            "Not a 'who am I' archaeology:"
        )
    elif band == "P3":
        spark = (
            "Seeds below are background; the pass's main work is list_intents and the queue. "
            "Do not drift into the web instead of the scheduler:"
        )
    else:
        spark = (
            "Seeds below are sparks for the web. Take ONE -> search_web / fetch_url / browse_web. "
            "Do not get stuck in search_memory instead of the world:"
        )

    parts = [
        f"Curiosity seeds — {spark}",
        f"Lane id={lane_id}: {lane_text}",
    ]
    if mem:
        parts.append("Memory seeds (prefer USER: / world facts):")
        parts.extend(mem)
    if diary:
        parts.append("Diary seeds (non-solitary only):")
        parts.extend(diary)
    if not mem and not diary:
        if band == "P2":
            parts.append(
                "No seeds in DB — still run search_memory / search_by_time for stale rows."
            )
        elif band == "P3":
            parts.append("No seeds in DB — still run list_intents.")
        else:
            parts.append(
                "No seeds in DB — pick any concrete external topic and search_web."
            )
    return "\n".join(parts)


def _activity_hint() -> str:
    elapsed = minutes_since_last_user_turn()
    if elapsed is None:
        return "No live turns recorded yet in agent_activity."
    if elapsed < 5:
        return (
            f"USER was active {elapsed:.0f} min ago — they are NOT silent. "
            "Do not claim silence. Keep this pass on the lane; "
            "do not retell the live dialogue."
        )
    if elapsed < 60:
        return f"Last live USER turn was {elapsed:.0f} min ago."
    hours = elapsed / 60.0
    return f"Last live USER turn was {hours:.1f} h ago — USER may be away."


def _compose(*, task: str, mind_limit: int = 10) -> str:
    # Mind Hub timeline lives in SYS (retrieve) — do not duplicate it here.
    _ = mind_limit
    parts = [task.strip(), "", _activity_hint()]
    anchors = _anchor_block()
    if anchors:
        parts.extend(["", anchors])
    # Policy in YAML solitary_prompt — pass frame is dynamic only.
    parts.extend(
        [
            "",
            "Pass frame (dynamic; obey solitary task above):",
            "- One main lane = Lane (P1|P2|P3). Tools are bound — call them.",
            "- Mind Hub in SYS — trust it; do not claim silence when USER_IN is recent.",
            "- Do not repeat an OUTREACH that already went out recently.",
        ]
    )
    return "\n".join(parts)


def build_solitary_user_message() -> str:
    task = load_solitary_prompt() or (
        "Free time: P1 web->save_memory; P2 hygiene by id; "
        "P3 list_intents/schedule_intent. Facts only from tool output. "
        "If you want to write to USER — message_user."
    )
    return _compose(task=task)


__all__ = ["build_solitary_user_message", "last_solitary_lane_id"]
