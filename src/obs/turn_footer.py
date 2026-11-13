"""Turn summary line for Telegram (matches REPL ``→ node`` log style)."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from src.obs.logs import format_elapsed_ms
from src.utils import last_real_human_index


def tools_used_in_turn(messages: list) -> list[str]:
    """Tool names invoked after the last real USER message in *messages*."""
    start = last_real_human_index(messages)
    if start is None:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for m in messages[start + 1 :]:
        if isinstance(m, ToolMessage):
            name = str(getattr(m, "name", None) or "?").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def format_turn_log_footer(
    *,
    elapsed_ms: float | None,
    tools: list[str],
    node_timings: dict[str, float] | None = None,
) -> str:
    """One-line footer: ``**→ turn 12.5s**  search_web, fetch_url``.

    ``elapsed_ms=None`` → ``—`` (orphan / preempted pass after ``message_user``).
    """
    _ = node_timings  # kept for callers; node breakdown not shown in Telegram footer
    if elapsed_ms is None:
        dur = "—"
    else:
        dur = format_elapsed_ms(elapsed_ms)
    head = f"**→ turn {dur}**"
    clean_tools = [t.strip() for t in tools if (t or "").strip()]
    if clean_tools:
        return f"{head}  {', '.join(clean_tools)}"
    return head


def orphan_outreach_footer(
    *,
    elapsed_ms: float | None = None,
    tools: list[str] | None = None,
) -> str:
    """Footer when the graph never finished a normal stamp (cancel / crash).

    Prefer a real ``elapsed_ms`` (wall clock of the interrupted pass). Only
    omit duration when the caller truly has none (legacy / unknown).
    """
    return format_turn_log_footer(
        elapsed_ms=elapsed_ms,
        tools=list(tools) if tools is not None else ["message_user"],
    )


def footer_from_turn(turn: object) -> str:
    """Build the standard Telegram footer from an ``AgentTurnResult``-like object."""
    raw_elapsed = getattr(turn, "elapsed_ms", None)
    if raw_elapsed is None:
        elapsed: float | None = None
    else:
        try:
            elapsed = float(raw_elapsed)
        except (TypeError, ValueError):
            elapsed = None
    tools = list(getattr(turn, "tools_used", ()) or ())
    timings = getattr(turn, "node_timings", None)
    return format_turn_log_footer(
        elapsed_ms=elapsed,
        tools=tools,
        node_timings=timings if isinstance(timings, dict) else None,
    )


def format_summary_timing_line(elapsed_ms: float) -> str:
    """Follow-up line after deferred session summary."""
    return f"**→ summarize_session {format_elapsed_ms(elapsed_ms)}**"


def append_turn_footer(reply: str, *, elapsed_ms: float, tools: list[str]) -> str:
    body = (reply or "").strip() or "(no reply text)"
    footer = format_turn_log_footer(elapsed_ms=elapsed_ms, tools=tools)
    return f"{body}\n\n{footer}"
