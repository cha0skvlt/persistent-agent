from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM

from src.config import settings
from src.memory import get_default_memory
from src.obs.logs import log_event
from src.obs.spans import traced_node
from src.reply.visible import strip_redacted_thinking
from src.state.agent_state import AgentState, summarize_session_trim_updates
from src.utils import get_message_text

_log = logging.getLogger(__name__)

_DEFAULT_SUMMARY_LLM_CAP = 80
_DEFAULT_FORCE_SUMMARY_LLM_CAP = 24

_SESSION_SUMMARY_SYS_HEADER = "[Session summary]"

# Recent dialogue kept in checkpoint after compress (anchor for the next reply).
_SUMMARY_TRIM_KEEP_RECENT = 2


def is_first_turn_after_summary(state: AgentState) -> bool:
    """True on the first live turn after ``summarize_session`` updated the checkpoint."""
    last = int(state.get("last_summary_turn", 0) or 0)
    if last <= 0:
        return False
    turn = int(state.get("turn_count", 0) or 0)
    return turn == last + 1


def session_summary_sys_block(state: AgentState) -> str | None:
    """SYS injection for rolling session summary (with post-summary guard)."""
    ss = (state.get("session_summary") or "").strip()
    if not ss:
        return None
    if is_first_turn_after_summary(state):
        # Omit the block entirely on the first turn after compress — otherwise the
        # model treats the summary wall as the user prompt (tail was trimmed).
        return None
    return "\n".join([_SESSION_SUMMARY_SYS_HEADER, ss])


def _summary_input_message_cap(*, force: bool = False) -> int:
    if force:
        cap = int(getattr(settings, "summary_llm_max_messages_force", 0) or 0)
        if cap > 0:
            return cap
    cap = int(getattr(settings, "summary_llm_max_messages", 0) or 0)
    return cap if cap > 0 else _DEFAULT_SUMMARY_LLM_CAP


def session_summary_due(state: AgentState) -> bool:
    """True when rolling summary should run."""
    every_n = int(getattr(settings, "summary_every_n", 0) or 0)
    if every_n <= 0:
        return False

    turns = int(state.get("turn_count", 0) or 0)
    if turns < every_n or turns % every_n != 0:
        return False

    last = int(state.get("last_summary_turn", 0) or 0)
    if turns > 0 and turns <= last:
        return False

    max_ctx = int(getattr(settings, "context_window_messages", 0) or 0)
    if max_ctx <= 0:
        return False

    msgs = list(state.get("messages") or [])
    offset = _effective_session_offset(state, len(msgs))
    session_msgs = msgs[offset:]
    if len(session_msgs) > max_ctx:
        prefix = list(session_msgs[:-max_ctx])
    else:
        prefix = list(session_msgs)
    return len(prefix) >= 2


def _effective_session_offset(state: AgentState, n_messages: int) -> int:
    """Ignore checkpoint offset when process session_uuid does not match persisted anchor."""
    raw = int(state.get("session_message_offset", 0) or 0)
    sid = (state.get("session_uuid") or "").strip()
    anchor = (state.get("session_uuid_anchor") or "").strip()
    if sid and anchor and sid != anchor:
        return max(n_messages, 0)
    if sid and not anchor:
        return max(n_messages, 0)
    return max(0, raw)


def _transcript_for_summary(msg: BaseMessage) -> str:
    from src.speaker import speaker_label

    line = get_message_text(msg, style="transcript")
    if isinstance(msg, AIMessage):
        speaker = speaker_label()
        for prefix in (f"{speaker}: ", "Agent: "):
            if line.startswith(prefix):
                body = strip_redacted_thinking(line[len(prefix) :])
                return f"{speaker}: {body}"
    return line


def _incremental_summarize(
    prior_summary: str, messages: list[BaseMessage], max_chars: int, llm: Any
) -> str:
    last_n_messages = "\n".join(_transcript_for_summary(m) for m in messages)
    prior = (prior_summary or "").strip()
    user_content = (
        f"Previous summary: {prior}\n\n"
        f"New messages:\n{last_n_messages}\n\n"
        f"Update the summary."
    )
    system = (
        f"Produce an updated session summary ≤{max_chars} characters. "
        "Merge the previous summary with the new messages. "
        "Keep names, numbers, facts, agreements, and decisions. "
        "Plain text only, no markdown. In English."
    )
    from src.obs.llm_invoke import invoke_llm

    res = invoke_llm(
        llm,
        [SystemMessage(content=system), HumanMessage(content=user_content)],
        role="summarize",
        config={"tags": [TAG_NOSTREAM]},
    )
    out = getattr(res, "content", "")
    if not isinstance(out, str):
        out = str(out)
    return out.strip()[:max_chars]


@traced_node("summarize_session")
def summarize_session(state: AgentState, config: RunnableConfig, *, llm: Any) -> dict:
    log_event("node.start", node="summarize_session")
    t0 = time.perf_counter()
    try:
        if not session_summary_due(state):
            return {}

        max_ctx = int(getattr(settings, "context_window_messages", 0) or 0)
        msgs = list(state.get("messages") or [])
        offset = _effective_session_offset(state, len(msgs))

        session_msgs = msgs[offset:]
        if len(session_msgs) > max_ctx:
            prefix = list(session_msgs[:-max_ctx])
        else:
            prefix = list(session_msgs)

        cap = _summary_input_message_cap(force=False)
        chunk = prefix[-cap:] if len(prefix) > cap else prefix

        prior_summary = (state.get("session_summary") or "").strip()
        turns = int(state.get("turn_count", 0) or 0)

        compressed = _incremental_summarize(
            prior_summary, chunk, int(settings.summary_max_chars), llm
        )
        if not compressed:
            return {}

        out: dict = {"session_summary": compressed, "last_summary_turn": turns}
        trim_ops = summarize_session_trim_updates(
            msgs,
            offset=offset,
            session_msgs=session_msgs,
            max_ctx=max_ctx,
            keep_recent=_SUMMARY_TRIM_KEEP_RECENT,
        )
        if trim_ops:
            out["messages"] = trim_ops

        # Persist to DB if possible (best-effort); same conversations row as snapshot (DEFAULT_AGENT_ID).
        try:
            get_default_memory().update_conversation_summary(
                settings.default_agent_id, compressed
            )
        except Exception as e:
            _log.warning("summary persist skipped: %s", e)

        return out
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_event("node.timing", node="summarize_session", elapsed_ms=elapsed_ms)


def summarize_session_node(state: AgentState, config: RunnableConfig) -> dict:
    """Graph-shaped entry for tests; production uses ``session_summary_runner``."""
    from src import llm_factory

    return summarize_session(state, config, llm=llm_factory.get_llm())
