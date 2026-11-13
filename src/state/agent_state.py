from typing import Annotated, NotRequired, Sequence, TypedDict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages

from src.config import settings
from src.reply.visible import strip_redacted_thinking

_ARCHIVED_PREFIX = "[archived:"


def _message_text(msg: BaseMessage) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return strip_redacted_thinking(content)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return strip_redacted_thinking("".join(parts))
    return strip_redacted_thinking(str(content))


def _is_archived_summary(msg: BaseMessage) -> bool:
    return _message_text(msg).lstrip().startswith(_ARCHIVED_PREFIX)


def _is_tool_call_pair_start(msg: BaseMessage) -> bool:
    if _is_archived_summary(msg):
        return False
    if not isinstance(msg, AIMessage):
        return False
    tool_calls = getattr(msg, "tool_calls", None)
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def _tool_result_preview(result_msg: ToolMessage) -> str:
    result_text = _message_text(result_msg).strip()
    if len(result_text) > 80:
        result_text = result_text[:77] + "…"
    return result_text or "∅"


def _summarize_tool_pair(call_msg: AIMessage, result_msg: ToolMessage) -> AIMessage:
    tool_name = getattr(result_msg, "name", None) or "tool"
    tool_calls = getattr(call_msg, "tool_calls", None) or []
    if tool_calls and isinstance(tool_calls[0], dict):
        tool_name = tool_calls[0].get("name") or tool_name
    preview = _tool_result_preview(result_msg)
    return AIMessage(content=f"[archived: {tool_name} → {preview!r}]", id=str(uuid4()))


def _summarize_tool_run(call_msg: AIMessage, tool_msgs: list[ToolMessage]) -> AIMessage:
    """One archived line for an AIMessage with one or more tool results."""
    tool_calls = getattr(call_msg, "tool_calls", None) or []
    by_id = {str(getattr(tm, "tool_call_id", "") or ""): tm for tm in tool_msgs}
    parts: list[str] = []
    if tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or "tool")
            tid = str(tc.get("id") or "")
            tm = by_id.get(tid)
            if tm is not None:
                parts.append(f"{name} → {_tool_result_preview(tm)!r}")
    if not parts:
        for tm in tool_msgs:
            name = getattr(tm, "name", None) or "tool"
            parts.append(f"{name} → {_tool_result_preview(tm)!r}")
    return AIMessage(content=f"[archived: {'; '.join(parts)}]", id=str(uuid4()))


def _collapse_tool_pairs(
    merged: list[BaseMessage], *, collapse_before: int
) -> list[BaseMessage]:
    """Replace old AIMessage+ToolMessage runs with compact archived summaries."""
    result: list[BaseMessage] = []
    i = 0
    while i < len(merged):
        msg = merged[i]
        if i < collapse_before and _is_tool_call_pair_start(msg):
            j = i + 1
            tool_msgs: list[ToolMessage] = []
            while j < len(merged) and isinstance(merged[j], ToolMessage):
                tool_msgs.append(merged[j])
                j += 1
            if tool_msgs and isinstance(msg, AIMessage):
                result.append(_summarize_tool_run(msg, tool_msgs))
                i = j
                continue
        result.append(msg)
        i += 1
    return result


def _ensure_message_ids(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for msg in messages:
        if getattr(msg, "id", None):
            out.append(msg)
        else:
            out.append(msg.model_copy(update={"id": str(uuid4())}))
    return out


def _reconcile_messages(
    current: Sequence[BaseMessage],
    target: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Build RemoveMessage ops (+ new messages) to reach ``target`` checkpoint tail.

    Uses only ids already present on ``current`` — never invents ids for removals
    (that produced RemoveMessage ops for ids missing from the checkpoint).
    """
    tgt = _ensure_message_ids(list(target))
    target_ids = {m.id for m in tgt if m.id}
    current_ids = {getattr(m, "id", None) for m in current if getattr(m, "id", None)}

    ops: list[BaseMessage] = []
    for msg in current:
        mid = getattr(msg, "id", None)
        if mid and mid not in target_ids:
            ops.append(RemoveMessage(id=mid))

    for msg in tgt:
        if msg.id and msg.id not in current_ids:
            ops.append(msg)
    return ops


def checkpoint_overflow_updates(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage] | None:
    """If checkpoint exceeds cap, return updates that drop older messages."""
    max_cp = int(settings.checkpoint_max_messages)
    if max_cp <= 0:
        return None
    if len(messages) <= max_cp:
        return None
    pruned = prune_messages_for_llm(list(messages))
    target = pruned[-max_cp:]
    ops = _reconcile_messages(messages, target)
    return ops or None


def summarize_session_trim_updates(
    messages: Sequence[BaseMessage],
    *,
    offset: int,
    session_msgs: Sequence[BaseMessage],
    max_ctx: int,
    keep_recent: int = 2,
) -> list[BaseMessage] | None:
    """After summary, drop summarized session dialogue from checkpoint (summary stays in SYS)."""
    off = max(0, int(offset))
    sess = list(session_msgs)
    keep_n = max(0, int(keep_recent))
    if len(sess) > max_ctx:
        keep_session = list(sess[-max_ctx:])
    elif keep_n > 0:
        keep_session = list(sess[-keep_n:])
    else:
        keep_session = []
    target = prune_messages_for_llm(list(messages)[:off] + keep_session)
    ops = _reconcile_messages(messages, target)
    return ops or None


def prune_messages_for_llm(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Collapse stale tool pairs and cap tail (same rules as checkpoint reducer)."""
    merged = list(messages)
    keep_pairs = max(0, int(settings.tool_collapse_keep_recent_pairs))
    collapse_before = max(0, len(merged) - keep_pairs * 2)
    merged = _collapse_tool_pairs(merged, collapse_before=collapse_before)
    max_keep = int(settings.checkpoint_max_messages)
    if max_keep > 0:
        capped = merged[-max_keep:]
        # Qwen's chat template raises "No user query found in messages" when the
        # kept window has no user turn — this happens when many tool rounds push
        # the original ask past the cap, leaving only AI/Tool messages. Anchor
        # the most recent HumanMessage at the front so a user query always exists.
        if not any(isinstance(m, HumanMessage) for m in capped):
            last_human = next(
                (m for m in reversed(merged) if isinstance(m, HumanMessage)), None
            )
            if last_human is not None:
                room = max_keep - 1
                capped = [last_human] + (capped[-room:] if room > 0 else [])
        merged = capped
    return merged


def _messages_reducer(
    left: Sequence[BaseMessage], right: Sequence[BaseMessage]
) -> Sequence[BaseMessage]:
    """Append messages, collapse old tool pairs, cap checkpoint history."""
    if not right:
        return prune_messages_for_llm(list(left))
    if any(isinstance(m, RemoveMessage) for m in right):
        left_ids = {getattr(m, "id", None) for m in left if getattr(m, "id", None)}
        safe_right = [
            m for m in right if not isinstance(m, RemoveMessage) or m.id in left_ids
        ]
        merged = add_messages(left, safe_right)
    else:
        merged = list(left) + list(right)
    return prune_messages_for_llm(merged)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], _messages_reducer]
    retrieved_memories: list[str]
    # First-person diary lines for system injection (not KG/MEM).
    retrieved_self_reflections: NotRequired[list[str]]
    current_time: str
    session_summary: str
    # Messages restored from checkpoint when the current session began (for summary gating).
    session_message_offset: int
    # UUID issued once per process (`main`); used to reset session_message_offset after restart.
    session_uuid: NotRequired[str]
    # Last session_uuid committed to the checkpoint (detect process restart vs same session).
    session_uuid_anchor: NotRequired[str]
    # Whether to run reflect_node after this turn (false when the latest reply requests tools).
    needs_reflect: NotRequired[bool]
    # Incremented once per graph entry (update_time); used for REFLECT_EVERY_N throttle.
    turn_count: NotRequired[int]
    # ToolNode executions this turn; reset in update_time (caps agent→tools loops).
    tool_rounds: NotRequired[int]
    # Monotonic clock at turn entry — wall-clock budget for the tool loop.
    turn_started_at: NotRequired[float]
    # One compact record per tool round: what ran, with what, and whether it
    # yielded anything. Lives in state rather than being re-derived from
    # messages because the message window is trimmed — a long loop would
    # otherwise forget its own first steps and start repeating them.
    tool_ledger: NotRequired[list[dict]]
    # Last turn_count when reflect_node completed a run.
    last_reflect_turn: NotRequired[int]
    # Last turn_count when summarize_session persisted.
    last_summary_turn: NotRequired[int]
    # Single routing contract for the current USER turn (set by route_turn node).
    turn_plan: NotRequired[dict]
    # Mind Hub timeline lines (cross-channel continuity).
    retrieved_mind_timeline: NotRequired[list[str]]
    # Override diary source for reflect_node (e.g. solitary).
    diary_source: NotRequired[str]
