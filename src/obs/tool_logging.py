"""Tool I/O observability — one INFO ``tool.result`` line per executed tool."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from src.obs.log_format import clip_words
from src.obs.logs import log_event
from src.state.agent_state import AgentState

_PREVIEW_MAX_CHARS = 220

_EMPTY_STRINGS = frozenset({"", "[]", "null", "None", "∅"})


def tool_result_is_empty(content: Any) -> bool:
    """True when a tool returned no usable payload ([] / null / empty string)."""
    if content is None:
        return True
    if isinstance(content, list):
        return len(content) == 0
    if isinstance(content, str):
        return content.strip() in _EMPTY_STRINGS
    return False


def _find_tool_call_args(messages: list[Any], tool_call_id: str) -> dict[str, Any]:
    if not tool_call_id:
        return {}
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        for tc in m.tool_calls or []:
            if not isinstance(tc, dict):
                continue
            if str(tc.get("id") or "") == tool_call_id:
                args = tc.get("args")
                return args if isinstance(args, dict) else {}
    return {}


def _query_preview(args: dict[str, Any]) -> str:
    for key in ("query", "name", "subject", "content"):
        val = args.get(key)
        if val:
            return clip_words(str(val))
    if args:
        return clip_words(str(args))
    return ""


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def log_tool_results(
    prior_messages: list[Any],
    new_messages: list[Any],
) -> None:
    """One INFO ``tool.result`` per ToolMessage: size + compact preview (or ∅)."""
    combined = list(prior_messages) + list(new_messages)
    for m in new_messages:
        if not isinstance(m, ToolMessage):
            continue
        name = str(getattr(m, "name", None) or "?")
        content = getattr(m, "content", None)
        if tool_result_is_empty(content):
            tcid = str(getattr(m, "tool_call_id", None) or "")
            args = _find_tool_call_args(combined, tcid)
            log_event(
                "tool.result",
                name=name,
                empty=True,
                chars=0,
                query=_query_preview(args),
            )
            continue
        text = _content_text(content)
        log_event(
            "tool.result",
            name=name,
            empty=False,
            chars=len(text),
            preview=clip_words(text[:_PREVIEW_MAX_CHARS], max_words=24),
        )


def invoke_tool_node_logged(
    tool_node: Any,
    state: AgentState,
    config: Any,
) -> dict[str, Any]:
    """Run LangGraph ``ToolNode`` and log each tool's result at INFO."""
    prior = list(state.get("messages") or [])
    out = tool_node.invoke(state, config)
    new_msgs = out.get("messages") or []
    log_tool_results(prior, new_msgs)
    try:
        from src.graph.turn_evidence import note_tool_text

        for m in new_msgs:
            if isinstance(m, ToolMessage):
                note_tool_text(getattr(m, "content", ""))
    except Exception:
        pass
    return out
