"""DEBUG LLM-context dumps — full SYS/messages go to a separate rotating file.

Full prompt dumps used to flood the main log at DEBUG (~90% of tail lines),
drowning INFO events. The main log keeps one compact ``llm.invoke`` header;
complete payloads land in ``~/Library/Logs/agent-context.log`` (rotating,
in-process handler — safe because only this process writes it).
"""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.obs.logs import log_debug_event

_CONTEXT_LOG_MAX_BYTES = 32 * 1024 * 1024
_CONTEXT_LOG_BACKUPS = 1


def _context_logger() -> logging.Logger:
    lg = logging.getLogger("agent.context")
    if not lg.handlers:
        path = Path.home() / "Library" / "Logs" / "agent-context.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=_CONTEXT_LOG_MAX_BYTES,
            backupCount=_CONTEXT_LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
    return lg


def context_debug_event(event: str, **fields: Any) -> None:
    """JSON line into the context dump file (never the main log)."""
    payload = {"event": event, "ts_ms": int(time.time() * 1000), **fields}
    try:
        line = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        line = json.dumps({"event": event, "error": "non_serializable"})
    try:
        _context_logger().debug(line)
    except Exception:
        pass


def _msg_role(m: Any) -> str:
    cls = type(m).__name__
    if cls == "HumanMessage":
        return "user"
    if cls == "AIMessage":
        return "agent"
    if cls == "ToolMessage":
        return "tool"
    if cls == "SystemMessage":
        return "sys"
    return cls.lower()


def _msg_text(m: Any) -> str:
    content = getattr(m, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def emit_llm_context(
    path: str,
    sys_text: str,
    messages: list[Any],
    tools_names: str,
) -> None:
    """Compact header in the main log; full SYS + messages into the context file."""
    non_sys = [m for m in messages if _msg_role(m) != "sys"]
    log_debug_event(
        "llm.invoke",
        path=path,
        msgs=len(non_sys),
        sys_chars=len(sys_text),
        tools=tools_names,
    )
    context_debug_event("llm.sys", path=path, text=sys_text)
    for i, m in enumerate(non_sys):
        role = _msg_role(m)
        text = _msg_text(m)
        tc = getattr(m, "tool_calls", None)
        tc_names = (
            ",".join(t.get("name", "?") for t in tc)
            if isinstance(tc, list) and tc
            else ""
        )
        tool_name = (getattr(m, "name", "") or "") if role == "tool" else ""
        context_debug_event(
            "llm.ctx_msg",
            path=path,
            idx=i,
            role=role,
            text=text,
            tool_calls=tc_names,
            tool_name=tool_name,
        )
