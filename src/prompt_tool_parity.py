"""Prompt ↔ runtime tool parity helpers (CI contract tests)."""

from __future__ import annotations

import re
from pathlib import Path

from src.selftest.checks import EXPECTED_AGENT_TOOLS
from src.system_prompt_text import read_yaml_doc

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_PATH = _REPO_ROOT / "prompts" / "agent" / "system_prompt.yml"

_PROMPT_TOOL_RE = re.compile(r"`([a-z][a-z0-9_]*)`")

# Mentioned in prompt as disabled, never-call, or optional future — not bound in graph.
PROMPT_EXEMPT_TOOLS = frozenset(
    {
        "get_current_time",
        "save_fact_to_graph",
        "add_triple",
        "describe_runtime",
    }
)

_SCAN_TOOL_NAMES = frozenset(EXPECTED_AGENT_TOOLS | PROMPT_EXEMPT_TOOLS)
_BARE_TOOL_RE = re.compile(
    r"\b("
    + "|".join(re.escape(n) for n in sorted(_SCAN_TOOL_NAMES, key=len, reverse=True))
    + r")\b"
)

_SELFTEST_PROTOCOL_KEYS = ("selftest_protocol",)
_TOOLS_PROTOCOL_KEY = "tools_protocol"


def tools_referenced_in_text(text: str) -> frozenset[str]:
    body = text or ""
    found = set(_PROMPT_TOOL_RE.findall(body))
    found.update(_BARE_TOOL_RE.findall(body))
    return frozenset(found)


def selftest_protocol_tools(*, path: Path = _PROMPT_PATH) -> frozenset[str]:
    doc = read_yaml_doc(path)
    found: set[str] = set()
    for key in _SELFTEST_PROTOCOL_KEYS:
        found.update(tools_referenced_in_text(str(doc.get(key) or "")))
    return frozenset(found)


def tools_protocol_tools(*, path: Path = _PROMPT_PATH) -> frozenset[str]:
    doc = read_yaml_doc(path)
    return tools_referenced_in_text(str(doc.get(_TOOLS_PROTOCOL_KEY) or ""))


def bound_agent_tool_names() -> frozenset[str]:
    from src.graph import workflow

    return frozenset(
        str(getattr(t, "name", "") or "")
        for t in workflow.tools
        if getattr(t, "name", None)
    )


def prompt_runtime_drift(
    *, path: Path = _PROMPT_PATH
) -> tuple[frozenset[str], frozenset[str]]:
    """Return (missing_from_runtime, exempt_but_referenced)."""
    bound = bound_agent_tool_names()
    selftest = selftest_protocol_tools(path=path)
    protocol = tools_protocol_tools(path=path)
    required = (selftest | protocol) - PROMPT_EXEMPT_TOOLS
    missing = required - bound
    exempt_referenced = (selftest | protocol) & PROMPT_EXEMPT_TOOLS
    return missing, exempt_referenced


def assert_prompt_runtime_parity(*, path: Path = _PROMPT_PATH) -> None:
    missing, _ = prompt_runtime_drift(path=path)
    if missing:
        raise AssertionError(
            "Prompt references tools not bound in workflow: "
            + ", ".join(sorted(missing))
        )
    extra = bound_agent_tool_names() - EXPECTED_AGENT_TOOLS
    if extra:
        raise AssertionError(
            "Bound tools missing from EXPECTED_AGENT_TOOLS: " + ", ".join(sorted(extra))
        )


__all__ = [
    "PROMPT_EXEMPT_TOOLS",
    "assert_prompt_runtime_parity",
    "bound_agent_tool_names",
    "prompt_runtime_drift",
    "selftest_protocol_tools",
    "tools_protocol_tools",
    "tools_referenced_in_text",
]
