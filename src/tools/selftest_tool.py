"""Agent tool: architectural self-diagnostic (read-only)."""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import settings
from src.selftest.core import run_all_checks


@tool
def run_selftest() -> str:
    """Run a full architectural self-diagnostic: database, vector store,
    Ollama models, embeddings, registered tools, identity, recent vitals.
    Returns a structured status report. Read-only — changes nothing.

    Live: when USER asks (selftest / "self-check").
    Solitary/intent: allowed by your will (rare). Read-only; no network probe.
    """
    report = run_all_checks(settings, probe_network=False)
    return report.to_agent_text()


__all__ = ["run_selftest"]
