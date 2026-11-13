#!/usr/bin/env python3
"""DEBUG verification for the memory retrieve layer (Agent Core 3.4).

Runs static grep checks and simulated turn scenarios with LOG_LEVEL=DEBUG-style
trace output. Use --live to exercise real Postgres + Ollama retrieve (optional).

Usage:
  LOG_LEVEL=DEBUG PYTHONPATH=. venv/bin/python3 scripts/debug_memory.py
  make debug-memory
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from src.graph.junk_gate import is_junk
from src.graph.retrieval_node import (
    _SELF_REFLECTION_IDENTITY_TOP_K,
    _SELF_REFLECTION_TOP_K,
    _self_reflection_top_k,
    build_self_reflection_sys_block,
    retrieve_memory,
)
from src.graph.workflow import _run_world_facts_reflect
from src.obs.console_format import ConsoleFormatter
from src.obs.logs import setup_agent_event_logger


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


class _EventTap:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def install(self) -> None:
        import src.graph.retrieval_node as retrieval_node
        import src.graph.workflow as workflow
        import src.obs.logs as logs

        self._orig_logs = logs.log_event
        self._orig_retrieval = retrieval_node.log_event
        self._orig_workflow = workflow.log_event

        def _capture(event: str, **fields: Any) -> None:
            self.events.append((event, dict(fields)))
            self._orig_logs(event, **fields)

        logs.log_event = _capture  # type: ignore[method-assign]
        retrieval_node.log_event = _capture  # type: ignore[method-assign]
        workflow.log_event = _capture  # type: ignore[method-assign]

    def restore(self) -> None:
        import src.graph.retrieval_node as retrieval_node
        import src.graph.workflow as workflow
        import src.obs.logs as logs

        logs.log_event = self._orig_logs  # type: ignore[method-assign]
        retrieval_node.log_event = self._orig_retrieval  # type: ignore[method-assign]
        workflow.log_event = self._orig_workflow  # type: ignore[method-assign]

    def last(self, event: str) -> dict[str, Any] | None:
        for ev, fields in reversed(self.events):
            if ev == event:
                return fields
        return None


class _TrackingMem:
    def __init__(self) -> None:
        self.saved: list[str] = []
        self.triples: list[tuple[str, str, str]] = []

    def save(self, content: str, **kwargs: Any) -> None:
        self.saved.append(content)

    def save_fact_to_graph(self, s: str, p: str, o: str) -> bool:
        self.triples.append((s, p, o))
        return True


def _configure_debug_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(ConsoleFormatter())
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    setup_agent_event_logger()
    for noisy in ("httpx", "httpcore", "urllib3", "ollama"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _grep(pattern: str, path: str) -> int:
    proc = subprocess.run(
        ["rg", "-n", pattern, path],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode


def static_checks() -> list[Check]:
    checks: list[Check] = []

    decay = subprocess.run(
        ["rg", "-n", r"time_decay|MEMORY_DECAY|memory_halflife", "src/memory.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    decay_hits = [
        ln
        for ln in (decay.stdout or "").splitlines()
        if ln and "battery" not in ln.casefold()
    ]
    checks.append(
        Check(
            "no time_decay in src/memory",
            len(decay_hits) == 0,
            "clean" if not decay_hits else "; ".join(decay_hits[:3]),
        )
    )

    traverse_rc = _grep(r"traverse_graph|batch_traverse", "src/graph/retrieval_node.py")
    checks.append(
        Check(
            "no traverse_graph in retrieval_node",
            traverse_rc != 0,
            (
                "zero matches"
                if traverse_rc != 0
                else "found traverse_graph on retrieve path"
            ),
        )
    )

    resolve_rc = _grep(
        r"resolve_entity|resolve_kg_entity", "src/graph/retrieval_node.py"
    )
    checks.append(
        Check(
            "no resolve_entity in retrieval_node",
            resolve_rc != 0,
            (
                "zero matches"
                if resolve_rc != 0
                else "found entity resolution on retrieve path"
            ),
        )
    )

    bump = subprocess.run(
        ["rg", "-n", r"importance\s*\+=|bump.*importance", "src/memory.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    checks.append(
        Check(
            "no dedup importance bump in src/memory.py",
            bump.returncode != 0,
            (
                "zero matches"
                if bump.returncode != 0
                else (bump.stdout or "").strip()[:120]
            ),
        )
    )

    return checks


def check_routing_helpers() -> list[Check]:
    return [
        Check(
            "junk_gate rejects bare Shalom",
            is_junk("Shalom AGENT!", user_text="Shalom", agent_text="Hello!"),
            "is_junk=True",
        ),
        Check(
            "self_reflection top_k default=4",
            _SELF_REFLECTION_TOP_K == 4,
            str(_SELF_REFLECTION_TOP_K),
        ),
        Check(
            "self_reflection identity top_k=6",
            _self_reflection_top_k("who are you")
            == _SELF_REFLECTION_IDENTITY_TOP_K
            == 6,
            f"identity={_self_reflection_top_k('who are you')}",
        ),
        Check(
            "self_reflection block header (English)",
            (
                build_self_reflection_sys_block(["— (2026-01-01) test line"]) or ""
            ).startswith("From my recent reflections:"),
            (build_self_reflection_sys_block(["— (2026-01-01) test line"]) or "")[:40],
        ),
    ]


def scenario_retrieve_skip_greeting(tap: _EventTap) -> Check:
    out = retrieve_memory({"messages": [HumanMessage(content="Shalom")]})
    skip = tap.last("retrieve.skip")
    return Check(
        "retrieve skips greeting (no KG path)",
        out == {"retrieved_memories": [], "retrieved_self_reflections": []}
        and skip is not None,
        f"skip={skip!r} out={out!r}",
    )


def scenario_retrieve_mocked(tap: _EventTap) -> Check:
    import src.graph.retrieval_node as rn

    mem = MagicMock()
    mem.search.return_value = [
        ("id-1", "saved memory line about phylactery stack details", 0.88, 0.7),
    ]
    mem.fetch_memory_embeddings_by_ids.return_value = {"id-1": [1.0, 0.0]}
    self_ref = MagicMock()
    self_ref.search.return_value = [
        ("I notice the void is older than me.", 0.9, None),
    ]

    orig_get = rn._get_mem
    orig_self = rn.get_default_self_reflection
    try:
        rn._get_mem = lambda: mem  # type: ignore[assignment]
        rn.get_default_self_reflection = lambda: self_ref  # type: ignore[assignment]
        out = retrieve_memory(
            {
                "messages": [
                    HumanMessage(
                        content="what is the phylactery database stack in persistent-agent"
                    )
                ],
                "flags": {},
            }
        )
    finally:
        rn._get_mem = orig_get  # type: ignore[assignment]
        rn.get_default_self_reflection = orig_self  # type: ignore[assignment]

    result = tap.last("retrieve.result") or {}
    mem_lines = out.get("retrieved_memories") or []
    self_lines = out.get("retrieved_self_reflections") or []
    has_kg = any(str(m).startswith("[KG]") for m in mem_lines)
    has_mem = any(str(m).startswith("[MEM]") for m in mem_lines)
    ok = result.get("kg") == 0 and has_mem and len(self_lines) >= 1 and not has_kg
    return Check(
        "retrieve mocked: kg=0, [MEM]+[SELF], no [KG]",
        ok,
        f"result={result} mem={mem_lines!r} self={len(self_lines)}",
    )


def scenario_reflect(
    tap: _EventTap,
    *,
    name: str,
    llm_json: str,
    user: str,
    agent: str,
    expect_saved: int,
    expect_triples: int,
    expect_facts_event: int | None = None,
) -> Check:
    import src.graph.workflow as wf

    tracking = _TrackingMem()
    orig = wf._invoke_reflect_llm
    wf._invoke_reflect_llm = lambda *_a, **_k: llm_json  # type: ignore[assignment]
    try:
        wf.settings.reflect_every_n = 1  # type: ignore[attr-defined]
        _run_world_facts_reflect(user, agent, tracking, turn_count=1)
    finally:
        wf._invoke_reflect_llm = orig  # type: ignore[assignment]

    extract = tap.last("reflect.extract") or {}
    facts_saved = int(extract.get("facts") or 0)
    ok = (
        len(tracking.saved) == expect_saved
        and len(tracking.triples) == expect_triples
        and (expect_facts_event is None or facts_saved == expect_facts_event)
    )
    return Check(
        name,
        ok,
        f"saved={len(tracking.saved)} triples={len(tracking.triples)} "
        f"reflect.extract={extract}",
    )


def scenario_retrieve_live(tap: _EventTap) -> Check:
    try:
        import psycopg

        from src.config import settings
        from src.ollama_client import ollama_http_client

        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        ollama_http_client().embeddings(model=settings.embed_model, prompt="ping")
    except Exception as e:
        return Check("retrieve live (skipped)", True, f"deps unavailable: {e}")

    out = retrieve_memory(
        {
            "messages": [
                HumanMessage(
                    content="what is the phylactery database stack in persistent-agent"
                )
            ],
            "flags": {},
        }
    )
    result = tap.last("retrieve.result") or {}
    mem_lines = out.get("retrieved_memories") or []
    has_kg = any(str(m).startswith("[KG]") for m in mem_lines)
    ok = result.get("kg") == 0 and not has_kg
    return Check(
        "retrieve live: kg=0, no [KG] lines",
        ok,
        f"result={result} mem_count={len(mem_lines)}",
    )


def _print_checks(title: str, checks: list[Check]) -> int:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    failures = 0
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        print(f"  [{mark}] {c.name}")
        if not c.ok or c.detail:
            print(f"         {c.detail}")
        if not c.ok:
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DEBUG verify memory retrieve layer (Agent Core 3.4)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run retrieve against real Postgres + Ollama (optional)",
    )
    args = parser.parse_args()

    _configure_debug_logging()
    print(
        "memory layer DEBUG verification (LOG_LEVEL=DEBUG trace on stderr)\n",
        flush=True,
    )

    failures = 0
    failures += _print_checks("Static checks", static_checks())
    failures += _print_checks("Routing / junk-gate helpers", check_routing_helpers())

    tap = _EventTap()
    tap.install()
    try:
        scenario_checks = [
            scenario_retrieve_skip_greeting(tap),
            scenario_retrieve_mocked(tap),
            scenario_reflect(
                tap,
                name="reflect Shalom -> junk-gate (facts=0)",
                llm_json='{"facts":["Shalom AGENT!"],"triples":[]}',
                user="Shalom",
                agent="Hello!",
                expect_saved=0,
                expect_triples=0,
                expect_facts_event=0,
            ),
            scenario_reflect(
                tap,
                name="reflect relational fact → memories only",
                llm_json=(
                    '{"facts":["Alex lives in Porto near the central river '
                    'embankment district"],"triples":[]}'
                ),
                user="u",
                agent="a",
                expect_saved=1,
                expect_triples=0,
                expect_facts_event=1,
            ),
            scenario_reflect(
                tap,
                name="reflect prose → memories only",
                llm_json=(
                    '{"facts":["USER believes time is linear yet illusory '
                    'and that matters to him"],"triples":[]}'
                ),
                user="u",
                agent="a",
                expect_saved=1,
                expect_triples=0,
                expect_facts_event=1,
            ),
        ]
        if args.live:
            scenario_checks.append(scenario_retrieve_live(tap))
        failures += _print_checks("Scenario trace checks", scenario_checks)
    finally:
        tap.restore()

    print(f"\n{'=' * 60}")
    if failures:
        print(f"RESULT: {failures} check(s) FAILED")
        return 1
    print("RESULT: all checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
