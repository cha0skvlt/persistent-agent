"""Tests for selftest core types and runner."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.selftest.core import (
    CheckResult,
    CheckStatus,
    SelfTestReport,
    run_all_checks,
)


def test_report_ok_false_only_on_fail():
    r = SelfTestReport(
        results=[
            CheckResult("a", CheckStatus.WARN, "w"),
            CheckResult("b", CheckStatus.OK, "ok"),
        ]
    )
    assert r.ok is True
    r.results.append(CheckResult("c", CheckStatus.FAIL, "bad"))
    assert r.ok is False


def test_report_ok_ignores_skip():
    r = SelfTestReport(results=[CheckResult("x", CheckStatus.SKIP, "skip")])
    assert r.ok is True


def test_to_text_and_to_dict():
    r = SelfTestReport(
        results=[
            CheckResult("postgres", CheckStatus.OK, "up", latency_ms=12),
        ]
    )
    text = r.to_text()
    assert "[OK] postgres: up (12ms)" in text
    assert "selftest PASS" in text
    data = r.to_dict()
    assert data["ok"] is True
    assert data["results"][0]["status"] == "ok"
    payload = json.loads(r.to_json())
    assert payload["results"][0]["name"] == "postgres"


def test_to_agent_text_omits_skip():
    r = SelfTestReport(
        results=[
            CheckResult("postgres", CheckStatus.OK, "up"),
            CheckResult(
                "web_tools",
                CheckStatus.SKIP,
                "live probe disabled (use --probe-network)",
            ),
        ]
    )
    cli = r.to_text()
    agent = r.to_agent_text()
    assert "[SKIP] web_tools" in cli
    assert "12 checks" not in cli  # only 2 results in this fixture
    assert "(2 checks)" in cli
    assert "[SKIP]" not in agent
    assert "web_tools" not in agent
    assert "(1 checks)" in agent
    assert "[OK] postgres" in agent


def test_run_all_checks_never_raises(monkeypatch):
    def boom(_s, *, probe_network=False):
        raise RuntimeError("kaboom")

    import src.selftest.checks as chk

    monkeypatch.setattr(chk, "check_postgres", boom)
    for name in (
        "check_pgvector",
        "check_tables",
        "check_ollama_reachable",
        "check_reasoning_model",
        "check_embed_model",
        "check_agent_tools",
        "check_identity",
        "check_heartbeat",
        "check_memory_stats",
        "check_config_sanity",
        "check_web_tools",
        "check_role_review_schema",
    ):
        monkeypatch.setattr(
            chk,
            name,
            lambda _s, *, probe_network=False, n=name: CheckResult(
                n.removeprefix("check_"), CheckStatus.OK, "ok"
            ),
        )
    report = run_all_checks(MagicMock(), probe_network=False)
    assert report.results[0].status == CheckStatus.FAIL
    assert "kaboom" in report.results[0].detail
    assert len(report.results) == 13
