"""Agent run_selftest tool tests."""

from __future__ import annotations

from unittest.mock import patch

from src.selftest.core import CheckResult, CheckStatus, SelfTestReport
from src.tools.selftest_tool import run_selftest


def test_run_selftest_invokes_core_once():
    report = SelfTestReport(results=[CheckResult("postgres", CheckStatus.OK, "up")])
    with patch("src.tools.selftest_tool.run_all_checks", return_value=report) as run:
        out = run_selftest.invoke({})
    run.assert_called_once()
    assert run.call_args.kwargs.get("probe_network") is False
    assert "[OK] postgres" in out


def test_run_selftest_uses_agent_text_not_cli_skip():
    report = SelfTestReport(
        results=[
            CheckResult("postgres", CheckStatus.OK, "up"),
            CheckResult("web_tools", CheckStatus.SKIP, "live probe disabled"),
        ]
    )
    with patch("src.tools.selftest_tool.run_all_checks", return_value=report):
        out = run_selftest.invoke({})
    assert "[SKIP]" not in out
    assert "web_tools" not in out
    assert "[OK] postgres" in out
