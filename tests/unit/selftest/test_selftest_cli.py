"""CLI entry tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.selftest import __main__ as cli
from src.selftest.core import CheckResult, CheckStatus, SelfTestReport


def test_main_text_exit_ok(capsys):
    report = SelfTestReport(results=[CheckResult("postgres", CheckStatus.OK, "up")])
    with patch("src.selftest.__main__.run_all_checks", return_value=report):
        code = cli.main([])
    assert code == 0
    assert "postgres" in capsys.readouterr().out


def test_main_json_exit_fail(capsys):
    report = SelfTestReport(results=[CheckResult("postgres", CheckStatus.FAIL, "down")])
    with patch("src.selftest.__main__.run_all_checks", return_value=report):
        code = cli.main(["--json"])
    assert code == 1
    assert '"ok": false' in capsys.readouterr().out.lower()


def test_cli_entry_success():
    with patch("src.selftest.__main__.main", return_value=0):
        with pytest.raises(SystemExit) as exc:
            cli.cli_entry()
    assert exc.value.code == 0


def test_main_probe_network_flag():
    with patch("src.selftest.__main__.run_all_checks") as run:
        run.return_value = SelfTestReport(results=[])
        cli.main(["--probe-network"])
    run.assert_called_once()
    assert run.call_args.kwargs.get("probe_network") is True
