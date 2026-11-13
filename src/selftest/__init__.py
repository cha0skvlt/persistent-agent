"""Architectural self-diagnostic (read-only)."""

from src.selftest.core import CheckResult, CheckStatus, SelfTestReport, run_all_checks

__all__ = [
    "CheckResult",
    "CheckStatus",
    "SelfTestReport",
    "run_all_checks",
]
