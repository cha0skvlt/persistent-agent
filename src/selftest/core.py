"""Self-test types and runner (single source of truth for CLI + agent tool)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.config import Settings


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    latency_ms: int | None = None


@dataclass
class SelfTestReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no FAIL; WARN/SKIP do not fail the report."""
        return all(r.status != CheckStatus.FAIL for r in self.results)

    def to_text(self) -> str:
        return self._format_text(self.results, total=len(self.results))

    def to_agent_text(self) -> str:
        """Render for agent replay: omit SKIP (not probed in turn path).

        Showing SKIP to the model invites rounding \"not checked\" → \"stable\".
        CLI ``to_text()`` still includes SKIP for the operator.
        """
        shown = [r for r in self.results if r.status != CheckStatus.SKIP]
        return self._format_text(shown, total=len(shown))

    def _format_text(self, results: list[CheckResult], *, total: int) -> str:
        lines: list[str] = []
        for r in results:
            mark = r.status.value.upper()
            line = f"[{mark}] {r.name}: {r.detail}"
            if r.latency_ms is not None:
                line += f" ({r.latency_ms}ms)"
            lines.append(line)
        summary = "PASS" if self.ok else "FAIL"
        lines.append(f"--- selftest {summary} ({total} checks) ---")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "results": [
                {
                    **asdict(r),
                    "status": r.status.value,
                }
                for r in self.results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def run_all_checks(
    settings: Settings,
    *,
    probe_network: bool = False,
) -> SelfTestReport:
    """Run every check; never raises — returns a full report."""
    from src.selftest import checks as chk

    report = SelfTestReport()
    runners = [
        chk.check_postgres,
        chk.check_pgvector,
        chk.check_tables,
        chk.check_ollama_reachable,
        chk.check_reasoning_model,
        chk.check_embed_model,
        chk.check_agent_tools,
        chk.check_identity,
        chk.check_heartbeat,
        chk.check_memory_stats,
        chk.check_config_sanity,
        chk.check_web_tools,
        chk.check_role_review_schema,
    ]
    for fn in runners:
        try:
            result = fn(settings, probe_network=probe_network)
        except Exception as exc:  # pragma: no cover — checks must not raise
            result = CheckResult(
                name=getattr(fn, "__name__", "unknown").removeprefix("check_"),
                status=CheckStatus.FAIL,
                detail=f"check crashed: {exc}",
            )
        report.results.append(result)
    return report


__all__ = [
    "CheckResult",
    "CheckStatus",
    "SelfTestReport",
    "run_all_checks",
]
