"""CLI entry: python -m src.selftest"""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.selftest.core import run_all_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agent architectural self-test (read-only)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report (to_dict)",
    )
    parser.add_argument(
        "--probe-network",
        action="store_true",
        help="Run live web fetch probe (≤5s)",
    )
    args = parser.parse_args(argv)

    report = run_all_checks(
        get_settings(),
        probe_network=bool(args.probe_network),
    )
    if args.json:
        print(report.to_json())
    else:
        print(report.to_text())
    return 0 if report.ok else 1


def cli_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    cli_entry()
