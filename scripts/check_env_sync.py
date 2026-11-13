#!/usr/bin/env python3
"""Compare ``Settings`` env keys with ``.env.example`` (exit 1 on drift)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pydantic import AliasChoices

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENV_EXAMPLE = ROOT / ".env.example"

# Documented in .env.example for ops / Makefile — not pydantic Settings fields.
EXAMPLE_ONLY_KEYS = frozenset(
    {
        "INSTALL_DEPS",
    }
)

# Valid Settings keys omitted from the template (tests / rare overrides).
SETTINGS_OPTIONAL_IN_EXAMPLE = frozenset(
    {
        "TEST_DATABASE_URL",
    }
)


def _settings_env_keys() -> set[str]:
    from src.config import Settings

    keys: set[str] = set()
    for name, field in Settings.model_fields.items():
        alias = field.validation_alias
        if isinstance(alias, AliasChoices):
            keys.add(str(alias.choices[0]))
        elif alias is not None:
            keys.add(str(alias))
        else:
            keys.add(name.upper())
    return keys


def _parse_env_example(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match:
            keys.add(match.group(1))
    return keys


def main() -> int:
    if not ENV_EXAMPLE.is_file():
        print(f"missing {ENV_EXAMPLE}", file=sys.stderr)
        return 1

    settings_keys = _settings_env_keys()
    example_keys = _parse_env_example(ENV_EXAMPLE)

    missing_in_example = sorted(
        settings_keys - example_keys - SETTINGS_OPTIONAL_IN_EXAMPLE
    )
    unknown_in_example = sorted(example_keys - settings_keys - EXAMPLE_ONLY_KEYS)

    ok = not missing_in_example and not unknown_in_example
    if ok:
        print("env sync ok")
        return 0

    if missing_in_example:
        print("Settings keys missing from .env.example:", file=sys.stderr)
        for key in missing_in_example:
            print(f"  - {key}", file=sys.stderr)
    if unknown_in_example:
        print("Unknown keys in .env.example (no Settings field):", file=sys.stderr)
        for key in unknown_in_example:
            print(f"  - {key}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
