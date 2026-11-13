"""UTF-8 strings for tests; Cyrillic lives only under ``tests/fixtures/``."""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "test_locale_strings.json"
LOCALE: dict[str, str] = json.loads(_FIXTURES.read_text(encoding="utf-8"))
