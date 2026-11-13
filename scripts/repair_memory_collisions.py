#!/usr/bin/env python3
"""Strip polluted entity aliases and remove obvious KG mis-attachments (hygiene)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import psycopg  # noqa: E402

from src.config import settings  # noqa: E402
from src.kg.entities import (  # noqa: E402
    _MAX_AUTO_ALIAS_LEN,
    must_resolve_canonical_only,
)

# Aliases that must not live on another entity's ``aliases`` array.
_STRIP_ALIAS_LOWER = {
    "agent",
    "assistant",
    "the agent",
}


def _should_strip_alias(alias: str, canonical: str) -> bool:
    al = (alias or "").strip().lower()
    canon_l = (canonical or "").strip().lower()
    if not al or al == canon_l:
        return False
    protected = _STRIP_ALIAS_LOWER
    if al in protected or must_resolve_canonical_only(alias):
        return True
    if len((alias or "").strip()) > _MAX_AUTO_ALIAS_LEN:
        return True
    # Persona/user tokens attached to non-person entities (e.g. MacBook).
    if any(tok in al for tok in ("agent", "user")):
        if not any(tok in canon_l for tok in ("agent", "user")):
            return True
    return False


def _audit(conn) -> list[str]:
    lines: list[str] = []
    aid = settings.default_agent_id

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, canonical, aliases
            FROM entities WHERE agent_id = %s
            """,
            (aid,),
        )
        for eid, canonical, aliases in cur.fetchall():
            als = aliases if isinstance(aliases, list) else json.loads(aliases or "[]")
            for a in als:
                if _should_strip_alias(a, canonical):
                    lines.append(
                        f"STRIP_ALIAS entity={canonical!r} id={eid} alias={a!r}"
                    )

    return lines


def _apply(conn) -> int:
    aid = settings.default_agent_id
    aliases_stripped = 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, canonical, aliases
            FROM entities WHERE agent_id = %s
            """,
            (aid,),
        )
        for eid, canonical, aliases in cur.fetchall():
            als = aliases if isinstance(aliases, list) else json.loads(aliases or "[]")
            kept = [a for a in als if not _should_strip_alias(a, canonical)]
            if len(kept) != len(als):
                aliases_stripped += len(als) - len(kept)
                cur.execute(
                    """
                    UPDATE entities
                    SET aliases = %s::jsonb
                    WHERE id = %s::uuid AND agent_id = %s
                    """,
                    (json.dumps(kept, ensure_ascii=False), eid, aid),
                )

    conn.commit()
    return aliases_stripped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs (default: audit only)",
    )
    args = parser.parse_args()

    with psycopg.connect(settings.database_url) as conn:
        issues = _audit(conn)
        if args.apply:
            if issues:
                n_alias = _apply(conn)
                print(f"Applied: stripped {n_alias} alias(es)")
                issues = _audit(conn)
            else:
                print("Nothing to repair.")

    if not issues:
        print("No alias collisions detected.")
        return 0

    print(f"Found {len(issues)} issue(s):")
    for line in issues:
        print(f"  {line}")
    if not args.apply:
        print("\nRe-run with --apply to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
