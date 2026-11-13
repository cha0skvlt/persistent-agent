"""CLI commands for cryptographic identity management.

Used by Makefile targets:
  make init-identity       — generate Ed25519 keypair, store in DB + Keychain
  make show-identity       — print public key and identity metadata
  make export-identity FILE=path  — dump identity to JSON file
"""

from __future__ import annotations

import base64
import getpass
import json
import sys
from pathlib import Path

import psycopg

from src.config import settings
from src.identity import keypair as kp
from src.identity.keychain import PhylacteryError, store_passphrase


def _pg_conninfo() -> str:
    url = settings.database_url
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


def cmd_init_identity() -> None:
    """Generate Ed25519 keypair, encrypt, store in DB + Keychain. Idempotent-refusal."""
    aid = settings.default_agent_id
    with psycopg.connect(_pg_conninfo()) as conn:
        row = conn.execute(
            "SELECT id FROM agent_identity WHERE agent = %s LIMIT 1",
            (aid,),
        ).fetchone()
        if row is not None:
            raise PhylacteryError(
                "Identity already exists. Phylactery is forever. Aborting."
            )

    # Non-interactive path for installers: passphrase from the environment.
    import os

    env_pass = (os.environ.get("AGENT_IDENTITY_PASSPHRASE") or "").strip()
    if env_pass:
        if len(env_pass) < 16:
            print(
                "AGENT_IDENTITY_PASSPHRASE must be at least 16 characters.",
                file=sys.stderr,
            )
            sys.exit(1)
        passphrase = env_pass
    else:
        # Prompt for passphrase (stdin may be redirected in tests).
        while True:
            try:
                passphrase = getpass.getpass("Passphrase (≥16 chars): ")
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.", file=sys.stderr)
                sys.exit(1)
            if len(passphrase) < 16:
                print("Passphrase must be at least 16 characters.", file=sys.stderr)
                continue
            try:
                confirm = getpass.getpass("Confirm passphrase: ")
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.", file=sys.stderr)
                sys.exit(1)
            if passphrase != confirm:
                print("Passphrases do not match. Try again.", file=sys.stderr)
                continue
            break

    private_pem, public_pem = kp.generate()
    encrypted = kp.encrypt_private(private_pem, passphrase, public_pem)

    # Keychain first. If it fails, no row is written and the command can be
    # retried. The reverse order leaves an encrypted key in the database with no
    # passphrase to open it — and init-identity refuses to run twice.
    try:
        store_passphrase(passphrase)
    except Exception as exc:
        print(
            f"Could not store the passphrase in Keychain: {exc}\n"
            "Nothing was written to the database. Fix Keychain access and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with psycopg.connect(_pg_conninfo()) as conn:
            conn.execute(
                """
                INSERT INTO agent_identity (agent, public_key_pem, private_key_encrypted)
                VALUES (%s, %s, %s)
                """,
                (aid, public_pem.decode("utf-8"), encrypted),
            )
            conn.commit()
    except Exception as exc:
        # The passphrase stays in Keychain on purpose: it is harmless on its own
        # and may be needed to recover an identity written by a partial run.
        print(
            f"Identity row was not written: {exc}\n"
            "The passphrase is in Keychain. Fix the database and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Phylactery created. Save this public key publicly:")
    print(public_pem.decode("utf-8"))


def cmd_show_identity() -> None:
    """Print public key + metadata. No passphrase required."""
    with psycopg.connect(_pg_conninfo()) as conn:
        row = conn.execute(
            "SELECT agent, public_key_pem, created_at, version "
            "FROM agent_identity WHERE agent = %s LIMIT 1",
            (settings.default_agent_id,),
        ).fetchone()
    if row is None:
        print("No identity found. Run: make init-identity", file=sys.stderr)
        sys.exit(1)
    agent, public_pem, created_at, version = row
    print(f"Agent id:   {agent}")
    print(f"Created:    {created_at}")
    print(f"Version:    {version}")
    print("Public key:")
    print(public_pem)


def cmd_export_identity(path: str) -> None:
    """Export identity to a JSON file (for machine migration)."""
    with psycopg.connect(_pg_conninfo()) as conn:
        row = conn.execute(
            "SELECT agent, public_key_pem, private_key_encrypted, created_at, version "
            "FROM agent_identity WHERE agent = %s LIMIT 1",
            (settings.default_agent_id,),
        ).fetchone()
    if row is None:
        print("No identity found. Run: make init-identity", file=sys.stderr)
        sys.exit(1)
    agent, public_pem, encrypted_bytes, created_at, version = row
    payload = {
        "agent": agent,
        "public_key_pem": public_pem,
        "private_key_encrypted": base64.b64encode(bytes(encrypted_bytes)).decode(),
        "created_at": str(created_at),
        "version": version,
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Identity exported to {dest}")
    print(
        "IMPORTANT: Keep this file encrypted. Restore with manual DB INSERT + Keychain import."
    )


def cmd_store_backup_passphrase() -> None:
    """Interactive: store backup encryption passphrase in Keychain."""
    from src.identity.keychain import store_backup_passphrase

    while True:
        try:
            passphrase = getpass.getpass("Backup passphrase (>=16 chars): ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(1)
        if len(passphrase) < 16:
            print("Passphrase must be at least 16 characters.", file=sys.stderr)
            continue
        try:
            confirm = getpass.getpass("Confirm backup passphrase: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(1)
        if passphrase != confirm:
            print("Passphrases do not match.", file=sys.stderr)
            continue
        break

    store_backup_passphrase(passphrase)
    print("Backup passphrase stored in Keychain (agent.phylactery.backup).")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phylactery identity management")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("init", help="Generate and store new Ed25519 identity")
    sub.add_parser("show", help="Print public key and metadata")
    sub.add_parser(
        "store-backup-passphrase",
        help="Store backup encryption passphrase in Keychain",
    )
    exp = sub.add_parser("export", help="Export identity to JSON file")
    exp.add_argument("file", help="Output path")
    args = parser.parse_args()

    if args.cmd == "init":
        cmd_init_identity()
    elif args.cmd == "show":
        cmd_show_identity()
    elif args.cmd == "export":
        cmd_export_identity(args.file)
    elif args.cmd == "store-backup-passphrase":
        cmd_store_backup_passphrase()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
