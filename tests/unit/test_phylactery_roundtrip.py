"""backup encrypt/decrypt roundtrip (no Postgres required)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest


@pytest.mark.skipif(not shutil.which("openssl"), reason="openssl not installed")
def test_openssl_aes256cbc_roundtrip():
    secret = b"agent-test-payload-" + os.urandom(64)
    passphrase = "unit-test-passphrase-123456"

    with tempfile.TemporaryDirectory() as td:
        plain = os.path.join(td, "plain.bin")
        enc = os.path.join(td, "plain.enc")
        out = os.path.join(td, "out.bin")
        with open(plain, "wb") as f:
            f.write(secret)

        subprocess.run(
            [
                "openssl",
                "enc",
                "-aes-256-cbc",
                "-salt",
                "-pbkdf2",
                "-iter",
                "600000",
                "-pass",
                f"pass:{passphrase}",
                "-in",
                plain,
                "-out",
                enc,
            ],
            check=True,
            timeout=60,
        )
        assert os.path.getsize(enc) > 0

        subprocess.run(
            [
                "openssl",
                "enc",
                "-aes-256-cbc",
                "-d",
                "-pbkdf2",
                "-iter",
                "600000",
                "-pass",
                f"pass:{passphrase}",
                "-in",
                enc,
                "-out",
                out,
            ],
            check=True,
            timeout=60,
        )

        with open(out, "rb") as f:
            got = f.read()
        assert got == secret


@pytest.mark.skipif(
    not shutil.which("pg_dump") or not os.environ.get("TEST_DATABASE_URL"),
    reason="needs pg_dump and TEST_DATABASE_URL",
)
def test_pg_dump_custom_roundtrip_smoke():
    """Optional integration: dump TEST_DATABASE_URL to custom format (unencrypted)."""
    url = os.environ["TEST_DATABASE_URL"].replace(
        "postgresql+psycopg://", "postgresql://"
    )
    with tempfile.TemporaryDirectory() as td:
        dump_path = os.path.join(td, "t.dump")
        subprocess.run(
            ["pg_dump", url, "-Fc", "-f", dump_path],
            check=True,
            timeout=120,
        )
        assert os.path.getsize(dump_path) > 0
