"""Add agent_identity table for Ed25519 cryptographic identity."""

from __future__ import annotations

from alembic import op

revision = "0011_agent_identity"
down_revision = "0010_vitals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_identity (
            id SERIAL PRIMARY KEY,
            agent TEXT NOT NULL UNIQUE,
            public_key_pem TEXT NOT NULL,
            private_key_encrypted BYTEA NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            version INTEGER NOT NULL DEFAULT 1
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_identity")
