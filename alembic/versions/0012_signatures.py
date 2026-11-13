"""Add signature column to self_reflections.

``vitals.signature`` was already added in 0010_vitals.
"""

from __future__ import annotations

from alembic import op

revision = "0012_signatures"
down_revision = "0011_agent_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE self_reflections
            ADD COLUMN IF NOT EXISTS signature TEXT
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE self_reflections DROP COLUMN IF EXISTS signature")
