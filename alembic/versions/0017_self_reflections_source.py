"""Add source column to self_reflections (solitary cycle)."""

from __future__ import annotations

from alembic import op

revision = "0017_self_reflections_source"
down_revision = "0016_dpo_gold_pairs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE self_reflections "
        "ADD COLUMN IF NOT EXISTS source VARCHAR(32) NOT NULL DEFAULT 'reflect'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE self_reflections DROP COLUMN IF EXISTS source")
