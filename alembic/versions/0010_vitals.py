"""Add vitals table for heart daemon ticks."""

from __future__ import annotations

from alembic import op

revision = "0010_vitals"
down_revision = "0009_self_reflections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS vitals (
            tick_id BIGSERIAL PRIMARY KEY,
            tick_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            heart_pid INTEGER NOT NULL,
            body_metrics JSONB NOT NULL,
            phylactery_status TEXT NOT NULL
                CHECK (phylactery_status IN ('ok', 'degraded', 'fail')),
            signature TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS vitals_tick_at_idx ON vitals (tick_at DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS vitals_tick_at_idx")
    op.execute("DROP TABLE IF EXISTS vitals")
