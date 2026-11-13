"""Agent activity table — last live turn timestamp (time-sense / activity)."""

from __future__ import annotations

from alembic import op

revision = "0018_agent_activity"
down_revision = "0017_self_reflections_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_activity (
          agent_id VARCHAR(64) PRIMARY KEY,
          last_turn_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_activity")
