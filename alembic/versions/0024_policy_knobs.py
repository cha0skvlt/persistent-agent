"""Durable behavioral policy knobs (not diary / not identity).

Revision ID: 0024_policy_knobs
Revises: 0023_causal_lessons
"""

from __future__ import annotations

from alembic import op

revision = "0024_policy_knobs"
down_revision = "0023_causal_lessons"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS policy_knobs (
          agent_id TEXT NOT NULL,
          key TEXT NOT NULL,
          value DOUBLE PRECISION NOT NULL,
          lesson_id UUID NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          PRIMARY KEY (agent_id, key)
        )
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS policy_knobs")
