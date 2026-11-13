"""Unique exact lesson text per agent (dedupe races).

Revision ID: 0025_causal_lessons_unique
Revises: 0024_policy_knobs
"""

from __future__ import annotations

from alembic import op

revision = "0025_causal_lessons_unique"
down_revision = "0024_policy_knobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop exact duplicates keeping the newest row per (agent, change, because).
    op.execute("""
        DELETE FROM causal_lessons a
        USING causal_lessons b
        WHERE a.agent_id = b.agent_id
          AND a.change_text = b.change_text
          AND a.because_text = b.because_text
          AND a.created_at < b.created_at
        """)
    op.execute("""
        DELETE FROM causal_lessons a
        USING causal_lessons b
        WHERE a.agent_id = b.agent_id
          AND a.change_text = b.change_text
          AND a.because_text = b.because_text
          AND a.created_at = b.created_at
          AND a.id < b.id
        """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_causal_lessons_agent_texts
        ON causal_lessons (agent_id, change_text, because_text)
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_causal_lessons_agent_texts")
