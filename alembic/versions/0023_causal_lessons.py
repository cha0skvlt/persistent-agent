"""Causal lessons — structured “changed X because Y” rows.

Revision ID: 0023_causal_lessons
Revises: 0022_outbox_footer
"""

from __future__ import annotations

from alembic import op

revision = "0023_causal_lessons"
down_revision = "0022_outbox_footer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS causal_lessons (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id TEXT NOT NULL,
          change_text TEXT NOT NULL,
          because_text TEXT NOT NULL,
          evidence_ids TEXT[] NOT NULL DEFAULT '{}',
          embedding vector(768),
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_causal_lessons_agent_created "
        "ON causal_lessons (agent_id, created_at DESC)"
    )
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_causal_lessons_embedding
        ON causal_lessons
        USING hnsw (embedding vector_cosine_ops)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS causal_lessons")
