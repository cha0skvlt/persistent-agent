"""Causal lessons: importance weight (agent lesson --importance).

The lesson CLI writes importance and the skill treats high-importance lessons
as standing orders, but 0023 never created the column.

Revision ID: 0027_causal_lessons_importance
Revises: 0026_scheduler_resilience
"""

from __future__ import annotations

from alembic import op

revision = "0027_causal_lessons_importance"
down_revision = "0026_scheduler_resilience"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE causal_lessons "
        "ADD COLUMN IF NOT EXISTS importance REAL NOT NULL DEFAULT 0.5"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_causal_lessons_importance "
        "ON causal_lessons (agent_id, importance DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_causal_lessons_importance")
    op.execute("ALTER TABLE causal_lessons DROP COLUMN IF EXISTS importance")
