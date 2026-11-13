"""Scheduler resilience: retry budget for intents, per-row backoff for outbox.

intents: attempts + last_error (exponential backoff, terminal 'failed').
outbox:  attempts + next_attempt_at (per-row backoff, terminal 'dead').

Revision ID: 0026_scheduler_resilience
Revises: 0025_causal_lessons_unique
"""

from __future__ import annotations

from alembic import op

revision = "0026_scheduler_resilience"
down_revision = "0025_causal_lessons_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE intents ADD COLUMN IF NOT EXISTS attempts INT NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE intents ADD COLUMN IF NOT EXISTS last_error TEXT")
    op.execute(
        "ALTER TABLE outbox ADD COLUMN IF NOT EXISTS attempts INT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE outbox ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE outbox DROP COLUMN IF EXISTS next_attempt_at")
    op.execute("ALTER TABLE outbox DROP COLUMN IF EXISTS attempts")
    op.execute("ALTER TABLE intents DROP COLUMN IF EXISTS last_error")
    op.execute("ALTER TABLE intents DROP COLUMN IF EXISTS attempts")
