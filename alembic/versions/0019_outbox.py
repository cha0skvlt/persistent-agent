"""Outbox table — proactive outreach message queue."""

from __future__ import annotations

from alembic import op

revision = "0019_outbox"
down_revision = "0018_agent_activity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS outbox (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
          created_at TIMESTAMPTZ DEFAULT now(),
          trigger_type VARCHAR(32) NOT NULL,
          draft_text TEXT NOT NULL,
          status VARCHAR(16) NOT NULL DEFAULT 'pending',
          sent_at TIMESTAMPTZ
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_outbox_status_created "
        "ON outbox (status, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_outbox_status_created")
    op.execute("DROP TABLE IF EXISTS outbox")
