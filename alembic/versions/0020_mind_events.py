"""Mind events — unified cross-channel conversation timeline (Mind Hub)."""

from __future__ import annotations

from alembic import op

revision = "0020_mind_events"
down_revision = "0019_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS mind_events (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
          kind VARCHAR(32) NOT NULL,
          channel VARCHAR(32),
          thread_id VARCHAR(128),
          content TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mind_events_agent_created "
        "ON mind_events (agent_id, created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_mind_events_agent_created")
    op.execute("DROP TABLE IF EXISTS mind_events")
