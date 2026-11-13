"""Intents table — persistent scheduled intentions (Pulse).

Status lifecycle (VARCHAR, not a Postgres enum):
  pending → due → running → done | cancelled

  pending: scheduled, waiting for run_at
  due:     heart marked run_at reached (ready for reverie executor)
  running: intent pass claimed the row
  done:    one-shot intent completed
  cancelled: agent or operator cancelled before run
"""

from __future__ import annotations

from alembic import op

revision = "0021_intents"
down_revision = "0020_mind_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS intents (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL,
          text TEXT NOT NULL,
          run_at TIMESTAMPTZ NOT NULL,
          recur_rule VARCHAR(64),
          status VARCHAR(16) NOT NULL DEFAULT 'pending',
          created_by VARCHAR(16) NOT NULL DEFAULT 'agent',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_run_at TIMESTAMPTZ
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_intents_agent_status_run_at "
        "ON intents (agent_id, status, run_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_intents_agent_status_run_at")
    op.execute("DROP TABLE IF EXISTS intents")
