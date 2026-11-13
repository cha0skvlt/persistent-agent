"""DPO gold pairs — offline role-review training data (offline DPO)."""

from __future__ import annotations

from alembic import op

revision = "0016_dpo_gold_pairs"
down_revision = "0015_postgres_exporter_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS dpo_gold_pairs (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
          thread_id VARCHAR(256) NOT NULL DEFAULT '',
          user_text TEXT NOT NULL,
          rejected_reply TEXT NOT NULL,
          chosen_reply TEXT NOT NULL,
          critique TEXT NOT NULL DEFAULT '',
          role_failed BOOLEAN NOT NULL DEFAULT true,
          model VARCHAR(128) NOT NULL DEFAULT '',
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dpo_gold_pairs_created_at "
        "ON dpo_gold_pairs (created_at);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dpo_gold_pairs_agent_id "
        "ON dpo_gold_pairs (agent_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dpo_gold_pairs;")
