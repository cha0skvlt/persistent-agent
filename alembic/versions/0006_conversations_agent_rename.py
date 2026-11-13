"""Align legacy conversations.agent_id with DEFAULT_AGENT_ID (agent).

Some installs already ran the manual SQL
    UPDATE conversations SET agent_id = 'agent' WHERE agent_id = 'miqu_witch';
which only renames if 'agent' does not already exist. With UNIQUE(agent_id)
from 0003_conv_unique_agent, a plain UPDATE collides whenever both legacy
('miqu_witch' / 'main_session_v1') and 'agent' rows coexist.

This migration is idempotent and collision-safe:
  - pick the "richest" row among {'agent', 'miqu_witch', 'main_session_v1'}:
    longest summary first (rolling summary is the canonical session memory),
    then longest messages (single-turn snapshot, less valuable);
  - UPSERT it into 'agent';
  - delete remaining legacy rows.

No-op when no rows match.
"""

from __future__ import annotations

from alembic import op

revision = "0006_conversations_agent_rename"
down_revision = "0005_memory_legacy_agent_ids"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        WITH winner AS (
            SELECT messages, summary
            FROM conversations
            WHERE agent_id IN ('agent', 'miqu_witch', 'main_session_v1')
            ORDER BY COALESCE(LENGTH(summary), 0) DESC,
                     LENGTH(messages::text) DESC NULLS LAST
            LIMIT 1
        )
        INSERT INTO conversations (agent_id, messages, summary)
        SELECT 'agent', messages, summary FROM winner
        ON CONFLICT (agent_id) DO UPDATE
          SET messages = EXCLUDED.messages,
              summary  = EXCLUDED.summary
        """)
    op.execute("""
        DELETE FROM conversations
        WHERE agent_id IN ('miqu_witch', 'main_session_v1')
        """)


def downgrade() -> None:
    pass
