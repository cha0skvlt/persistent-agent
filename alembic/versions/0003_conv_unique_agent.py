from __future__ import annotations

from alembic import op

revision = "0003_conv_unique_agent"
down_revision = "0002_kg_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Dedupe: keep one row per agent_id (newest created_at, then largest id).
    op.execute("""
        DELETE FROM conversations c
        WHERE c.id NOT IN (
          SELECT id FROM (
            SELECT DISTINCT ON (agent_id) id
            FROM conversations
            ORDER BY agent_id, created_at DESC NULLS LAST, id DESC
          ) keep_rows
        );
        """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_conversations_agent_id
          ON conversations (agent_id);
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_conversations_agent_id;")
