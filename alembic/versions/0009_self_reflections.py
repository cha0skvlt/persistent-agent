"""Add self_reflections table for first-person agent narrative."""

from __future__ import annotations

from alembic import op

revision = "0009_self_reflections"
down_revision = "0008_memories_type_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS self_reflections (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
          content TEXT NOT NULL,
          embedding vector(768),
          importance REAL DEFAULT 0.5,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_self_reflections_agent_id
          ON self_reflections (agent_id);
        """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_self_reflections_created_at
          ON self_reflections (created_at);
        """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS self_reflections_embedding_idx
          ON self_reflections
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS self_reflections_embedding_idx;")
    op.execute("DROP INDEX IF EXISTS idx_self_reflections_created_at;")
    op.execute("DROP INDEX IF EXISTS idx_self_reflections_agent_id;")
    op.execute("DROP TABLE IF EXISTS self_reflections;")
