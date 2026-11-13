"""Composite filters + entity trigram index."""

from __future__ import annotations

from alembic import op

revision = "0013_hot_perf_indexes"
down_revision = "0012_signatures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_agent_type
          ON memories (agent_id, memory_type);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_kg_agent_subject
          ON knowledge_graph (agent_id, subject_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_entities_canonical_trgm
          ON entities USING gin (lower(canonical) gin_trgm_ops);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_entities_canonical_trgm;")
    op.execute("DROP INDEX IF EXISTS idx_kg_agent_subject;")
    op.execute("DROP INDEX IF EXISTS idx_memories_agent_type;")
