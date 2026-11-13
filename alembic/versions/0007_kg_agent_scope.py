"""Scope knowledge-graph entities and edges by agent_id."""

from __future__ import annotations

from alembic import op

revision = "0007_kg_agent_scope"
down_revision = "0006_conversations_agent_rename"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE entities
          ADD COLUMN IF NOT EXISTS agent_id VARCHAR(64) NOT NULL DEFAULT 'agent';
        """)
    op.execute("""
        ALTER TABLE knowledge_graph
          ADD COLUMN IF NOT EXISTS agent_id VARCHAR(64) NOT NULL DEFAULT 'agent';
        """)

    op.execute("DROP INDEX IF EXISTS uq_entities_canonical_lower;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_agent_canonical_lower
          ON entities (agent_id, (lower(canonical)));
        """)

    op.execute("DROP INDEX IF EXISTS uq_knowledge_graph_triple_hash;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_graph_agent_triple_hash
          ON knowledge_graph (agent_id, triple_hash);
        """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_graph_agent_id
          ON knowledge_graph (agent_id);
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_knowledge_graph_agent_id;")
    op.execute("DROP INDEX IF EXISTS uq_knowledge_graph_agent_triple_hash;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_graph_triple_hash
          ON knowledge_graph (triple_hash);
        """)
    op.execute("DROP INDEX IF EXISTS uq_entities_agent_canonical_lower;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_canonical_lower
          ON entities ((lower(canonical)));
        """)
    op.execute("ALTER TABLE knowledge_graph DROP COLUMN IF EXISTS agent_id;")
    op.execute("ALTER TABLE entities DROP COLUMN IF EXISTS agent_id;")
