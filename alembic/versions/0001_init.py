from __future__ import annotations

from alembic import op

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS memories (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL,
          memory_type VARCHAR(32) NOT NULL,
          content TEXT NOT NULL,
          embedding vector(768),
          metadata JSONB DEFAULT '{}',
          importance FLOAT DEFAULT 0.5,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories (agent_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_metadata_gin ON memories USING gin (metadata);"
    )
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          agent_id VARCHAR(64) NOT NULL,
          messages JSONB NOT NULL DEFAULT '[]',
          summary TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_agent_id ON conversations (agent_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at);"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_graph (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          subject VARCHAR(256) NOT NULL,
          predicate VARCHAR(256) NOT NULL,
          object TEXT NOT NULL,
          embedding vector(768),
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_knowledge_graph_created_at ON knowledge_graph (created_at);"
    )
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_graph_embedding ON knowledge_graph
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledge_graph;")
    op.execute("DROP TABLE IF EXISTS conversations;")
    op.execute("DROP TABLE IF EXISTS memories;")
