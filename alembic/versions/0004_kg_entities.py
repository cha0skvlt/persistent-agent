from __future__ import annotations

from alembic import op

revision = "0004_kg_entities"
down_revision = "0003_conv_unique_agent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS entities (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          canonical TEXT NOT NULL,
          aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
          kind TEXT,
          embedding vector(768),
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_canonical_lower
          ON entities ((lower(canonical)));
        """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_entities_aliases_gin
          ON entities USING gin (aliases);
        """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)

    op.execute("""
        ALTER TABLE knowledge_graph
          ADD COLUMN IF NOT EXISTS subject_id UUID,
          ADD COLUMN IF NOT EXISTS object_id UUID;
        """)

    op.execute("""
        INSERT INTO entities (canonical, aliases, kind)
        SELECT d.name, '[]'::jsonb, NULL
        FROM (
          SELECT DISTINCT ON (lower(trim(t))) trim(t) AS name
          FROM (
            SELECT subject AS t FROM knowledge_graph
            UNION ALL
            SELECT object FROM knowledge_graph
          ) q
          WHERE trim(t) <> ''
          ORDER BY lower(trim(t)), trim(t)
        ) AS d(name)
        WHERE NOT EXISTS (
          SELECT 1 FROM entities e WHERE lower(trim(e.canonical)) = lower(trim(d.name))
        );
        """)

    op.execute("""
        UPDATE knowledge_graph kg
        SET subject_id = e.id
        FROM entities e
        WHERE lower(trim(e.canonical)) = lower(trim(kg.subject));
        """)
    op.execute("""
        UPDATE knowledge_graph kg
        SET object_id = e.id
        FROM entities e
        WHERE lower(trim(e.canonical)) = lower(trim(kg.object));
        """)

    op.execute("""
        DELETE FROM knowledge_graph a
        USING knowledge_graph b
        WHERE a.subject_id IS NOT DISTINCT FROM b.subject_id
          AND a.predicate = b.predicate
          AND a.object_id IS NOT DISTINCT FROM b.object_id
          AND a.id > b.id;
        """)

    op.execute("DROP INDEX IF EXISTS uq_knowledge_graph_triple_hash;")

    op.execute("""
        UPDATE knowledge_graph
        SET triple_hash = md5(subject_id::text || '|' || predicate || '|' || object_id::text)
        WHERE subject_id IS NOT NULL AND object_id IS NOT NULL;
        """)

    op.execute("""
        DELETE FROM knowledge_graph a
        USING knowledge_graph b
        WHERE a.triple_hash = b.triple_hash AND a.id > b.id;
        """)

    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_graph_triple_hash
          ON knowledge_graph (triple_hash);
        """)

    op.execute("""
        ALTER TABLE knowledge_graph
          ALTER COLUMN subject_id SET NOT NULL,
          ALTER COLUMN object_id SET NOT NULL;
        """)

    op.execute("""
        ALTER TABLE knowledge_graph
          ADD CONSTRAINT fk_knowledge_graph_subject
            FOREIGN KEY (subject_id) REFERENCES entities(id) ON DELETE CASCADE;
        """)
    op.execute("""
        ALTER TABLE knowledge_graph
          ADD CONSTRAINT fk_knowledge_graph_object
            FOREIGN KEY (object_id) REFERENCES entities(id) ON DELETE CASCADE;
        """)


def downgrade() -> None:
    op.execute(
        "ALTER TABLE knowledge_graph DROP CONSTRAINT IF EXISTS fk_knowledge_graph_subject;"
    )
    op.execute(
        "ALTER TABLE knowledge_graph DROP CONSTRAINT IF EXISTS fk_knowledge_graph_object;"
    )
    op.execute("DROP INDEX IF EXISTS uq_knowledge_graph_triple_hash;")
    op.execute("""
        UPDATE knowledge_graph
        SET triple_hash = md5(subject || '|' || predicate || '|' || object)
        WHERE subject IS NOT NULL AND predicate IS NOT NULL AND object IS NOT NULL;
        """)
    op.execute("""
        DELETE FROM knowledge_graph a
        USING knowledge_graph b
        WHERE a.triple_hash = b.triple_hash AND a.id > b.id;
        """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_graph_triple_hash
          ON knowledge_graph (triple_hash);
        """)
    op.execute("""
        ALTER TABLE knowledge_graph
          DROP COLUMN IF EXISTS subject_id,
          DROP COLUMN IF EXISTS object_id;
        """)
    op.execute("DROP INDEX IF EXISTS idx_entities_embedding;")
    op.execute("DROP INDEX IF EXISTS idx_entities_aliases_gin;")
    op.execute("DROP INDEX IF EXISTS uq_entities_canonical_lower;")
    op.execute("DROP TABLE IF EXISTS entities;")
