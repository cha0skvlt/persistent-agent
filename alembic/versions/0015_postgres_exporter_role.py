"""Create read-only ``postgres_exporter`` role for Prometheus scraping."""

from __future__ import annotations

from alembic import op

revision = "0015_postgres_exporter_role"
down_revision = "0014_agent_id_agent_to_agent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'postgres_exporter') THEN
                CREATE ROLE postgres_exporter LOGIN;
            END IF;
        END
        $$;
    """)
    op.execute("ALTER ROLE postgres_exporter SET SEARCH_PATH TO public, pg_catalog")
    op.execute("GRANT pg_monitor TO postgres_exporter")
    # The database name is whatever this deployment uses, not a hardcoded default.
    op.execute("""
        DO $$
        BEGIN
            EXECUTE format(
                'GRANT CONNECT ON DATABASE %I TO postgres_exporter',
                current_database()
            );
        END
        $$;
    """)
    op.execute("GRANT USAGE ON SCHEMA public TO postgres_exporter")
    op.execute(
        "GRANT SELECT ON memories, knowledge_graph, entities, self_reflections, vitals "
        "TO postgres_exporter"
    )
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'checkpoints'
            ) THEN
                EXECUTE 'GRANT SELECT ON checkpoints TO postgres_exporter';
            END IF;
        END
        $$;
    """)


def downgrade() -> None:
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM postgres_exporter")
    op.execute("REVOKE ALL ON SCHEMA public FROM postgres_exporter")
    op.execute("""
        DO $$
        BEGIN
            EXECUTE format(
                'REVOKE CONNECT ON DATABASE %I FROM postgres_exporter',
                current_database()
            );
        END
        $$;
    """)
    op.execute("REVOKE pg_monitor FROM postgres_exporter")
    op.execute("DROP ROLE IF EXISTS postgres_exporter")
