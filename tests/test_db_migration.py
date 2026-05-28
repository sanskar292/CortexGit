import os
import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError
from alembic.config import Config
from alembic import command
from cortexgit.db.models import HAS_PGVECTOR

TEST_DB_URL_SYNC = os.getenv("TEST_DB_URL_SYNC", "sqlite:///cortexgit_test.db")
TEST_DB_URL_ASYNC = os.getenv("TEST_DB_URL_ASYNC", "sqlite+aiosqlite:///cortexgit_test.db")

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    # If using postgresql, connect to default postgres DB to create/drop the test database
    if "postgresql" in TEST_DB_URL_SYNC:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Drop test database if it exists
        cursor.execute("DROP DATABASE IF EXISTS cortexgit_test;")
        # Create test database
        cursor.execute("CREATE DATABASE cortexgit_test;")
        
        cursor.close()
        conn.close()

    # Run Alembic migrations programmatically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_ini_path = os.path.join(base_dir, "alembic.ini")
    
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DB_URL_ASYNC)
    command.upgrade(alembic_cfg, "head")

    yield

    # Teardown: Drop the test database or remove SQLite file at the end of testing
    if "postgresql" in TEST_DB_URL_SYNC:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute("DROP DATABASE IF EXISTS cortexgit_test;")
        cursor.close()
        conn.close()
    else:
        # SQLite cleanup
        db_file = "cortexgit_test.db"
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass

def test_tables_and_columns_exist():
    engine = create_engine(TEST_DB_URL_SYNC)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    # 1. Confirm all four tables exist
    expected_tables = {"event_log", "entity_registry", "snapshot_store", "conflict_log"}
    for table in expected_tables:
        assert table in tables, f"Expected table {table} not found in database."

    # 2. Confirm columns in event_log
    columns_event_log = {col["name"]: col["type"].__class__.__name__ for col in inspector.get_columns("event_log")}
    assert "event_id" in columns_event_log
    assert "session_id" in columns_event_log
    assert "agent_id" in columns_event_log
    assert "event_type" in columns_event_log
    assert "payload" in columns_event_log
    assert "created_at" in columns_event_log

    # 3. Confirm columns in entity_registry
    columns_entity_registry = {col["name"]: col["type"].__class__.__name__ for col in inspector.get_columns("entity_registry")}
    assert "key" in columns_entity_registry
    assert "value" in columns_entity_registry
    assert "agent_id" in columns_entity_registry
    assert "event_id" in columns_entity_registry
    assert "updated_at" in columns_entity_registry

    # 4. Confirm columns in snapshot_store
    columns_snapshot_store = {col["name"]: col["type"].__class__.__name__ for col in inspector.get_columns("snapshot_store")}
    assert "snapshot_id" in columns_snapshot_store
    assert "event_range" in columns_snapshot_store
    assert "summary" in columns_snapshot_store
    assert "entities_mentioned" in columns_snapshot_store
    assert "embedding" in columns_snapshot_store
    assert "created_at" in columns_snapshot_store

    # 5. Confirm columns in conflict_log
    columns_conflict_log = {col["name"]: col["type"].__class__.__name__ for col in inspector.get_columns("conflict_log")}
    assert "conflict_id" in columns_conflict_log
    assert "key" in columns_conflict_log
    assert "existing_value" in columns_conflict_log
    assert "proposed_value" in columns_conflict_log
    assert "existing_event_id" in columns_conflict_log
    assert "proposed_event_id" in columns_conflict_log
    assert "resolved" in columns_conflict_log
    assert "created_at" in columns_conflict_log

    engine.dispose()

def test_event_log_append_only_constraint():
    engine = create_engine(TEST_DB_URL_SYNC)
    
    # Insert an event
    event_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO event_log (event_id, session_id, agent_id, event_type, payload, created_at)
                VALUES (:event_id, 'test_session', 'test_agent', 'USER', '{"test": true}', :created_at)
            """),
            {"event_id": str(event_id), "created_at": datetime.now(timezone.utc).isoformat() if "sqlite" in TEST_DB_URL_SYNC else datetime.now(timezone.utc)}
        )

    # Try to update the event (should raise exception)
    with pytest.raises(DBAPIError) as exc_info:
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE event_log SET agent_id = 'malicious_agent' WHERE event_id = :event_id"),
                {"event_id": str(event_id)}
            )
    assert "Updates and deletes are not allowed on this table" in str(exc_info.value)

    # Try to delete the event (should raise exception)
    with pytest.raises(DBAPIError) as exc_info:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM event_log WHERE event_id = :event_id"),
                {"event_id": str(event_id)}
            )
    assert "Updates and deletes are not allowed on this table" in str(exc_info.value)

    engine.dispose()

def test_snapshot_store_immutability_constraint():
    engine = create_engine(TEST_DB_URL_SYNC)
    
    # Insert a snapshot
    snapshot_id = uuid.uuid4()
    embedding_val = [0.1] * 1536 if HAS_PGVECTOR else [0.1, 0.2]
    embedding_val_str = str(embedding_val) if "sqlite" in TEST_DB_URL_SYNC else embedding_val
    event_range_val = "1,10" if "sqlite" in TEST_DB_URL_SYNC else text("int4range(1, 10)")
    
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO snapshot_store (snapshot_id, event_range, summary, entities_mentioned, embedding, created_at)
                VALUES (:snapshot_id, :event_range, 'test snapshot', :entities, :embedding, :created_at)
            """),
            {
                "snapshot_id": str(snapshot_id),
                "event_range": event_range_val,
                "entities": '["entity1", "entity2"]' if "sqlite" in TEST_DB_URL_SYNC else ["entity1", "entity2"],
                "embedding": embedding_val_str,
                "created_at": datetime.now(timezone.utc).isoformat() if "sqlite" in TEST_DB_URL_SYNC else datetime.now(timezone.utc)
            }
        )

    # Try to update the snapshot (should raise exception)
    with pytest.raises(DBAPIError) as exc_info:
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE snapshot_store SET summary = 'updated summary' WHERE snapshot_id = :snapshot_id"),
                {"snapshot_id": str(snapshot_id)}
            )
    assert "Updates and deletes are not allowed on this table" in str(exc_info.value)

    # Note: deleting snapshots should be allowed because the trigger only runs BEFORE UPDATE
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM snapshot_store WHERE snapshot_id = :snapshot_id"),
            {"snapshot_id": str(snapshot_id)}
        )
        assert result.rowcount == 1

    engine.dispose()


def test_reg_indexes_exist():
    engine = create_engine(TEST_DB_URL_SYNC)
    inspector = inspect(engine)
    
    # 1. Get indexes for entity_nodes
    indexes_nodes = inspector.get_indexes("entity_nodes")
    node_index_names = {idx["name"] for idx in indexes_nodes if idx["name"]}
    assert "ix_entity_nodes_entity_name" in node_index_names
    assert "ix_entity_nodes_agent_id" in node_index_names
    assert "ix_entity_nodes_ttl_expiry" in node_index_names

    # 2. Get indexes for entity_edges
    indexes_edges = inspector.get_indexes("entity_edges")
    edge_index_names = {idx["name"] for idx in indexes_edges if idx["name"]}
    assert "ix_entity_edges_source_target" in edge_index_names

    # 3. Get indexes for node_hits
    indexes_hits = inspector.get_indexes("node_hits")
    hit_index_names = {idx["name"] for idx in indexes_hits if idx["name"]}
    assert "ix_node_hits_node_id_timestamp" in hit_index_names

    engine.dispose()
