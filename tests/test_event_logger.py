import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit import CortexGit
from cortexgit.db.models import Base, EventLog, EventType
from cortexgit.core.event_log import EventLogger

from tests.db_helper import TEST_DATABASE_URL, test_engine, TestingSessionLocal

@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    """Ensure pytest-asyncio runs tests correctly."""
    return "asyncio"

@pytest.fixture(scope="module", autouse=True)
def create_test_db():
    if "postgresql" in TEST_DATABASE_URL:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute("DROP DATABASE IF EXISTS cortexgit_test;")
        cursor.execute("CREATE DATABASE cortexgit_test;")
        cursor.close()
        conn.close()

    yield

    asyncio.run(test_engine.dispose())

    if "postgresql" in TEST_DATABASE_URL:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute("DROP DATABASE IF EXISTS cortexgit_test;")
        cursor.close()
        conn.close()

@pytest.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        if "postgresql" in TEST_DATABASE_URL:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            except Exception:
                pass

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    async with test_engine.begin() as conn:
        if "postgresql" in TEST_DATABASE_URL:
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION prevent_update_delete()
                RETURNS TRIGGER AS $$
                BEGIN
                    RAISE EXCEPTION 'Updates and deletes are not allowed on this table';
                END;
                $$ LANGUAGE plpgsql;
            """))
            await conn.execute(text("DROP TRIGGER IF EXISTS enforce_event_log_append_only ON event_log;"))
            await conn.execute(text("""
                CREATE TRIGGER enforce_event_log_append_only
                BEFORE UPDATE OR DELETE ON event_log
                FOR EACH ROW EXECUTE FUNCTION prevent_update_delete();
            """))
        else:
            await conn.execute(text("DROP TRIGGER IF EXISTS enforce_event_log_append_only_update;"))
            await conn.execute(text("DROP TRIGGER IF EXISTS enforce_event_log_append_only_delete;"))
            await conn.execute(text("""
                CREATE TRIGGER enforce_event_log_append_only_update
                BEFORE UPDATE ON event_log
                BEGIN
                    SELECT RAISE(FAIL, 'Updates and deletes are not allowed on this table');
                END;
            """))
            await conn.execute(text("""
                CREATE TRIGGER enforce_event_log_append_only_delete
                BEFORE DELETE ON event_log
                BEGIN
                    SELECT RAISE(FAIL, 'Updates and deletes are not allowed on this table');
                END;
            """))
        
    yield
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session


@pytest.mark.anyio
async def test_valid_event_write():
    """Valid event write via SDK returns valid event object."""
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    event = await memory.log_event(
        session_id="session-123",
        agent_id="agent-456",
        event_type="user",
        payload={"message": "hello world"}
    )
    
    assert event.event_id is not None
    assert event.created_at is not None
    
    # Query DB to confirm successful write
    async with TestingSessionLocal() as session:
        db_event = await session.get(EventLog, event.event_id)
        assert db_event is not None
        assert db_event.session_id == "session-123"
        assert db_event.agent_id == "agent-456"
        assert db_event.event_type == EventType.USER
        assert db_event.payload == {"message": "hello world"}


@pytest.mark.anyio
async def test_invalid_event_type_rejected():
    """Invalid event_type raises ValueError in the EventLogger."""
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    with pytest.raises(ValueError) as exc_info:
        await memory.log_event(
            session_id="session-123",
            agent_id="agent-456",
            event_type="invalid_type",
            payload={"message": "hello world"}
        )
    assert "Invalid event_type" in str(exc_info.value)


@pytest.mark.anyio
async def test_two_writes_produce_separate_event_ids():
    """Two writes produce two separate event_ids."""
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    e1 = await memory.log_event(
        session_id="session-123",
        agent_id="agent-456",
        event_type="user",
        payload={"num": 1}
    )
    e2 = await memory.log_event(
        session_id="session-123",
        agent_id="agent-456",
        event_type="system",
        payload={"num": 2}
    )
    
    assert e1.event_id != e2.event_id


@pytest.mark.anyio
async def test_append_only_constraint_holds_application_layer(db_session):
    """The append-only constraint holds from the application layer."""
    logger = EventLogger(db_session)
    event = await logger.log_event(
        session_id="session-999",
        agent_id="agent-999",
        event_type="agent",
        payload={"action": "test"}
    )
    
    assert event.event_id is not None
    
    # Attempt an UPDATE on this event (should fail on commit)
    event.agent_id = "malicious_agent"
    with pytest.raises(DBAPIError) as exc_info:
        await db_session.commit()
    assert "Updates and deletes are not allowed on this table" in str(exc_info.value)
    
    await db_session.rollback()
    
    # Attempt a DELETE on this event (should fail on commit)
    await db_session.delete(event)
    with pytest.raises(DBAPIError) as exc_info:
        await db_session.commit()
    assert "Updates and deletes are not allowed on this table" in str(exc_info.value)
    
    await db_session.rollback()
