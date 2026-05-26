import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit import CortexGit
from cortexgit.core.memory import ConflictError
from cortexgit.db.models import Base, EventLog, EntityRegistry, ConflictLog, EventType

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


@pytest.mark.anyio
async def test_clean_write_succeeds():
    """Clean write succeeds and returns True."""
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    # Create event first
    event = await memory.log_event(
        session_id="session-123",
        agent_id="agent_alpha",
        event_type="user",
        payload={"message": "initialize task"}
    )

    result = await memory.write_entity(
        key="agent_alpha.current_task",
        value={"task": "initialize registry", "priority": 1},
        agent_id="agent_alpha",
        event_id=event.event_id
    )
    
    assert result is True
    
    # Verify in database
    async with TestingSessionLocal() as session:
        db_res = await session.execute(
            select(EntityRegistry).where(EntityRegistry.key == "agent_alpha.current_task")
        )
        entity = db_res.scalar_one_or_none()
        assert entity is not None
        assert entity.value == {"task": "initialize registry", "priority": 1}
        assert entity.agent_id == "agent_alpha"


@pytest.mark.anyio
async def test_idempotent_write_succeeds():
    """Second write with same key and same value returns False (idempotent)."""
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    # Create events
    ev1 = await memory.log_event(
        session_id="session-123",
        agent_id="agent_alpha",
        event_type="user",
        payload={"message": "first event"}
    )
    ev2 = await memory.log_event(
        session_id="session-123",
        agent_id="agent_beta",
        event_type="user",
        payload={"message": "second event"}
    )

    # First write
    r1 = await memory.write_entity(
        key="project.goal",
        value="build memory core",
        agent_id="agent_alpha",
        event_id=ev1.event_id
    )
    assert r1 is True
    
    # Second write (same key, same value)
    r2 = await memory.write_entity(
        key="project.goal",
        value="build memory core",
        agent_id="agent_beta",
        event_id=ev2.event_id
    )
    assert r2 is False  # Idempotent write returns False


@pytest.mark.anyio
async def test_conflicting_write_rejected():
    """
    Second write with same key and different value raises ConflictError.
    Also proves:
    - Conflict is written to the conflict log.
    - Conflicting write does not appear in the entity registry.
    """
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    # Create events
    ev1 = await memory.log_event(
        session_id="session-123",
        agent_id="agent_alpha",
        event_type="user",
        payload={"message": "port event"}
    )
    ev2 = await memory.log_event(
        session_id="session-123",
        agent_id="agent_beta",
        event_type="user",
        payload={"message": "alternate port event"}
    )

    # First write (success)
    r1 = await memory.write_entity(
        key="shared.port",
        value=8000,
        agent_id="agent_alpha",
        event_id=ev1.event_id
    )
    assert r1 is True
    
    # Second write (conflict)
    with pytest.raises(ConflictError) as exc_info:
        await memory.write_entity(
            key="shared.port",
            value=8080,
            agent_id="agent_beta",
            event_id=ev2.event_id
        )
    
    # Verify in database
    async with TestingSessionLocal() as session:
        # 1. Conflicting write does NOT mutate entity registry
        res_reg = await session.execute(
            select(EntityRegistry).where(EntityRegistry.key == "shared.port")
        )
        entity = res_reg.scalar_one_or_none()
        assert entity is not None
        assert entity.value == 8000  # Remains original value

        # 2. Conflict is recorded in the ConflictLog table
        res_conflict = await session.execute(
            select(ConflictLog).where(ConflictLog.key == "shared.port")
        )
        conflict = res_conflict.scalar_one_or_none()
        assert conflict is not None
        assert conflict.existing_value == 8000
        assert conflict.proposed_value == 8080
        assert conflict.resolved is False
        assert conflict.existing_event_id == ev1.event_id
        assert conflict.proposed_event_id == ev2.event_id


@pytest.mark.anyio
async def test_concurrent_writes_produce_conflict():
    """
    Simultaneous concurrent writes with differing values to the same key
    should result in one success and one ConflictError.
    """
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    ev1 = await memory.log_event(
        session_id="session-123",
        agent_id="agent_alpha",
        event_type="user",
        payload={"message": "concurrent write A"}
    )
    ev2 = await memory.log_event(
        session_id="session-123",
        agent_id="agent_beta",
        event_type="user",
        payload={"message": "concurrent write B"}
    )

    # We perform two concurrent writes to 'concurrent.key' with different values
    res = await asyncio.gather(
        memory.write_entity(
            key="concurrent.key",
            value="value_a",
            agent_id="agent_alpha",
            event_id=ev1.event_id
        ),
        memory.write_entity(
            key="concurrent.key",
            value="value_b",
            agent_id="agent_beta",
            event_id=ev2.event_id
        ),
        return_exceptions=True
    )
    
    # One must succeed (return True) and the other must be ConflictError
    success = False
    conflict = False
    for r in res:
        if r is True:
            success = True
        elif isinstance(r, ConflictError):
            conflict = True
            
    assert success is True
    assert conflict is True
