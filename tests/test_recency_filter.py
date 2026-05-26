import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit.db.models import Base, EventLog, EventType
from cortexgit.core.recency_filter import RecencyFilter

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
async def test_returns_last_k_events_in_order(db_session):
    """Returns last k events in chronological order."""
    now = datetime.now(timezone.utc)
    session_id = "session-k-test"
    
    # Insert 5 events with staggered timestamps
    events = []
    for i in range(1, 6):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"index": i},
            created_at=now - timedelta(seconds=(6 - i) * 10)
        )
        db_session.add(event)
        events.append(event)
    await db_session.commit()

    rf = RecencyFilter(db_session)
    # Fetch last 3 events (should be indexes 3, 4, 5)
    result = await rf.get_recent_events(session_id, k=3)
    
    assert len(result) == 3
    # Check chronological order (oldest to newest)
    assert result[0].payload["index"] == 3
    assert result[1].payload["index"] == 4
    assert result[2].payload["index"] == 5


@pytest.mark.anyio
async def test_returns_fewer_than_k_events(db_session):
    """Returns fewer than k if not enough events exist in the session."""
    now = datetime.now(timezone.utc)
    session_id = "session-fewer"
    
    # Insert only 2 events
    for i in range(1, 3):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"index": i},
            created_at=now - timedelta(seconds=(3 - i) * 10)
        )
        db_session.add(event)
    await db_session.commit()

    rf = RecencyFilter(db_session)
    # Request k=5, but only 2 exist
    result = await rf.get_recent_events(session_id, k=5)
    
    assert len(result) == 2
    assert result[0].payload["index"] == 1
    assert result[1].payload["index"] == 2


@pytest.mark.anyio
async def test_returns_empty_list_for_unknown_session_id(db_session):
    """Returns empty list if no events exist for the session_id."""
    rf = RecencyFilter(db_session)
    result = await rf.get_recent_events("unknown-session-123", k=10)
    assert result == []


@pytest.mark.anyio
async def test_k_is_configurable_per_call(db_session):
    """Verifies that k is configurable per call and defaults to 20."""
    now = datetime.now(timezone.utc)
    session_id = "session-config-k"
    
    # Insert 25 events (more than the default k=20)
    for i in range(1, 26):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"index": i},
            created_at=now - timedelta(seconds=(26 - i) * 10)
        )
        db_session.add(event)
    await db_session.commit()

    rf = RecencyFilter(db_session)
    
    # 1. Test default k=20
    res_default = await rf.get_recent_events(session_id)
    assert len(res_default) == 20
    assert res_default[0].payload["index"] == 6
    assert res_default[-1].payload["index"] == 25

    # 2. Test k=5
    res_five = await rf.get_recent_events(session_id, k=5)
    assert len(res_five) == 5
    assert res_five[0].payload["index"] == 21
    assert res_five[-1].payload["index"] == 25
