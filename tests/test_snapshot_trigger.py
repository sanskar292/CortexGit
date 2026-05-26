import pytest
import asyncio
import os
import uuid
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit.db.models import Base, EventLog, SnapshotStore, EventType
from cortexgit.llm.snapshot_trigger import should_snapshot, SnapshotTrigger

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
        
    yield
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session


@pytest.mark.anyio
async def test_returns_false_when_event_count_is_below_threshold(db_session):
    """Returns False when the count of new events since the last snapshot is below the threshold."""
    session_id = "session-1"
    
    # Insert 49 events
    for i in range(49):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": f"event {i}"}
        )
        db_session.add(event)
    await db_session.commit()
    
    # Threshold is 50, count is 49. should_snapshot should be False.
    res = await should_snapshot(session_id, db_session, threshold=50)
    assert res is False


@pytest.mark.anyio
async def test_returns_true_when_event_count_meets_threshold(db_session):
    """Returns True when the count of new events since the last snapshot meets or exceeds the threshold."""
    session_id = "session-2"
    
    event_range_val = "1,51" if "sqlite" in TEST_DATABASE_URL else text("int4range(1, 51)")
    snapshot = SnapshotStore(
        snapshot_id=uuid.uuid4(),
        session_id=session_id,
        event_range=event_range_val,
        summary="first 50 events summary",
        entities_mentioned=[],
        embedding=[0.1] * 1536
    )
    db_session.add(snapshot)
    
    # 2. Insert 100 events in this session
    for i in range(100):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": f"event {i}"}
        )
        db_session.add(event)
    await db_session.commit()
    
    # Total events: 100. Last snapshot upper bound: 51 (so last snapshotted is 50).
    # New events since last snapshot: 100 - 50 = 50.
    # Meets threshold of 50. should_snapshot should be True.
    res = await should_snapshot(session_id, db_session, threshold=50)
    assert res is True
    
    # Below threshold of 60. should_snapshot should be False.
    res_60 = await should_snapshot(session_id, db_session, threshold=60)
    assert res_60 is False


@pytest.mark.anyio
async def test_returns_true_when_no_snapshots_exist_and_meets_threshold(db_session):
    """Returns True when no snapshots exist and event count meets the threshold."""
    session_id = "session-3"
    
    # Insert 50 events
    for i in range(50):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": f"event {i}"}
        )
        db_session.add(event)
    await db_session.commit()
    
    # Count is 50, no snapshots. Meets threshold 50. should_snapshot should be True.
    res = await should_snapshot(session_id, db_session, threshold=50)
    assert res is True
    
    # should be False for threshold 51
    res_51 = await should_snapshot(session_id, db_session, threshold=51)
    assert res_51 is False


@pytest.mark.anyio
async def test_threshold_is_configurable_via_env(db_session, monkeypatch):
    """Verifies that the threshold is configurable via SNAPSHOT_THRESHOLD env var."""
    session_id = "session-4"
    
    # Insert 20 events
    for i in range(20):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": f"event {i}"}
        )
        db_session.add(event)
    await db_session.commit()
    
    # Default is 50, so should be False.
    res_default = await should_snapshot(session_id, db_session)
    assert res_default is False
    
    # Set env var SNAPSHOT_THRESHOLD to "20"
    monkeypatch.setenv("SNAPSHOT_THRESHOLD", "20")
    res_env = await should_snapshot(session_id, db_session)
    assert res_env is True


@pytest.mark.anyio
async def test_snapshot_trigger_class(db_session):
    """Verifies that the SnapshotTrigger class operates correctly."""
    session_id = "session-5"
    
    # Insert 15 events
    for i in range(15):
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": f"event {i}"}
        )
        db_session.add(event)
    await db_session.commit()
    
    # Trigger with limit 10
    trigger_10 = SnapshotTrigger(db_session, trigger_limit=10)
    assert await trigger_10.check_trigger(session_id) is True
    
    # Trigger with limit 20
    trigger_20 = SnapshotTrigger(db_session, trigger_limit=20)
    assert await trigger_20.check_trigger(session_id) is False
