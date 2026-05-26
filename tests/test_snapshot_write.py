import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit import CortexGit
from cortexgit.db.models import Base, EventLog, SnapshotStore, EventType
from cortexgit.core.write_back_gate import ValidationError

from tests.db_helper import TEST_DATABASE_URL, test_engine, TestingSessionLocal

import cortexgit.db.database
cortexgit.db.database.AsyncSessionLocal = TestingSessionLocal

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


@pytest.mark.anyio
@patch("cortexgit.core.memory.summarize")
@patch("cortexgit.core.memory.write_snapshot")
async def test_snapshot_written_when_threshold_met(mock_write, mock_summarize, monkeypatch):
    """Snapshot is written to store when the trigger threshold is met (e.g. 5 events)."""
    # Set threshold to 5
    monkeypatch.setenv("SNAPSHOT_THRESHOLD", "5")
    
    # Mock embeddings and summarization
    mock_summarize.return_value = {
        "summary": "This is a valid mock summary of the events.",
        "entities_mentioned": ["user", "system"],
        "event_range": [1, 5]
    }
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    session_id = "test-session-snapshot"
    
    # Post 4 events (should NOT trigger snapshot)
    for i in range(4):
        await memory.log_event(
            session_id=session_id,
            agent_id="agent-1",
            event_type="user",
            payload={"text": f"event {i}"}
        )
        
    # Check SnapshotStore and mock calls (should be 0)
    assert mock_summarize.call_count == 0
    assert mock_write.call_count == 0

    # Post 5th event (reaches threshold 5, triggers snapshot)
    await memory.log_event(
        session_id=session_id,
        agent_id="agent-1",
        event_type="user",
        payload={"text": "event 5"}
    )

    # Confirm snapshot trigger and write were called
    assert mock_summarize.call_count == 1
    assert mock_write.call_count == 1
    
    args, kwargs = mock_write.call_args
    assert args[0] == session_id
    assert args[1] == {
        "summary": "This is a valid mock summary of the events.",
        "entities_mentioned": ["user", "system"],
        "event_range": [1, 5]
    }


@pytest.mark.anyio
@patch("cortexgit.core.memory.summarize")
async def test_failed_summarization_does_not_write_partial_snapshot(mock_summarize, monkeypatch):
    """If summarization raises ValidationError, it does not write any snapshot and does not crash."""
    monkeypatch.setenv("SNAPSHOT_THRESHOLD", "3")
    
    # Mock summarizer to raise ValidationError
    mock_summarize.side_effect = ValidationError("Invalid schema output from mock summarizer")
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    session_id = "test-session-fail"
    
    # Post 3 events to trigger the background task
    for i in range(3):
        await memory.log_event(
            session_id=session_id,
            agent_id="agent-1",
            event_type="user",
            payload={"text": f"event {i}"}
        )

    # Confirm that no snapshot was written in the DB
    async with TestingSessionLocal() as session:
        snaps = (await session.execute(select(SnapshotStore))).scalars().all()
        assert len(snaps) == 0
