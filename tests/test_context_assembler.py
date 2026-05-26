import pytest
import asyncio
import uuid
import json
from datetime import datetime, timezone
from unittest.mock import patch
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit.db.models import Base, EventLog, ConflictLog, SnapshotStore, EntityRegistry, EventType
from cortexgit.core.context_assembler import assemble

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_test"

# Dedicated engine and session for running context assembler tests
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestingSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    """Ensure pytest-asyncio runs tests correctly."""
    return "asyncio"

@pytest.fixture(scope="module", autouse=True)
def create_test_db():
    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("DROP DATABASE IF EXISTS cortexgit_test;")
    cursor.execute("CREATE DATABASE cortexgit_test;")
    cursor.close()
    conn.close()

    yield

    asyncio.run(test_engine.dispose())

    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("DROP DATABASE IF EXISTS cortexgit_test;")
    cursor.close()
    conn.close()

@pytest.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
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
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_returns_all_four_keys_in_output(mock_embed, db_session):
    """Returns all four keys (events, snapshots, entities, conflicts) in output."""
    mock_embed.return_value = [0.1] * 1536
    
    result = await assemble("goal", "session-1", 1000, db_session)
    assert isinstance(result, dict)
    assert "events" in result
    assert "snapshots" in result
    assert "entities" in result
    assert "conflicts" in result


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_returns_empty_lists_when_stores_are_empty(mock_embed, db_session):
    """Returns empty lists/dict when stores are empty."""
    mock_embed.return_value = [0.1] * 1536
    
    result = await assemble("goal", "session-1", 1000, db_session)
    assert result == {
        "events": [],
        "snapshots": [],
        "entities": {},
        "conflicts": []
    }


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_budget_of_zero_returns_empty_result(mock_embed, db_session):
    """A budget of zero returns empty lists/dict regardless of data in stores."""
    mock_embed.return_value = [0.1] * 1536
    
    # Insert an event first
    event_id = uuid.uuid4()
    e = EventLog(
        event_id=event_id,
        session_id="session-1",
        agent_id="agent-1",
        event_type=EventType.USER,
        payload={"msg": "hello"},
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(e)
    await db_session.commit()

    result = await assemble("goal", "session-1", 0, db_session)
    assert result == {
        "events": [],
        "snapshots": [],
        "entities": {},
        "conflicts": []
    }


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_never_exceeds_budget_tokens(mock_embed, db_session):
    """Assembled output never exceeds budget_tokens limit."""
    mock_embed.return_value = [0.1] * 1536
    
    # 1. Insert 5 events
    event_ids = []
    for i in range(5):
        event_id = uuid.uuid4()
        event_ids.append(event_id)
        e = EventLog(
            event_id=event_id,
            session_id="session-1",
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": f"event {i}"},
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(e)
    await db_session.commit()

    # 2. Insert 2 conflicts
    for i in range(2):
        c = ConflictLog(
            conflict_id=uuid.uuid4(),
            key=f"conflict.key.{i}",
            existing_value="val1",
            proposed_value="val2",
            existing_event_id=event_ids[0],
            proposed_event_id=event_ids[1],
            resolved=False,
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(c)
    await db_session.commit()

    # 3. Request assemble with a very tight budget (e.g. 50 tokens)
    budget = 50
    result = await assemble("goal", "session-1", budget, db_session)

    # 4. Calculate total tokens inside the result
    total_tokens = 0
    for item in result["conflicts"]:
        total_tokens += len(json.dumps(item)) // 4
    for item in result["events"]:
        total_tokens += len(json.dumps(item)) // 4
    for item in result["snapshots"]:
        total_tokens += len(json.dumps(item)) // 4
    for key, val in result["entities"].items():
        total_tokens += len(json.dumps({key: val})) // 4

    assert total_tokens <= budget


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_conflicts_appear_before_events(mock_embed, db_session):
    """Unresolved conflicts appear before events in packing priority."""
    mock_embed.return_value = [0.1] * 1536
    
    # Insert 1 event
    event_id = uuid.uuid4()
    e = EventLog(
        event_id=event_id,
        session_id="session-1",
        agent_id="agent-1",
        event_type=EventType.USER,
        payload={"msg": "important event"},
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(e)
    await db_session.commit()

    # Insert 1 conflict
    conflict_id = uuid.uuid4()
    c = ConflictLog(
        conflict_id=conflict_id,
        key="project.goal",
        existing_value="value1",
        proposed_value="value2",
        existing_event_id=event_id,
        proposed_event_id=event_id,
        resolved=False,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(c)
    await db_session.commit()

    # Calculate token size of conflict dict
    c_dict = {
        "conflict_id": str(c.conflict_id),
        "key": c.key,
        "existing_value": c.existing_value,
        "proposed_value": c.proposed_value,
        "existing_event_id": str(c.existing_event_id),
        "proposed_event_id": str(c.proposed_event_id),
        "resolved": c.resolved,
        "created_at": c.created_at.isoformat() if c.created_at else None
    }
    conflict_tokens = len(json.dumps(c_dict)) // 4

    # Set budget exactly equal to conflict token cost.
    # It should only have enough room to pack the conflict, skipping the event!
    result = await assemble("goal", "session-1", conflict_tokens, db_session)
    
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["key"] == "project.goal"
    assert len(result["events"]) == 0
