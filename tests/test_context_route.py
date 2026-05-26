import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import patch
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit import CortexGit
from cortexgit.db.models import Base, EventLog, ConflictLog, EventType

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_test"

# Dedicated engine and session for running context SDK tests
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


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_valid_request_returns_all_keys(mock_embed):
    """Valid request returns the four expected keys in context."""
    mock_embed.return_value = [0.1] * 1536
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    # Write an event
    async with TestingSessionLocal() as db_session:
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id="session-1",
            agent_id="agent-1",
            event_type=EventType.USER,
            payload={"msg": "hello"},
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(event)
        await db_session.commit()

    data = await memory.get_context(
        goal="build backend",
        budget_tokens=1000,
        session_id="session-1"
    )
    
    assert "events" in data
    assert "snapshots" in data
    assert "entities" in data
    assert "conflicts" in data
    assert len(data["events"]) == 1
    assert data["events"][0]["payload"] == {"msg": "hello"}


@pytest.mark.anyio
async def test_invalid_inputs_raise_value_error():
    """Missing goal, budget_tokens <= 0, empty or whitespace strings raise ValueError."""
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    # 1. Missing goal (empty string)
    with pytest.raises(ValueError):
        await memory.get_context(goal="", budget_tokens=1000, session_id="session-1")

    # 2. Whitespace-only goal
    with pytest.raises(ValueError):
        await memory.get_context(goal="   ", budget_tokens=1000, session_id="session-1")

    # 3. budget_tokens of 0
    with pytest.raises(ValueError):
        await memory.get_context(goal="build", budget_tokens=0, session_id="session-1")

    # 4. Negative budget_tokens
    with pytest.raises(ValueError):
        await memory.get_context(goal="build", budget_tokens=-10, session_id="session-1")

    # 5. Missing session_id (empty string)
    with pytest.raises(ValueError):
        await memory.get_context(goal="build", budget_tokens=1000, session_id="")

    # 6. Whitespace session_id
    with pytest.raises(ValueError):
        await memory.get_context(goal="build", budget_tokens=1000, session_id="  ")


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_empty_stores_return_empty_lists(mock_embed):
    """Clean stores return empty lists/dicts, not errors."""
    mock_embed.return_value = [0.1] * 1536
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    data = await memory.get_context(
        goal="build backend",
        budget_tokens=1000,
        session_id="empty-session"
    )
    
    assert data == {
        "events": [],
        "snapshots": [],
        "entities": {},
        "conflicts": []
    }
