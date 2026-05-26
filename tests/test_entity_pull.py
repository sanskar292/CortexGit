import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit.db.models import Base, EventLog, EntityRegistry, EventType
from cortexgit.core.entity_pull import entity_pull

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_test"

# Dedicated engine and session for running entity pull tests
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


async def create_helper_event(session: AsyncSession) -> uuid.UUID:
    """Helper to insert a valid event log entry to fulfill FK constraints."""
    event_id = uuid.uuid4()
    event = EventLog(
        event_id=event_id,
        session_id="session-test",
        agent_id="agent-test",
        event_type=EventType.SYSTEM,
        payload={"msg": "test event context"},
        created_at=datetime.now(timezone.utc)
    )
    session.add(event)
    await session.commit()
    return event_id


@pytest.mark.anyio
async def test_returns_matching_entities_for_goal(db_session):
    """Returns matching entities for a goal string."""
    event_id = await create_helper_event(db_session)
    
    # Insert two entities
    e1 = EntityRegistry(
        key="project.goal",
        value={"goal": "build backend"},
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    e2 = EntityRegistry(
        key="agent.status",
        value="active",
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add_all([e1, e2])
    await db_session.commit()

    # Query with a goal that contains "project" token
    result = await entity_pull("project task", db_session)
    assert result == {"project.goal": {"goal": "build backend"}}


@pytest.mark.anyio
async def test_returns_empty_dict_on_no_match(db_session):
    """Returns empty dict when no entities match."""
    event_id = await create_helper_event(db_session)
    
    e1 = EntityRegistry(
        key="agent.status",
        value="active",
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(e1)
    await db_session.commit()

    result = await entity_pull("unknown goal", db_session)
    assert result == {}


@pytest.mark.anyio
async def test_matching_is_case_insensitive(db_session):
    """Simple substring match is case-insensitive."""
    event_id = await create_helper_event(db_session)
    
    e1 = EntityRegistry(
        key="PROJECT.goal",
        value="build memory",
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(e1)
    await db_session.commit()

    # Query with a lowercase goal token
    result = await entity_pull("project", db_session)
    assert result == {"PROJECT.goal": "build memory"}


@pytest.mark.anyio
async def test_returns_multiple_matches(db_session):
    """Returns multiple matches when multiple keys match any goal token."""
    event_id = await create_helper_event(db_session)
    
    e1 = EntityRegistry(
        key="agent_alpha.task",
        value="initialize",
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    e2 = EntityRegistry(
        key="agent_beta.task",
        value="verify",
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    e3 = EntityRegistry(
        key="project.goal",
        value="build core",
        agent_id="agent-test",
        event_id=event_id,
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add_all([e1, e2, e3])
    await db_session.commit()

    # Query with a goal that contains "agent" token (matches key 1 and 2)
    result = await entity_pull("agent status", db_session)
    assert len(result) == 2
    assert "agent_alpha.task" in result
    assert "agent_beta.task" in result
    assert result["agent_alpha.task"] == "initialize"
    assert result["agent_beta.task"] == "verify"
