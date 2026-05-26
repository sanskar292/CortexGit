import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit import CortexGit
from cortexgit.db.models import Base, EventLog, EntityRegistry, ConflictLog
from cortexgit.core.write_back_gate import ValidationError

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_test"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestingSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

import cortexgit.db.database
cortexgit.db.database.AsyncSessionLocal = TestingSessionLocal

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
@patch("cortexgit.core.memory.extract_entities")
async def test_valid_extraction_writes_entities(mock_extract):
    """Valid extraction outputs are successfully parsed and written to EntityRegistry."""
    mock_extract.return_value = {
        "updates": [
            {"key": "project.status", "value": "active"},
            {"key": "agent.current_goal", "value": "debugging tests"}
        ]
    }
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    session_id = "test-session-extraction"
    
    # We log the event, which internally runs the background pipeline
    event = await memory.log_event(
        session_id=session_id,
        agent_id="agent-1",
        event_type="user",
        payload={"text": "let's deploy project cortexgit"}
    )
        
    async with TestingSessionLocal() as session:
        entities = (await session.execute(select(EntityRegistry))).scalars().all()
        assert len(entities) == 2
        
        entity_map = {e.key: e.value for e in entities}
        assert entity_map["project.status"] == "active"
        assert entity_map["agent.current_goal"] == "debugging tests"


@pytest.mark.anyio
@patch("cortexgit.core.memory.extract_entities")
async def test_failed_extraction_does_not_write_partial_entities(mock_extract):
    """If extraction raises a ValidationError, no updates are processed and request does not crash."""
    mock_extract.side_effect = ValidationError("Gate validation failed")
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    session_id = "test-session-failed"
    
    event = await memory.log_event(
        session_id=session_id,
        agent_id="agent-1",
        event_type="thought",
        payload={"text": "thinking about errors"}
    )
        
    async with TestingSessionLocal() as session:
        entities = (await session.execute(select(EntityRegistry))).scalars().all()
        assert len(entities) == 0


@pytest.mark.anyio
@patch("cortexgit.llm.entity_extractor.AsyncAnthropic")
async def test_key_with_invalid_pattern_rejected_by_gate(mock_anthropic_class):
    """LLM output with keys that don't match the lowercase alphanumeric + dot/underscore pattern are rejected by gate."""
    from cortexgit.llm.entity_extractor import extract_entities
    
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='{"updates": [{"key": "Project.Status", "value": "active"}]}')
    ]
    mock_client.messages.create.return_value = mock_message
    
    with pytest.raises(ValidationError) as exc_info:
        await extract_entities({"payload": {"text": "hello"}})
        
    assert "pattern" in str(exc_info.value).lower() or "validation failed" in str(exc_info.value).lower()


@pytest.mark.anyio
@patch("cortexgit.llm.entity_extractor.AsyncAnthropic")
async def test_key_with_hyphen_rejected_by_gate(mock_anthropic_class):
    """LLM output with keys containing hyphens are rejected by gate."""
    from cortexgit.llm.entity_extractor import extract_entities
    
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='{"updates": [{"key": "project-status", "value": "active"}]}')
    ]
    mock_client.messages.create.return_value = mock_message
    
    with pytest.raises(ValidationError) as exc_info:
        await extract_entities({"payload": {"text": "hello"}})
        
    assert "pattern" in str(exc_info.value).lower() or "validation failed" in str(exc_info.value).lower()


@pytest.mark.anyio
@patch("cortexgit.core.memory.extract_entities")
async def test_empty_updates_writes_nothing(mock_extract):
    """If extraction updates array is empty, it succeeds cleanly and writes nothing to registry."""
    mock_extract.return_value = {
        "updates": []
    }
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    session_id = "test-session-empty"
    
    event = await memory.log_event(
        session_id=session_id,
        agent_id="agent-1",
        event_type="user",
        payload={"text": "nothing to extract here"}
    )
        
    async with TestingSessionLocal() as session:
        entities = (await session.execute(select(EntityRegistry))).scalars().all()
        assert len(entities) == 0


@pytest.mark.anyio
@patch("cortexgit.core.memory.extract_entities")
async def test_extraction_conflict_logged(mock_extract):
    """If extraction writes a key that already exists with a different value, conflict is logged and EntityRegistry is not overwritten."""
    async with TestingSessionLocal() as session:
        evt = EventLog(
            event_id=uuid.uuid4(),
            session_id="test-session-conflict",
            agent_id="agent-1",
            event_type="user",
            payload={"text": "initial event"}
        )
        session.add(evt)
        await session.commit()
        
        entity = EntityRegistry(
            key="project.status",
            value="active",
            agent_id="agent-1",
            event_id=evt.event_id
        )
        session.add(entity)
        await session.commit()

    mock_extract.return_value = {
        "updates": [
            {"key": "project.status", "value": "completed"}
        ]
    }
    
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    session_id = "test-session-conflict"
    
    event = await memory.log_event(
        session_id=session_id,
        agent_id="agent-1",
        event_type="user",
        payload={"text": "project has finished"}
    )
        
    async with TestingSessionLocal() as session:
        conflicts = (await session.execute(select(ConflictLog))).scalars().all()
        assert len(conflicts) == 1
        assert conflicts[0].key == "project.status"
        assert conflicts[0].existing_value == "active"
        assert conflicts[0].proposed_value == "completed"
        
        entities = (await session.execute(select(EntityRegistry))).scalars().all()
        assert len(entities) == 1
        assert entities[0].value == "active"
