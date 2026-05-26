import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import patch
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from cortexgit.db.models import Base, SnapshotStore, HAS_PGVECTOR
from cortexgit.retrieval.semantic_recall import semantic_recall

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
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_returns_empty_list_when_no_snapshots(mock_embed, db_session):
    """Returns empty list when no snapshots exist yet."""
    mock_embed.return_value = [0.1] * 1536
    
    result = await semantic_recall("goal string", db_session, top_n=5)
    assert result == []


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_returns_top_n_snapshots(mock_embed, db_session):
    """Returns top_n snapshots when more exist than top_n."""
    mock_embed.return_value = [0.1] * 1536
    
    # Write 5 snapshots
    for i in range(5):
        event_range_val = "1,10" if "sqlite" in TEST_DATABASE_URL else text("int4range(1, 10)")
        snapshot = SnapshotStore(
            snapshot_id=uuid.uuid4(),
            event_range=event_range_val,
            summary=f"snapshot {i}",
            entities_mentioned=["entity"],
            embedding=[0.1] * 1536,
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(snapshot)
    await db_session.commit()

    result = await semantic_recall("goal string", db_session, top_n=3)
    assert len(result) == 3


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_results_ordered_by_similarity(mock_embed, db_session):
    """Results are strictly ordered by similarity (most similar first)."""
    # Query vector matches Snapshot A perfectly
    mock_embed.return_value = [1.0, 0.0] + [0.0] * 1534
    
    # Snapshot A: perfect match
    event_range_a = "1,10" if "sqlite" in TEST_DATABASE_URL else text("int4range(1, 10)")
    snap_a = SnapshotStore(
        snapshot_id=uuid.uuid4(),
        event_range=event_range_a,
        summary="first",
        entities_mentioned=[],
        embedding=[1.0, 0.0] + [0.0] * 1534,
        created_at=datetime.now(timezone.utc)
    )
    
    # Snapshot B: completely orthogonal (lowest similarity)
    event_range_b = "11,20" if "sqlite" in TEST_DATABASE_URL else text("int4range(11, 20)")
    snap_b = SnapshotStore(
        snapshot_id=uuid.uuid4(),
        event_range=event_range_b,
        summary="second",
        entities_mentioned=[],
        embedding=[0.0, 1.0] + [0.0] * 1534,
        created_at=datetime.now(timezone.utc)
    )
    
    # Snapshot C: partial match (middle similarity)
    event_range_c = "21,30" if "sqlite" in TEST_DATABASE_URL else text("int4range(21, 30)")
    snap_c = SnapshotStore(
        snapshot_id=uuid.uuid4(),
        event_range=event_range_c,
        summary="third",
        entities_mentioned=[],
        embedding=[0.707, 0.707] + [0.0] * 1534,
        created_at=datetime.now(timezone.utc)
    )

    db_session.add_all([snap_b, snap_c, snap_a])
    await db_session.commit()

    result = await semantic_recall("goal", db_session, top_n=3)
    
    assert len(result) == 3
    # Check strict descending similarity order: A ("first"), C ("third"), B ("second")
    assert result[0].summary == "first"
    assert result[1].summary == "third"
    assert result[2].summary == "second"
