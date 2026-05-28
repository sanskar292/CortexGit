"""
End-to-end integration test for Proactive Surface Injection.

Covers both SQLite and PostgreSQL.

Scenario:
1. Agent writes 20 events over time
2. Some entities become high-importance (many hits, many connections)
3. Query with a goal that doesn't mention high-importance entity
4. Verify high-importance entity is injected into context anyway
5. Verify it appears before low-importance entities
"""

import os
import uuid
import asyncio
import json
from unittest.mock import patch, MagicMock

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import cortexgit.db.database
from api.main import app
from cortexgit import CortexGit
from cortexgit.db.models import Base, EntityNode, EntityEdge, NodeHit, HitType
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import update_centrality


# Determine if PostgreSQL is available locally
def is_postgres_available() -> bool:
    try:
        conn = psycopg2.connect(
            "postgresql://postgres:password@localhost:5432/postgres",
            connect_timeout=3
        )
        conn.close()
        return True
    except Exception:
        return False


# SQLite database setups
SQLITE_DB_URL = "sqlite+aiosqlite:///cortexgit_injection_integration.db"
sqlite_engine = create_async_engine(SQLITE_DB_URL, echo=False)
SqliteSessionLocal = sessionmaker(sqlite_engine, class_=AsyncSession, expire_on_commit=False)


# PostgreSQL database setups
POSTGRES_DB_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_reg_injection_test"
postgres_engine = create_async_engine(POSTGRES_DB_URL, echo=False, poolclass=NullPool)
PostgresSessionLocal = sessionmaker(postgres_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    return "asyncio"


# Shared integration test scenario runner
async def run_injection_integration_scenario(engine, session_local, db_url):
    # Patch the module-level AsyncSessionLocal in cortexgit.db.database
    # to redirect the API background tasks to our test database
    original_session_local = cortexgit.db.database.AsyncSessionLocal
    cortexgit.db.database.AsyncSessionLocal = session_local

    try:
        # Create all tables dynamically
        async with engine.begin() as conn:
            if "postgresql" in db_url:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                except Exception:
                    pass
            await conn.run_sync(Base.metadata.create_all)

        client = TestClient(app)
        session_id = f"test-session-{uuid.uuid4().hex[:8]}"
        agent_id = "test-agent"

        # relational entity graph updates response maps
        reg_mock_responses = {
            "Introduce important_project": {
                "updates": [
                    {
                        "entity_name": "important_project",
                        "entity_type": "project",
                        "properties": {
                            "description": "Highly important project with many connections",
                            "status": "active"
                        },
                        "connected_to": [
                            { "target_entity": "anchor_a", "relation_type": "part" },
                            { "target_entity": "anchor_b", "relation_type": "part" },
                            { "target_entity": "anchor_c", "relation_type": "part" },
                            { "target_entity": "anchor_d", "relation_type": "part" }
                        ]
                    }
                ]
            },
            "Introduce unimportant_project": {
                "updates": [
                    {
                        "entity_name": "unimportant_project",
                        "entity_type": "project",
                        "properties": {
                            "description": "Low importance project with few connections",
                            "status": "active"
                        },
                        "connected_to": [
                            { "target_entity": "anchor_a", "relation_type": "part" }
                        ]
                    }
                ]
            }
        }

        async def mock_extract_entities(event_dict, provider=None):
            return {"updates": []}

        async def mock_extract_reg_entities(event_dict, provider=None):
            text_content = event_dict.get("payload", {}).get("text", "")
            for key, val in reg_mock_responses.items():
                if key in text_content:
                    return val
            return {"updates": []}

        async def mock_should_snapshot(session_id, session):
            return False

        # Apply mocks to prevent external network or LLM execution overhead
        with patch("cortexgit.llm.entity_extractor.extract_entities", side_effect=mock_extract_entities), \
             patch("cortexgit.llm.entity_extractor.extract_reg_entities", side_effect=mock_extract_reg_entities), \
             patch("cortexgit.llm.snapshot_trigger.should_snapshot", side_effect=mock_should_snapshot), \
             patch("cortexgit.retrieval.semantic_recall.embed_text", return_value=[0.1] * 1536):

            # 1. Agent writes 20 events over time
            events_payloads = [
                "Introduce important_project with its key parameters",
                "Introduce unimportant_project to complete task"
            ] + [f"Informational agent log event number {i}" for i in range(3, 21)]

            for i, text_content in enumerate(events_payloads, start=1):
                payload = {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "event_type": "agent",
                    "payload": {"text": text_content}
                }
                res = client.post("/events", json=payload)
                assert res.status_code == 201

            # Wait briefly for concurrent background task execution to persist graph items
            await asyncio.sleep(0.5)

            # Assert database has successfully created nodes with correct centrality
            async with session_local() as session:
                repo = GraphRepository(session)
                important_node = await repo.get_node("important_project", agent_id)
                unimportant_node = await repo.get_node("unimportant_project", agent_id)
                assert important_node is not None, "important_project node was not created"
                assert unimportant_node is not None, "unimportant_project node was not created"

                # Verify degree centrality (4.0 vs 1.0)
                assert important_node.degree_centrality >= 4.0
                assert unimportant_node.degree_centrality == 1.0

            # 2. Some entities become high-importance (many hits, many connections)
            # Elevate important_project importance by hitting it multiple times via queries
            # We query 6 times to record hits in background
            memory = CortexGit(database_url=db_url)
            for _ in range(6):
                await memory.get_context(
                    goal="Query targeting important_project details",
                    budget_tokens=9999,
                    session_id=session_id,
                    agent_id=agent_id
                )
                await asyncio.sleep(0.05)

            # Elevate unimportant_project slightly by hitting it only once
            await memory.get_context(
                goal="Query targeting unimportant_project details",
                budget_tokens=9999,
                session_id=session_id,
                agent_id=agent_id
            )
            await asyncio.sleep(0.1)

            # Let's verify hit frequency counts and calculated importance
            async with session_local() as session:
                repo = GraphRepository(session)
                important_node = await repo.get_node("important_project", agent_id)
                unimportant_node = await repo.get_node("unimportant_project", agent_id)

                assert important_node.hit_frequency >= 5
                assert unimportant_node.hit_frequency >= 1

                important_importance = float(important_node.degree_centrality) * float(important_node.hit_frequency)
                unimportant_importance = float(unimportant_node.degree_centrality) * float(unimportant_node.hit_frequency)

                assert important_importance > unimportant_importance

            # 3. Query with a goal that doesn't mention high-importance entity
            # Let's query with "unrelated_goal" which doesn't match important_project or unimportant_project
            ctx = await memory.get_context(
                goal="unrelated_goal",
                budget_tokens=9999,
                session_id=session_id,
                use_reg=True,
                agent_id=agent_id
            )

            # 4. Verify high-importance entity is injected into context anyway
            assert "important_project" in ctx["entities"], "important_project was not injected in context"
            injected_entities = ctx.get("metadata", {}).get("injected_entities", [])
            assert "important_project" in injected_entities, "important_project was not marked as injected in metadata"

            # 5. Verify it appears before low-importance entities
            # Dict keys preserve insertion order, representing descending packing priority
            entity_keys = list(ctx["entities"].keys())
            assert "important_project" in entity_keys

            if "unimportant_project" in entity_keys:
                important_idx = entity_keys.index("important_project")
                unimportant_idx = entity_keys.index("unimportant_project")
                assert important_idx < unimportant_idx, f"important_project ({important_idx}) should appear before unimportant_project ({unimportant_idx})"

    finally:
        # Restore module-level session factory
        cortexgit.db.database.AsyncSessionLocal = original_session_local

        # Drop tables cleanly
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_injection_integration_sqlite():
    """
    Run injection integration test on SQLite database.
    """
    # Clean up any existing SQLite database file
    for suffix in ["", "-shm", "-wal"]:
        path = f"cortexgit_injection_integration.db{suffix}"
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    try:
        await run_injection_integration_scenario(sqlite_engine, SqliteSessionLocal, SQLITE_DB_URL)
    finally:
        # Clean up database files
        await sqlite_engine.dispose()
        for suffix in ["", "-shm", "-wal"]:
            path = f"cortexgit_injection_integration.db{suffix}"
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.skipif(
    not is_postgres_available(),
    reason="Requires a running PostgreSQL database at localhost:5432"
)
async def test_injection_integration_postgres():
    """
    Run injection integration test on PostgreSQL database.
    """
    # Create the PostgreSQL database if it does not exist
    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS cortexgit_reg_injection_test;")
    cur.execute("CREATE DATABASE cortexgit_reg_injection_test;")
    cur.close()
    conn.close()

    try:
        await run_injection_integration_scenario(postgres_engine, PostgresSessionLocal, POSTGRES_DB_URL)
    finally:
        await postgres_engine.dispose()
        conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute("DROP DATABASE IF EXISTS cortexgit_reg_injection_test;")
        cur.close()
        conn.close()
