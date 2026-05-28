"""
Integration test for Relational Entity Graph (REG) multi-agent coordination.

Requirements:
  - PostgreSQL running at localhost:5432 (password: password)
  - pgvector extension installed
  - Real ANTHROPIC_API_KEY in environment
  - Real OPENAI_API_KEY in environment

Run with:
  python -m pytest tests/test_reg_multi_agent.py -m integration -s -v
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Override providers dynamically to use real Anthropic LLM and OpenAI Embeddings as requested
os.environ["CORTEXGIT_LLM_PROVIDER"] = "anthropic"
os.environ["CORTEXGIT_EMBEDDING_PROVIDER"] = "openai"

import asyncio
import uuid
from datetime import datetime, timezone, timedelta

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from fastapi.testclient import TestClient

import cortexgit.db.database
from api.main import app
from cortexgit import CortexGit
from cortexgit.core.context_assembler import assemble
from cortexgit.db.models import Base, EntityNode, EntityEdge, NodeHit, HitType
from cortexgit.graph.graph_repository import GraphRepository

# ──────────────────────────────────────────────────────────────────────────────
# Test database setup
# ──────────────────────────────────────────────────────────────────────────────

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_reg_multi_test"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestingSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

# Redirect the module-level AsyncSessionLocal so background tasks use the test DB
cortexgit.db.database.AsyncSessionLocal = TestingSessionLocal


@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
def create_test_db():
    """Create the e2e test database; tear it down after the module."""
    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS cortexgit_reg_multi_test;")
    cur.execute("CREATE DATABASE cortexgit_reg_multi_test;")
    cur.close()
    conn.close()

    yield

    asyncio.run(test_engine.dispose())

    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS cortexgit_reg_multi_test;")
    cur.close()
    conn.close()


@pytest.fixture(scope="module", autouse=True)
async def setup_db(create_test_db):
    """Enable pgvector and create all tables once for the module."""
    async with test_engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception:
            pass  # pgvector may already exist or not be available; fallback handles it

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Helper to fuzzy find a node in the database
async def find_node_by_fuzzy_name(session: AsyncSession, name: str) -> EntityNode:
    stmt = select(EntityNode)
    res = await session.execute(stmt)
    nodes = res.scalars().all()
    clean_target = name.lower().replace("_", "").replace("-", "").replace(".", "")
    for node in nodes:
        clean_node = node.entity_name.lower().replace("_", "").replace("-", "").replace(".", "")
        if clean_target in clean_node or clean_node in clean_target:
            return node
    return None


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY environment variable"
)
async def test_reg_multi_agent_isolation_and_shared_state():
    """
    Simulates a multi-agent Relational Entity Graph (REG) session with real LLM calls:
    1. Create two agents: agent_alpha and agent_beta.
    2. agent_alpha writes events mentioning ProjectX and Person1.
    3. agent_beta writes events mentioning ProjectY and Person1.
    4. Verify both projects have high degree centrality and edges linking them to the shared Person1.
    5. Verify query routing and importance scores are correctly isolated to each agent's active projects.
    """
    client = TestClient(app)
    session_id_alpha = f"multi-session-alpha-{uuid.uuid4().hex[:8]}"
    session_id_beta = f"multi-session-beta-{uuid.uuid4().hex[:8]}"

    # ──────────────────────────────────────────────────────────────────────
    # 1. Agent Alpha logs events mentioning ProjectX and Person1
    # ──────────────────────────────────────────────────────────────────────
    events_alpha = [
        "Starting a new corporate sprint for ProjectX. Lead developer Person1 is assigned to this project.",
        "Person1 has finalized the database schema designs and backend repositories for ProjectX.",
        "A full performance review of ProjectX's core gateway was completed by Person1.",
        "Person1 successfully configured the production deployment environment for ProjectX."
    ]

    print("\n[Multi-Agent] Posting Agent Alpha events...")
    for text_content in events_alpha:
        payload = {
            "session_id": session_id_alpha,
            "agent_id": "agent_alpha",
            "event_type": "agent",
            "payload": {"text": text_content}
        }
        res = client.post("/events", json=payload)
        assert res.status_code == 201

    # ──────────────────────────────────────────────────────────────────────
    # 2. Agent Beta logs events mentioning ProjectY and Person1
    # ──────────────────────────────────────────────────────────────────────
    events_beta = [
        "Initializing the development cycle for ProjectY. Developer Person1 has been assigned here as well.",
        "Person1 finished setting up the frontend components and repository structure for ProjectY.",
        "Person1 resolved a major UI synchronization defect on ProjectY.",
        "ProjectY has raised its automated test suite coverage to 90% under Person1's supervision."
    ]

    print("[Multi-Agent] Posting Agent Beta events...")
    for text_content in events_beta:
        payload = {
            "session_id": session_id_beta,
            "agent_id": "agent_beta",
            "event_type": "agent",
            "payload": {"text": text_content}
        }
        res = client.post("/events", json=payload)
        assert res.status_code == 201

    # Wait briefly to let extraction pipelines completely finish writing
    await asyncio.sleep(1.0)

    # ──────────────────────────────────────────────────────────────────────
    # 3. Verify Graph Topology, Centrality, and Shared State (Person1)
    # ──────────────────────────────────────────────────────────────────────
    async with TestingSessionLocal() as session:
        node_project_x = await find_node_by_fuzzy_name(session, "ProjectX")
        node_project_y = await find_node_by_fuzzy_name(session, "ProjectY")
        node_person_1 = await find_node_by_fuzzy_name(session, "Person1")

        assert node_project_x is not None, "ProjectX node was not created!"
        assert node_project_y is not None, "ProjectY node was not created!"
        assert node_person_1 is not None, "Person1 node was not created!"

        print(f"[Multi-Agent] Verified nodes exist in database:")
        print(f"  - ProjectX node: name='{node_project_x.entity_name}' (agent={node_project_x.agent_id})")
        print(f"  - ProjectY node: name='{node_project_y.entity_name}' (agent={node_project_y.agent_id})")
        print(f"  - Person1 node: name='{node_person_1.entity_name}' (agent={node_person_1.agent_id})")

        # Centrality verification: Both projects are highly connected to Person1
        assert node_project_x.degree_centrality >= 1.0, f"ProjectX degree centrality is too low: {node_project_x.degree_centrality}"
        assert node_project_y.degree_centrality >= 1.0, f"ProjectY degree centrality is too low: {node_project_y.degree_centrality}"

        # Verify edges: Person1 has edges to both ProjectX and ProjectY
        edges_res = await session.execute(select(EntityEdge))
        edges = edges_res.scalars().all()
        edge_pairs = {(e.source_node_id, e.target_node_id) for e in edges}

        has_x_edge = (
            (node_project_x.node_id, node_person_1.node_id) in edge_pairs or
            (node_person_1.node_id, node_project_x.node_id) in edge_pairs
        )
        has_y_edge = (
            (node_project_y.node_id, node_person_1.node_id) in edge_pairs or
            (node_person_1.node_id, node_project_y.node_id) in edge_pairs
        )

        assert has_x_edge, "No edge created between ProjectX and Person1!"
        assert has_y_edge, "No edge created between ProjectY and Person1!"
        print("[Multi-Agent] Edges successfully verified! Shared state confirmed for Person1.")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Context Query and Isolation Verification
    # ──────────────────────────────────────────────────────────────────────
    memory = CortexGit(database_url=TEST_DATABASE_URL)

    # Agent Alpha Queries context for "ProjectX"
    print("\n[Multi-Agent] Querying context for Agent Alpha...")
    async with TestingSessionLocal() as session:
        ctx_alpha = await assemble(
            goal="Tell me about ProjectX and developer status",
            session_id=session_id_alpha,
            budget_tokens=8000,
            session=session,
            embedding_provider=memory.embedding_provider,
            use_reg=True,
            agent_id="agent_alpha"
        )
    
    # Wait for non-blocking reinforcement hits
    await asyncio.sleep(0.5)

    async with TestingSessionLocal() as session:
        node_project_x = await find_node_by_fuzzy_name(session, "ProjectX")
        assert node_project_x.hit_frequency >= 1, "Hit frequency on ProjectX did not increment!"
        
        entities_keys_alpha = list(ctx_alpha["entities"].keys())
        print(f"  - Agent Alpha Context Entities: {entities_keys_alpha}")
        assert any("projectx" in k.lower() or "project_x" in k.lower() for k in entities_keys_alpha), (
            "ProjectX did not appear in Agent Alpha's entities context!"
        )
        # Isolation assertion: agent_alpha should not pull ProjectY since ProjectY belongs to agent_beta
        assert not any("projecty" in k.lower() or "project_y" in k.lower() for k in entities_keys_alpha), (
            "ProjectY incorrectly appeared in Agent Alpha's context!"
        )

    # Agent Beta Queries context for "ProjectY"
    print("[Multi-Agent] Querying context for Agent Beta...")
    async with TestingSessionLocal() as session:
        ctx_beta = await assemble(
            goal="Tell me about ProjectY and developer status",
            session_id=session_id_beta,
            budget_tokens=8000,
            session=session,
            embedding_provider=memory.embedding_provider,
            use_reg=True,
            agent_id="agent_beta"
        )

    # Wait for reinforcement hits
    await asyncio.sleep(0.5)

    async with TestingSessionLocal() as session:
        node_project_y = await find_node_by_fuzzy_name(session, "ProjectY")
        assert node_project_y.hit_frequency >= 1, "Hit frequency on ProjectY did not increment!"
        
        entities_keys_beta = list(ctx_beta["entities"].keys())
        print(f"  - Agent Beta Context Entities: {entities_keys_beta}")
        assert any("projecty" in k.lower() or "project_y" in k.lower() for k in entities_keys_beta), (
            "ProjectY did not appear in Agent Beta's entities context!"
        )
        # Isolation assertion: agent_beta should not pull ProjectX since ProjectX belongs to agent_alpha
        assert not any("projectx" in k.lower() or "project_x" in k.lower() for k in entities_keys_beta), (
            "ProjectX incorrectly appeared in Agent Beta's context!"
        )

    print("\n" + "=" * 60)
    print("REG MULTI-AGENT ISOLATION & SHARED STATE TEST PASSED SUCCESSFULLY")
    print("=" * 60 + "\n")
