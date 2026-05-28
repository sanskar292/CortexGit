"""
End-to-end integration test for the Relational Entity Graph (REG) pipeline.

Requirements:
  - PostgreSQL running at localhost:5432 (password: password)
  - pgvector extension installed
  - Real ANTHROPIC_API_KEY in environment
  - Real OPENAI_API_KEY in environment

Run with:
  python -m pytest tests/test_reg_end_to_end.py -m integration -s -v
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
from cortexgit.graph.expiration import expire_old_nodes

# ──────────────────────────────────────────────────────────────────────────────
# Test database setup
# ──────────────────────────────────────────────────────────────────────────────

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_reg_e2e_test"

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
    cur.execute("DROP DATABASE IF EXISTS cortexgit_reg_e2e_test;")
    cur.execute("CREATE DATABASE cortexgit_reg_e2e_test;")
    cur.close()
    conn.close()

    yield

    asyncio.run(test_engine.dispose())

    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS cortexgit_reg_e2e_test;")
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
async def test_reg_end_to_end_pipeline():
    """
    Simulates a realistic agent session using REG with real LLM calls:
    1. Writes 20 events mentioning ProjectA, ProjectB, Person1, Person2.
    2. Lets background tasks extract entities, build graph nodes and edges.
    3. Queries context for 'Tell me about ProjectA' and verifies correct graph topology.
    4. Asserts hit frequency sequential updates.
    5. Back-dates nodes and calls delete_expired_nodes to verify LRU eviction.
    """
    client = TestClient(app)
    session_id = f"reg-e2e-session-{uuid.uuid4().hex[:8]}"
    agent_id = "cortex-planner-v1"

    # 1. Coherent series of 20 events focusing heavily on ProjectA and lightly on ProjectB
    events_payload = [
        "Starting a new sprint for ProjectA. Lead developer Person1 and lead manager Person2 are assigned to the project.",
        "Person1 completed setting up the GitHub repository and CI/CD pipelines for ProjectA.",
        "Person2 finalized the user experience guidelines and design mockups for ProjectA.",
        "A database architecture layout for ProjectA was drafted and approved by Person1.",
        "We identified a potential performance bottleneck in ProjectA's payment processing. Person1 is investigating.",
        "Let's also spin up a secondary small project called ProjectB. Person1 will write a quick prototype for it.",
        "Person2 hosted the first weekly design review session for ProjectA today.",
        "ProjectB is now on standby as we prioritize ProjectA sprint goals.",
        "Person1 finished writing unit tests for ProjectA's core services, raising coverage to 85%.",
        "Person2 coordinated with external stakeholders to align on ProjectA's integration milestones.",
        "Person1 successfully resolved the connection pooling bug in ProjectA's database adapter.",
        "UI components for ProjectA's payment dashboard were completed and pushed by Person2.",
        "Person1 deployed the beta staging environment for ProjectA.",
        "We are seeing highly positive internal feedback on ProjectA's design elements built by Person2.",
        "Person1 is preparing the production migration script for ProjectA's databases.",
        "ProjectB's prototype was successfully archived. No further updates are planned for ProjectB.",
        "Person2 conducted a full accessibility audit on ProjectA's user dashboard.",
        "Automated stress testing on ProjectA was completed successfully under Person1's supervision.",
        "Person2 draft-updated the release notes and user guide for ProjectA.",
        "Ready to deploy. ProjectA is fully operational, led by developer Person1 and designer Person2."
    ]

    print("\n[E2E] Posting 20 realistic events to the FastAPI server...")
    for i, text_content in enumerate(events_payload, start=1):
        payload = {
            "session_id": session_id,
            "agent_id": agent_id,
            "event_type": "agent" if i % 2 == 0 else "user",
            "payload": {"text": text_content}
        }
        res = client.post("/events", json=payload)
        assert res.status_code == 201, f"Failed to post event {i}: {res.text}"

    # Wait briefly to ensure any extra background handlers or tasks finish completely
    await asyncio.sleep(1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Verify nodes & edges were created correctly
    # ──────────────────────────────────────────────────────────────────────
    async with TestingSessionLocal() as session:
        node_project_a = await find_node_by_fuzzy_name(session, "ProjectA")
        node_project_b = await find_node_by_fuzzy_name(session, "ProjectB")
        node_person_1 = await find_node_by_fuzzy_name(session, "Person1")
        node_person_2 = await find_node_by_fuzzy_name(session, "Person2")

        assert node_project_a is not None, "ProjectA node was not created!"
        assert node_project_b is not None, "ProjectB node was not created!"
        assert node_person_1 is not None, "Person1 node was not created!"
        assert node_person_2 is not None, "Person2 node was not created!"

        print(f"[E2E] Verified all nodes exist in database:")
        print(f"  - ProjectA node: name='{node_project_a.entity_name}'")
        print(f"  - ProjectB node: name='{node_project_b.entity_name}'")
        print(f"  - Person1 node: name='{node_person_1.entity_name}'")
        print(f"  - Person2 node: name='{node_person_2.entity_name}'")

        # Confirm degree centrality for ProjectA > ProjectB
        assert node_project_a.degree_centrality > node_project_b.degree_centrality, (
            f"Expected ProjectA centrality ({node_project_a.degree_centrality}) "
            f"> ProjectB centrality ({node_project_b.degree_centrality})"
        )
        print(f"[E2E] Centrality verified: ProjectA ({node_project_a.degree_centrality}) > ProjectB ({node_project_b.degree_centrality})")

        # Confirm edges (ProjectA ↔ Person1, ProjectA ↔ Person2)
        edges_res = await session.execute(select(EntityEdge))
        edges = edges_res.scalars().all()
        edge_pairs = {(e.source_node_id, e.target_node_id) for e in edges}

        has_p1_edge = (
            (node_project_a.node_id, node_person_1.node_id) in edge_pairs or
            (node_person_1.node_id, node_project_a.node_id) in edge_pairs
        )
        has_p2_edge = (
            (node_project_a.node_id, node_person_2.node_id) in edge_pairs or
            (node_person_2.node_id, node_project_a.node_id) in edge_pairs
        )

        assert has_p1_edge, "Edge between ProjectA and Person1 was not created!"
        assert has_p2_edge, "Edge between ProjectA and Person2 was not created!"
        print("[E2E] Edges successfully verified!")

    # ──────────────────────────────────────────────────────────────────────
    # Query context for 'Tell me about ProjectA'
    # ──────────────────────────────────────────────────────────────────────
    memory = CortexGit(database_url=TEST_DATABASE_URL)
    
    # Track initial hit frequency of ProjectA node
    async with TestingSessionLocal() as session:
        node_project_a = await find_node_by_fuzzy_name(session, "ProjectA")
        initial_hits = node_project_a.hit_frequency

    # Assemble context for ProjectA
    async with TestingSessionLocal() as session:
        ctx_a = await assemble(
            goal="Tell me about ProjectA",
            session_id=session_id,
            budget_tokens=8000,
            session=session,
            embedding_provider=memory.embedding_provider,
            use_reg=True,
            agent_id=agent_id
        )

    # Let non-blocking record_hit_in_background tasks finish
    await asyncio.sleep(0.5)

    async with TestingSessionLocal() as session:
        node_project_a = await find_node_by_fuzzy_name(session, "ProjectA")
        hits_after_query1 = node_project_a.hit_frequency
        assert hits_after_query1 > initial_hits, "Hit frequency did not increase after query!"
        print(f"[E2E] Hit frequency increased correctly: {initial_hits} -> {hits_after_query1}")

        # Verify that ProjectA entities appear first or are prioritized
        entities_keys = list(ctx_a["entities"].keys())
        print(f"[E2E] Entities in ProjectA query context: {entities_keys}")
        assert any("projecta" in k.lower() or "project_a" in k.lower() for k in entities_keys), (
            "ProjectA did not appear in context entities!"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Query context for 'Person1'
    # ──────────────────────────────────────────────────────────────────────
    async with TestingSessionLocal() as session:
        ctx_p1 = await assemble(
            goal="Person1",
            session_id=session_id,
            budget_tokens=8000,
            session=session,
            embedding_provider=memory.embedding_provider,
            use_reg=True,
            agent_id=agent_id
        )

    # Let reinforcement tasks finish
    await asyncio.sleep(0.5)

    async with TestingSessionLocal() as session:
        node_person_1 = await find_node_by_fuzzy_name(session, "Person1")
        node_project_a = await find_node_by_fuzzy_name(session, "ProjectA")
        
        importance_p1 = float(node_person_1.degree_centrality) * float(node_person_1.hit_frequency)
        importance_pa = float(node_project_a.degree_centrality) * float(node_project_a.hit_frequency)

        assert importance_p1 <= importance_pa, (
            f"Expected Person1 importance ({importance_p1}) "
            f"to be less than or equal to ProjectA importance ({importance_pa})"
        )
        print(f"[E2E] Importance verified: Person1 ({importance_p1}) <= ProjectA ({importance_pa})")

    # ──────────────────────────────────────────────────────────────────────
    # Let 7+ days pass (Back-date nodes to test TTL expiration and retention)
    # ──────────────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    async with TestingSessionLocal() as session:
        # Backdate ALL nodes in the database to 10 days ago (expired status)
        stmt = select(EntityNode)
        res = await session.execute(stmt)
        all_nodes = res.scalars().all()
        for node in all_nodes:
            node.ttl_expiry = now - timedelta(days=10)
        await session.commit()
        print("\n[E2E] Back-dated all nodes to 10 days in the past (expired).")

    # Now, execute a context query that reinforces ONLY ProjectA, Person1, and Person2
    print("[E2E] Triggering reinforcement hits for ProjectA...")
    async with TestingSessionLocal() as session:
        await assemble(
            goal="Tell me about ProjectA and team members Person1 and Person2",
            session_id=session_id,
            budget_tokens=8000,
            session=session,
            embedding_provider=memory.embedding_provider,
            use_reg=True,
            agent_id=agent_id
        )

    # Let background reinforcement tasks finish
    await asyncio.sleep(0.8)

    # Verify that reinforced nodes have active (refreshed) TTLs, while ProjectB remains expired
    async with TestingSessionLocal() as session:
        db_project_a = await find_node_by_fuzzy_name(session, "ProjectA")
        db_project_b = await find_node_by_fuzzy_name(session, "ProjectB")

        # ProjectA was reinforced: its TTL is now in the future
        assert db_project_a.ttl_expiry > now, "ProjectA TTL was not refreshed!"
        # ProjectB was NOT reinforced: its TTL is still in the past (expired)
        assert db_project_b.ttl_expiry < now, "ProjectB TTL was incorrectly refreshed!"
        
        print(f"[E2E] TTL verification before deletion:")
        print(f"  - ProjectA TTL Expiry: {db_project_a.ttl_expiry} (Active, > {now})")
        print(f"  - ProjectB TTL Expiry: {db_project_b.ttl_expiry} (Expired, < {now})")

    # Call delete_expired_nodes() to run eviction
    deleted_count = await expire_old_nodes()
    print(f"[E2E] Cleanup task executed. Deleted {deleted_count} expired node(s).")
    assert deleted_count >= 1, "Expected at least one expired node to be deleted!"

    # Verify that ProjectB was successfully deleted, but ProjectA was retained
    async with TestingSessionLocal() as session:
        db_project_a = await find_node_by_fuzzy_name(session, "ProjectA")
        db_project_b = await find_node_by_fuzzy_name(session, "ProjectB")

        assert db_project_a is not None, "High-hit node ProjectA was incorrectly deleted!"
        assert db_project_b is None, "Low-hit expired node ProjectB was not deleted!"
        print("[E2E] Cache eviction successfully verified! ProjectA retained, ProjectB evicted.")

    print("\n" + "=" * 60)
    print("REG E2E INTEGRATION TEST PASSED SUCCESSFULLY")
    print("=" * 60 + "\n")
