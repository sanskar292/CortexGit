import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import (
    Base,
    EntityNode,
    EntityEdge,
    NodeHit,
    EntityNodeType,
    HitType,
)
from cortexgit.graph.graph_repository import GraphRepository

@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    """Ensure pytest-asyncio runs tests correctly."""
    return "asyncio"

@pytest.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_create_node_idempotent():
    """
    Test that create_node is idempotent within the same agent,
    and that different agents owning the same entity_name get separate nodes.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # 1. Create a node for agent-abc
        node_id_1 = await repo.create_node(
            entity_name="CortexGit",
            entity_type="project",
            description="Persistent memory SDK",
            status="active",
            agent_id="agent-abc",
        )
        assert isinstance(node_id_1, uuid.UUID)

        # 2. Call create_node again with SAME agent — must be idempotent
        node_id_1b = await repo.create_node(
            entity_name="CortexGit",
            entity_type="project",
            description="Alternate description",
            status="inactive",
            agent_id="agent-abc",
        )
        assert node_id_1 == node_id_1b, "Same agent + same name must return the same node_id"

        # 3. Call create_node with a DIFFERENT agent — must create a separate node
        node_id_2 = await repo.create_node(
            entity_name="CortexGit",
            entity_type="project",
            description="Agent XYZ owns its own node",
            status="active",
            agent_id="agent-xyz",
        )
        assert node_id_1 != node_id_2, "Different agents must receive separate nodes for the same entity_name"

        # 4. Verify original node details are preserved for agent-abc
        db_node = await session.get(EntityNode, node_id_1)
        assert db_node is not None
        assert db_node.entity_name == "CortexGit"
        assert db_node.agent_id == "agent-abc"
        assert db_node.description == "Persistent memory SDK"  # original write preserved


@pytest.mark.anyio
async def test_create_edge_weight_increment():
    """
    Test that create_edge increments weight on duplicate relationship calls.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # 1. Create source and target nodes
        src_id = await repo.create_node("NodeA", "concept", "Source", "active", "agent-1")
        tgt_id = await repo.create_node("NodeB", "concept", "Target", "active", "agent-1")

        # 2. Create relationship edge
        edge_id_1 = await repo.create_edge(src_id, tgt_id, "relates_to")
        assert isinstance(edge_id_1, uuid.UUID)

        # 3. Create same relationship edge again (should increment weight)
        edge_id_2 = await repo.create_edge(src_id, tgt_id, "relates_to")
        assert edge_id_1 == edge_id_2

        # 4. Verify in DB
        db_edge = await session.get(EntityEdge, edge_id_1)
        assert db_edge is not None
        assert db_edge.weight == 2.0


@pytest.mark.anyio
async def test_record_hit_updates_frequency_and_ttl():
    """
    Test that record_hit increments frequency, logs the hit, and refreshes the TTL.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # 1. Create a node and back-date its TTL to simulate aging
        node_id = await repo.create_node("HitNode", "person", "Hit Target", "active", "agent-1")
        
        node = await session.get(EntityNode, node_id)
        old_ttl = datetime.now(timezone.utc) - timedelta(hours=12)
        node.ttl_expiry = old_ttl
        node.hit_frequency = 0
        await session.commit()

        # 2. Record the hit
        await repo.record_hit(node_id, hit_type="query", session_id="session-123")

    # Verify updates in database
    async with TestingSessionLocal() as session:
        # Check node updates
        node = await session.get(EntityNode, node_id)
        assert node.hit_frequency == 1
        assert node.last_hit is not None
        # Assert TTL is refreshed to roughly 7 days from now
        node_ttl = node.ttl_expiry
        if node_ttl.tzinfo is None:
            node_ttl = node_ttl.replace(tzinfo=timezone.utc)
        assert node_ttl > datetime.now(timezone.utc) + timedelta(days=6)


        # Check hit record creation
        db_hits = await session.execute(
            select(NodeHit).where(NodeHit.node_id == node_id)
        )
        hits = db_hits.scalars().all()
        assert len(hits) == 1
        assert hits[0].hit_type == HitType.QUERY
        assert hits[0].session_id == "session-123"


@pytest.mark.anyio
async def test_delete_expired_nodes_cascades():
    """
    Test that delete_expired_nodes deletes nodes that have expired TTLs and cascades to their edges/hits.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # 1. Create two nodes
        expired_node_id = await repo.create_node("ExpiredNode", "concept", "Expired", "active", "agent-1")
        active_node_id = await repo.create_node("ActiveNode", "concept", "Active", "active", "agent-1")

        # 2. Create edge between them
        edge_id = await repo.create_edge(expired_node_id, active_node_id, "linked_to")

        # 3. Record hit on expired node
        await repo.record_hit(expired_node_id, "query", "session-expired")

        # 4. Artificially back-date TTL of the expired node to the past
        node_exp = await session.get(EntityNode, expired_node_id)
        node_exp.ttl_expiry = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        node_act = await session.get(EntityNode, active_node_id)
        node_act.ttl_expiry = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        await session.commit()

        # 5. Call delete_expired_nodes
        deleted_count = await repo.delete_expired_nodes()
        assert deleted_count == 1

    # Verify deletions cascaded in DB
    async with TestingSessionLocal() as session:
        # Nodes checking
        db_exp_node = await session.get(EntityNode, expired_node_id)
        db_act_node = await session.get(EntityNode, active_node_id)
        assert db_exp_node is None
        assert db_act_node is not None

        # Edges checking (edge referenced expired node, should be cascade deleted)
        db_edge = await session.get(EntityEdge, edge_id)
        assert db_edge is None

        # Hits checking (hit referenced expired node, should be cascade deleted)
        db_hits = await session.execute(
            select(NodeHit).where(NodeHit.node_id == expired_node_id)
        )
        hits = db_hits.scalars().all()
        assert len(hits) == 0
