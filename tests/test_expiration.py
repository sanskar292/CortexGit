import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EntityNode, EntityEdge, NodeHit
from cortexgit.graph.graph_repository import GraphRepository, INITIAL_TTL
from cortexgit.graph.expiration import expire_old_nodes

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
async def test_expired_nodes_deleted_active_nodes_retained():
    """Verify that nodes with expired TTL are deleted and active nodes are retained."""
    now = datetime.now(timezone.utc)
    
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # 1. Create active node
        active_id = await repo.create_node(
            entity_name="ActiveNode",
            entity_type="project",
            description="Active node description",
            status="active",
            agent_id="agent-1"
        )
        
        # 2. Create expired node
        expired_id = await repo.create_node(
            entity_name="ExpiredNode",
            entity_type="concept",
            description="Expired node description",
            status="active",
            agent_id="agent-1"
        )
        
        # Manually back-date expired node's TTL to the past
        node_expired = await session.get(EntityNode, expired_id)
        node_expired.ttl_expiry = now - timedelta(minutes=10)
        
        # Ensure active node has TTL in the future
        node_active = await session.get(EntityNode, active_id)
        node_active.ttl_expiry = now + timedelta(hours=2)
        
        await session.commit()

    # Call cleanup task
    deleted_count = await expire_old_nodes()
    assert deleted_count == 1

    # Verify database contents
    async with TestingSessionLocal() as session:
        db_expired = await session.get(EntityNode, expired_id)
        db_active = await session.get(EntityNode, active_id)
        
        assert db_expired is None, "Expired node was not deleted"
        assert db_active is not None, "Active node was incorrectly deleted"

@pytest.mark.anyio
async def test_cascade_delete_removes_connected_edges():
    """Verify that deleting an expired node cascade deletes any edges referencing it."""
    now = datetime.now(timezone.utc)
    
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # Create an expired node and an active node
        exp_id = await repo.create_node("ExpiredTarget", "concept", "Expired", "active", "agent-1")
        act_id = await repo.create_node("ActiveSource", "concept", "Active", "active", "agent-1")
        
        # Create a relationship edge between them
        edge_id = await repo.create_edge(act_id, exp_id, "references")
        
        # Back-date the target node so it expires
        node_exp = await session.get(EntityNode, exp_id)
        node_exp.ttl_expiry = now - timedelta(minutes=5)
        await session.commit()

    # Call cleanup task
    deleted_count = await expire_old_nodes()
    assert deleted_count == 1

    # Verify edge and node deletion in database
    async with TestingSessionLocal() as session:
        db_exp_node = await session.get(EntityNode, exp_id)
        db_act_node = await session.get(EntityNode, act_id)
        db_edge = await session.get(EntityEdge, edge_id)
        
        assert db_exp_node is None, "Expired node was not deleted"
        assert db_act_node is not None, "Active node should not be deleted"
        assert db_edge is None, "Edge referencing deleted node was not cascade deleted"

@pytest.mark.anyio
async def test_record_hit_resets_ttl_expiry():
    """Verify that calling record_hit() resets a node's ttl_expiry to now + INITIAL_TTL."""
    now = datetime.now(timezone.utc)
    
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # 1. Create node
        node_id = await repo.create_node("HitNode", "person", "Hit Target", "active", "agent-1")
        
        # 2. Artificially back-date TTL to simulate it being close to expiry
        node = await session.get(EntityNode, node_id)
        approaching_expiry = now + timedelta(minutes=5)
        node.ttl_expiry = approaching_expiry
        node.hit_frequency = 0
        await session.commit()
        
        # 3. Call record_hit() to reset TTL
        await repo.record_hit(node_id, hit_type="query", session_id="session-999")

    # Verify updates in database
    async with TestingSessionLocal() as session:
        db_node = await session.get(EntityNode, node_id)
        
        assert db_node.hit_frequency == 1
        
        # Assert TTL was reset to INITIAL_TTL in the future
        node_ttl = db_node.ttl_expiry
        if node_ttl.tzinfo is None:
            node_ttl = node_ttl.replace(tzinfo=timezone.utc)
            
        expected_expiry = datetime.now(timezone.utc) + INITIAL_TTL
        # Check that TTL is close to the expected expiry (within 10 seconds tolerance)
        assert abs((node_ttl - expected_expiry).total_seconds()) < 10
