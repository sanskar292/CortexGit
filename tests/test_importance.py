import pytest
import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EntityNode
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import update_centrality
from cortexgit.graph.importance import (
    calculate_importance,
    rank_nodes_by_importance,
    get_top_k_important_nodes,
    get_high_importance_nodes,
)

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
async def test_isolated_nodes_importance_zero():
    """
    Isolated nodes with no hits have importance 0.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        node_id = await repo.create_node(
            "IsolatedNode", "concept", "Isolated", "active", "agent-1"
        )
        
        # Calculate degree centrality and update it (should be 0.0)
        await update_centrality(node_id, session)
        
        # Fresh read
        importance = await calculate_importance(node_id, session)
        assert importance == 0.0


@pytest.mark.anyio
async def test_connected_nodes_with_hits_importance_nonzero():
    """
    Connected nodes with hits have non-zero importance.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # Create two connected nodes
        node_a = await repo.create_node("NodeA", "concept", "A", "active", "agent-1")
        node_b = await repo.create_node("NodeB", "concept", "B", "active", "agent-1")
        await repo.create_edge(node_a, node_b, "connected_to")
        
        # Record 3 hits on NodeA
        await repo.record_hit(node_a, "query", "session-1")
        await repo.record_hit(node_a, "query", "session-1")
        await repo.record_hit(node_a, "query", "session-1")
        
        # Update centrality for NodeA (should be 1.0)
        await update_centrality(node_a, session)
        
        importance = await calculate_importance(node_a, session)
        # degree_centrality (1.0) * hit_frequency (3) = 3.0
        assert importance == 3.0


@pytest.mark.anyio
async def test_ranking_correctness():
    """
    Verify ranking sorts by importance descending and filters out expired nodes.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # Create nodes
        n1 = await repo.create_node("Node1", "concept", "N1", "active", "agent-A")
        n2 = await repo.create_node("Node2", "concept", "N2", "active", "agent-A")
        n3 = await repo.create_node("Node3", "concept", "N3", "active", "agent-A")
        n_expired = await repo.create_node("NodeExpired", "concept", "Expired", "active", "agent-A")
        n_other_agent = await repo.create_node("NodeOther", "concept", "Other", "active", "agent-B")
        
        # Make n1 have centrality=2.0, hits=3 -> importance=6.0
        # n1 connected to n2 and n3
        await repo.create_edge(n1, n2, "link")
        await repo.create_edge(n1, n3, "link")
        await repo.record_hit(n1, "query", "session-1")
        await repo.record_hit(n1, "query", "session-1")
        await repo.record_hit(n1, "query", "session-1")
        await update_centrality(n1, session)
        
        # Make n2 have centrality=1.0, hits=4 -> importance=4.0
        # n2 connected to n1 only
        await repo.record_hit(n2, "query", "session-1")
        await repo.record_hit(n2, "query", "session-1")
        await repo.record_hit(n2, "query", "session-1")
        await repo.record_hit(n2, "query", "session-1")
        await update_centrality(n2, session)
        
        # Make n3 have centrality=1.0, hits=0 -> importance=0.0
        await update_centrality(n3, session)
        
        # Set n_expired as expired in the past
        db_n_expired = await session.get(EntityNode, n_expired)
        db_n_expired.ttl_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        # Give n_expired huge centrality and hits, but it should still be ignored because it is expired
        db_n_expired.degree_centrality = 10.0
        db_n_expired.hit_frequency = 10
        
        await session.commit()
        
        # Rank nodes for agent-A
        ranked = await rank_nodes_by_importance("agent-A", session)
        
        # Should return exactly 3 nodes: Node1, Node2, Node3 (NodeExpired is expired, NodeOther belongs to another agent)
        assert len(ranked) == 3
        assert ranked[0].node_id == n1
        assert ranked[1].node_id == n2
        assert ranked[2].node_id == n3
        
        assert ranked[0].degree_centrality * ranked[0].hit_frequency == 6.0
        assert ranked[1].degree_centrality * ranked[1].hit_frequency == 4.0
        assert ranked[2].degree_centrality * ranked[2].hit_frequency == 0.0


@pytest.mark.anyio
async def test_top_k_returns_exactly_k_or_fewer():
    """
    get_top_k_important_nodes returns exactly k nodes or fewer if fewer exist.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        n1 = await repo.create_node("Node1", "concept", "N1", "active", "agent-A")
        n2 = await repo.create_node("Node2", "concept", "N2", "active", "agent-A")
        
        # Verify k = 1 returns exactly 1 node
        top_1 = await get_top_k_important_nodes("agent-A", 1, session)
        assert len(top_1) == 1
        
        # Verify k = 5 returns exactly 2 nodes (fewer than k, since only 2 exist)
        top_5 = await get_top_k_important_nodes("agent-A", 5, session)
        assert len(top_5) == 2


@pytest.mark.anyio
async def test_get_high_importance_nodes_filtering_and_limiting():
    """
    Verify get_high_importance_nodes correct filtering based on threshold and limiting by top_k.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # Create three nodes for agent-X
        na = await repo.create_node("NodeA", "concept", "N_A", "active", "agent-X")
        nb = await repo.create_node("NodeB", "concept", "N_B", "active", "agent-X")
        nc = await repo.create_node("NodeC", "concept", "N_C", "active", "agent-X")
        
        # Node A: 3 edges, 3 hits = 9.0 importance
        await repo.create_edge(na, nb, "link")
        await repo.create_edge(na, nc, "link")
        await repo.create_edge(na, na, "link") # self-loop or anchor link
        await repo.record_hit(na, "query", "session-1")
        await repo.record_hit(na, "query", "session-1")
        await repo.record_hit(na, "query", "session-1")
        await update_centrality(na, session)
        
        # Node B: 2 edges, 2 hits = 4.0 importance
        await repo.create_edge(nb, nc, "link")
        await repo.record_hit(nb, "query", "session-1")
        await repo.record_hit(nb, "query", "session-1")
        await update_centrality(nb, session)
        
        # Node C: 1 edge, 1 hit = 1.0 importance
        await repo.record_hit(nc, "query", "session-1")
        await update_centrality(nc, session)
        
        # 1. Test threshold filtering
        # Threshold = 5.0 -> should only return NodeA
        high_nodes_5 = await get_high_importance_nodes("agent-X", 5.0, session)
        assert len(high_nodes_5) == 1
        assert high_nodes_5[0].node_id == na
        
        # Threshold = 3.0 -> should return NodeA and NodeB
        high_nodes_3 = await get_high_importance_nodes("agent-X", 3.0, session)
        assert len(high_nodes_3) == 2
        assert high_nodes_3[0].node_id == na
        assert high_nodes_3[1].node_id == nb
        
        # Threshold = 0.0 -> should return NodeA, NodeB, and NodeC
        high_nodes_0 = await get_high_importance_nodes("agent-X", 0.0, session)
        assert len(high_nodes_0) == 3
        
        # 2. Test top_k limiting
        # Threshold = 0.0, top_k = 2 -> should return only NodeA and NodeB (top 2 ranked by importance)
        limited_nodes = await get_high_importance_nodes("agent-X", 0.0, session, top_k=2)
        assert len(limited_nodes) == 2
        assert limited_nodes[0].node_id == na
        assert limited_nodes[1].node_id == nb
