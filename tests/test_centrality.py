import pytest
import asyncio
import uuid
from sqlalchemy import select
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EntityNode
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import (
    calculate_degree_centrality,
    update_centrality,
    recalculate_all_centrality,
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
async def test_isolated_node_has_zero_centrality():
    """
    Verify that degree centrality is 0.0 for isolated nodes.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        node_id = await repo.create_node("IsolatedNode", "concept", "Isolated", "active", "agent-1")
        
        centrality = await calculate_degree_centrality(node_id, session)
        assert centrality == 0.0


@pytest.mark.anyio
async def test_centrality_increments_with_edges():
    """
    Verify that degree centrality correctly counts edges where the node is source OR target,
    and increments as new edges are created.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # Create three nodes
        node_a = await repo.create_node("NodeA", "concept", "A", "active", "agent-1")
        node_b = await repo.create_node("NodeB", "concept", "B", "active", "agent-1")
        node_c = await repo.create_node("NodeC", "concept", "C", "active", "agent-1")

        # Initial centrality: all 0.0
        assert await calculate_degree_centrality(node_a, session) == 0.0
        assert await calculate_degree_centrality(node_b, session) == 0.0
        assert await calculate_degree_centrality(node_c, session) == 0.0

        # Add edge between A and B
        await repo.create_edge(node_a, node_b, "linked_to")
        assert await calculate_degree_centrality(node_a, session) == 1.0
        assert await calculate_degree_centrality(node_b, session) == 1.0
        assert await calculate_degree_centrality(node_c, session) == 0.0

        # Add edge between A and C (A is source)
        await repo.create_edge(node_a, node_c, "references")
        assert await calculate_degree_centrality(node_a, session) == 2.0
        assert await calculate_degree_centrality(node_b, session) == 1.0
        assert await calculate_degree_centrality(node_c, session) == 1.0

        # Add edge between C and A (A is target)
        await repo.create_edge(node_c, node_a, "depends_on")
        # Direct self-loop or distinct relation type edge count incremented
        assert await calculate_degree_centrality(node_a, session) == 3.0
        assert await calculate_degree_centrality(node_c, session) == 2.0


@pytest.mark.anyio
async def test_update_centrality_persists_to_db():
    """
    Verify that update_centrality persists the calculated degree centrality to the database.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        node_a = await repo.create_node("NodeA", "concept", "A", "active", "agent-1")
        node_b = await repo.create_node("NodeB", "concept", "B", "active", "agent-1")
        await repo.create_edge(node_a, node_b, "knows")

        # Before updating, the DB value should be the default (0.0)
        db_node_before = await session.get(EntityNode, node_a)
        assert db_node_before.degree_centrality == 0.0

        # Run update
        updated_val = await update_centrality(node_a, session)
        assert updated_val == 1.0

    # Read back in a fresh transaction to verify persistence
    async with TestingSessionLocal() as session:
        db_node_after = await session.get(EntityNode, node_a)
        assert db_node_after.degree_centrality == 1.0


@pytest.mark.anyio
async def test_recalculate_all_centrality_updates_all():
    """
    Verify that recalculate_all_centrality correctly calculates and updates centrality on all nodes.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create three nodes
        node_a = await repo.create_node("NodeA", "concept", "A", "active", "agent-1")
        node_b = await repo.create_node("NodeB", "concept", "B", "active", "agent-1")
        node_c = await repo.create_node("NodeC", "concept", "C", "active", "agent-1")

        # Create edges to make actual centrality values different
        await repo.create_edge(node_a, node_b, "link1")
        await repo.create_edge(node_a, node_c, "link2")

        # Run recalculation for all
        updated_count = await recalculate_all_centrality(session)
        assert updated_count == 3

    # Verify all are persisted in DB
    async with TestingSessionLocal() as session:
        db_node_a = await session.get(EntityNode, node_a)
        db_node_b = await session.get(EntityNode, node_b)
        db_node_c = await session.get(EntityNode, node_c)

        assert db_node_a.degree_centrality == 2.0
        assert db_node_b.degree_centrality == 1.0
        assert db_node_c.degree_centrality == 1.0
