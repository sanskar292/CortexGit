"""
Tests for entity_pull_with_reg():
- Returns entities ranked by importance (degree_centrality × hit_frequency) descending
- Unconnected entities (no edges, no hits) rank lower than connected+hit entities
- Top-K filtering returns exactly K nodes (or fewer if fewer match)

Uses SQLite via db_helper.py (no PostgreSQL required).
"""
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EntityNode
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import update_centrality
from cortexgit.core.entity_pull import entity_pull_with_reg


@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_entities_ranked_by_importance():
    """
    entity_pull_with_reg() returns nodes sorted by importance descending.
    High importance = high degree_centrality × high hit_frequency.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create three "billing" nodes with different importance levels
        n_high = await repo.create_node("billing_core", "concept", "High importance billing", "active", "agent-1")
        n_mid  = await repo.create_node("billing_service", "concept", "Mid importance billing", "active", "agent-1")
        n_low  = await repo.create_node("billing_note", "concept", "Low importance billing", "active", "agent-1")

        # Wire n_high: 2 edges + 4 hits → importance = 2.0 × 4 = 8.0
        n_aux1 = await repo.create_node("aux_one", "concept", "Aux1", "active", "agent-1")
        n_aux2 = await repo.create_node("aux_two", "concept", "Aux2", "active", "agent-1")
        await repo.create_edge(n_high, n_aux1, "linked_to")
        await repo.create_edge(n_high, n_aux2, "linked_to")
        for _ in range(4):
            await repo.record_hit(n_high, "query", "session-1")
        await update_centrality(n_high, session)

        # Wire n_mid: 1 edge + 2 hits → importance = 1.0 × 2 = 2.0
        await repo.create_edge(n_mid, n_aux1, "linked_to")
        for _ in range(2):
            await repo.record_hit(n_mid, "query", "session-1")
        await update_centrality(n_mid, session)

        # n_low: 0 edges + 0 hits → importance = 0.0
        await update_centrality(n_low, session)

        # Goal "billing" matches all three nodes by name
        results = await entity_pull_with_reg("billing", "agent-1", session)

        # All three should be returned (top_k defaults to 5)
        assert len(results) == 3

        # Check ordering: high > mid > low
        assert results[0]["entity_name"] == "billing_core"
        assert results[1]["entity_name"] == "billing_service"
        assert results[2]["entity_name"] == "billing_note"

        # Verify importance values
        assert results[0]["importance"] == 8.0
        assert results[1]["importance"] == 2.0
        assert results[2]["importance"] == 0.0


@pytest.mark.anyio
async def test_unconnected_entities_rank_lower():
    """
    Nodes with no edges and no hits have importance 0 and appear at the bottom.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Connected + hit node
        n_connected = await repo.create_node("project_alpha", "project", "Connected", "active", "agent-2")
        n_isolated  = await repo.create_node("project_beta", "concept", "Isolated", "active", "agent-2")
        n_anchor    = await repo.create_node("anchor_node", "concept", "Anchor", "active", "agent-2")

        # Connect and hit n_connected: centrality=1.0, hits=3 → importance=3.0
        await repo.create_edge(n_connected, n_anchor, "uses")
        for _ in range(3):
            await repo.record_hit(n_connected, "query", "session-x")
        await update_centrality(n_connected, session)

        # n_isolated stays unconnected, zero hits → importance=0.0
        await update_centrality(n_isolated, session)

        results = await entity_pull_with_reg("project", "agent-2", session)

        # Both project nodes matched
        names = [r["entity_name"] for r in results]
        assert "project_alpha" in names
        assert "project_beta" in names

        # Connected node comes first
        assert results[0]["entity_name"] == "project_alpha"
        assert results[0]["importance"] == 3.0
        assert results[-1]["importance"] == 0.0


@pytest.mark.anyio
async def test_top_k_filtering():
    """
    Top-K limits the result: returns exactly K nodes when more than K match.
    Returns fewer when fewer nodes match the goal.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create 6 nodes that all match "widget"
        for i in range(6):
            nid = await repo.create_node(f"widget_{i}", "concept", f"Widget {i}", "active", "agent-3")
            await update_centrality(nid, session)

        # top_k=5 (default) → should return 5 out of 6
        results_default = await entity_pull_with_reg("widget", "agent-3", session)
        assert len(results_default) == 5

        # top_k=3 → should return exactly 3
        results_3 = await entity_pull_with_reg("widget", "agent-3", session, top_k=3)
        assert len(results_3) == 3

        # top_k=10 with only 6 matches → returns 6 (fewer than K)
        results_10 = await entity_pull_with_reg("widget", "agent-3", session, top_k=10)
        assert len(results_10) == 6

        # top_k=1 → returns exactly 1
        results_1 = await entity_pull_with_reg("widget", "agent-3", session, top_k=1)
        assert len(results_1) == 1


@pytest.mark.anyio
async def test_no_match_returns_empty_list():
    """
    When no entity_name contains any goal token, return an empty list.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        await repo.create_node("cortex_memory", "concept", "Cortex memory node", "active", "agent-4")

        results = await entity_pull_with_reg("zzz_nonexistent_token", "agent-4", session)
        assert results == []


@pytest.mark.anyio
async def test_returned_dict_has_required_fields():
    """
    Each dict in the result list must contain expected REG fields.
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        nid = await repo.create_node("search_engine", "project", "Search engine node", "active", "agent-5")
        await update_centrality(nid, session)

        results = await entity_pull_with_reg("search", "agent-5", session)
        assert len(results) == 1

        node = results[0]
        assert "node_id" in node
        assert "entity_name" in node
        assert "entity_type" in node
        assert "degree_centrality" in node
        assert "hit_frequency" in node
        assert "importance" in node
        assert node["entity_name"] == "search_engine"
        assert node["importance"] == 0.0   # no edges, no hits
