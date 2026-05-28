"""
Tests for Proactive Surface Injection (inject_high_mass_nodes).

Covers:
- Injection returns nodes not in semantic results
- Injection respects token budget
- Injection is idempotent

Uses SQLite via db_helper.py.
"""
import json
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EventLog, EventType, SnapshotStore, PortableInt4Range
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import update_centrality
from cortexgit.graph.injection import inject_high_mass_nodes
from cortexgit.core.context_assembler import assemble


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


async def make_reg_node(repo, session, name, entity_type, agent_id, n_edges=0, n_hits=0):
    node_id = await repo.create_node(name, entity_type, f"Desc {name}", "active", agent_id)

    for i in range(n_edges):
        anchor = await repo.create_node(f"anchor_{name}_{i}", "concept", "anc", "active", agent_id)
        await repo.create_edge(node_id, anchor, "linked")

    for _ in range(n_hits):
        await repo.record_hit(node_id, "query", "sess")

    await update_centrality(node_id, session)
    return node_id


@pytest.mark.anyio
async def test_injection_excludes_semantic_results():
    """
    Injection returns nodes not in semantic results (snapshots.entities_mentioned).
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create three nodes with different importances
        # Node A: 3 edges, 3 hits = 9.0 importance
        await make_reg_node(repo, session, "entity_A", "concept", "agent-X", n_edges=3, n_hits=3)
        # Node B: 4 edges, 4 hits = 16.0 importance (should be top, but will be excluded)
        await make_reg_node(repo, session, "entity_B", "concept", "agent-X", n_edges=4, n_hits=4)
        # Node C: 2 edges, 2 hits = 4.0 importance
        await make_reg_node(repo, session, "entity_C", "concept", "agent-X", n_edges=2, n_hits=2)

        # Build mock semantic results (snapshots) mentioning "entity_B"
        mock_snapshot = SnapshotStore(
            snapshot_id=uuid.uuid4(),
            session_id="s1",
            event_range=PortableInt4Range(1, 10),
            summary="Mentions B",
            entities_mentioned=["entity_B"],
            embedding=[0.1] * 1536,
            created_at=datetime.now(timezone.utc),
        )

        # Call inject_high_mass_nodes
        injected = await inject_high_mass_nodes(
            goal="test",
            agent_id="agent-X",
            session=session,
            k=2,
            semantic_results=[mock_snapshot],
        )

        names = [n.entity_name for n in injected]
        # B must be excluded even though it has the highest importance.
        # A and C should be returned.
        assert "entity_B" not in names
        assert "entity_A" in names
        assert "entity_C" in names
        assert len(injected) == 2


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_injection_respects_token_budget(mock_embed):
    """
    Verification that proactive surface injection respects context token budgets.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create high mass nodes
        await make_reg_node(repo, session, "important_node_one", "concept", "agent-Y", n_edges=5, n_hits=5)
        await make_reg_node(repo, session, "important_node_two", "concept", "agent-Y", n_edges=4, n_hits=4)

        # Assemble with a tight token budget
        # We want to check that the total token usage never exceeds the budget.
        budget = 50
        result = await assemble(
            goal="test",
            session_id="sess-y",
            budget_tokens=budget,
            session=session,
            use_reg=True,
            agent_id="agent-Y",
        )

        total_tokens = 0
        for item in result["conflicts"]:
            total_tokens += len(json.dumps(item)) // 4
        for item in result["events"]:
            total_tokens += len(json.dumps(item)) // 4
        for item in result["snapshots"]:
            total_tokens += len(json.dumps(item)) // 4
        for key, val in result["entities"].items():
            total_tokens += len(json.dumps({key: val})) // 4

        assert total_tokens <= budget
        # With a tight budget, at most one of the injected entities should fit (or none).
        # We verify it doesn't exceed.
        assert len(result["entities"]) <= 1


@pytest.mark.anyio
async def test_injection_is_idempotent():
    """
    Calling inject_high_mass_nodes multiple times with the same parameters
    returns the exact same set of nodes (read-only query function).
    """
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        await make_reg_node(repo, session, "node_1", "concept", "agent-Z", n_edges=3, n_hits=3)
        await make_reg_node(repo, session, "node_2", "concept", "agent-Z", n_edges=2, n_hits=2)

        res_1 = await inject_high_mass_nodes(
            goal="test",
            agent_id="agent-Z",
            session=session,
            k=2,
        )

        res_2 = await inject_high_mass_nodes(
            goal="test",
            agent_id="agent-Z",
            session=session,
            k=2,
        )

        names_1 = [n.entity_name for n in res_1]
        names_2 = [n.entity_name for n in res_2]

        assert names_1 == names_2
        assert len(names_1) == 2
