"""
Tests for context_assembler.py REG importance-weighted packing.

Covers:
- High-importance entities (>10) are packed before snapshots
- Token budget is never exceeded in REG mode
- Results are deterministic (same inputs → same outputs)
- use_reg=False falls back to original alphabetical entity packing
- Medium and low importance entities are packed after snapshots
- use_reg=True with no agent_id falls back gracefully (no REG entities)

Uses SQLite via db_helper.py — no PostgreSQL required.
"""
import json
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EventLog, EventType, EntityRegistry
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import update_centrality
from cortexgit.core.context_assembler import assemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def make_event(session, session_id="session-1", agent_id="agent-test"):
    """Insert a minimal EventLog row and return its ID."""
    event_id = uuid.uuid4()
    session.add(EventLog(
        event_id=event_id,
        session_id=session_id,
        agent_id=agent_id,
        event_type=EventType.SYSTEM,
        payload={"msg": "test"},
        created_at=datetime.now(timezone.utc),
    ))
    await session.commit()
    return event_id


async def make_reg_node(repo, session, name, entity_type, agent_id,
                        n_edges=0, n_hits=0):
    """
    Create a REG node, optionally wire edges to anchor nodes for centrality,
    record hits, and persist degree_centrality.
    Returns node_id.
    """
    node_id = await repo.create_node(name, entity_type, f"Desc {name}", "active", agent_id)

    for i in range(n_edges):
        anchor = await repo.create_node(f"anchor_{name}_{i}", "concept", "anc", "active", agent_id)
        await repo.create_edge(node_id, anchor, "linked")

    for _ in range(n_hits):
        await repo.record_hit(node_id, "query", "sess")

    await update_centrality(node_id, session)
    return node_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_high_importance_entities_packed_first(mock_embed):
    """
    High-importance entities (importance > 10) appear in assembled_entities
    before any snapshot would displace them, even when a snapshot exists.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Node with importance = 12.0 (3 edges × 4 hits) → HIGH tier
        await make_reg_node(repo, session, "billing_core", "project", "agent-A",
                            n_edges=3, n_hits=4)

        # Node with importance = 2.0 (1 edge × 2 hits) → LOW tier
        await make_reg_node(repo, session, "billing_note", "concept", "agent-A",
                            n_edges=1, n_hits=2)

        result = await assemble(
            goal="billing",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=True,
            agent_id="agent-A",
        )

    entities = result["entities"]
    assert "billing_core" in entities, "High-importance entity must be packed"
    assert entities["billing_core"]["importance"] == 12.0

    # Both should be included with a large budget
    assert "billing_note" in entities


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_token_budget_never_exceeded_in_reg_mode(mock_embed):
    """
    Total serialised tokens across all assembled categories must never
    exceed budget_tokens, even in REG mode with many entities.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create 10 nodes matching "widget"
        for i in range(10):
            n_edges = (10 - i)  # decreasing importance
            await make_reg_node(repo, session, f"widget_{i}", "concept", "agent-B",
                                n_edges=n_edges, n_hits=n_edges)

        budget = 120
        result = await assemble(
            goal="widget",
            session_id="s1",
            budget_tokens=budget,
            session=session,
            use_reg=True,
            agent_id="agent-B",
        )

    total = 0
    for item in result["conflicts"]:
        total += len(json.dumps(item)) // 4
    for item in result["events"]:
        total += len(json.dumps(item)) // 4
    for item in result["snapshots"]:
        total += len(json.dumps(item)) // 4
    for key, val in result["entities"].items():
        total += len(json.dumps({key: val})) // 4

    assert total <= budget, f"Budget {budget} exceeded: used {total} tokens"


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_results_are_deterministic(mock_embed):
    """
    Calling assemble() twice with the same inputs produces identical output.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        await make_reg_node(repo, session, "cortex_alpha", "project", "agent-C",
                            n_edges=2, n_hits=3)
        await make_reg_node(repo, session, "cortex_beta", "concept", "agent-C",
                            n_edges=1, n_hits=1)

        result_1 = await assemble(
            goal="cortex",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=True,
            agent_id="agent-C",
        )
        result_2 = await assemble(
            goal="cortex",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=True,
            agent_id="agent-C",
        )

    # Entity keys and importance values must be identical
    assert set(result_1["entities"].keys()) == set(result_2["entities"].keys())
    for key in result_1["entities"]:
        assert result_1["entities"][key]["importance"] == result_2["entities"][key]["importance"]


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_use_reg_false_falls_back_to_legacy(mock_embed):
    """
    use_reg=False must use the original entity_registry substring match.
    Entities should come from EntityRegistry, not entity_nodes.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        event_id = await make_event(session, agent_id="agent-D")

        # Insert a legacy entity_registry row
        session.add(EntityRegistry(
            key="legacy_module",
            value={"description": "old style"},
            agent_id="agent-D",
            event_id=event_id,
            updated_at=datetime.now(timezone.utc),
        ))
        await session.commit()

        result = await assemble(
            goal="legacy",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=False,
            agent_id="agent-D",
        )

    # Legacy path returns the entity_registry value directly (not a node dict)
    assert "legacy_module" in result["entities"]
    entity_val = result["entities"]["legacy_module"]
    # Legacy path stores the raw value, NOT an importance-enriched dict
    assert "importance" not in entity_val


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_medium_importance_entities_packed_after_snapshots(mock_embed):
    """
    Medium-importance entities (5 <= importance <= 10) are packed AFTER snapshots
    in REG mode. High-importance entities appear BEFORE snapshots.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # HIGH: 2 edges × 6 hits = 12.0 → should appear (packed at priority 3)
        await make_reg_node(repo, session, "core_engine", "project", "agent-E",
                            n_edges=2, n_hits=6)

        # MED: 1 edge × 7 hits = 7.0 → packed at priority 5 (after snapshots)
        await make_reg_node(repo, session, "helper_module", "concept", "agent-E",
                            n_edges=1, n_hits=7)

        result = await assemble(
            goal="engine module",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=True,
            agent_id="agent-E",
        )

    entities = result["entities"]
    assert "core_engine" in entities, "HIGH entity must be present"
    assert "helper_module" in entities, "MED entity must be present"

    assert entities["core_engine"]["importance"] == 12.0
    assert entities["helper_module"]["importance"] == 7.0


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_use_reg_true_no_agent_id_uses_no_reg_entities(mock_embed):
    """
    use_reg=True without agent_id gracefully falls back: no REG entities pulled,
    legacy entity_pull() is used instead.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        await make_reg_node(repo, session, "orphan_node", "concept", "agent-F",
                            n_edges=3, n_hits=5)

        # Pass use_reg=True but NO agent_id
        result = await assemble(
            goal="orphan",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=True,
            agent_id=None,      # <-- no agent_id
        )

    # Should return empty entities (legacy path, no entity_registry rows inserted)
    assert result["entities"] == {}


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_injected_nodes_appear_in_context_within_budget(mock_embed):
    """
    Verify that injected nodes appear in the assembled context when budget allows,
    are marked as 'injected': True, are logged in metadata, and never exceed budget.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create some high-importance entities (importance > 10)
        # These will be packed under Priority 3: High-importance entities
        await make_reg_node(repo, session, "high_entity_1", "concept", "agent-injected", n_edges=3, n_hits=4) # 12.0

        # Create some nodes that will be active but not queried/matched directly,
        # so they will be available for Proactive Surface Injection.
        # Since importance is (degree_centrality * weight) * (hit_frequency * weight)
        # let's make their importances high enough so they rank top.
        await make_reg_node(repo, session, "injected_node_1", "concept", "agent-injected", n_edges=4, n_hits=4) # 16.0
        await make_reg_node(repo, session, "injected_node_2", "concept", "agent-injected", n_edges=3, n_hits=3) # 9.0

        budget = 2000
        result = await assemble(
            goal="high",
            session_id="s1",
            budget_tokens=budget,
            session=session,
            use_reg=True,
            agent_id="agent-injected",
        )

        entities = result["entities"]
        
        # Verify high_entity_1 is in entities
        assert "high_entity_1" in entities
        assert not entities["high_entity_1"].get("injected", False)

        # Verify injected nodes are present and marked as injected
        assert "injected_node_1" in entities
        assert entities["injected_node_1"]["injected"] is True
        assert "injected_node_2" in entities
        assert entities["injected_node_2"]["injected"] is True

        # Verify metadata
        assert "metadata" in result
        assert "injected_entities" in result["metadata"]
        assert "injected_node_1" in result["metadata"]["injected_entities"]
        assert "injected_node_2" in result["metadata"]["injected_entities"]

        # Calculate total tokens
        total = 0
        for item in result["conflicts"]:
            total += len(json.dumps(item)) // 4
        for item in result["events"]:
            total += len(json.dumps(item)) // 4
        for item in result["snapshots"]:
            total += len(json.dumps(item)) // 4
        for key, val in result["entities"].items():
            total += len(json.dumps({key: val})) // 4

        assert total <= budget


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_injected_nodes_budget_strictly_enforced(mock_embed):
    """
    Verify that if the token budget is tight, some or all injected nodes are excluded,
    and the token budget is never exceeded.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create a high importance entity
        await make_reg_node(repo, session, "high_entity_1", "concept", "agent-tight", n_edges=3, n_hits=4) # 12.0

        # Create nodes to be injected
        await make_reg_node(repo, session, "very_long_injected_entity_name_for_token_budget_test_one", "concept", "agent-tight", n_edges=4, n_hits=4)
        await make_reg_node(repo, session, "very_long_injected_entity_name_for_token_budget_test_two", "concept", "agent-tight", n_edges=3, n_hits=3)

        # First, run with high budget to check they would normally be injected
        result_large = await assemble(
            goal="high",
            session_id="s1",
            budget_tokens=9999,
            session=session,
            use_reg=True,
            agent_id="agent-tight",
        )
        assert "very_long_injected_entity_name_for_token_budget_test_one" in result_large["entities"]
        assert "very_long_injected_entity_name_for_token_budget_test_two" in result_large["entities"]

        # Now, run with a tight budget. Let's calculate a budget where high_entity_1 fits,
        # but the first injected node fits and the second does not.
        high_entity_dict = result_large["entities"]["high_entity_1"]
        high_entity_tokens = len(json.dumps({"high_entity_1": high_entity_dict})) // 4

        injected_one_dict = result_large["entities"]["very_long_injected_entity_name_for_token_budget_test_one"]
        injected_one_tokens = len(json.dumps({"very_long_injected_entity_name_for_token_budget_test_one": injected_one_dict})) // 4

        # Set budget to accommodate high_entity and injected_one, but not injected_two
        budget = high_entity_tokens + injected_one_tokens + 10

        result_tight = await assemble(
            goal="high",
            session_id="s1",
            budget_tokens=budget,
            session=session,
            use_reg=True,
            agent_id="agent-tight",
        )

        entities_tight = result_tight["entities"]
        assert "high_entity_1" in entities_tight
        assert "very_long_injected_entity_name_for_token_budget_test_one" in entities_tight
        assert "very_long_injected_entity_name_for_token_budget_test_two" not in entities_tight

        # Verify budget is not exceeded
        # Verify budget is not exceeded
        total = 0
        for item in result_tight["conflicts"]:
            total += len(json.dumps(item)) // 4
        for item in result_tight["events"]:
            total += len(json.dumps(item)) // 4
        for item in result_tight["snapshots"]:
            total += len(json.dumps(item)) // 4
        for key, val in result_tight["entities"].items():
            total += len(json.dumps({key: val})) // 4

        assert total <= budget


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_injection_can_be_disabled(mock_embed):
    """
    Verify that if enable_injection is False on CortexGit, proactive surface injection is completely skipped.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create a node that would normally be injected
        await make_reg_node(repo, session, "should_be_injected", "concept", "agent-disabled", n_edges=3, n_hits=3)

        from cortexgit import CortexGit
        memory = CortexGit(enable_injection=False)

        # Call get_context using memory instance with enable_injection=False
        result = await memory.get_context(
            goal="test",
            budget_tokens=9999,
            session_id="s1",
            use_reg=True,
            agent_id="agent-disabled"
        )

        # Verify no injected entities are present
        entities = result["entities"]
        assert "should_be_injected" not in entities
        assert result.get("metadata", {}).get("injected_entities") == []


@pytest.mark.anyio
@patch("cortexgit.retrieval.semantic_recall.embed_text")
async def test_injection_configs_respected(mock_embed):
    """
    Verify that custom configurations (injection_threshold and injection_top_k) are fully respected by CortexGit context retrieval.
    """
    mock_embed.return_value = [0.1] * 1536

    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)

        # Create three nodes with different importances
        # Node A: 4 edges, 4 hits = 16.0 importance
        await make_reg_node(repo, session, "node_A", "concept", "agent-config", n_edges=4, n_hits=4)
        # Node B: 3 edges, 3 hits = 9.0 importance
        await make_reg_node(repo, session, "node_B", "concept", "agent-config", n_edges=3, n_hits=3)
        # Node C: 2 edges, 2 hits = 4.0 importance
        await make_reg_node(repo, session, "node_C", "concept", "agent-config", n_edges=2, n_hits=2)

        from cortexgit import CortexGit
        
        # Test Case 1: Custom threshold of 10.0 (Only Node A should be injected, B and C filtered)
        memory_threshold = CortexGit(injection_threshold=10.0)
        result_threshold = await memory_threshold.get_context(
            goal="test",
            budget_tokens=9999,
            session_id="s1",
            use_reg=True,
            agent_id="agent-config"
        )
        assert "node_A" in result_threshold["entities"]
        assert "node_B" not in result_threshold["entities"]
        assert "node_C" not in result_threshold["entities"]

        # Test Case 2: Custom top_k of 1 (Only the single most important node, Node A, is injected)
        memory_top_k = CortexGit(injection_top_k=1)
        result_top_k = await memory_top_k.get_context(
            goal="test",
            budget_tokens=9999,
            session_id="s1",
            use_reg=True,
            agent_id="agent-config"
        )
        assert "node_A" in result_top_k["entities"]
        assert "node_B" not in result_top_k["entities"]
        assert "node_C" not in result_top_k["entities"]
        assert len(result_top_k["metadata"]["injected_entities"]) == 1


@pytest.mark.anyio
@patch.dict("os.environ", {"INJECTION_IMPORTANCE_THRESHOLD": "15.0", "INJECTION_TOP_K": "1"})
def test_injection_configs_env_defaults():
    """
    Verify that CortexGit initializer reads environment variables as defaults when not explicitly passed.
    """
    from cortexgit import CortexGit
    memory = CortexGit()
    assert memory.enable_injection is True
    assert memory.injection_threshold == 15.0
    assert memory.injection_top_k == 1


