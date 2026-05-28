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
async def test_reg_schema_operations():
    """
    Test case to verify the Relational Entity Graph (REG) schema.
    Specifically:
    1. Creates a test entity node.
    2. Creates an edge between two nodes.
    3. Records a hit.
    4. Confirms all three tables can be read back.
    """
    async with TestingSessionLocal() as session:
        # 1. Create two test entity nodes
        node1_id = uuid.uuid4()
        node1 = EntityNode(
            node_id=node1_id,
            entity_name="CortexGit",
            entity_type=EntityNodeType.PROJECT,
            description="Persistent memory SDK for LLM agents",
            status="active",
            degree_centrality=1.0,
            hit_frequency=5,
            last_hit=datetime.now(timezone.utc),
            ttl_expiry=datetime.now(timezone.utc) + timedelta(days=1),
            created_at=datetime.now(timezone.utc),
            agent_id="agent-007",
        )

        node2_id = uuid.uuid4()
        node2 = EntityNode(
            node_id=node2_id,
            entity_name="Antigravity",
            entity_type=EntityNodeType.PERSON,
            description="Advanced AI coding assistant",
            status="ready",
            degree_centrality=1.0,
            hit_frequency=10,
            last_hit=datetime.now(timezone.utc),
            ttl_expiry=datetime.now(timezone.utc) + timedelta(days=2),
            created_at=datetime.now(timezone.utc),
            agent_id="agent-007",
        )

        session.add(node1)
        session.add(node2)
        await session.commit()

    async with TestingSessionLocal() as session:
        # Verify both nodes can be read back
        db_res = await session.execute(select(EntityNode).order_by(EntityNode.entity_name))
        nodes = db_res.scalars().all()
        assert len(nodes) == 2
        assert nodes[0].entity_name == "Antigravity"
        assert nodes[1].entity_name == "CortexGit"

        # 2. Create an edge between the two nodes
        edge_id = uuid.uuid4()
        edge = EntityEdge(
            edge_id=edge_id,
            source_node_id=node1_id,
            target_node_id=node2_id,
            relation_type="developed_by",
            weight=1.5,
            created_at=datetime.now(timezone.utc),
        )
        session.add(edge)

        # 3. Record a hit on node1
        hit_id = uuid.uuid4()
        hit = NodeHit(
            hit_id=hit_id,
            node_id=node1_id,
            hit_type=HitType.QUERY,
            hit_timestamp=datetime.now(timezone.utc),
            session_id="session-xyz",
        )
        session.add(hit)
        await session.commit()

    # 4. Confirms all three tables can be read back
    async with TestingSessionLocal() as session:
        # Check node_hits table
        db_hits = await session.execute(select(NodeHit).where(NodeHit.hit_id == hit_id))
        read_hit = db_hits.scalar_one_or_none()
        assert read_hit is not None
        assert read_hit.node_id == node1_id
        assert read_hit.hit_type == HitType.QUERY
        assert read_hit.session_id == "session-xyz"

        # Check entity_edges table
        db_edges = await session.execute(select(EntityEdge).where(EntityEdge.edge_id == edge_id))
        read_edge = db_edges.scalar_one_or_none()
        assert read_edge is not None
        assert read_edge.source_node_id == node1_id
        assert read_edge.target_node_id == node2_id
        assert read_edge.relation_type == "developed_by"
        assert read_edge.weight == 1.5

        # Check entity_nodes table
        db_nodes = await session.execute(select(EntityNode).where(EntityNode.node_id == node1_id))
        read_node = db_nodes.scalar_one_or_none()
        assert read_node is not None
        assert read_node.entity_name == "CortexGit"
        assert read_node.entity_type == EntityNodeType.PROJECT
        assert read_node.degree_centrality == 1.0
        assert read_node.hit_frequency == 5
