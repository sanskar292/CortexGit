import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EntityNode, EntityRegistry, NodeHit, HitType
from cortexgit.graph.graph_repository import GraphRepository, INITIAL_TTL
from cortexgit.core.entity_pull import entity_pull
from cortexgit.core.context_assembler import assemble

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
async def test_hit_on_entity_pull():
    """Verify that pulling an entity increments hit frequency, resets TTL, and creates a hit log."""
    now = datetime.now(timezone.utc)
    session_id = "test-session-pull"
    
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # 1. Create a graph node
        node_id = await repo.create_node(
            entity_name="billing-service",
            entity_type="project",
            description="Payments microservice",
            status="active",
            agent_id="agent-1"
        )
        
        # 2. Also register it as an entity in the flat registry
        registry_entry = EntityRegistry(
            key="billing-service",
            value="in_progress_state",
            agent_id="agent-1",
            event_id=uuid_id_val() # helper to get valid event UUID or construct one
        )
        session.add(registry_entry)
        await session.commit()
        
        # Back-date the node's TTL and hit frequency
        node = await session.get(EntityNode, node_id)
        node.ttl_expiry = now + timedelta(minutes=10)
        node.hit_frequency = 0
        await session.commit()

    # Perform the entity pull
    async with TestingSessionLocal() as session:
        matched = await entity_pull(goal="billing", session=session, session_id=session_id)
        assert "billing-service" in matched

    # Sleep a short duration to let non-blocking asyncio background task complete
    await asyncio.sleep(0.5)

    # Verify updates in the database
    async with TestingSessionLocal() as session:
        db_node = await session.get(EntityNode, node_id)
        assert db_node.hit_frequency == 1
        
        # Assert TTL has been refreshed
        node_ttl = db_node.ttl_expiry
        if node_ttl.tzinfo is None:
            node_ttl = node_ttl.replace(tzinfo=timezone.utc)
        expected_expiry = datetime.now(timezone.utc) + INITIAL_TTL
        assert abs((node_ttl - expected_expiry).total_seconds()) < 10

        # Assert hit log entry exists
        hits_res = await session.execute(select(NodeHit).where(NodeHit.node_id == node_id))
        hits = hits_res.scalars().all()
        assert len(hits) == 1
        assert hits[0].hit_type == HitType.QUERY
        assert hits[0].session_id == session_id

@pytest.mark.anyio
async def test_hit_on_context_assembler():
    """Verify that assembling context triggers reinforcement hits on injected entities."""
    now = datetime.now(timezone.utc)
    session_id = "test-session-assembler"
    
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        
        # Create a graph node
        node_id = await repo.create_node(
            entity_name="monolith-migration",
            entity_type="concept",
            description="Decomposing monolith",
            status="active",
            agent_id="agent-1"
        )
        
        # Register it in flat registry
        registry_entry = EntityRegistry(
            key="monolith-migration",
            value="active_migration",
            agent_id="agent-1",
            event_id=uuid_id_val()
        )
        session.add(registry_entry)
        await session.commit()
        
        # Back-date the node
        node = await session.get(EntityNode, node_id)
        node.ttl_expiry = now + timedelta(minutes=5)
        node.hit_frequency = 0
        await session.commit()

    # Call assemble()
    async with TestingSessionLocal() as session:
        context = await assemble(
            goal="migration",
            session_id=session_id,
            budget_tokens=4000,
            session=session
        )
        assert "monolith-migration" in context["entities"]

    # Sleep to drain non-blocking background tasks
    await asyncio.sleep(0.5)

    # Verify hits (should increment by 2: 1 from entity_pull, 1 from assembler packaging)
    async with TestingSessionLocal() as session:
        db_node = await session.get(EntityNode, node_id)
        assert db_node.hit_frequency == 2
        
        node_ttl = db_node.ttl_expiry
        if node_ttl.tzinfo is None:
            node_ttl = node_ttl.replace(tzinfo=timezone.utc)
        expected_expiry = datetime.now(timezone.utc) + INITIAL_TTL
        assert abs((node_ttl - expected_expiry).total_seconds()) < 10

        hits_res = await session.execute(select(NodeHit).where(NodeHit.node_id == node_id))
        hits = hits_res.scalars().all()
        assert len(hits) == 2
        assert all(h.hit_type == HitType.QUERY for h in hits)

@pytest.mark.anyio
async def test_multiple_consecutive_hits_increment():
    """Verify that multiple consecutive pulls sequentially increment the hit frequency."""
    session_id = "test-session-multi"
    
    async with TestingSessionLocal() as session:
        repo = GraphRepository(session)
        node_id = await repo.create_node("multi-node", "concept", "Multiple", "active", "agent-1")
        
        registry_entry = EntityRegistry(
            key="multi-node",
            value="multi_val",
            agent_id="agent-1",
            event_id=uuid_id_val()
        )
        session.add(registry_entry)
        await session.commit()

    # Trigger 3 hits consecutively
    for _ in range(3):
        async with TestingSessionLocal() as session:
            await entity_pull(goal="multi", session=session, session_id=session_id)
        await asyncio.sleep(0.2)  # pause briefly to let task run

    # Sleep slightly longer to guarantee final task completion
    await asyncio.sleep(0.5)

    # Assert hit frequency is exactly 3
    async with TestingSessionLocal() as session:
        db_node = await session.get(EntityNode, node_id)
        assert db_node.hit_frequency == 3
        
        hits_res = await session.execute(select(NodeHit).where(NodeHit.node_id == node_id))
        hits = hits_res.scalars().all()
        assert len(hits) == 3

def uuid_id_val():
    import uuid
    return uuid.uuid4()
