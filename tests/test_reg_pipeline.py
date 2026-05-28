import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from api.main import app
from tests.db_helper import test_engine, TestingSessionLocal
from cortexgit.db.models import Base, EntityNode, EntityEdge
from sqlalchemy import select

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
@patch("cortexgit.llm.entity_extractor.extract_entities", new_callable=AsyncMock)
@patch("cortexgit.llm.entity_extractor.extract_reg_entities", new_callable=AsyncMock)
async def test_post_event_triggers_reg_pipeline(mock_extract_reg, mock_extract_flat):
    # Set mock return values
    mock_extract_flat.return_value = {"updates": []}
    mock_extract_reg.return_value = {
        "updates": [
            {
                "entity_name": "billing-service",
                "entity_type": "project",
                "properties": {
                    "description": "Microservice for managing payments",
                    "status": "in_progress"
                },
                "connected_to": [
                    {
                        "target_entity": "monolith-migration",
                        "relation_type": "part_of"
                    },
                    {
                        "target_entity": "alice@company.com",
                        "relation_type": "owned_by"
                    }
                ]
            }
        ]
    }

    # Initialize TestClient
    client = TestClient(app)

    # Post an event
    payload = {
        "session_id": "test-session-123",
        "agent_id": "test-agent-456",
        "event_type": "user",
        "payload": {"text": "Setting up the billing-service repository for monolith-migration migration."}
    }
    
    # Send request
    response = client.post("/events", json=payload)
    assert response.status_code == 201
    response_data = response.json()
    assert response_data["session_id"] == "test-session-123"
    assert response_data["agent_id"] == "test-agent-456"

    # In FastAPI, background tasks queued with TestClient are run synchronously before returning.
    # Verify in the database.
    async with TestingSessionLocal() as session:
        # 1. Confirm nodes were created in entity_nodes table
        nodes_res = await session.execute(select(EntityNode))
        nodes = nodes_res.scalars().all()
        
        node_names = [n.entity_name for n in nodes]
        assert "billing-service" in node_names
        assert "monolith-migration" in node_names
        assert "alice@company.com" in node_names
        
        # Verify billing-service node details
        billing_node = next(n for n in nodes if n.entity_name == "billing-service")
        assert billing_node.entity_type.value == "project"
        assert billing_node.description == "Microservice for managing payments"
        assert billing_node.status == "in_progress"
        assert billing_node.agent_id == "test-agent-456"

        # Verify target nodes default to 'concept'
        monolith_node = next(n for n in nodes if n.entity_name == "monolith-migration")
        assert monolith_node.entity_type.value == "concept"
        assert monolith_node.description is None
        
        alice_node = next(n for n in nodes if n.entity_name == "alice@company.com")
        assert alice_node.entity_type.value == "concept"
        assert alice_node.description is None

        # 2. Confirm edges were created in entity_edges table
        edges_res = await session.execute(select(EntityEdge))
        edges = edges_res.scalars().all()
        assert len(edges) == 2
        
        # Confirm relations and weights
        edge_relations = {(e.source_node_id, e.target_node_id, e.relation_type) for e in edges}
        assert (billing_node.node_id, monolith_node.node_id, "part_of") in edge_relations
        assert (billing_node.node_id, alice_node.node_id, "owned_by") in edge_relations

        # 3. Confirm degree_centrality was updated
        assert billing_node.degree_centrality == 2.0
