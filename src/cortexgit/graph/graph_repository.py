import uuid
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.models import EntityNode, EntityEdge, NodeHit, EntityNodeType, HitType

INITIAL_TTL_DAYS = int(os.getenv("INITIAL_TTL_DAYS", "7"))
INITIAL_TTL = timedelta(days=INITIAL_TTL_DAYS)


class GraphRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_node(
        self,
        entity_name: str,
        entity_type: str,
        description: Optional[str],
        status: Optional[str],
        agent_id: str,
    ) -> uuid.UUID:
        """
        Create a new entity node or return the existing one's ID (idempotent).
        Initializes degree_centrality to 0.0, hit_frequency to 0, and TTL default to 7 days.
        """
        # Checks if (entity_name, agent_id) pair already exists — scoped per agent
        result = await self.session.execute(
            select(EntityNode).where(
                EntityNode.entity_name == entity_name,
                EntityNode.agent_id == agent_id,
            )
        )
        existing_node = result.scalar_one_or_none()
        if existing_node:
            return existing_node.node_id

        # Create new node
        node_id = uuid.uuid4()
        node = EntityNode(
            node_id=node_id,
            entity_name=entity_name,
            entity_type=EntityNodeType(entity_type.lower()),
            description=description,
            status=status,
            degree_centrality=0.0,
            hit_frequency=0,
            ttl_expiry=datetime.now(timezone.utc) + INITIAL_TTL,
            created_at=datetime.now(timezone.utc),
            agent_id=agent_id,
        )
        self.session.add(node)
        await self.session.commit()
        return node_id

    async def create_edge(
        self,
        source_node_id: uuid.UUID,
        target_node_id: uuid.UUID,
        relation_type: str,
    ) -> uuid.UUID:
        """
        Create a new relationship edge or increment its weight if it already exists.
        Checks the unique constraint: (source_node_id, target_node_id, relation_type).
        """
        # Check unique constraint: (source, target, relation_type)
        result = await self.session.execute(
            select(EntityEdge).where(
                EntityEdge.source_node_id == source_node_id,
                EntityEdge.target_node_id == target_node_id,
                EntityEdge.relation_type == relation_type,
            )
        )
        existing_edge = result.scalar_one_or_none()
        if existing_edge:
            existing_edge.weight += 1.0
            await self.session.commit()
            return existing_edge.edge_id

        # Create new edge
        edge_id = uuid.uuid4()
        edge = EntityEdge(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            weight=1.0,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(edge)
        await self.session.commit()
        return edge_id

    async def record_hit(self, node_id: uuid.UUID, hit_type: str, session_id: str) -> None:
        """
        Record a traversal or retrieval hit on a node.
        Increments hit_frequency by 1, updates last_hit, creates a NodeHit record,
        and resets ttl_expiry to now + 7 days.
        """
        result = await self.session.execute(
            select(EntityNode).where(EntityNode.node_id == node_id)
        )
        node = result.scalar_one_or_none()
        if not node:
            raise ValueError(f"Node with ID '{node_id}' not found.")

        now = datetime.now(timezone.utc)
        
        # Increment frequency and update last_hit
        node.hit_frequency += 1
        node.last_hit = now
        
        # Reset ttl_expiry
        node.ttl_expiry = now + INITIAL_TTL

        # Create node hit record
        hit = NodeHit(
            hit_id=uuid.uuid4(),
            node_id=node_id,
            hit_type=HitType(hit_type.lower()),
            hit_timestamp=now,
            session_id=session_id,
        )
        self.session.add(hit)
        await self.session.commit()

    async def get_node(self, entity_name: str, agent_id: str = None) -> Optional[EntityNode]:
        """
        Fetch a node by its entity_name.
        When agent_id is provided (recommended) the lookup is scoped to that agent.
        Falls back to a global lookup when agent_id is None (legacy callers).
        Returns None if not found.
        """
        conditions = [EntityNode.entity_name == entity_name]
        if agent_id is not None:
            conditions.append(EntityNode.agent_id == agent_id)
        result = await self.session.execute(
            select(EntityNode).where(*conditions)
        )
        return result.scalar_one_or_none()

    async def get_nodes_by_agent(self, agent_id: str) -> List[EntityNode]:
        """
        Fetch all nodes associated with a specific agent.
        """
        result = await self.session.execute(
            select(EntityNode).where(EntityNode.agent_id == agent_id)
        )
        return list(result.scalars().all())

    async def delete_expired_nodes(self) -> int:
        """
        Delete all nodes where ttl_expiry is in the past.
        Cascades the deletion to edges and hits referencing those nodes.
        Returns the count of deleted nodes.

        Uses a lightweight scalar subquery to collect only node_ids, avoiding
        materialising full ORM objects into Python memory.
        """
        now = datetime.now(timezone.utc)

        # Collect expired node_ids without fetching full ORM rows
        id_result = await self.session.execute(
            select(EntityNode.node_id).where(EntityNode.ttl_expiry < now)
        )
        expired_ids = id_result.scalars().all()
        if not expired_ids:
            return 0

        # Cascade: edges
        await self.session.execute(
            delete(EntityEdge).where(
                (EntityEdge.source_node_id.in_(expired_ids)) |
                (EntityEdge.target_node_id.in_(expired_ids))
            )
        )

        # Cascade: hits
        await self.session.execute(
            delete(NodeHit).where(NodeHit.node_id.in_(expired_ids))
        )

        # Nodes
        await self.session.execute(
            delete(EntityNode).where(EntityNode.node_id.in_(expired_ids))
        )

        await self.session.commit()
        return len(expired_ids)
