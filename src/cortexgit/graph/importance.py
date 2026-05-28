import os
import uuid
from datetime import datetime, timezone
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.models import EntityNode

HIT_FREQUENCY_WEIGHT = float(os.getenv("HIT_FREQUENCY_WEIGHT", "1.0"))
DEGREE_CENTRALITY_WEIGHT = float(os.getenv("DEGREE_CENTRALITY_WEIGHT", "1.0"))


async def calculate_importance(node_id: uuid.UUID, session: AsyncSession) -> float:
    """
    Fetch node from entity_nodes and calculate its importance.
    importance = (degree_centrality * DEGREE_CENTRALITY_WEIGHT) * (hit_frequency * HIT_FREQUENCY_WEIGHT)
    """
    result = await session.execute(
        select(EntityNode).where(EntityNode.node_id == node_id)
    )
    node = result.scalar_one_or_none()
    if not node:
        raise ValueError(f"Node with ID '{node_id}' not found.")

    deg_weighted = float(node.degree_centrality) * DEGREE_CENTRALITY_WEIGHT
    hit_weighted = float(node.hit_frequency) * HIT_FREQUENCY_WEIGHT
    return deg_weighted * hit_weighted


async def rank_nodes_by_importance(
    agent_id: str,
    session: AsyncSession,
    top_k: int = None,
) -> List[EntityNode]:
    """
    Fetch all active nodes (ttl_expiry > now) for agent.
    Sort by weighted importance (degree_centrality * hit_frequency) descending.

    If top_k is provided, only the top-k results are returned — avoids sorting the
    entire node set when the caller only needs the highest-ranked entries.
    """
    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(EntityNode).where(
            EntityNode.agent_id == agent_id,
            EntityNode.ttl_expiry > now
        )
    )
    nodes = list(result.scalars().all())

    # Sort descending by weighted importance
    nodes.sort(
        key=lambda n: (float(n.degree_centrality) * DEGREE_CENTRALITY_WEIGHT) * (float(n.hit_frequency) * HIT_FREQUENCY_WEIGHT),
        reverse=True
    )
    return nodes[:top_k] if top_k is not None else nodes


async def get_top_k_important_nodes(
    agent_id: str, k: int, session: AsyncSession
) -> List[EntityNode]:
    """
    Return the top-k importance-ranked active nodes for agent.
    Delegates to rank_nodes_by_importance with top_k set for efficiency.
    """
    return await rank_nodes_by_importance(agent_id, session, top_k=k)


async def get_high_importance_nodes(
    agent_id: str,
    threshold: float,
    session: AsyncSession,
    top_k: int = None,
) -> List[EntityNode]:
    """
    Fetches all active nodes for the agent where weighted importance is strictly greater than the threshold.
    The returned list is sorted by importance in descending order.
    If top_k is provided, limits the returned list to at most top_k nodes.
    """
    # 1. Fetch and rank all active nodes for this agent
    nodes = await rank_nodes_by_importance(agent_id, session)

    # 2. Filter nodes where importance > threshold
    filtered_nodes = []
    for node in nodes:
        importance = (
            float(node.degree_centrality) * DEGREE_CENTRALITY_WEIGHT
        ) * (
            float(node.hit_frequency) * HIT_FREQUENCY_WEIGHT
        )
        if importance > threshold:
            filtered_nodes.append(node)

    # 3. Limit to top_k if specified
    return filtered_nodes[:top_k] if top_k is not None else filtered_nodes
