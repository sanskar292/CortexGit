# Core Entity Pull module (Phase 2)
import re
import asyncio
import logging
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from cortexgit.db.models import EntityRegistry, EntityNode
from cortexgit.graph.importance import HIT_FREQUENCY_WEIGHT, DEGREE_CENTRALITY_WEIGHT

async def record_hit_in_background(entity_name: str, session_id: str):
    """
    Non-blocking background helper to fetch an entity node and record a retrieval hit.
    Uses a fresh AsyncSessionLocal transaction to prevent request session collisions.
    """
    from cortexgit.db.database import AsyncSessionLocal
    from cortexgit.graph.graph_repository import GraphRepository
    
    logger = logging.getLogger(__name__)
    try:
        async with AsyncSessionLocal() as session:
            repo = GraphRepository(session)
            node = await repo.get_node(entity_name)
            if node:
                await repo.record_hit(node.node_id, hit_type="query", session_id=session_id)
    except Exception as e:
        logger.exception(f"Failed to record hit in background for entity '{entity_name}': {e}")

async def entity_pull(goal: str, session: AsyncSession, session_id: str = None) -> dict:
    """
    Tokenizes the goal string by splitting on spaces and punctuation.
    Returns all entity registry entries whose key contains any goal token.
    Returns a dict of key -> value.
    Relevance is simple substring match — no LLM, no embeddings.
    Returns empty dict if no matches.
    
    If session_id is provided, automatically schedules background hit reinforcement tasks
    for all retrieved entity graph nodes.
    """
    # 1. Tokenize the goal string (extracting alphanumeric words, lowercased)
    tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', goal) if t]
    if not tokens:
        return {}

    # 2. Query all entity registry entries
    stmt = select(EntityRegistry)
    result = await session.execute(stmt)
    entities = result.scalars().all()

    # 3. Filter entries based on case-insensitive substring match
    matched = {}
    for entity in entities:
        key_lower = entity.key.lower()
        if any(token in key_lower for token in tokens):
            matched[entity.key] = entity.value

    # 4. Trigger graph node retrieval reinforcement in the background
    if session_id and matched:
        for key in matched.keys():
            asyncio.create_task(record_hit_in_background(key, session_id))

    return matched

DEFAULT_TOP_K = 5


async def entity_pull_with_reg(
    goal: str,
    agent_id: str,
    session: AsyncSession,
    session_id: str = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    REG-powered entity retrieval (Layer 3 upgrade).
    
    1. Tokenizes goal string using the same regex as entity_pull().
    2. Matches tokens against entity_nodes.entity_name (case-insensitive substring).
    3. Calculates importance = degree_centrality × hit_frequency for each match.
    4. Sorts matched nodes by importance descending.
    5. Returns top_k nodes as a list of dicts.
    
    Schedules background hit reinforcement for returned nodes when session_id is provided.
    """
    logger = logging.getLogger(__name__)

    # 1. Tokenize goal
    tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', goal) if t]
    if not tokens:
        return []

    # 2. Query entity_nodes for the given agent
    result = await session.execute(
        select(EntityNode).where(EntityNode.agent_id == agent_id)
    )
    nodes = result.scalars().all()

    # 3. Filter by substring match on entity_name
    matched = [
        node for node in nodes
        if any(token in node.entity_name.lower() for token in tokens)
    ]

    # 4. Sort by weighted importance (consistent with rank_nodes_by_importance)
    matched.sort(
        key=lambda n: (
            float(n.degree_centrality) * DEGREE_CENTRALITY_WEIGHT
        ) * (
            float(n.hit_frequency) * HIT_FREQUENCY_WEIGHT
        ),
        reverse=True,
    )

    # 5. Slice top_k
    top_nodes = matched[:top_k]

    # 6. Trigger hit reinforcement in background for all returned nodes
    if session_id and top_nodes:
        for node in top_nodes:
            asyncio.create_task(
                record_hit_in_background(node.entity_name, session_id)
            )

    # 7. Serialize to dicts
    return [
        {
            "node_id": str(node.node_id),
            "entity_name": node.entity_name,
            "entity_type": node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
            "description": node.description,
            "status": node.status,
            "degree_centrality": float(node.degree_centrality),
            "hit_frequency": node.hit_frequency,
            "importance": (
                float(node.degree_centrality) * DEGREE_CENTRALITY_WEIGHT
            ) * (
                float(node.hit_frequency) * HIT_FREQUENCY_WEIGHT
            ),
        }
        for node in top_nodes
    ]


class EntityPull:
    def __init__(self, session: AsyncSession, session_id: str = None):
        self.session = session
        self.session_id = session_id

    async def pull_relevant_entities(self, goal: str) -> dict:
        """Fetches entity registry keys relevant to the goal string using simple substring match."""
        return await entity_pull(goal, self.session, session_id=self.session_id)
