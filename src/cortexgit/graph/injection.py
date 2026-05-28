from typing import List, Set
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.models import EntityNode
from cortexgit.graph.importance import rank_nodes_by_importance

async def inject_high_mass_nodes(
    goal: str,
    agent_id: str,
    session: AsyncSession,
    k: int = 3,
    semantic_results: list = None,
) -> List[EntityNode]:
    """
    Get top-K important active nodes for the agent, filtering out those
    already mentioned/present in the semantic recall results.
    """
    # 1. Fetch top-K*2 nodes ranked by importance (over-fetch to allow for exclusion filtering)
    ranked_nodes = await rank_nodes_by_importance(agent_id, session, top_k=k * 2)

    # 2. Extract excluded entity names from semantic recall results (snapshots)
    exclude_names: Set[str] = set()
    if semantic_results:
        for snapshot in semantic_results:
            # handle both SnapshotStore ORM objects and serialised snapshot dicts
            if isinstance(snapshot, dict):
                entities_list = snapshot.get("entities_mentioned")
            else:
                entities_list = getattr(snapshot, "entities_mentioned", None)
            
            if entities_list:
                for entity in entities_list:
                    exclude_names.add(entity)

    # 3. Filter out excluded nodes
    filtered = [node for node in ranked_nodes if node.entity_name not in exclude_names]

    # 4. Return top K
    return filtered[:k]


async def inject_high_importance_nodes(
    goal: str,
    agent_id: str,
    session: AsyncSession,
    k: int = 3,
    semantic_results: list = None,
) -> List[EntityNode]:
    """
    Get top-K important active nodes for the agent, filtering out those
    already mentioned/present in the semantic recall results.
    Identical to inject_high_mass_nodes.
    """
    return await inject_high_mass_nodes(
        goal=goal,
        agent_id=agent_id,
        session=session,
        k=k,
        semantic_results=semantic_results,
    )

