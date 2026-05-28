import uuid
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.models import EntityNode, EntityEdge


async def calculate_degree_centrality(node_id: uuid.UUID, session: AsyncSession) -> float:
    """
    Count all edges where node_id is the source OR target.
    Returns the raw degree centrality as a float.
    """
    result = await session.execute(
        select(func.count(EntityEdge.edge_id)).where(
            or_(
                EntityEdge.source_node_id == node_id,
                EntityEdge.target_node_id == node_id
            )
        )
    )
    count = result.scalar() or 0
    return float(count)


async def update_centrality(node_id: uuid.UUID, session: AsyncSession) -> float:
    """
    Calculate the degree centrality for a node, persist it to the database,
    and return the updated value.
    """
    centrality = await calculate_degree_centrality(node_id, session)
    result = await session.execute(
        select(EntityNode).where(EntityNode.node_id == node_id)
    )
    node = result.scalar_one_or_none()
    if node:
        node.degree_centrality = centrality
        await session.commit()
    return centrality


async def recalculate_all_centrality(session: AsyncSession) -> int:
    """
    Recalculate and update the degree centrality of all registered entity nodes.
    Returns the total count of nodes updated.
    """
    result = await session.execute(select(EntityNode))
    nodes = result.scalars().all()
    
    count = 0
    for node in nodes:
        await update_centrality(node.node_id, session)
        count += 1
        
    return count
