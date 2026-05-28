import os
import logging
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.database import AsyncSessionLocal
from cortexgit.graph.graph_repository import GraphRepository

logger = logging.getLogger(__name__)

async def expire_old_nodes(session: AsyncSession = None) -> int:
    """
    Remove nodes whose TTL has expired from the Relational Entity Graph.
    Does NOT affect raw events in the CortexGit append-only Event Log.
    Cascades deletions to entity edges automatically.
    Returns the count of deleted nodes.
    """
    if session is not None:
        repo = GraphRepository(session)
        deleted_count = await repo.delete_expired_nodes()
        if deleted_count > 0:
            logger.info(f"REG Expiration: successfully evicted {deleted_count} expired nodes.")
        return deleted_count

    async with AsyncSessionLocal() as session:
        repo = GraphRepository(session)
        deleted_count = await repo.delete_expired_nodes()
        if deleted_count > 0:
            logger.info(f"REG Expiration: successfully evicted {deleted_count} expired nodes.")
        return deleted_count

async def start_background_expiration():
    """
    Asynchronous background task loop that runs continuously in the FastAPI application.
    Evicts expired entity nodes at a configurable hourly interval.
    """
    # Configurable schedule interval, defaults to 3600 seconds (1 hour)
    interval = int(os.getenv("EXPIRATION_INTERVAL_SECONDS", "3600"))
    logger.info(f"REG Expiration Scheduler: background cleanup task started (interval: {interval}s).")
    
    while True:
        try:
            await asyncio.sleep(interval)
            deleted_count = await expire_old_nodes()
            if deleted_count > 0:
                logger.info(f"REG Expiration Scheduler: successfully cleared {deleted_count} expired nodes.")
        except asyncio.CancelledError:
            logger.info("REG Expiration Scheduler: background cleanup task canceled/stopped.")
            break
        except Exception as e:
            logger.exception(f"REG Expiration Scheduler: error encountered in cleanup loop: {e}")
