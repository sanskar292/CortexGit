# Core Recency Filter module (Phase 2)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from cortexgit.db.models import EventLog

class RecencyFilter:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_recent_events(self, session_id: str, k: int = 20) -> list[EventLog]:
        """
        Fetches the last k events from the event log for a given session.
        Returns events in chronological order (oldest to newest).
        Returns an empty list if no events exist for the session.
        """
        stmt = (
            select(EventLog)
            .where(EventLog.session_id == session_id)
            .order_by(EventLog.created_at.desc())
            .limit(k)
        )
        result = await self.session.execute(stmt)
        events = list(result.scalars().all())
        
        # Since they were loaded from most recent to oldest (desc),
        # reverse the list to restore chronological order (oldest to newest).
        events.reverse()
        return events
