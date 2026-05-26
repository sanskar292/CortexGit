import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from cortexgit.db.models import SnapshotStore, EventLog

async def should_snapshot(session_id: str, db: AsyncSession, threshold: int = None) -> bool:
    """
    Counts events since the last snapshot for this session.
    Returns True if the count is greater than or equal to the threshold (default 50).
    
    This function is pure read — no writes, no LLM calls.
    """
    if threshold is None:
        threshold_str = os.getenv("SNAPSHOT_THRESHOLD", "50")
        try:
            threshold = int(threshold_str)
        except ValueError:
            threshold = 50

    # 1. Count the total number of events for this session
    stmt_count = (
        select(func.count(EventLog.event_id))
        .where(EventLog.session_id == session_id)
    )
    res_count = await db.execute(stmt_count)
    total_events = res_count.scalar() or 0

    # 2. Get the latest snapshot for this session
    stmt_snap = (
        select(SnapshotStore)
        .where(SnapshotStore.session_id == session_id)
        .order_by(SnapshotStore.created_at.desc())
        .limit(1)
    )
    res_snap = await db.execute(stmt_snap)
    latest_snapshot = res_snap.scalars().first()

    # 3. Determine the number of events since the last snapshot
    if latest_snapshot is None:
        new_events = total_events
    else:
        event_range = latest_snapshot.event_range
        last_event_index = 0
        if event_range is not None:
            if hasattr(event_range, "upper") and not isinstance(event_range, str):
                upper = event_range.upper
                upper_inc = getattr(event_range, "upper_inc", False)
                # If upper is None (unbounded upper), treat last_event_index as total_events
                if upper is None:
                    last_event_index = total_events
                else:
                    last_event_index = upper if upper_inc else (upper - 1)
            else:
                try:
                    if isinstance(event_range, str):
                        parts = event_range.split(",")
                        last_event_index = int(parts[1]) - 1
                    else:
                        last_event_index = int(event_range[1]) - 1
                except Exception:
                    last_event_index = 0
        
        new_events = max(0, total_events - last_event_index)

    return new_events >= threshold


class SnapshotTrigger:
    def __init__(self, session: AsyncSession, trigger_limit: int = None):
        self.session = session
        if trigger_limit is None:
            threshold_str = os.getenv("SNAPSHOT_THRESHOLD", "50")
            try:
                self.trigger_limit = int(threshold_str)
            except ValueError:
                self.trigger_limit = 50
        else:
            self.trigger_limit = trigger_limit

    async def check_trigger(self, session_id: str) -> bool:
        """Determines if the session events count crosses trigger_limit since last snapshot."""
        return await should_snapshot(session_id, self.session, self.trigger_limit)
