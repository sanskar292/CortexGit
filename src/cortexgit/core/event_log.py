# Core event logger module (Phase 1)
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.models import EventLog, EventType


class EventLogger:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_event(self, session_id: str, agent_id: str, event_type: str, payload: dict) -> EventLog:
        """
        Append event to PostgreSQL event log.
        - Validates event_type against the EventType enum.
        - Automatically creates UUID and timestamp.
        """
        # 1. Validate event_type against model enum
        try:
            # EventType values are system, user, agent, action, observation, thought, error
            # Try to match the case-insensitive or direct string
            enum_event_type = EventType(event_type.lower())
        except ValueError:
            valid_types = [e.value for e in EventType]
            raise ValueError(
                f"Invalid event_type: '{event_type}'. Must be one of: {', '.join(valid_types)}"
            )

        # 2. Build the event model instance
        event = EventLog(
            event_id=uuid.uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            event_type=enum_event_type,
            payload=payload,
            created_at=datetime.now(timezone.utc)
        )

        # 3. Write to PostgreSQL database
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        
        return event
