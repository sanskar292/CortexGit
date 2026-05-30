# Core conflict detector module (Phase 1)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from typing import Any
import uuid
from datetime import datetime, timezone
from cortexgit.db.models import EntityRegistry, ConflictLog

class ConflictDetector:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def detect_conflict(self, key: str, proposed_value: Any) -> EntityRegistry | None:
        """
        Check if key exists in EntityRegistry.
        If it exists and value differs, return the existing registry record (indicating a conflict).
        If it does not exist, or exists with the exact same value, return None (no conflict).

        Note: WITH FOR UPDATE locks an existing row but cannot prevent a concurrent INSERT on a
        missing key. The IntegrityError path in EntityRegistryHandler.write_entity handles that
        race and must log the conflict rather than silently discarding the write.
        """
        result = await self.session.execute(
            select(EntityRegistry).where(EntityRegistry.key == key).with_for_update()
        )
        existing_entity = result.scalar_one_or_none()

        if existing_entity is not None:
            if existing_entity.value != proposed_value:
                return existing_entity  # Conflict detected!
        return None

    async def log_conflict(
        self,
        key: str,
        existing_value: Any,
        proposed_value: Any,
        existing_event_id: uuid.UUID,
        proposed_event_id: uuid.UUID
    ) -> ConflictLog:
        """Log the conflict to ConflictLog."""
        conflict = ConflictLog(
            conflict_id=uuid.uuid4(),
            key=key,
            existing_value=existing_value,
            proposed_value=proposed_value,
            existing_event_id=existing_event_id,
            proposed_event_id=proposed_event_id,
            resolved=False,
            created_at=datetime.now(timezone.utc)
        )
        self.session.add(conflict)
        await self.session.commit()
        await self.session.refresh(conflict)
        return conflict
