# Core entity registry module (Phase 1)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from datetime import datetime, timezone
from cortexgit.db.models import EntityRegistry, EventLog, EventType


class EntityRegistryHandler:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def write_entity(self, key: str, value: any, agent_id: str, event_id: uuid.UUID) -> bool:
        """
        Write entity cleanly to the registry.
        Assumes conflict check has already passed.
        Returns True if a new write occurred, False if it was idempotent.
        """
        # Check if already exists (for idempotency checks)
        result = await self.session.execute(
            select(EntityRegistry).where(EntityRegistry.key == key).with_for_update()
        )
        existing_entity = result.scalar_one_or_none()

        if existing_entity is not None:
            # Same key, same value -> idempotent success, do nothing
            if existing_entity.value == value:
                return False
            else:
                # Differing value is a conflict; should have been caught in detector
                raise ValueError(
                    f"Entity registry write collision on key '{key}' with differing value."
                )

        # Clean write
        entity = EntityRegistry(
            key=key,
            value=value,
            agent_id=agent_id,
            event_id=event_id,
            updated_at=datetime.now(timezone.utc)
        )
        self.session.add(entity)
        await self.session.commit()
        return True
