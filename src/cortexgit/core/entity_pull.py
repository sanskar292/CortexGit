# Core Entity Pull module (Phase 2)
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from cortexgit.db.models import EntityRegistry

async def entity_pull(goal: str, session: AsyncSession) -> dict:
    """
    Tokenizes the goal string by splitting on spaces and punctuation.
    Returns all entity registry entries whose key contains any goal token.
    Returns a dict of key -> value.
    Relevance is simple substring match — no LLM, no embeddings.
    Returns empty dict if no matches.
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

    return matched

class EntityPull:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def pull_relevant_entities(self, goal: str) -> dict:
        """Fetches entity registry keys relevant to the goal string using simple substring match."""
        return await entity_pull(goal, self.session)
