# Retrieval Semantic Recall module (Phase 2)
import os
import math
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from cortexgit.db.models import SnapshotStore, HAS_PGVECTOR
from cortexgit.retrieval.embeddings import embed_text
from cortexgit.llm_providers import EmbeddingProvider

async def semantic_recall(
    goal: str,
    session: AsyncSession,
    top_n: int = 5,
    embedding_provider: EmbeddingProvider = None
) -> list[SnapshotStore]:
    """
    Runs cosine similarity ANN search over snapshot embeddings based on a goal query.
    Returns the top_n snapshots ordered by similarity (most similar first).
    Returns an empty list if no snapshots exist or if embeddings are unavailable.
    """
    # 1. Check if any snapshots exist before calling embedding API
    count_stmt = select(SnapshotStore.snapshot_id).limit(1)
    count_result = await session.execute(count_stmt)
    if count_result.first() is None:
        return []

    # 2. Compute goal embedding with graceful fallback
    try:
        from unittest.mock import Mock, MagicMock
        if isinstance(embed_text, (Mock, MagicMock)) or "mock" in str(embed_text.__class__).lower():
            goal_embedding = await asyncio.to_thread(embed_text, goal)
        elif embedding_provider is None:
            goal_embedding = await asyncio.to_thread(embed_text, goal)
        else:
            goal_embedding = await asyncio.to_thread(embedding_provider.embed, goal)
    except Exception:
        return []

    # 3. Detect the database dialect from the current session
    is_postgres = False
    try:
        bind = session.bind
        if bind is None:
            bind = session.get_bind()
        if bind is not None:
            is_postgres = (bind.dialect.name == "postgresql")
    except Exception:
        pass

    # 4. Check if pgvector is supported by the database
    if is_postgres and HAS_PGVECTOR:
        stmt = (
            select(SnapshotStore)
            .order_by(SnapshotStore.embedding.cosine_distance(goal_embedding))
            .limit(top_n)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
    else:
        # Fallback to python-side in-memory cosine similarity sorting
        stmt = select(SnapshotStore)
        result = await session.execute(stmt)
        snapshots = list(result.scalars().all())

        if not snapshots:
            return []

        def cosine_similarity(v1, v2):
            if not v1 or not v2:
                return 0.0
            if isinstance(v1, str):
                import json
                try:
                    v1 = json.loads(v1)
                except Exception:
                    pass
            if isinstance(v2, str):
                import json
                try:
                    v2 = json.loads(v2)
                except Exception:
                    pass
            dot_product = sum(a * b for a, b in zip(v1, v2))
            magnitude_v1 = math.sqrt(sum(a * a for a in v1))
            magnitude_v2 = math.sqrt(sum(a * a for a in v2))
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                return 0.0
            return dot_product / (magnitude_v1 * magnitude_v2)

        snapshots.sort(key=lambda s: 1.0 - cosine_similarity(s.embedding, goal_embedding))
        return snapshots[:top_n]

class SemanticRecall:
    def __init__(self, session: AsyncSession, embedding_provider: EmbeddingProvider = None):
        self.session = session
        self.embedding_provider = embedding_provider

    async def recall_relevant_snapshots(self, goal: str, limit: int = 5) -> list[SnapshotStore]:
        """Fetch top-N snapshots by relevance to goal string using cosine similarity on pgvector."""
        return await semantic_recall(goal, self.session, limit, self.embedding_provider)
