# Core SDK memory module (Phase 3)
import os
import logging
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from cortexgit.db.database import AsyncSessionLocal, engine
from cortexgit.core.event_log import EventLogger, EventLog
from cortexgit.core.conflict_detector import ConflictDetector
from cortexgit.core.entity_registry import EntityRegistryHandler
from cortexgit.core.context_assembler import assemble
from cortexgit.llm.entity_extractor import extract_entities
from cortexgit.llm.snapshot_trigger import should_snapshot
from cortexgit.llm.summarizer import summarize, write_snapshot
from cortexgit.llm_providers import (
    LLMProvider,
    EmbeddingProvider,
    create_llm_provider,
    create_embedding_provider,
)

class ConflictError(Exception):
    """Exception raised when an entity write conflict is detected."""
    pass

class CortexGit:
    def __init__(
        self,
        database_url: str = None,
        llm_provider: LLMProvider = None,
        embedding_provider: EmbeddingProvider = None,
        enable_injection: bool = True,
        injection_threshold: float = None,
        injection_top_k: int = None
    ):
        """
        Initialize the CortexGit SDK persistent memory client.
        If database_url is provided, it configures a new engine.
        Otherwise, it uses the default engine configured from the environment.

        Args:
            database_url: Optional database connection URL. Defaults to DATABASE_URL env var.
            llm_provider: Optional custom LLM provider instance. Defaults to env-configured provider.
            embedding_provider: Optional custom embedding provider instance. Defaults to env-configured provider.
            enable_injection: Whether proactive surface injection is enabled. Defaults to True.
            injection_threshold: Optional minimum importance score for injection. Defaults to INJECTION_IMPORTANCE_THRESHOLD env var, else 5.0.
            injection_top_k: Optional maximum number of injected entities. Defaults to INJECTION_TOP_K env var, else 3.
        """
        if database_url:
            from sqlalchemy.pool import NullPool
            self.engine = create_async_engine(database_url, echo=False, poolclass=NullPool)
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        else:
            self.engine = engine
            self.session_factory = AsyncSessionLocal

        self._initialized = False

        # Set up LLM and Embedding Providers
        self.llm_provider = llm_provider or create_llm_provider(
            os.getenv("CORTEXGIT_LLM_PROVIDER") or (
                "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
            )
        )
        self.embedding_provider = embedding_provider or create_embedding_provider(
            os.getenv("CORTEXGIT_EMBEDDING_PROVIDER") or "openai"
        )

        self.enable_injection = enable_injection
        
        if injection_threshold is None:
            try:
                injection_threshold = float(os.getenv("INJECTION_IMPORTANCE_THRESHOLD", "5.0"))
            except ValueError:
                injection_threshold = 5.0
        self.injection_threshold = injection_threshold

        if injection_top_k is None:
            try:
                injection_top_k = int(os.getenv("INJECTION_TOP_K", "3"))
            except ValueError:
                injection_top_k = 3
        self.injection_top_k = injection_top_k


    async def _ensure_tables(self):
        """Create all tables if they don't exist yet."""
        if self._initialized:
            return
        from cortexgit.db.models import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True

    async def log_event(self, session_id: str, agent_id: str, event_type: str, payload: dict) -> EventLog:
        """
        Append an event to the persistent event log.
        Triggers the entity extraction and snapshot generation pipelines in the background.
        """
        await self._ensure_tables()
        async with self.session_factory() as session:
            logger = EventLogger(session)
            event = await logger.log_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type=event_type,
                payload=payload
            )
            
            # Run the extraction and snapshot background pipeline
            await self._run_background_pipeline(event, session_id, agent_id)
            
            return event

    async def _run_background_pipeline(self, event: EventLog, session_id: str, agent_id: str):
        # 1. Run Entity Extraction
        async with self.session_factory() as session:
            try:
                event_dict = {
                    "event_id": str(event.event_id),
                    "session_id": event.session_id,
                    "agent_id": event.agent_id,
                    "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat() if event.created_at else None
                }
                extraction_result = await extract_entities(event_dict, self.llm_provider)
                
                detector = ConflictDetector(session)
                handler = EntityRegistryHandler(session)
                
                for update in extraction_result.get("updates", []):
                    key = update["key"]
                    value = update["value"]
                    
                    conflict = await detector.detect_conflict(key, value)
                    if conflict:
                        await detector.log_conflict(
                            key=key,
                            existing_value=conflict.value,
                            proposed_value=value,
                            existing_event_id=conflict.event_id,
                            proposed_event_id=event.event_id
                        )
                    else:
                        await handler.write_entity(
                            key=key,
                            value=value,
                            agent_id=agent_id,
                            event_id=event.event_id
                        )
            except Exception:
                logging.getLogger(__name__).exception(
                    "[cortexgit] entity_extraction pipeline failed for event_id=%s session_id=%s",
                    str(event.event_id),
                    session_id,
                )

        # 2. Run Snapshot Trigger
        async with self.session_factory() as session:
            try:
                if await should_snapshot(session_id, session):
                    stmt = select(EventLog).where(EventLog.session_id == session_id).order_by(EventLog.created_at.asc())
                    res = await session.execute(stmt)
                    events = res.scalars().all()
                    
                    events_list = []
                    for e in events:
                         events_list.append({
                             "event_id": str(e.event_id),
                             "session_id": e.session_id,
                             "agent_id": e.agent_id,
                             "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
                             "payload": e.payload,
                             "created_at": e.created_at.isoformat() if e.created_at else None
                         })
                    
                    summary_output = await summarize(events_list, self.llm_provider)
                    await write_snapshot(session_id, summary_output, session, self.embedding_provider)
            except Exception:
                logging.getLogger(__name__).exception(
                    "[cortexgit] snapshot pipeline failed for session_id=%s",
                    session_id,
                )

    async def get_context(
        self,
        goal: str,
        budget_tokens: int,
        session_id: str,
        use_reg: bool = True,
        agent_id: str = None
    ) -> dict:
        """
        Retrieves context containing recent events, relevant snapshots, entities, and open conflicts
        packed cleanly under the budget token limit.

        Raises:
            ValueError: If goal or session_id are empty/whitespace, or if budget_tokens <= 0.
        """
        if not goal or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be greater than zero")
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        await self._ensure_tables()
        async with self.session_factory() as session:
            return await assemble(
                goal=goal,
                session_id=session_id,
                budget_tokens=budget_tokens,
                session=session,
                embedding_provider=self.embedding_provider,
                use_reg=use_reg,
                agent_id=agent_id,
                enable_injection=self.enable_injection,
                injection_threshold=self.injection_threshold,
                injection_top_k=self.injection_top_k,
            )

    async def write_entity(self, key: str, value: any, agent_id: str, event_id: str) -> bool:
        """
        Directly writes an entity to the EntityRegistry.
        Checks for conflicts first, logs conflict and raises ConflictError if one is found.
        """
        await self._ensure_tables()
        event_uuid = uuid.UUID(event_id) if isinstance(event_id, str) else event_id
        async with self.session_factory() as session:
            detector = ConflictDetector(session)
            handler = EntityRegistryHandler(session)
            
            conflict = await detector.detect_conflict(key, value)
            if conflict:
                await detector.log_conflict(
                    key=key,
                    existing_value=conflict.value,
                    proposed_value=value,
                    existing_event_id=conflict.event_id,
                    proposed_event_id=event_uuid
                )
                raise ConflictError(f"Conflict detected on key '{key}'")
            
            try:
                return await handler.write_entity(
                    key=key,
                    value=value,
                    agent_id=agent_id,
                    event_id=event_uuid
                )
            except Exception as e:
                # Handle unique constraint collisions gracefully (crucial for SQLite concurrency)
                await session.rollback()
                async with self.session_factory() as check_session:
                    check_detector = ConflictDetector(check_session)
                    conflict = await check_detector.detect_conflict(key, value)
                    if conflict:
                        await check_detector.log_conflict(
                            key=key,
                            existing_value=conflict.value,
                            proposed_value=value,
                            existing_event_id=conflict.event_id,
                            proposed_event_id=event_uuid
                        )
                        raise ConflictError(f"Conflict detected on key '{key}'")
                raise e
