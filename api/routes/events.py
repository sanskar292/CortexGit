from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Dict, Any
from uuid import UUID
from datetime import datetime
import logging
import uuid

from cortexgit.db.database import AsyncSessionLocal
from cortexgit.core.event_log import EventLogger
from cortexgit.db.models import EventType

router = APIRouter()
logger = logging.getLogger(__name__)

class EventCreate(BaseModel):
    session_id: str
    agent_id: str
    event_type: str
    payload: Dict[str, Any]

class EventResponse(BaseModel):
    event_id: UUID
    session_id: str
    agent_id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True
        orm_mode = True

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def run_entity_extraction_pipeline(event_dict: dict, llm_provider=None):
    from cortexgit.llm.entity_extractor import extract_entities
    from cortexgit.core.conflict_detector import ConflictDetector
    from cortexgit.core.entity_registry import EntityRegistryHandler
    from cortexgit.core.write_back_gate import ValidationError
    
    try:
        extraction_result = await extract_entities(event_dict, llm_provider)
        
        async with AsyncSessionLocal() as session:
            detector = ConflictDetector(session)
            handler = EntityRegistryHandler(session)
            event_id = uuid.UUID(event_dict["event_id"])
            
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
                        proposed_event_id=event_id
                    )
                else:
                    await handler.write_entity(
                        key=key,
                        value=value,
                        agent_id=event_dict.get("agent_id", "unknown"),
                        event_id=event_id
                    )
            await session.commit()
    except ValidationError as e:
        logger.error(f"Flat entity extraction validation failed: {e}")
    except Exception as e:
        logger.error(f"Flat entity extraction pipeline failed: {e}")

async def run_reg_extraction_pipeline(event_dict: dict, llm_provider=None):
    from cortexgit.llm.entity_extractor import extract_reg_entities
    from cortexgit.graph.graph_repository import GraphRepository
    from cortexgit.graph.centrality import update_centrality
    from cortexgit.core.write_back_gate import ValidationError
    
    try:
        result = await extract_reg_entities(event_dict, llm_provider)
        
        async with AsyncSessionLocal() as session:
            repo = GraphRepository(session)
            
            for entity in result.get("updates", []):
                node_id = await repo.create_node(
                    entity_name=entity["entity_name"],
                    entity_type=entity["entity_type"],
                    description=entity.get("properties", {}).get("description"),
                    status=entity.get("properties", {}).get("status"),
                    agent_id=event_dict.get("agent_id", "unknown"),
                )
                
                for connection in entity.get("connected_to", []):
                    target_id = await repo.create_node(
                        entity_name=connection["target_entity"],
                        entity_type="concept",
                        description=None,
                        status=None,
                        agent_id=event_dict.get("agent_id", "unknown"),
                    )
                    await repo.create_edge(node_id, target_id, connection["relation_type"])
                
                # Update centrality for both endpoints — both gained a new edge
                await update_centrality(node_id, session)
                await update_centrality(target_id, session)
    except ValidationError as e:
        logger.error(f"REG extraction validation failed: {e}")
    except Exception as e:
        logger.error(f"REG extraction pipeline error: {e}")

async def run_snapshot_pipeline(session_id: str, llm_provider=None, embedding_provider=None):
    from cortexgit.llm.snapshot_trigger import should_snapshot
    from cortexgit.llm.summarizer import summarize, write_snapshot
    from cortexgit.db.models import EventLog
    from sqlalchemy import select
    
    try:
        async with AsyncSessionLocal() as session:
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
                
                summary_output = await summarize(events_list, llm_provider)
                await write_snapshot(session_id, summary_output, session, embedding_provider)
    except Exception as e:
        logger.error(f"Snapshot pipeline failed for session_id={session_id}: {e}")

@router.post("/events", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def post_event(
    event_in: EventCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db)
):
    try:
        try:
            # Check event type validation
            EventType(event_in.event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid event_type '{event_in.event_type}'"
            )
            
        logger_handler = EventLogger(session)
        event = await logger_handler.log_event(
            session_id=event_in.session_id,
            agent_id=event_in.agent_id,
            event_type=event_in.event_type.lower(),
            payload=event_in.payload
        )
        
        event_dict = {
            "event_id": str(event.event_id),
            "session_id": event.session_id,
            "agent_id": event.agent_id,
            "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
            "payload": event.payload,
            "created_at": event.created_at.isoformat() if event.created_at else None
        }
        
        background_tasks.add_task(run_entity_extraction_pipeline, event_dict)
        background_tasks.add_task(run_reg_extraction_pipeline, event_dict)
        background_tasks.add_task(run_snapshot_pipeline, event.session_id)
        
        return event
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to write event")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal database error: {str(e)}"
        )
