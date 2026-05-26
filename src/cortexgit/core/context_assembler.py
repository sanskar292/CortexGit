# Core Context Assembler module (Phase 2)
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from cortexgit.db.models import ConflictLog
from cortexgit.core.recency_filter import RecencyFilter
from cortexgit.retrieval.semantic_recall import semantic_recall
from cortexgit.core.entity_pull import entity_pull


def serialize_conflict(c: ConflictLog) -> dict:
    return {
        "conflict_id": str(c.conflict_id),
        "key": c.key,
        "existing_value": c.existing_value,
        "proposed_value": c.proposed_value,
        "existing_event_id": str(c.existing_event_id),
        "proposed_event_id": str(c.proposed_event_id),
        "resolved": c.resolved,
        "created_at": c.created_at.isoformat() if c.created_at else None
    }

def serialize_event(e) -> dict:
    return {
        "event_id": str(e.event_id),
        "session_id": e.session_id,
        "agent_id": e.agent_id,
        "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
        "payload": e.payload,
        "created_at": e.created_at.isoformat() if e.created_at else None
    }

def serialize_snapshot(s) -> dict:
    event_range_val = None
    if s.event_range is not None:
        try:
            event_range_val = [s.event_range.lower, s.event_range.upper]
        except Exception:
            event_range_val = list(s.event_range)

    return {
        "snapshot_id": str(s.snapshot_id),
        "event_range": event_range_val,
        "summary": s.summary,
        "entities_mentioned": s.entities_mentioned,
        "created_at": s.created_at.isoformat() if s.created_at else None
    }

async def assemble(goal: str, session_id: str, budget_tokens: int, session: AsyncSession) -> dict:
    """
    Assembles context containing recent events, relevant snapshots, entities, and open conflicts.
    Enforces strict token budget limit. Priority order:
    1. Conflicts (first)
    2. Recent events (second)
    3. Snapshots (third)
    4. Entities (fourth)
    Estimates tokens as: len(json.dumps(item)) // 4
    Stops adding items when budget would be exceeded.
    Returns: { events: [], snapshots: [], entities: {}, conflicts: [] }
    """
    # 1. Fetch data from internal modules
    # Fetch open (unresolved) conflicts
    stmt = select(ConflictLog).where(ConflictLog.resolved == False)
    result = await session.execute(stmt)
    conflicts = list(result.scalars().all())
    # Sort conflicts deterministically by ID to guarantee same output for same inputs
    conflicts.sort(key=lambda c: str(c.conflict_id))

    # Fetch recent events
    rf = RecencyFilter(session)
    events = await rf.get_recent_events(session_id)

    # Fetch relevant snapshots
    snapshots = await semantic_recall(goal, session)

    # Fetch relevant entities
    entities = await entity_pull(goal, session)

    # 2. Pack results into the budget
    current_tokens = 0
    assembled_conflicts = []
    assembled_events = []
    assembled_snapshots = []
    assembled_entities = {}

    # Category 1: Conflicts (Priority 1)
    for c in conflicts:
        c_dict = serialize_conflict(c)
        tokens = len(json.dumps(c_dict)) // 4
        if current_tokens + tokens <= budget_tokens:
            assembled_conflicts.append(c_dict)
            current_tokens += tokens
        else:
            break

    # Category 2: Recent events (Priority 2)
    for e in events:
        e_dict = serialize_event(e)
        tokens = len(json.dumps(e_dict)) // 4
        if current_tokens + tokens <= budget_tokens:
            assembled_events.append(e_dict)
            current_tokens += tokens
        else:
            break

    # Category 3: Snapshots (Priority 3)
    for s in snapshots:
        s_dict = serialize_snapshot(s)
        tokens = len(json.dumps(s_dict)) // 4
        if current_tokens + tokens <= budget_tokens:
            assembled_snapshots.append(s_dict)
            current_tokens += tokens
        else:
            break

    # Category 4: Entities (Priority 4)
    # Sort keys deterministically to guarantee same inputs produce same output
    sorted_entity_keys = sorted(entities.keys())
    for key in sorted_entity_keys:
        val = entities[key]
        pair_dict = {key: val}
        tokens = len(json.dumps(pair_dict)) // 4
        if current_tokens + tokens <= budget_tokens:
            assembled_entities[key] = val
            current_tokens += tokens
        else:
            break

    return {
        "events": assembled_events,
        "snapshots": assembled_snapshots,
        "entities": assembled_entities,
        "conflicts": assembled_conflicts
    }

class ContextAssembler:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def assemble_context(self, session_id: str, goal: str, budget_tokens: int) -> dict:
        """Assembles context containing recent events, relevant snapshots, entities, and conflicts."""
        return await assemble(goal, session_id, budget_tokens, self.session)
