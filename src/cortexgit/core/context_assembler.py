# Core Context Assembler module (Phase 2)
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from cortexgit.db.models import ConflictLog
from cortexgit.core.recency_filter import RecencyFilter
from cortexgit.retrieval.semantic_recall import semantic_recall
from cortexgit.core.entity_pull import entity_pull, entity_pull_with_reg
from cortexgit.llm_providers import EmbeddingProvider


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

async def assemble(
    goal: str,
    session_id: str,
    budget_tokens: int,
    session: AsyncSession,
    embedding_provider: EmbeddingProvider = None,
    use_reg: bool = True,
    agent_id: str = None,
    enable_injection: bool = True,
    injection_threshold: float = None,
    injection_top_k: int = None,
) -> dict:
    """
    Assembles context containing recent events, relevant snapshots, entities, and open conflicts.
    Enforces strict token budget limit.

    Legacy priority order (use_reg=False):
      1. Conflicts
      2. Recent events
      3. Snapshots
      4. Entities (alphabetically sorted)

    REG priority order (use_reg=True, default in v0.2.0):
      1. Conflicts
      2. Recent events
      3. High-importance entities  (importance > 10)
      4. Snapshots
      5. Medium-importance entities (5 <= importance <= 10)
      6. Low-importance entities   (importance < 5)

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
    snapshots = await semantic_recall(goal, session, embedding_provider=embedding_provider)

    # Fetch relevant entities — REG-powered or legacy
    if use_reg and agent_id:
        # Returns list[dict] sorted by importance descending
        entities_reg = await entity_pull_with_reg(
            goal, agent_id, session, session_id=session_id
        )
        entities = None  # not used in REG path
    else:
        entities = await entity_pull(goal, session, session_id=session_id)
        entities_reg = None  # not used in legacy path

    # Retrieve injected high-importance nodes
    injected_nodes = []
    if use_reg and agent_id and enable_injection:
        from cortexgit.graph.injection import inject_high_importance_nodes
        k_val = injection_top_k if injection_top_k is not None else 3
        injected_nodes = await inject_high_importance_nodes(
            goal=goal,
            agent_id=agent_id,
            session=session,
            k=k_val,
            semantic_results=snapshots,
        )
        if injection_threshold is not None:
            injected_nodes = [
                node for node in injected_nodes
                if (float(node.degree_centrality) * float(node.hit_frequency)) > injection_threshold
            ]

    # 2. Pack results into the budget
    current_tokens = 0
    assembled_conflicts = []
    assembled_events = []
    assembled_snapshots = []
    assembled_entities = {}
    injected_names = []

    # Category 1: Conflicts (Priority 1 — unchanged in both modes)
    for c in conflicts:
        c_dict = serialize_conflict(c)
        tokens = len(json.dumps(c_dict)) // 4
        if current_tokens + tokens <= budget_tokens:
            assembled_conflicts.append(c_dict)
            current_tokens += tokens
        else:
            break

    # Category 2: Recent events (Priority 2 — unchanged in both modes)
    for e in events:
        e_dict = serialize_event(e)
        tokens = len(json.dumps(e_dict)) // 4
        if current_tokens + tokens <= budget_tokens:
            assembled_events.append(e_dict)
            current_tokens += tokens
        else:
            break

    if use_reg and entities_reg is not None:
        # ----------------------------------------------------------------
        # REG packing path — importance-weighted priority order:
        #   Priority 3: High-importance entities from entity_pull (importance > 10)
        #   Priority 4: Snapshots
        #   Priority 5: Injected high-importance nodes (if budget allows)
        # ----------------------------------------------------------------
        HIGH_THRESHOLD = 10.0

        # entity_pull_with_reg() already returns nodes sorted descending by importance.
        # Partition into tiers while preserving that inner order.
        high_entities = [n for n in entities_reg if n["importance"] > HIGH_THRESHOLD]

        def pack_reg_entities(entity_list: list) -> None:
            """Pack REG entity dicts into assembled_entities within budget."""
            nonlocal current_tokens
            for node_dict in entity_list:
                tokens = len(json.dumps({node_dict["entity_name"]: node_dict})) // 4
                if current_tokens + tokens <= budget_tokens:
                    assembled_entities[node_dict["entity_name"]] = node_dict
                    current_tokens += tokens
                else:
                    break

        # Priority 3: High-importance entities (packed before snapshots)
        pack_reg_entities(high_entities)

        # Priority 4: Snapshots
        for s in snapshots:
            s_dict = serialize_snapshot(s)
            tokens = len(json.dumps(s_dict)) // 4
            if current_tokens + tokens <= budget_tokens:
                assembled_snapshots.append(s_dict)
                current_tokens += tokens
            else:
                break

        # Priority 5: Injected high-importance nodes (if budget allows)
        for node in injected_nodes:
            if node.entity_name in assembled_entities:
                continue
            node_dict = {
                "node_id": str(node.node_id),
                "entity_name": node.entity_name,
                "entity_type": node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
                "description": node.description,
                "status": node.status,
                "degree_centrality": float(node.degree_centrality),
                "hit_frequency": node.hit_frequency,
                "importance": float(node.degree_centrality) * float(node.hit_frequency),
                "injected": True,
            }
            tokens = len(json.dumps({node.entity_name: node_dict})) // 4
            if current_tokens + tokens <= budget_tokens:
                assembled_entities[node.entity_name] = node_dict
                current_tokens += tokens
                injected_names.append(node.entity_name)
            else:
                break


    else:
        # ----------------------------------------------------------------
        # Legacy packing path — original order (completely unchanged):
        #   Priority 3: Snapshots
        #   Priority 4: Entities (alphabetically sorted)
        # ----------------------------------------------------------------

        # Category 3: Snapshots (Priority 3)
        for s in snapshots:
            s_dict = serialize_snapshot(s)
            tokens = len(json.dumps(s_dict)) // 4
            if current_tokens + tokens <= budget_tokens:
                assembled_snapshots.append(s_dict)
                current_tokens += tokens
            else:
                break

        # Category 4: Entities (Priority 4) — sort keys deterministically
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

    # Reinforce hit recording in background for injected entities
    if session_id and assembled_entities:
        import asyncio
        from cortexgit.core.entity_pull import record_hit_in_background

        async def record_hit_with_delay(key, session_id):
            await asyncio.sleep(0.1)
            await record_hit_in_background(key, session_id)

        for key in assembled_entities.keys():
            asyncio.create_task(record_hit_with_delay(key, session_id))

    ret = {
        "events": assembled_events,
        "snapshots": assembled_snapshots,
        "entities": assembled_entities,
        "conflicts": assembled_conflicts,
    }
    if agent_id is not None:
        ret["metadata"] = {
            "injected_entities": injected_names
        }
    return ret

class ContextAssembler:
    def __init__(self, session: AsyncSession, embedding_provider: EmbeddingProvider = None):
        self.session = session
        self.embedding_provider = embedding_provider

    async def assemble_context(
        self,
        session_id: str,
        goal: str,
        budget_tokens: int,
        use_reg: bool = True,
        agent_id: str = None,
        enable_injection: bool = True,
        injection_threshold: float = None,
        injection_top_k: int = None,
    ) -> dict:
        """Assembles context containing recent events, relevant snapshots, entities, and conflicts.

        REG mode (use_reg=True, default in v0.2.0):
          Entities are importance-weighted: high (>10) packed before snapshots,
          medium (5–10) and low (<5) packed after snapshots.

        Legacy mode (use_reg=False):
          Falls back to original entity_registry substring match with alphabetical packing.

        agent_id is required when use_reg=True; ignored in legacy mode.
        """
        return await assemble(
            goal,
            session_id,
            budget_tokens,
            self.session,
            self.embedding_provider,
            use_reg=use_reg,
            agent_id=agent_id,
            enable_injection=enable_injection,
            injection_threshold=injection_threshold,
            injection_top_k=injection_top_k,
        )
