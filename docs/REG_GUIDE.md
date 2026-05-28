# Relational Entity Graph (REG) — Developer Guide

This guide describes how to configure, use, and monitor the **Relational Entity Graph (REG)** feature in CortexGit (introduced in version `v0.2.0`).

---

## 1. What is REG?

The **Relational Entity Graph (REG)** is a persistent graph memory system designed for LLM agents. Rather than relying on computationally heavy, unstructured, or unreliable semantic vector models alone, REG structures your agent's memory using standard Graph/DBMS principles:

* **Nodes**: Named entities extracted from agent events (projects, concepts, users, decisions).
* **Edges**: Causal and structural links formed automatically when two entities co-occur in the same context or event window.
* **Weights**: edges increment in strength on repeated co-occurrence.
* **Eviction**: Old, unreferenced nodes expire automatically via deterministic LRU and TTL bounds, leaving the core append-only Event Log intact but keeping the active context window clean.

---

## 2. How Does REG Improve Retrieval?

By structuring entity memories as a network, REG upgrades Layer 3 (Retrieval) to use **1st-Degree Neighbor Traversal** and **Importance-Weighted context packing**:

1. **Topology-based Importance**: Entity importance is calculated deterministically at query time using graph metrics rather than slow, fuzzy LLM judgments:
   $$\text{Importance} = (\text{Degree Centrality} \times \text{Centrality Weight}) \times (\text{Hit Frequency} \times \text{Hit Weight})$$
2. **Proactive Surface Injection**: High-importance entities are surfaced into context, even if they aren't semantically matched to the query, giving the agent a reliable sense of "core context."
3. **Structured Priority Packing**:
   * **Priority 3 — High Importance ($>10$)**: Packed *before* snapshots to ensure critical entities are never displaced.
   * **Priority 5 — Medium Importance ($5\text{–}10$)**: Packed after snapshots.
   * **Priority 6 — Low Importance ($<5$)**: Packed last.

---

## 3. How to Use REG in Your Agent Code

REG is fully integrated and **enabled by default** in CortexGit `v0.2.0` when `agent_id` is supplied.

### Initializing and Logging Events
```python
from cortexgit import CortexGit

# Initialize memory (loads variables from .env by default)
memory = CortexGit()

# Log events as usual. Entity extraction and graph topology pipelines
# are triggered in the background automatically.
await memory.log_event(
    session_id="session-123",
    agent_id="cortex-planner-v1",
    event_type="user",
    payload={"text": "Starting migration on ProjectX. Assign developer Person1 to the project."}
)
```

### Retrieving Assembled Context
When querying context, the assembler automatically incorporates importance-ranked entities and applies proactive surface injection:

```python
context = await memory.get_context(
    goal="Tell me about ProjectX",
    session_id="session-123",
    budget_tokens=8000
)

# Access prioritised entities
for key, entity in context["entities"].items():
    print(f"Entity: {key}")
    print(f"  - Importance: {entity['importance']}")
    print(f"  - Injected Proactively: {entity.get('injected', False)}")
```

---

## 4. Configuration Options

You can tune the REG mechanics by setting the following environment variables in your `.env` file:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `INITIAL_TTL_DAYS` | `7` | Base Time-To-Live (TTL) for new entity nodes (in days). |
| `HIT_FREQUENCY_WEIGHT` | `1.0` | Tuning weight multiplier for the hit frequency component. |
| `DEGREE_CENTRALITY_WEIGHT` | `1.0` | Tuning weight multiplier for the degree centrality component. |

> [!TIP]
> Increase `HIT_FREQUENCY_WEIGHT` to prioritize entities that are frequently queried by the agent. 
> Increase `DEGREE_CENTRALITY_WEIGHT` to prioritize highly connected core domain concepts and project scopes.

---

## 5. Monitoring & Direct Access

If you need direct programmatic control, you can import graph utilities directly from the package root:

```python
from cortexgit import (
    GraphRepository,
    calculate_importance,
    rank_nodes_by_importance,
    expire_old_nodes
)
```

### Querying Entity Importance
To query the dynamic importance score of a specific node:

```python
async with memory.session_factory() as session:
    repo = GraphRepository(session)
    node = await repo.get_node("ProjectX")
    if node:
        importance = await calculate_importance(node.node_id, session)
        print(f"ProjectX Importance: {importance}")
```

### Listing All Entities for an Agent
To list all entity nodes associated with a specific agent:

```python
async with memory.session_factory() as session:
    repo = GraphRepository(session)
    nodes = await repo.get_nodes_by_agent("cortex-planner-v1")
    for n in nodes:
        print(f"Entity: {n.entity_name} | Type: {n.entity_type} | Centrality: {n.degree_centrality}")
```

### Manual TTL Cache Eviction
While cache eviction can be scheduled (via `pg_cron` or standard worker routines), you can trigger node pruning manually:

```python
# Triggers eviction of expired nodes (cascades automatically to edges and hits)
deleted_count = await expire_old_nodes()
print(f"Evicted {deleted_count} expired entity nodes.")
```
