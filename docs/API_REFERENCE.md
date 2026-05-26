# CortexGit API Reference

This document provides a complete reference for the public API of the `CortexGit` client SDK.

---

## Class: `CortexGit`

The main interface for persistent, event-sourced memory, context retrieval, and entity registry management.

```python
from cortexgit import CortexGit
```

### Table of Contents
1. [`__init__`](#__init__)
2. [`log_event`](#log_event)
3. [`get_context`](#get_context)
4. [`write_entity`](#write_entity)

---

### `__init__`

#### Signature
```python
def __init__(self, database_url: str = None)
```

#### Description
Initializes the CortexGit client.

#### Parameters
* **`database_url`** *(str, optional)*: Connection URL for the database (e.g., `sqlite+aiosqlite:///cortexgit.db` or `postgresql+asyncpg://...`). If not provided, it defaults to the `DATABASE_URL` environment variable.

#### Return Value
* **`None`**

#### Code Example
```python
# Initialize using the DATABASE_URL environment variable
memory = CortexGit()

# Initialize using an explicit database connection string
memory = CortexGit(database_url="sqlite+aiosqlite:///custom_memory.db")
```

#### Common Errors
* None.

---

### `log_event`

#### Signature
```python
async def log_event(self, session_id: str, agent_id: str, event_type: str, payload: dict) -> EventLog
```

#### Description
Appends a new event to the persistent event log and triggers background entity extraction and snapshotting.

#### Parameters
* **`session_id`** *(str)*: Unique identifier representing the current conversation session.
* **`agent_id`** *(str)*: Unique identifier of the agent logging the event.
* **`event_type`** *(str)*: Type of the event (case-insensitive). Must be one of: `"system"`, `"user"`, `"agent"`, `"action"`, `"observation"`, `"thought"`, `"error"`.
* **`payload`** *(dict)*: Dictionary payload containing the details of the event.

#### Return Value
* **`EventLog`** *(SQLAlchemy Model)*: The created database model instance representing the logged event, including its auto-assigned `event_id` and timestamp.

#### Code Example
```python
event = await memory.log_event(
    session_id="session-456",
    agent_id="bot-9",
    event_type="thought",
    payload={"reasoning": "Decided to search the web for architecture designs."}
)
print(f"Logged event ID: {event.event_id}")
```

#### Common Errors
* **`ValueError`**: Raised if `event_type` does not match one of the allowed event categories (e.g. `invalid_type`).

---

### `get_context`

#### Signature
```python
async def get_context(self, goal: str, budget_tokens: int, session_id: str) -> dict
```

#### Description
Retrieves an assembled dictionary of relevant context (recent events, semantic snapshots, registered entity facts, active conflicts) under a strict token budget.

#### Parameters
* **`goal`** *(str)*: The agent's current search goal or prompt query, used to retrieve semantically matching historical logs.
* **`budget_tokens`** *(int)*: The maximum token size allocated for the returned dictionary (must be greater than `0`).
* **`session_id`** *(str)*: The unique identifier representing the session to query.

#### Return Value
* **`dict`**: A dictionary containing four context keys:
  * `"events"` *(list)*: Recent and relevant event logs.
  * `"snapshots"` *(list)*: Relevant event log summary snapshots.
  * `"entities"` *(dict)*: Registered key-value entity facts.
  * `"conflicts"` *(list)*: Unresolved entity write collisions.

#### Code Example
```python
context = await memory.get_context(
    goal="What is the user's favorite programming language?",
    budget_tokens=3000,
    session_id="session-456"
)

# Accessing the assembled context components
print(context["entities"])   # e.g., {"user.language": "Python"}
print(context["conflicts"])  # Active entity conflicts
```

#### Common Errors
* **`ValueError`**: Raised if `goal` or `session_id` are empty/whitespace, or if `budget_tokens` is less than or equal to `0`.

---

### `write_entity`

#### Signature
```python
async def write_entity(self, key: str, value: any, agent_id: str, event_id: str) -> bool
```

#### Description
Directly writes a key-value fact to the permanent Entity Registry, logging a conflict if the key exists with a different value.

#### Parameters
* **`key`** *(str)*: The unique registry key/path for the fact (e.g., `project.architecture`).
* **`value`** *(any)*: The value/data to store under the key.
* **`agent_id`** *(str)*: The identifier of the agent performing the write.
* **`event_id`** *(str)*: UUID string of the event that supports or backs this registry update.

#### Return Value
* **`bool`**: Returns `True` if a new registry record was written. Returns `False` if the write was an idempotent match (same key, same value already exists).

#### Code Example
```python
try:
    updated = await memory.write_entity(
        key="project.status",
        value="Alpha development",
        agent_id="bot-9",
        event_id="3fa85f64-5717-4562-b3fc-2c963f66afa6"
    )
    if updated:
        print("Fact successfully written.")
    else:
        print("Duplicate write skipped (idempotent).")
except ConflictError as e:
    print(f"Failed to write: {e}")
```

#### Common Errors
* **`ConflictError`**: Raised if the key is already registered with a different value, causing a write collision.
* **`ValueError`**: Raised internally if the registry update fails validation or formatting.
