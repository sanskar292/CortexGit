# CortexGit — Architecture Reference
## For Claude Code: Read this file before every response. Every decision must be traceable to a section here.

---

## What This Is

A memory backend service for LLM agents. Agents call this API to write events and read context. Everything else — event storage, snapshotting, entity tracking, conflict detection, vector retrieval — is internal to this service.

This is infrastructure, not an agent, not an app.

---

## The One Rule

**LLMs produce content. Deterministic systems govern structure.**

Every LLM output must pass through the write-back gate before touching any store.
If validation fails: reject. Not retry. Not partial accept. Reject.
Freeform LLM judgment never sits in a database unchecked.

---

## API Surface (What Agents See)

Three endpoints. Nothing else is public.

```
POST /events
  Body: { session_id, agent_id, event_type, payload: {} }
  Returns: { event_id, timestamp }
  Rules: append-only, no LLM, no side effects beyond writing

POST /entities
  Body: { key, value, agent_id }
  Returns: { status: "ok" } or { status: "conflict", existing, proposed, event_ids }
  Rules: check for key collision first, never auto-resolve conflicts

GET /context
  Params: goal (string), budget_tokens (int), session_id
  Returns: { events: [], snapshots: [], entities: {}, conflicts: [] }
  Rules: deterministic packing, never exceed budget_tokens
```

---

## Stores (Internal)

### Event Log
- PostgreSQL or SQLite table (portable via SQLAlchemy)
- Append-only — no UPDATE, no DELETE ever
- Schema: event_id (uuid), session_id, agent_id, event_type (enum), payload (jsonb / JSON text), created_at (timestamptz)
- This is the source of truth. Every other store is derived from it.

### Entity Registry
- PostgreSQL or SQLite table
- Schema: key (text PK), value (jsonb / JSON text), agent_id, event_id (FK to event log), updated_at
- Keys are namespaced: agent_alpha.current_task for agent-local, project.goal for shared
- Conflict = key exists AND value differs from proposed value

### Snapshot Store
- PostgreSQL or SQLite table + pgvector extension (PostgreSQL) or in-memory cosine similarity (SQLite)
- Schema: snapshot_id (uuid), event_range (int4range / string), summary (text), entities_mentioned (text[] / JSON), embedding (vector(1536) / float[] / JSON), created_at
- Written by Summarizer LLM only
- Immutable after creation — no UPDATE ever
- Every write goes through write-back gate

### Conflict Log
- PostgreSQL or SQLite table
- Schema: conflict_id (uuid), key, existing_value, proposed_value, existing_event_id, proposed_event_id, resolved (bool), created_at
- Written by conflict detector only
- Never written by LLM

---

## Components (Build in This Order)

### 1. Database Schema + Migrations (Phase 1)
- Alembic for migrations
- All four tables above
- Append-only constraint on event log enforced at DB level
- Do not proceed until migrations run cleanly

### 2. Event Logger (Phase 1)
- POST /events handler
- Writes to event log only
- No LLM, no side effects
- Validate event_type against enum before writing

### 3. Entity Registry + Conflict Detector (Phase 1)
- POST /entities handler
- Check key existence first
- If conflict: write to conflict log, return conflict response, do not write to registry
- If clean: write to registry

### 4. Write-Back Gate (Phase 1)
- Standalone module: validate(output, schema_name) -> validated_object | raises ValidationError
- Schemas defined in /schemas/*.json
- Called by Summarizer and Entity Extractor before any store write
- No retry logic anywhere in this module

### 5. Recency Filter (Phase 2)
- Internal module used by GET /context
- Fetches last K events from event log for a session
- K is configurable, default 20
- Always included in context regardless of semantic score

### 6. Vector Index + Semantic Recall (Phase 2)
- pgvector on snapshot store
- Embed snapshot summaries using text-embedding-3-small
- ANN search by cosine similarity
- Return top-N snapshots by relevance to goal string
- Never embed raw events — snapshots only

### 7. Entity Pull (Phase 2)
- Internal module used by GET /context
- Fetches entity registry keys relevant to the goal string
- Relevance = simple substring match on key names against goal tokens
- Returns full entity dict for matched keys

### 8. Context Assembler (Phase 2)
- Internal module used by GET /context
- Inputs: recency_filter output, semantic_recall output, entity_pull output, conflict log (open conflicts only)
- Token budget enforcement: estimate tokens per item, pack until budget reached
- Priority order: conflicts first, recent events second, snapshots third, entities fourth
- Never exceed budget_tokens

### 9. Snapshot Trigger + Summarizer LLM (Phase 3)
- Trigger: fires when event log for a session crosses N new events since last snapshot (default N=50)
- Summarizer: calls Anthropic API with structured output prompt
- Output must match snapshot schema before write-back gate accepts it
- Snapshot schema: { summary: string, entities_mentioned: string[], event_range: [int, int] }
- If gate rejects: log failure, do not retry, fall back to raw events in context

### 10. Entity Extractor LLM (Phase 3)
- Runs after every POST /events call asynchronously
- Calls Anthropic API to extract entity updates from the new event
- Output must match entity extraction schema before write-back gate accepts it
- Entity extraction schema: { updates: [{ key: string, value: any }] }
- If gate rejects: log failure, skip this event's entities, continue

### 11. Responder LLM (Phase 3)
- Called externally — agents call GET /context then pass context to their own LLM
- This service does NOT expose a chat endpoint
- Do not build a responder LLM into this service

---

## Write-Back Gate Schemas

### snapshot_schema
```json
{
  "type": "object",
  "required": ["summary", "entities_mentioned", "event_range"],
  "additionalProperties": false,
  "properties": {
    "summary": { "type": "string", "minLength": 10 },
    "entities_mentioned": {
      "type": "array",
      "items": { "type": "string" }
    },
    "event_range": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 2,
      "maxItems": 2
    }
  }
}
```

### entity_extraction_schema
```json
{
  "type": "object",
  "required": ["updates"],
  "additionalProperties": false,
  "properties": {
    "updates": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["key", "value"],
        "additionalProperties": false,
        "properties": {
          "key": { "type": "string", "pattern": "^[a-z0-9_.]+$" },
          "value": {}
        }
      }
    }
  }
}
```

---

## LLM Prompts

### Summarizer System Prompt
```
You are a memory summarizer for an AI agent system.
You will receive a sequence of events from an agent session.
Summarize what happened, what decisions were made, and what facts were established.
Be specific. Do not generalize. Do not invent connections not present in the events.
Return only valid JSON matching this schema exactly:
{ "summary": string, "entities_mentioned": string[], "event_range": [int, int] }
No preamble. No explanation. No markdown. Raw JSON only.
```

### Entity Extractor System Prompt
```
You are an entity extractor for an AI agent memory system.
You will receive a single agent event.
Extract any named entities, decisions, goals, or facts that should be remembered.
Keys must be lowercase with dots and underscores only. Example: project.current_goal
Return only valid JSON matching this schema exactly:
{ "updates": [{ "key": string, "value": any }] }
No preamble. No explanation. No markdown. Raw JSON only.
If nothing should be extracted, return: { "updates": [] }
```

---

## What Not to Build

- No semantic diff engine — not solvable deterministically
- No importance scoring — not reliable, use recency as proxy
- No auto-merge of conflicts — flag and stop, human resolves
- No chat endpoint — agents bring their own LLM
- No branch/fork system — shared event log handles multi-agent
- No retry on write-back gate failure — reject is final

---

## Tech Stack

| Layer | Technology |
|---|---|---|
| Runtime | Python 3.10+ (async SDK) |
| Database (production) | PostgreSQL 15+ with pgvector |
| Database (development) | SQLite via aiosqlite |
| Vector search (native) | pgvector (PostgreSQL) |
| Vector search (fallback) | In-memory cosine similarity (SQLite) |
| Migrations | Alembic |
| Validation | jsonschema |
| LLM calls | Anthropic Python SDK / OpenAI SDK / OpenRouter / Ollama |
| Embeddings | OpenAI text-embedding-3-small / Ollama / OpenRouter |
| Testing | pytest + pytest-asyncio |

---

## Project Structure

```
src/
  cortexgit/
    __init__.py           # Public exports: CortexGit, EventLog, EntityRegistry
    core/
      memory.py           # Main CortexGit client class
      event_log.py        # Event logging (append-only)
      entity_registry.py  # Entity registry handler
      conflict_detector.py
      write_back_gate.py  # JSON schema validation for LLM outputs
      context_assembler.py
      recency_filter.py
      entity_pull.py
    llm_providers/
      base.py             # Abstract LLMProvider and EmbeddingProvider
      anthropic_provider.py
      openai_provider.py
      openrouter_provider.py
      ollama_provider.py
      provider_factory.py
    llm/
      summarizer.py
      entity_extractor.py
      snapshot_trigger.py
    retrieval/
      semantic_recall.py
      embeddings.py
    db/
      models.py
      database.py
      migrations/
    schemas/
      snapshot_schema.json
      entity_extraction_schema.json
tests/
  conftest.py
  test_event_logger.py
  test_entity_registry.py
  test_conflict_detector.py
  test_write_back_gate.py
  test_context_assembler.py
  test_retrieval.py
  ...
docs/
  ARCHITECTURE.md
  GETTING_STARTED.md
  API_REFERENCE.md
  PROVIDERS.md
```

---

## Rules for Claude Code

1. Read ARCHITECTURE.md at the start of every session
2. Read PROGRESS.md at the start of every session
3. Build only what the current phase requires
4. Ask before making any design decision not covered here
5. After every session, update PROGRESS.md with: what was built, what decisions were made, what the next session starts with
6. Never build the Responder LLM — agents use their own
7. Never add retry logic to the write-back gate
8. Never add UPDATE or DELETE to the event log
9. If unsure whether something needs an LLM: it probably doesn't

---

## PROGRESS.md Template (Create This File, Update Each Session)

```
## Last updated: [date]

## Phase: [1 / 2 / 3]

## What is built and tested:
- 

## Decisions made:
- 

## Known issues:
- 

## Next session starts with:
- 
```
