# CortexGit

Persistent memory for LLM agents. Event sourcing + semantic retrieval.

## The Problem

LLM agents are stateless. They forget context between sessions. They can't coordinate without explicit message passing. They have no audit trail.

## The Solution

CortexGit is an in-process memory system. Write events, retrieve context, persist facts. Works with any LLM, any agent framework.

## Installation

```bash
pip install cortexgit
```

## Quick Start

```python
from cortexgit import CortexGit
from anthropic import Anthropic

# Initialize memory (creates local SQLite database)
memory = CortexGit()
client = Anthropic()

def my_agent(user_query):
    # Retrieve relevant context from memory
    context = memory.get_context(
        goal=user_query,
        budget_tokens=4000
    )
    
    # Call Claude with context
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=f"You are a helpful agent. Memory: {context}",
        messages=[{"role": "user", "content": user_query}]
    )
    
    # Remember what happened
    memory.log_event("interaction", {
        "query": user_query,
        "response": response.content[0].text
    })
    
    return response.content[0].text

# Use it
print(my_agent("What is 2+2?"))
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Getting Started](docs/GETTING_STARTED.md)

## Configuration

By default, CortexGit uses SQLite (`sqlite+aiosqlite:///cortexgit.db`) which requires no external server or setup.

To use PostgreSQL in production:
1. Install PostgreSQL and create a database (e.g. `cortexgit`).
2. Set the `DATABASE_URL` environment variable:
   ```bash
   DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/cortexgit"
   ```

## Features

- ✅ Append-only event log (source of truth)
- ✅ Persistent entity registry with conflict detection
- ✅ Automatic snapshot generation
- ✅ Semantic retrieval over compressed memory
- ✅ Works with any LLM (Claude, GPT, local models)
- ✅ Single import, no server needed

## License

MIT
