# Getting Started with CortexGit

CortexGit provides persistent, event-sourced memory and semantic retrieval for LLM agents. This guide helps you set up CortexGit and integrate it with your LLM agents in minutes.

---

## 🚀 1. Installation

Install the library using `pip`:

```bash
pip install cortexgit
```

---

## 🛠️ 2. Environment Setup

Create a `.env` file in the root of your project:

```ini
# SQLite is used by default for local development
DATABASE_URL=sqlite+aiosqlite:///cortexgit.db

# API keys for LLM providers
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 💡 3. Quick Start Example

Here is a complete, copy-paste-able example of an agent using CortexGit for persistent context assembly and event logging, integrated with Anthropic's Claude.

```python
import asyncio
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from cortexgit import CortexGit

# Load environment variables (.env)
load_dotenv()

async def run_agent():
    # 1. Initialize CortexGit (uses DATABASE_URL from environment)
    memory = CortexGit()

    # 2. Initialize Anthropic client
    client = Anthropic()

    session_id = "user-session-123"
    agent_id = "helper-bot"

    user_query = "What did we decide about the project architecture yesterday?"

    # 3. Retrieve relevant context for the query within a token budget (e.g. 4000 tokens)
    context = await memory.get_context(
        goal=user_query,
        session_id=session_id,
        budget_tokens=4000
    )

    # 4. Format context and prompt the LLM
    system_prompt = f"""
    You are an intelligent engineering assistant.
    Here is the relevant memory context from the conversation history:
    {context}
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_query}]
    )
    
    agent_reply = response.content[0].text
    print(f"Claude's Reply:\n{agent_reply}\n")

    # 5. Log the interaction events to CortexGit
    # This automatically triggers entity extraction & conflict detection in the background
    await memory.log_event(
        session_id=session_id,
        agent_id=agent_id,
        event_type="user",
        payload={"query": user_query}
    )

    await memory.log_event(
        session_id=session_id,
        agent_id=agent_id,
        event_type="agent",
        payload={"response": agent_reply}
    )

# Run the async agent function
if __name__ == "__main__":
    asyncio.run(run_agent())
```

---

## ⚙️ 4. Configuration

When initializing `CortexGit`, you can specify custom connection URLs:

```python
# Pass database URL explicitly (e.g., PostgreSQL for production)
memory = CortexGit(database_url="postgresql+asyncpg://postgres:password@localhost:5432/cortex_prod")
```

### Database Engines Supported:
- **SQLite**: Great for local, in-process, serverless development (`sqlite+aiosqlite:///cortexgit.db`).
- **PostgreSQL**: Standard for production, highly optimized for vector operations using `pgvector` (`postgresql+asyncpg://user:pass@host:5432/dbname`).

---

## 🎯 5. Next Steps

- Explore the [API Reference](API_REFERENCE.md) to understand all available methods on the `CortexGit` client.
- Learn about internal mechanics in the [Architecture Guide](ARCHITECTURE.md) to see how conflicts are managed and event snapshots are taken.
