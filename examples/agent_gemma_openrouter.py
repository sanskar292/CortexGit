"""
CortexGit agent using Google Gemma 4 31B (free) via OpenRouter.

Run:
    pip install cortexgit openai python-dotenv
    python examples/agent_gemma_openrouter.py

Make sure .env has:
    OPENROUTER_API_KEY=sk-or-v1-your-key-here
    OPENROUTER_MODEL=google/gemma-4-31b-it:free
    OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
"""

import asyncio
import os
from dotenv import load_dotenv

from cortexgit import CortexGit
from cortexgit.core.memory import ConflictError
from cortexgit.llm_providers import create_llm_provider, create_embedding_provider

load_dotenv()

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set in your .env file or environment.")
        print("Please set it in .env, e.g.:")
        print("OPENROUTER_API_KEY=sk-or-v1-your-key-here")
        return

    # Explicitly configure CortexGit to use OpenRouter for this specific Gemma example
    llm = create_llm_provider(
        "openrouter",
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL") or "google/gemma-4-31b-it:free",
    )
    embedder = create_embedding_provider(
        "openrouter",
        api_key=api_key,
        embedding_model=os.getenv("OPENROUTER_EMBEDDING_MODEL") or "openai/text-embedding-3-small",
    )

    memory = CortexGit(
        database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///cortexgit.db"),
        llm_provider=llm,
        embedding_provider=embedder,
    )

    session_id = "gemma-session-1"
    agent_id = "gemma-agent"

    print("CortexGit + Gemma 4 31B (free via OpenRouter)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        ctx = await memory.get_context(
            goal=user_input,
            budget_tokens=4000,
            session_id=session_id,
        )

        system = (
            "You are a helpful AI assistant with persistent memory. "
            "Use the context below to remember past conversations and facts.\n"
            f"Context:\n{ctx}"
        )
        try:
            reply = memory.llm_provider.complete(system, user_input)
        except Exception as e:
            print(f"[LLM error] {e}")
            continue

        print(f"Agent: {reply}\n")

        event = await memory.log_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type="user",
            payload={"query": user_input},
        )
        await memory.log_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type="agent",
            payload={"response": reply},
        )

        try:
            await memory.write_entity(
                key="conversation.last_user_input",
                value=user_input,
                agent_id=agent_id,
                event_id=str(event.event_id),
            )
        except ConflictError:
            pass

if __name__ == "__main__":
    asyncio.run(main())
