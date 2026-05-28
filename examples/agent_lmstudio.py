import asyncio
import os
from dotenv import load_dotenv
from cortexgit import CortexGit
from cortexgit.llm_providers import OllamaProvider

load_dotenv()

async def main():
    # 1. Initialize local providers pointing to LM Studio
    llm = OllamaProvider(
        base_url="http://localhost:8000",
        model=os.getenv("OLLAMA_MODEL", "meta-llama-3-8b-instruct")
    )
    
    embedder = OllamaProvider(
        base_url="http://localhost:8000",
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    )

    # 2. Instantiate the memory client
    memory = CortexGit(
        database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///cortexgit.db"),
        llm_provider=llm,
        embedding_provider=embedder,
    )

    session_id = "lmstudio-session"
    agent_id = "lmstudio-agent"

    print("CortexGit Local Agent via LM Studio is running...")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        # Retrieve relevant memories and context for the goal
        ctx = await memory.get_context(
            goal=user_input,
            budget_tokens=1500,
            session_id=session_id,
        )

        system_prompt = (
            "You are a helpful AI assistant with persistent memory. "
            "Use the context below to remember past conversations and facts.\n"
            f"Context:\n{ctx}"
        )
        
        try:
            reply = memory.llm_provider.complete(system_prompt, user_input)
        except Exception as e:
            print(f"[LLM error] {e}")
            continue

        print(f"Agent: {reply}\n")

        # Save the current turn into persistent memory
        await memory.log_event(
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

if __name__ == "__main__":
    asyncio.run(main())
