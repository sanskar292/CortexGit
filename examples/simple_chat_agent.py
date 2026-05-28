# Requirements: pip install cortexgit anthropic
"""
Example 1: Simple Chat Agent with Persistent Memory

This agent demonstrates a basic multi-turn chat application with persistent memory
using CortexGit. It runs in an interactive loop in your terminal. On start,
it restores context from previous sessions and continues building conversational history.

If ANTHROPIC_API_KEY is not set in the environment, the script will gracefully
switch to MOCK MODE so you can run and experience the memory features immediately.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Import CortexGit classes
from cortexgit import CortexGit
from cortexgit.llm_providers import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError
from cortexgit.llm_providers import create_llm_provider, create_embedding_provider

# Load environment variables from .env if present
load_dotenv()

# --- Mock Providers for Zero-Setup execution ---

class MockLLMProvider(LLMProvider):
    """
    A high-fidelity Mock LLM Provider that replies conversationally based on user input,
    allowing developers to test the persistence loops without needing an API key.
    Handles background extraction and summarization prompts defensively with correct JSON shapes.
    """
    def complete(self, system_prompt: str, user_message: str) -> str:
        import json
        system_lower = system_prompt.lower()
        
        # 1. Background Entity Extraction handling
        if "entity extractor" in system_lower:
            return json.dumps({ "updates": [] })
            
        # 2. Background Relational Entity Graph handling
        elif "relational entity graph" in system_lower:
            return json.dumps({ "updates": [] })
            
        # 3. Background Memory Summarizer handling
        elif "memory summarizer" in system_lower:
            return json.dumps({
                "summary": "Mock summary of historical session.",
                "entities_mentioned": [],
                "event_range": [1, 10]
            })

        # 4. Standard conversational response
        msg = user_message.lower().strip()
        
        # Simple rule-based mock responses that feel conversational
        if "hello" in msg or "hi" in msg:
            return "Hello! I am a persistent memory agent powered by CortexGit. How can I help you today?"
        elif "remember" in msg or "recall" in msg:
            return "CortexGit stores all our conversational events in its database. Try asking 'what did we talk about?' or exit and restart the agent!"
        elif "who are you" in msg:
            return "I am a simple demo agent. Currently I'm running in Mock Mode because no Anthropic API key was found."
        else:
            return (
                f"I hear you! I've noted down: '{user_message}'.\n"
                "This fact has been recorded into my persistent event log, and will be loaded "
                "into my working context on our next turns."
            )

class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock Embedding Provider returning a dummy 1536-dimensional vector.
    Enables database semantic recall on SQLite/PostgreSQL without needing external API calls.
    """
    def embed(self, text: str) -> list[float]:
        # Return a deterministic 1536-dimensional mock embedding vector
        return [0.1] * 1536


async def main():
    print("=" * 60)
    print(" CortexGit - Simple Chat Agent with Persistent Memory")
    print("=" * 60)

    # 1. Setup Providers (Anthropic Claude + OpenAI/Mock Embeddings)
    # Check if Anthropic API key is available
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if api_key:
        print("[Status] ANTHROPIC_API_KEY found! Initializing Claude LLM...")
        llm = create_llm_provider("anthropic", api_key=api_key)
        
        # Check if OpenAI is available for real embeddings
        if os.getenv("OPENAI_API_KEY"):
            print("[Status] OPENAI_API_KEY found! Using OpenAI for semantic embeddings...")
            embedder = create_embedding_provider("openai")
        else:
            print("[Notice] OPENAI_API_KEY not found. Using MockEmbeddingProvider for recall.")
            embedder = MockEmbeddingProvider()
    else:
        print("[Warning] ANTHROPIC_API_KEY not found in your environment or .env file.")
        print("[Status] Running in MOCK MODE. Persistence features will still be fully operational!")
        print("[Tip] To use real Claude, set ANTHROPIC_API_KEY in your env or .env file.")
        print("-" * 60)
        llm = MockLLMProvider()
        embedder = MockEmbeddingProvider()

    # 2. Initialize CortexGit Persistent Memory Client
    # Points to a local 'cortexgit.db' SQLite file by default.
    try:
        memory = CortexGit(
            database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///cortexgit.db"),
            llm_provider=llm,
            embedding_provider=embedder,
            enable_injection=True
        )
    except Exception as e:
        print(f"[Error] Failed to initialize CortexGit: {e}", file=sys.stderr)
        return

    # Use a persistent session ID and agent ID so memory persists across restarts!
    session_id = "simple-chat-session"
    agent_id = "claude-chat-agent"

    print(f"[Memory] Active Session: '{session_id}' | Agent: '{agent_id}'")
    print("Type your message below. Type 'exit' or 'quit' to save and exit.")
    print("=" * 60)

    # 3. Conversational Loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Memory saved.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Your memory is safely stored. See you next session!")
            break

        # Step 3a: Get context from CortexGit (Semantic recall + recent events + entity registry)
        print("[Memory] Restoring context from past history...")
        try:
            context = await memory.get_context(
                goal=user_input,
                budget_tokens=2000,
                session_id=session_id,
                agent_id=agent_id
            )
        except Exception as e:
            print(f"[Memory Error] Could not retrieve context: {e}")
            context = {}

        # Step 3b: Construct System Prompt combining LLM instructions and retrieved memory
        system_prompt = (
            "You are a helpful AI assistant equipped with persistent memory.\n"
            "Below is the context retrieved from past turns/sessions. "
            "Use it to remember the user, facts, and conversation history.\n\n"
            f"--- PERSISTENT MEMORY CONTEXT ---\n{context}\n---------------------------------\n"
        )

        # Step 3c: Call LLM to complete the message
        print("[Agent] Thinking...")
        try:
            agent_response = memory.llm_provider.complete(
                system_prompt=system_prompt,
                user_message=user_input
            )
        except LLMError as e:
            print(f"[LLM Error] Failed to generate response: {e}")
            continue
        except Exception as e:
            print(f"[Error] Unexpected error during completion: {e}")
            continue

        print(f"Agent: {agent_response}")

        # Step 3d: Act & Log turns into CortexGit
        # Logging automatically triggers event-sourced updates and entity extraction in the background
        try:
            # Log the user's message
            await memory.log_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type="user",
                payload={"text": user_input}
            )
            # Log the agent's response
            await memory.log_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type="agent",
                payload={"text": agent_response}
            )
        except Exception as e:
            print(f"[Memory Error] Could not write to event log: {e}")


if __name__ == "__main__":
    # Run the async main loop
    asyncio.run(main())
