import asyncio
import os
from cortexgit import CortexGit
from openai import AsyncOpenAI  # Or use Anthropic client

# Set up API keys
os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")

# 1. Initialize CortexGit Memory Client
# Points to a local SQLite database file 'cortexgit.db' by default.
# Proactive surface injection is enabled by default.
memory = CortexGit(
    enable_injection=True, 
    injection_threshold=5.0,  # Surfaces entities with importance > 5.0
    injection_top_k=2         # Limits to top 2 injected entities
)

llm_client = AsyncOpenAI()

async def run_agent_turn(session_id: str, agent_id: str, user_prompt: str):
    print(f"\n--- [User Prompt]: {user_prompt} ---")

    # Step 2: RETRIEVE Relevant Context from Memory
    # Surfaces recent events, snapshots, active conflicts, and high-importance injected entities
    context = await memory.get_context(
        goal=user_prompt,
        budget_tokens=4000,
        session_id=session_id,
        agent_id=agent_id
    )

    print(f"[Memory Restored]: Injected Entities: {context.get('metadata', {}).get('injected_entities', [])}")

    # Step 3: REASON — Feed retrieved context to the LLM system instructions
    system_prompt = (
        "You are a helpful software engineering assistant.\n"
        "Here is the context/memory gathered from your past active sessions:\n"
        f"{context}\n"
    )
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",  # or gpt-4o-mini / claude-3-5-sonnet
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    agent_response = response.choices[0].message.content
    print(f"[Agent Response]: {agent_response}")

    # Step 4: ACT & LOG — Append the action/observation to the persistent event log
    # This automatically triggers the flat-extraction, REG-graph-extraction, and snapshotting in the background.
    await memory.log_event(
        session_id=session_id,
        agent_id=agent_id,
        event_type="agent",
        payload={
            "text": f"Agent responded to: {user_prompt}. Output: {agent_response}"
        }
    )

async def main():
    session_id = "e2e-real-agent-demo"
    agent_id = "cortex-developer-1"

    # Step 5: Run sequential agent turns to build history
    # Turn 1: Build relations
    await run_agent_turn(
        session_id, agent_id,
        "We are initializing the new CorePaymentGateway microservice today. "
        "Lead developer Alex is assigned to this project."
    )
    
    # Wait briefly for background entity and graph extraction tasks to complete
    await asyncio.sleep(1.5)

    # Turn 2: Reinforce the gateway relation to elevate its hit importance
    for _ in range(5):
        await run_agent_turn(
            session_id, agent_id,
            "Alex is completing the Stripe connection modules for the CorePaymentGateway."
        )
        await asyncio.sleep(0.5)

    # Turn 3: Query on an unrelated topic to verify proactive injection
    # Even though the prompt does not mention 'Alex' or 'CorePaymentGateway', 
    # they will be proactively injected into the context as high-importance nodes!
    await run_agent_turn(
        session_id, agent_id,
        "What are the best practices for writing Dockerfiles?"
    )

if __name__ == "__main__":
    asyncio.run(main())
