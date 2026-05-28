# Requirements: pip install cortexgit anthropic
"""
Example 4: Multi-Agent Coordination and Conflict Resolution

This agent demonstrates how two separate agents (Manager Agent and Developer Agent)
can coordinate and share memory via the same CortexGit instance (using a shared
database). It also shows CortexGit's automated conflict detection in action:

1. Agent A (Manager) writes a status value to the Entity Registry ("project.status" -> "active").
2. Agent B (Developer) restores context, seeing Agent A's logged events and registry entries.
3. Agent B attempts to update the status to a conflicting value ("project.status" -> "completed").
4. CortexGit automatically raises a ConflictError and logs the clash to the ConflictLog.
5. Both agents retrieve the open conflict via get_context() to trigger reconciliation.
6. The script demonstrates how to resolve the conflict directly using the database session.

No LLM API keys are required for this coordination and conflict resolution demo.
"""

import asyncio
import os
import sys
from sqlalchemy import select
from dotenv import load_dotenv

# Import CortexGit
from cortexgit import CortexGit
from cortexgit.core.memory import ConflictError
from cortexgit.db.models import ConflictLog, EntityRegistry
from cortexgit.llm_providers import LLMProvider, EmbeddingProvider
from cortexgit.llm_providers import create_llm_provider, create_embedding_provider

# Load environment variables
load_dotenv()

# --- Mock Providers for Zero-Setup execution ---

class MockLLMProvider(LLMProvider):
    def complete(self, system_prompt: str, user_message: str) -> str:
        import json
        system_lower = system_prompt.lower()
        if "entity extractor" in system_lower:
            return json.dumps({ "updates": [] })
        elif "relational entity graph" in system_lower:
            return json.dumps({ "updates": [] })
        elif "memory summarizer" in system_lower:
            return json.dumps({
                "summary": "Mock summary of historical session.",
                "entities_mentioned": [],
                "event_range": [1, 10]
            })
        return "Mock response"

class MockEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1536


async def main():
    print("=" * 60)
    print(" CortexGit - Multi-Agent Memory Coordination & Conflict Resolution")
    print("=" * 60)

    # 1. Initialize CortexGit Persistent Memory Client
    # Uses Mock Providers if API keys are missing to avoid background LLM extraction failures.
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
        llm = create_llm_provider(provider, api_key=api_key)
        embedder = create_embedding_provider("openai") if os.getenv("OPENAI_API_KEY") else MockEmbeddingProvider()
    else:
        llm = MockLLMProvider()
        embedder = MockEmbeddingProvider()

    try:
        memory = CortexGit(
            database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///cortexgit.db"),
            llm_provider=llm,
            embedding_provider=embedder
        )
    except Exception as e:
        print(f"[Error] Failed to initialize CortexGit: {e}", file=sys.stderr)
        return

    # Define unique agent IDs
    session_id = "coordination-session"
    agent_a_id = "manager-agent"
    agent_b_id = "developer-agent"

    print(f"[Memory] Shared Session: '{session_id}'")
    print(f"[Memory] Agent A: '{agent_a_id}' | Agent B: '{agent_b_id}'")
    print("=" * 60)

    # --- STEP 1: Agent A (Manager) Logs Event & Sets Entity Registry Key ---
    print("\n--- [Agent A: Manager Agent] ---")
    print("Manager Agent starts the 'PaymentGateway' project and registers its status...")
    
    # Log starting event
    event_a = await memory.log_event(
        session_id=session_id,
        agent_id=agent_a_id,
        event_type="user",
        payload={"action": "start_project", "project": "PaymentGateway", "details": "Initiated stripe connection module."}
    )
    
    # Directly write status as 'active' in the EntityRegistry
    try:
        await memory.write_entity(
            key="project.status",
            value="active",
            agent_id=agent_a_id,
            event_id=str(event_a.event_id)
        )
        print("Manager Agent successfully set 'project.status' -> 'active'")
    except Exception as e:
        print(f"Error writing entity: {e}")

    # --- STEP 2: Agent B (Developer) Restores Shared Context ---
    print("\n--- [Agent B: Developer Agent] ---")
    print("Developer Agent spins up, restores memory context, and inherits Manager's history...")
    
    # Retrieve context for Developer Agent (B)
    context_b = await memory.get_context(
        goal="What is the current status of the project?",
        budget_tokens=3000,
        session_id=session_id,
        agent_id=agent_b_id
    )
    
    # Print shared memory items
    print("\nDeveloper Agent's Restored Context:")
    print(f"  - Recent Events: {len(context_b.get('events', []))}")
    print(f"  - Registered Entities: {context_b.get('entities', {})}")
    print(f"  - Open Conflicts: {context_b.get('conflicts', [])}")

    # --- STEP 3: Agent B Attempts a Conflicting Update ---
    print("\n--- [Conflict Detection in Action] ---")
    print("Developer Agent tries to set 'project.status' -> 'completed' without coordinating...")
    
    # Log developer's event
    event_b = await memory.log_event(
        session_id=session_id,
        agent_id=agent_b_id,
        event_type="agent",
        payload={"action": "finish_work", "project": "PaymentGateway"}
    )
    
    # Writing a different value on the same key raises a ConflictError!
    try:
        await memory.write_entity(
            key="project.status",
            value="completed",
            agent_id=agent_b_id,
            event_id=str(event_b.event_id)
        )
    except ConflictError as e:
        print(f"\n[Conflict Caught!] {e}")
        print("CortexGit blocked the write! A conflict entry was automatically logged.")

    # --- STEP 4: Surface Conflict in the Next Turn ---
    print("\n--- [Surfacing the Conflict] ---")
    print("Retrieving context again to see if the conflict is surfaced to the agents...")
    
    context_recheck = await memory.get_context(
        goal="Check project conflicts.",
        budget_tokens=3000,
        session_id=session_id,
        agent_id=agent_a_id
    )
    
    conflicts = context_recheck.get("conflicts", [])
    if conflicts:
        print("\nOpen Conflicts surfaced in context:")
        for idx, conflict in enumerate(conflicts, 1):
            print(f"  {idx}. Key: '{conflict['key']}'")
            print(f"     Existing Value (Agent A): '{conflict['existing_value']}'")
            print(f"     Proposed Value (Agent B): '{conflict['proposed_value']}'")
            print(f"     Status: unresolved (resolved={conflict['resolved']})")

    # --- STEP 5: Reconciliation & Conflict Resolution ---
    print("\n--- [Resolving the Conflict] ---")
    print("Reconciling: Manager and Developer agree the project status is indeed 'completed'.")
    print("Resolving conflict and updating entity directly via SQLite DB session...")
    
    async with memory.session_factory() as session:
        try:
            # 1. Fetch the unresolved conflict for project.status
            stmt = select(ConflictLog).where(
                ConflictLog.key == "project.status",
                ConflictLog.resolved == False
            )
            res = await session.execute(stmt)
            conflict_record = res.scalar_one_or_none()
            
            if conflict_record:
                # 2. Mark the conflict as resolved
                conflict_record.resolved = True
                print("  - Marked conflict record in database as RESOLVED.")
                
                # 3. Update the value in the Entity Registry to the agreed 'completed' state
                entity_stmt = select(EntityRegistry).where(EntityRegistry.key == "project.status")
                entity_res = await session.execute(entity_stmt)
                entity = entity_res.scalar_one_or_none()
                
                if entity:
                    entity.value = "completed"
                    entity.agent_id = agent_b_id  # Update agent ownership to Agent B
                    entity.event_id = event_b.event_id
                    print("  - Updated registry 'project.status' -> 'completed'.")
                
                await session.commit()
                print("Conflict successfully resolved!")
        except Exception as e:
            print(f"Error during manual reconciliation: {e}")
            await session.rollback()

    # --- STEP 6: Final Verification ---
    print("\n--- [Final Verification] ---")
    print("Checking context once more to verify zero unresolved conflicts remain...")
    
    final_context = await memory.get_context(
        goal="Final verification.",
        budget_tokens=3000,
        session_id=session_id,
        agent_id=agent_a_id
    )
    
    print(f"  - Final Registered Entities: {final_context.get('entities', {})}")
    print(f"  - Final Open Conflicts: {final_context.get('conflicts', [])}")
    print("\nMulti-Agent Coordination and Conflict Resolution test successfully completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
