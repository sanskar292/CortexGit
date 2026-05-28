# Requirements: pip install cortexgit anthropic
"""
Example 3: Task Automation Agent showcasing Relation Entity Graph (REG)

This agent demonstrates CortexGit's Relation Entity Graph (REG) features.
REG calculates a dynamic "importance score" for entities using the formula:
    importance = degree_centrality * hit_frequency

By tracking tasks as nodes in the graph:
1. Adding links between tasks increases their connections (Degree Centrality).
2. Querying or interacting with tasks records a "hit" (Hit Frequency).
3. Stale tasks automatically decay or expire over time.

This script runs an interactive loop where you can add tasks, link them together,
record hits, and view the task list ranked dynamically by REG importance in a live table!
No LLM API keys are required for this graph demonstration.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Import CortexGit and Graph components
from cortexgit import CortexGit
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.centrality import recalculate_all_centrality
from cortexgit.graph.importance import rank_nodes_by_importance
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


def print_help():
    print("\nCommands:")
    print("  add <task_name>                 - Add a new task node to the graph")
    print("  link <task1> to <task2>        - Connect two tasks (increases Centrality!)")
    print("  hit <task_name>                 - Access/reference a task (increases Hit Frequency!)")
    print("  list                           - List all tasks ranked dynamically by REG importance")
    print("  complete <task_name>            - Mark a task as completed (removes from graph)")
    print("  help                           - Show this help menu")
    print("  quit                           - Exit the agent")


async def main():
    print("=" * 60)
    print(" CortexGit - Task Automation & REG Importance Agent")
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
            embedding_provider=embedder,
            enable_injection=True
        )
    except Exception as e:
        print(f"[Error] Failed to initialize CortexGit: {e}", file=sys.stderr)
        return

    # Create safe identifiers for the agent
    session_id = "task-manager-session"
    agent_id = "task-automation-agent"

    print(f"[Memory] Active Session: '{session_id}' | Agent: '{agent_id}'")
    print("No external LLM API key required for this Graph REG demo!")
    print_help()
    print("=" * 60)

    # 2. Command Interactive Loop
    while True:
        try:
            user_input = input("\nTaskAgent> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Memory saved.")
            break

        if not user_input:
            continue

        cmd_parts = user_input.split(" ", 1)
        action = cmd_parts[0].lower()
        args = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

        # --- COMMAND: QUIT ---
        if action in ("quit", "exit"):
            print("Goodbye!")
            break

        # --- COMMAND: HELP ---
        elif action == "help":
            print_help()

        # --- COMMAND: ADD ---
        elif action == "add":
            if not args:
                print("Error: Please specify a task name. E.g., 'add Write project report'")
                continue
            
            task_name = args
            print(f"[Graph] Adding task node: '{task_name}'...")
            
            # Write task to graph database via GraphRepository
            async with memory.session_factory() as session:
                repo = GraphRepository(session)
                try:
                    # Create node of type "concept" representing the task
                    node_id = await repo.create_node(
                        entity_name=task_name,
                        entity_type="concept",
                        description=f"Task: {task_name}",
                        status="pending",
                        agent_id=agent_id
                    )
                    # Log the addition event
                    await memory.log_event(
                        session_id=session_id,
                        agent_id=agent_id,
                        event_type="action",
                        payload={"action": "add_task", "task": task_name, "node_id": str(node_id)}
                    )
                    print(f"Task '{task_name}' successfully added (ID: {node_id}).")
                except Exception as e:
                    print(f"Error adding task: {e}")

        # --- COMMAND: LINK ---
        elif action == "link":
            if " to " not in args:
                print("Error: Format must be 'link <task1> to <task2>'")
                continue
            
            t1, t2 = [t.strip() for t in args.split(" to ", 1)]
            
            async with memory.session_factory() as session:
                repo = GraphRepository(session)
                # Fetch node models
                node1 = await repo.get_node(t1, agent_id)
                node2 = await repo.get_node(t2, agent_id)
                
                if not node1:
                    print(f"Error: Task '{t1}' not found.")
                    continue
                if not node2:
                    print(f"Error: Task '{t2}' not found.")
                    continue
                
                try:
                    # Create a directional edge linking them
                    edge_id = await repo.create_edge(node1.node_id, node2.node_id, "dependent_on")
                    
                    # Log action
                    await memory.log_event(
                        session_id=session_id,
                        agent_id=agent_id,
                        event_type="action",
                        payload={"action": "link_tasks", "from": t1, "to": t2, "edge_id": str(edge_id)}
                    )
                    
                    # Critical: Recalculate Centrality for all nodes to update degree counts
                    updated_count = await recalculate_all_centrality(session)
                    print(f"Linked '{t1}' to depend on '{t2}'! Recalculated centrality for {updated_count} nodes.")
                except Exception as e:
                    print(f"Error linking tasks: {e}")

        # --- COMMAND: HIT ---
        elif action == "hit":
            if not args:
                print("Error: Please specify task name. E.g., 'hit Write project report'")
                continue
            
            task_name = args
            
            async with memory.session_factory() as session:
                repo = GraphRepository(session)
                node = await repo.get_node(task_name, agent_id)
                
                if not node:
                    print(f"Error: Task '{task_name}' not found.")
                    continue
                
                try:
                    # Record query hit
                    await repo.record_hit(node.node_id, "query", session_id)
                    # Log hit
                    await memory.log_event(
                        session_id=session_id,
                        agent_id=agent_id,
                        event_type="action",
                        payload={"action": "access_task", "task": task_name}
                    )
                    print(f"Recorded hit on task '{task_name}'! (Hit Frequency incremented).")
                except Exception as e:
                    print(f"Error recording hit: {e}")

        # --- COMMAND: COMPLETE ---
        elif action == "complete":
            if not args:
                print("Error: Please specify task name. E.g., 'complete Write project report'")
                continue
            
            task_name = args
            
            async with memory.session_factory() as session:
                repo = GraphRepository(session)
                node = await repo.get_node(task_name, agent_id)
                
                if not node:
                    print(f"Error: Task '{task_name}' not found.")
                    continue
                
                try:
                    # Remove the node from the graph
                    # GraphRepository cascading will automatically clean up referencing edges and hits!
                    from sqlalchemy import delete
                    from cortexgit.db.models import EntityNode, EntityEdge, NodeHit
                    
                    # Cascade deletions manually or via helper
                    await session.execute(delete(EntityEdge).where(
                        (EntityEdge.source_node_id == node.node_id) |
                        (EntityEdge.target_node_id == node.node_id)
                    ))
                    await session.execute(delete(NodeHit).where(NodeHit.node_id == node.node_id))
                    await session.execute(delete(EntityNode).where(EntityNode.node_id == node.node_id))
                    await session.commit()
                    
                    # Log task completion
                    await memory.log_event(
                        session_id=session_id,
                        agent_id=agent_id,
                        event_type="action",
                        payload={"action": "complete_task", "task": task_name}
                    )
                    
                    # Recalculate centrality for remaining nodes
                    await recalculate_all_centrality(session)
                    print(f"Completed and removed task '{task_name}' from the graph.")
                except Exception as e:
                    print(f"Error completing task: {e}")

        # --- COMMAND: LIST ---
        elif action == "list":
            print("\nCalculating dynamic Relation Entity Graph (REG) importance...")
            async with memory.session_factory() as session:
                try:
                    # Retrieve nodes sorted by computed importance score
                    ranked_nodes = await rank_nodes_by_importance(agent_id, session)
                    
                    if not ranked_nodes:
                        print("No tasks found in memory! Use 'add <task_name>' to create one.")
                        continue
                    
                    # Format a beautiful table
                    print("-" * 85)
                    print(f"{'Task Name':<30} | {'Centrality (Edges)':<20} | {'Hits (Accesses)':<15} | {'REG Importance':<12}")
                    print("-" * 85)
                    
                    for n in ranked_nodes:
                        centrality = float(n.degree_centrality)
                        hits = int(n.hit_frequency)
                        # REG Formula: centrality * hits
                        importance = centrality * hits
                        print(f"{n.entity_name:<30} | {centrality:<20.1f} | {hits:<15d} | {importance:<12.1f}")
                    
                    print("-" * 85)
                    print("Formula: Importance = Centrality * Hits")
                except Exception as e:
                    print(f"Error fetching ranked tasks: {e}")
        
        else:
            print(f"Unknown command: '{action}'. Type 'help' to see valid commands.")


if __name__ == "__main__":
    asyncio.run(main())
