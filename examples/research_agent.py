# Requirements: pip install cortexgit anthropic
"""
Example 2: Research Agent with Compounding Memory

This agent takes a research topic as a command line argument, retrieves real factual
data from Wikipedia (using Python's standard urllib), and builds up a research dossier.
Every time you run the agent on the same topic, it:
1. Restores the prior research context from CortexGit.
2. Combines past learnings with new Wikipedia snippets.
3. Synthesizes an updated compound research dossier.
4. Directly registers key entity nodes (e.g. key terms, researchers, projects)
   into the CortexGit Entity Registry.

If no API keys are found in your environment, it gracefully runs in MOCK MODE,
enabling you to observe knowledge compounding across repeated sessions.
"""

import asyncio
import os
import sys
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

# Import CortexGit
from cortexgit import CortexGit
from cortexgit.core.memory import ConflictError
from cortexgit.llm_providers import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError
from cortexgit.llm_providers import create_llm_provider, create_embedding_provider

# Load environment variables
load_dotenv()

# --- Mock Providers for Zero-Setup execution ---

class MockLLMProvider(LLMProvider):
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

        # 4. Standard Wikipedia synthesis
        topic = "this topic"
        if "topic:" in user_message.lower():
            try:
                topic = user_message.split("topic:")[1].split("\n")[0].strip()
            except Exception:
                pass
        
        return (
            f"=== RESEARCH SUMMARY: {topic.upper()} ===\n"
            "This synthesized summary represents our accumulated knowledge on the topic.\n"
            f"Based on Wikipedia and our past sessions, {topic} is a significant area of development.\n"
            "• Key Insight: CortexGit memory successfully compounded this fact in the SQLite event log.\n"
            "• Next Steps: We should study related sub-concepts to expand the knowledge graph."
        )

class MockEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1536


# --- Wikipedia Fetcher using Standard Library ---

def fetch_wikipedia_intro(topic: str) -> str:
    """
    Fetches the introductory paragraph of a Wikipedia article using the MediaWiki API.
    Does not require any third-party libraries (uses urllib).
    """
    # Clean topic for URL
    encoded_topic = urllib.parse.quote(topic)
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro=1&explaintext=1&titles={encoded_topic}"
    
    headers = {"User-Agent": "CortexGitResearchAgent/1.0 (https://github.com/sanskar292/CortexGit)"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return ""
            
            # Extract first page content
            page_id = list(pages.keys())[0]
            if page_id == "-1":
                return ""  # Page not found
            
            return pages[page_id].get("extract", "")
    except Exception as e:
        print(f"[Wikipedia API] Fetch failed or offline ({e}). Using simulated fallback data.")
        return ""


async def main():
    print("=" * 60)
    print(" CortexGit - Knowledge Compounding Research Agent")
    print("=" * 60)

    # 1. Parse CLI arguments
    if len(sys.argv) < 2:
        print("Usage: python examples/research_agent.py \"<research topic>\"")
        print("\nUsing default topic: 'Quantum Computing'")
        topic = "Quantum Computing"
    else:
        topic = sys.argv[1].strip()

    # Create safe session and agent IDs based on the topic
    topic_slug = topic.lower().replace(" ", "-")
    session_id = f"research-session-{topic_slug}"
    agent_id = "researcher-agent"

    print(f"[Research] Topic: '{topic}'")
    print(f"[Memory] Session ID: '{session_id}' | Agent: '{agent_id}'")

    # 2. Setup Providers (Anthropic or OpenAI LLM + OpenAI or Mock Embeddings)
    # Prefer OpenAI or Anthropic depending on what is available
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if api_key:
        provider_name = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
        print(f"[Status] API key found! Initializing {provider_name.capitalize()} LLM...")
        llm = create_llm_provider(provider_name, api_key=api_key)
        
        if os.getenv("OPENAI_API_KEY"):
            embedder = create_embedding_provider("openai")
        else:
            embedder = MockEmbeddingProvider()
    else:
        print("[Warning] No API keys (OPENAI_API_KEY / ANTHROPIC_API_KEY) found.")
        print("[Status] Running in MOCK MODE with persistent SQLite backend.")
        llm = MockLLMProvider()
        embedder = MockEmbeddingProvider()

    # 3. Initialize CortexGit Memory Client
    try:
        memory = CortexGit(
            database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///cortexgit.db"),
            llm_provider=llm,
            embedding_provider=embedder,
        )
    except Exception as e:
        print(f"[Error] Failed to initialize CortexGit: {e}", file=sys.stderr)
        return

    # 4. Step 1: RETRIEVE existing research from CortexGit
    print("\n[Memory] Retrieving existing context for this research session...")
    try:
        context = await memory.get_context(
            goal=f"Show me everything we have researched on {topic} so far.",
            budget_tokens=3000,
            session_id=session_id,
            agent_id=agent_id
        )
    except Exception as e:
        print(f"[Memory Error] Could not restore context: {e}")
        context = {}

    has_past_memory = bool(context.get("events") or context.get("entities"))
    if has_past_memory:
        print("[Memory] Successfully restored past research! Restored items:")
        print(f"  - Events count: {len(context.get('events', []))}")
        print(f"  - Entity keys: {list(context.get('entities', {}).keys())}")
    else:
        print("[Memory] No prior sessions found. Starting fresh research.")

    # 5. Fetch new facts from Wikipedia (or use simulated fallback)
    print("\n[Wikipedia] Fetching facts from Wikipedia...")
    wiki_facts = fetch_wikipedia_intro(topic)
    
    if not wiki_facts:
        # High quality fallback simulated data
        wiki_facts = (
            f"{topic} is an active field of research involving numerous scientists and research laboratories globally. "
            "Recent advancements focus on improving efficiency, security, and scalability of practical applications. "
            "Collaborations between academic institutions and industry leaders are driving rapid developmental cycles."
        )
        print("[Wikipedia] Using high-fidelity simulated fallback facts.")
    else:
        print("[Wikipedia] Fetch successful! Sample text:")
        print(f"  \"{wiki_facts[:120]}...\"")

    # 6. Step 2: REASON & SYNTHESIZE using LLM
    print("\n[Agent] Synthesizing past learnings and new facts...")
    system_prompt = (
        "You are an expert research analyst.\n"
        "Compile the new findings with our historical research records below. "
        "Create an updated, unified research dossier. Do not repeat facts. Make it progressive.\n\n"
        f"--- HISTORICAL RESEARCH RECORDS (CortexGit Memory) ---\n{context}\n---------------------------------------\n"
    )
    
    user_message = (
        f"Topic: {topic}\n"
        f"New Wikipedia Facts: {wiki_facts}"
    )

    try:
        synthesis = memory.llm_provider.complete(system_prompt, user_message)
    except Exception as e:
        print(f"[LLM Error] Synthesis failed: {e}")
        return

    print("\n" + "=" * 60)
    print(synthesis)
    print("=" * 60)

    # 7. Step 3: ACT & LOG findings into memory
    print("\n[Memory] Recording synthesis to the persistent event log...")
    try:
        event = await memory.log_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type="agent",
            payload={
                "action": "research_synthesis",
                "topic": topic,
                "synthesis": synthesis
            }
        )
        print(f"[Memory] Logged research event: {event.event_id}")
    except Exception as e:
        print(f"[Memory Error] Logging event failed: {e}")
        return

    # 8. Step 4: Write key entities into the EntityRegistry (Dynamic facts)
    print("[Memory] Registering key factual entities...")
    try:
        # Record topic metadata
        await memory.write_entity(
            key=f"research.{topic_slug}.title",
            value=topic,
            agent_id=agent_id,
            event_id=str(event.event_id)
        )
        # Record a compound fact count
        facts_count = len(context.get("events", [])) + 1
        await memory.write_entity(
            key=f"research.{topic_slug}.session_runs",
            value=facts_count,
            agent_id=agent_id,
            event_id=str(event.event_id)
        )
        print("[Memory] Successfully updated Entity Registry! Run this script again to see memory compounding.")
    except ConflictError:
        print("[Memory] Note: Factual entity exists; skipped writing due to matching constraints.")
    except Exception as e:
        print(f"[Memory Error] Entity write failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
