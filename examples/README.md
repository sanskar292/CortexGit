# CortexGit Examples Overview & User Guide

Welcome to the **CortexGit Examples** directory! These files are designed to guide you through the core capabilities of CortexGit, starting from a basic single-agent loop to advanced multi-agent shared-memory systems and graph-based significance ranking.

All examples are **fully functional out of the box**. If you do not have external API keys (OpenAI or Anthropic), they will automatically and gracefully switch to a high-fidelity **Mock Mode** so you can see the event logs, database operations, and memory recall in action.

---

## The Examples

### 1. [Simple Chat Agent](file:///f:/aggin/CortexGit/examples/simple_chat_agent.py) (`simple_chat_agent.py`)
A basic interactive multi-turn chat application with persistent memory.
* **Demonstrates**: Basic agent loop, restoring working context on start, multi-turn dialogue saving, and event sourcing.
* **How to Run**:
  ```bash
  python examples/simple_chat_agent.py
  ```
* **What to Try**:
  1. Tell the agent your name and favorite coding language (e.g., *"Hi, I am Sanskar, and I love Rust!"*).
  2. Exchange a few dialogue turns.
  3. Exit the agent by typing `exit` or `quit`.
  4. Restart the agent (`python examples/simple_chat_agent.py`) and say: *"Do you remember who I am?"*
  5. Watch it recall your name and favorite language from history!

---

### 2. [Knowledge Compounding Research Agent](file:///f:/aggin/CortexGit/examples/research_agent.py) (`research_agent.py`)
An agent that fetches real Wikipedia introductions (using Python's standard `urllib`) to incrementally compile a factual research dossier.
* **Demonstrates**: Incremental knowledge compounding, Entity Registry updates, semantic retrieval across sessions, and Wikipedia API integration.
* **How to Run**:
  ```bash
  python examples/research_agent.py "Quantum Computing"
  ```
* **What to Try**:
  1. Run the script on a topic (e.g., *"Quantum Computing"* or *"Machine Learning"*).
  2. Examine the synthesized dossier printed in the terminal.
  3. Run the exact same command a second time.
  4. Notice the printout showing that it has **restored prior research history** and successfully **compounded** its findings in the Entity Registry!

---

### 3. [Task Automation & REG Graph Agent](file:///f:/aggin/CortexGit/examples/task_automation_agent.py) (`task_automation_agent.py`)
An interactive CLI task manager showing the **Relation Entity Graph (REG)** in action.
* **Demonstrates**: Adding nodes, connecting nodes with edges (Degree Centrality), recording node accesses (Hit Frequency), and dynamic importance ranking.
* **How to Run**:
  ```bash
  python examples/task_automation_agent.py
  ```
* **What to Try**:
  1. Add a few tasks:
     ```
     add Write project report
     add Design API endpoints
     add Set up database
     ```
  2. View the initial task list: type `list`. (All tasks will have centrality = 0, hits = 0, and importance = 0).
  3. Create dependencies/links to build connections (edges) and boost centrality:
     ```
     link Write project report to Design API endpoints
     link Design API endpoints to Set up database
     ```
  4. Record hits on critical tasks to simulate frequent accesses:
     ```
     hit Design API endpoints
     hit Design API endpoints
     ```
  5. Type `list` again. Notice how **Design API endpoints** has climbed to the top of the table due to its high centrality and multiple hits!
     * *REG Importance Formula:* `Importance = Centrality (Edges) * Hits`

---

### 4. [Multi-Agent Shared Memory Agent](file:///f:/aggin/CortexGit/examples/multi_agent_coordination.py) (`multi_agent_coordination.py`)
A simulation of two separate processes/agents (Manager Agent and Developer Agent) sharing memory via a single database.
* **Demonstrates**: Shared persistent memory, automated write conflicts (when two agents try to write different values to the same key), conflict surfacing in context, and direct DB reconciliation.
* **How to Run**:
  ```bash
  python examples/multi_agent_coordination.py
  ```
* **Expected Output**:
  1. **Agent A (Manager)** writes `project.status` -> `"active"` and logs starting the project.
  2. **Agent B (Developer)** initializes, retrieves context, and is able to see Agent A's history and active entities because they share the database!
  3. **Agent B** attempts to write `project.status` -> `"completed"`.
  4. CortexGit immediately blocks the write, throws a `ConflictError`, and records it in `ConflictLog`.
  5. Both agents retrieve the open conflict in their working context, signaling they are out of sync.
  6. The script resolves the conflict programmatically at the DB level, demonstrating clean alignment.

---

## Configuration & Environment Variables

To use real LLMs and semantic embeddings, create a `.env` file in the project root directory:

```env
# Choose between OpenAI or Anthropic for LLM
CORTEXGIT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-key-here

# Choice of Embedding Provider (Required for real semantic recall)
CORTEXGIT_EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-openai-key-here

# SQLite Database Location (Defaults to local cortexgit.db)
DATABASE_URL=sqlite+aiosqlite:///cortexgit.db
```

Enjoy pairing with CortexGit!
