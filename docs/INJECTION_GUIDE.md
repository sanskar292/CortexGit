# Proactive Surface Injection Guide

Proactive Surface Injection is a core retrieval feature of the CortexGit persistent agent memory system. It ensures that high-importance entities (projects, concepts, or people) from the Relational Entity Graph (REG) are automatically injected into the retrieval context, even if the agent's query goal does not explicitly mention them.

---

## What Proactive Surface Injection Does

In conventional retrieval systems, context is gathered exclusively via keyword, substring, or semantic matches against a specific query or goal. If an entity is not explicitly queried, it remains hidden.

Proactive Surface Injection upgrades this model. When getting context, the system:
1. Calculates importance scores for all entities in the Relational Entity Graph using their network topology and hit frequency.
2. Identifies "high-importance" entities whose scores exceed a configured threshold.
3. Automatically injects the most important entities into the returned context.
4. Transparency: Injected entities are marked with `"injected": True` in the returned JSON structure and recorded in the retrieval metadata.

---

## How It Improves Retrieval

Without proactive injection, an agent working on an urgent or highly critical corporate project might fail to receive context about that project if they ask an unrelated or abstract question.

By checking graph centrality (connections) and query frequency (recency/hits), Proactive Surface Injection ensures that:
* **Mission-Critical Context Surfacing**: High-mass elements (such as active master epics, leading developers, or core infrastructure nodes) are constantly present in the background of the agent's short-term context.
* **Implicit Associative Memory**: The agent receives critical contextual anchors automatically, avoiding "blind spots" in planning, reasoning, or decision-making cycles.

---

## Configuration Options

CortexGit allows fine-grained control over proactive injection via initializer arguments or environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `enable_injection` | N/A | `True` | Global toggle. If set to `False`, skips proactive injection entirely for backward compatibility. |
| `injection_threshold` | `INJECTION_IMPORTANCE_THRESHOLD` | `5.0` | Minimum importance score required for an entity node to be eligible for injection. |
| `injection_top_k` | `INJECTION_TOP_K` | `3` | Maximum number of injected entities packed into the context. |

### Importance Score Calculation
The importance of a node is calculated as:
$$\text{Importance} = (\text{Degree Centrality} \times \text{Centrality Weight}) \times (\text{Hit Frequency} \times \text{Hit Weight})$$

Where:
* **Degree Centrality**: The number of edges connected to the node.
* **Hit Frequency**: The cumulative count of context retrieval hits reinforced over time.

---

## Usage and Code Examples

### 1. Basic Initialization (Enabled by Default)

By default, proactive injection is enabled with default environment variable configurations:

```python
from cortexgit import CortexGit

# Initializes with enable_injection=True, threshold=5.0, top_k=3
memory = CortexGit()

# Retrieval automatically includes relevant injected nodes
context = await memory.get_context(
    goal="unrelated task planning",
    budget_tokens=4000,
    session_id="session-xyz",
    agent_id="cortex-agent-v1"
)
```

---

### 2. Custom Configuration

You can customize the injection parameters directly in the client constructor:

```python
from cortexgit import CortexGit

# Restrict injection to very high-importance nodes only (e.g. importance > 15.0)
# and retrieve at most 1 injected entity
memory = CortexGit(
    enable_injection=True,
    injection_threshold=15.0,
    injection_top_k=1
)
```

---

### 3. Disabling Proactive Injection

To disable proactive injection entirely for complete backward compatibility or context-saving purposes:

```python
from cortexgit import CortexGit

# Skipped entirely during get_context calls
memory = CortexGit(enable_injection=False)

context = await memory.get_context(
    goal="fetch standard queries only",
    budget_tokens=4000,
    session_id="session-xyz",
    agent_id="cortex-agent-v1"
)

# context["metadata"]["injected_entities"] will be empty []
# and no entities will carry the "injected": True property.
```

---

### 4. Reading Context Response Metadata

When injection occurs, the context JSON structure includes the injected nodes and lists them in the metadata for verification and debugging:

```python
# The returned context contains:
# {
#   "entities": {
#     "critical_payment_gateway": {
#       "value": "active",
#       "importance": 16.0,
#       "injected": True  <--- Flagged for transparency
#     }
#   },
#   "metadata": {
#     "injected_entities": ["critical_payment_gateway"]  <--- Listed in metadata
#   }
# }
```
