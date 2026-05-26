# Multi-Provider LLM & Embedding Support

CortexGit supports multiple LLM and Embedding API providers out-of-the-box. This enables agents to swap reasoning (LLM) and memory retrieval (embeddings) engines seamlessly depending on cost, performance, or local execution requirements.

---

## Supported Providers

1. **OpenAI** (Default for LLM and embeddings)
2. **Anthropic** (Default for LLM when only `ANTHROPIC_API_KEY` is set)
3. **OpenRouter** (For unified cloud access to 100+ models)
4. **Ollama** (For fully local execution with zero API keys)

---

## Setup & Configuration

Configure active providers and options in your `.env` file. See the examples below.

### 1. OpenAI (Default LLM & Embedding)
- **Use Case:** High quality general reasoning and fast embeddings.
- **Configuration:**
  ```env
  CORTEXGIT_LLM_PROVIDER=openai
  CORTEXGIT_EMBEDDING_PROVIDER=openai
  OPENAI_API_KEY=your_openai_api_key_here
  OPENAI_MODEL=gpt-4o-mini
  OPENAI_EMBEDDING_MODEL=text-embedding-3-small
  ```

### 2. Anthropic
- **Use Case:** Top-tier reasoning (Claude models) for entity extraction and complex summarization.
- **Note:** Anthropic does not natively host general embedding endpoints. If you choose Anthropic for LLM, couple it with another embedding provider (like OpenAI or Ollama).
- **Configuration:**
  ```env
  CORTEXGIT_LLM_PROVIDER=anthropic
  CORTEXGIT_EMBEDDING_PROVIDER=openai
  ANTHROPIC_API_KEY=your_anthropic_api_key_here
  ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
  ```

### 3. OpenRouter
- **Use Case:** Unified gateway accessing hundreds of models (e.g. LLaMA, Mistral, Gemini) with optional free tier options.
- **Configuration:**
  ```env
  CORTEXGIT_LLM_PROVIDER=openrouter
  CORTEXGIT_EMBEDDING_PROVIDER=openrouter
  OPENROUTER_API_KEY=your_openrouter_api_key_here
  OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct:free
  OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
  ```

### 4. Ollama (Fully Local)
- **Use Case:** Fully private, offline developer environments with zero running costs.
- **Configuration:**
  ```env
  CORTEXGIT_LLM_PROVIDER=ollama
  CORTEXGIT_EMBEDDING_PROVIDER=ollama
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=llama3
  OLLAMA_EMBEDDING_MODEL=nomic-embed-text
  ```

---

## Cost & Capability Comparison

| Provider | Core Strength | Default LLM | Default Embedding | Avg. LLM Cost (per 1M tokens) | Native Embeddings? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | General Purpose, Fast | `gpt-4o-mini` | `text-embedding-3-small` | ~$0.15 (Input) / ~$0.60 (Output) | Yes |
| **Anthropic** | Complex Reasoning | `claude-3-5-sonnet-20241022` | N/A | ~$3.00 (Input) / ~$15.00 (Output) | No |
| **OpenRouter** | Multi-Model Flexibility | `meta-llama/llama-3-8b-instruct:free` | `openai/text-embedding-3-small` | Variable (Free to Premium) | Yes (via gateway) |
| **Ollama** | 100% Local & Private | `llama3` | `nomic-embed-text` | **$0.00** (Free/Offline) | Yes |

---

## Programmatic Quick Start

You can also pass custom provider instances directly into `CortexGit` at runtime:

```python
from cortexgit import CortexGit
from cortexgit.llm_providers import OpenAIProvider, OllamaProvider

# 1. Use custom runtime options
llm = OpenAIProvider(api_key="custom_key", model="gpt-4o")
embeddings = OllamaProvider(base_url="http://localhost:11434", embedding_model="nomic-embed-text")

# 2. Instantiate client
client = CortexGit(
    database_url="sqlite+aiosqlite:///cortexgit.db",
    llm_provider=llm,
    embedding_provider=embeddings
)
```
