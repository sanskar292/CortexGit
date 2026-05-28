# Setting up LM Studio with CortexGit

LM Studio is a desktop application that lets you run open-source large language models (LLMs) locally on your own computer, fully offline and private. Since LM Studio hosts a local HTTP server, it can be used seamlessly as a local provider for CortexGit.

---

## 1. Install LM Studio

1. Download the installer for your operating system from the official [LM Studio Website](https://lmstudio.ai/).
2. Run the installer and open the LM Studio application.

---

## 2. Load a Model in LM Studio

1. **Search and Download:**
   - Click on the **Search (Magnifying Glass)** icon in the left sidebar.
   - Search for a popular GGUF model (e.g., `Meta-Llama-3-8B-Instruct` or `Qwen2.5-7B-Instruct`).
   - Click **Download** on your preferred quantization level (e.g., `Q4_K_M` or `Q8_0` depending on your RAM/VRAM).

2. **Start the Local Server:**
   - Click on the **Local Server (Double Arrow / Developer)** icon in the left sidebar.
   - Select your downloaded model from the dropdown at the top to load it into memory.
   - (Optional) Enable GPU acceleration if you have a compatible NVIDIA, Apple Silicon, or AMD graphics card.
   - In the settings on the right panel, verify the **Port** is set to `8000`. If it's different, you can adjust it or note the custom port.
   - Click **Start Server**. The server will start listening on `http://localhost:8000`.

---

## 3. Configure CortexGit to use LM Studio

You can use the local LM Studio instance by setting the environment variables in your `.env` file or programmatically.

### Option A: Environment Variables (`.env`)

Add or update the following values in your `.env` file:

```env
CORTEXGIT_LLM_PROVIDER=ollama
CORTEXGIT_EMBEDDING_PROVIDER=ollama

# Point Ollama base URL to the LM Studio default server
OLLAMA_BASE_URL=http://localhost:8000

# Specify the model identifier as it appears loaded in LM Studio
OLLAMA_MODEL=meta-llama-3-8b-instruct
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Option B: Programmatic Setup

You can initialize and pass the provider configuration directly in your Python code:

```python
from cortexgit import CortexGit
from cortexgit.llm_providers import OllamaProvider

# 1. Initialize the providers pointing to LM Studio on port 8000
local_llm = OllamaProvider(
    base_url="http://localhost:8000",
    model="meta-llama-3-8b-instruct"
)

local_embeddings = OllamaProvider(
    base_url="http://localhost:8000",
    embedding_model="nomic-embed-text"
)

# 2. Pass them to the CortexGit client
client = CortexGit(
    database_url="sqlite+aiosqlite:///cortexgit.db",
    llm_provider=local_llm,
    embedding_provider=local_embeddings
)
```

---

## 4. Troubleshooting & Tips

- **Server Connection Refused:** Make sure you clicked **Start Server** in the LM Studio local server tab and that no other application is using port 8000.
- **Model Name Match:** The model name specified in `model` or `OLLAMA_MODEL` must match the identifier or loaded model in the LM Studio server log/dropdown, although LM Studio often defaults to the currently active loaded model regardless of the exact requested model name.
- **Embeddings Support:** If you want to use local embeddings, make sure to load an embedding-capable model in LM Studio or use a separate local provider for embeddings.
