import json
import os
import urllib.request
from urllib.error import URLError, HTTPError
from cortexgit.llm_providers.base import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError

class OllamaProvider(LLMProvider, EmbeddingProvider):
    """Ollama provider for local LLM and embedding support using the standard HTTP API.

    Note on LM Studio Compatibility:
    - LM Studio uses an OpenAI-compatible API.
    - It works with OllamaProvider by changing the base_url.
    """

    def __init__(self, base_url: str = None, model: str = None, embedding_model: str = None):
        """Initialize the Ollama/LM Studio provider.

        Args:
            base_url: The base URL of the local server.
                - For Ollama, use http://localhost:11434
                - For LM Studio, use http://localhost:8000
            model: The name of the LLM model to use.
            embedding_model: The name of the embedding model to use.
        """
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip('/')
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    def complete(self, system_prompt: str, user_message: str) -> str:
        is_openai = "8000" in self.base_url or "1234" in self.base_url or "/v1" in self.base_url
        
        if is_openai:
            url = f"{self.base_url}/v1/chat/completions" if "/v1" not in self.base_url else f"{self.base_url}/chat/completions"
        else:
            url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                if is_openai or 'choices' in res_data:
                    if not res_data.get('choices'):
                        raise LLMError(f"Ollama returned an empty choices array. Response: {res_data}")
                    return res_data['choices'][0]['message']['content']
                return res_data['message']['content']
        except HTTPError as e:
            raise LLMError(f"Ollama HTTP error {e.code}: {e.reason}") from e
        except URLError as e:
            raise LLMError(f"Ollama connection error: {e.reason}") from e
        except Exception as e:
            raise LLMError(f"Ollama completion failed: {str(e)}") from e

    def embed(self, text: str) -> list[float]:
        is_openai = "8000" in self.base_url or "1234" in self.base_url or "/v1" in self.base_url
        
        if is_openai:
            url = f"{self.base_url}/v1/embeddings" if "/v1" not in self.base_url else f"{self.base_url}/embeddings"
        else:
            url = f"{self.base_url}/api/embed"

        payload = {
            "model": self.embedding_model,
            "input": text
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
                if is_openai or 'data' in res_data:
                    if not res_data.get('data'):
                        raise EmbeddingError(f"Ollama returned an empty data array. Response: {res_data}")
                    return res_data['data'][0]['embedding']

                # Check for standard 'embeddings' array or 'embedding' fallback
                embeddings = res_data.get('embeddings')
                if embeddings and isinstance(embeddings, list):
                    return embeddings[0]
                embedding = res_data.get('embedding')
                if embedding and isinstance(embedding, list):
                    return embedding
                raise EmbeddingError("Ollama response did not contain expected 'embeddings' or 'embedding' list.")
        except HTTPError as e:
            raise EmbeddingError(f"Ollama HTTP error {e.code}: {e.reason}") from e
        except URLError as e:
            raise EmbeddingError(f"Ollama connection error: {e.reason}") from e
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding failed: {str(e)}") from e
