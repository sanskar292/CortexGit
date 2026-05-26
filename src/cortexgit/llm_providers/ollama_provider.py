import json
import urllib.request
from urllib.error import URLError, HTTPError
from cortexgit.llm_providers.base import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError

class OllamaProvider(LLMProvider, EmbeddingProvider):
    """Ollama provider for local LLM and embedding support using the standard HTTP API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3", embedding_model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embedding_model = embedding_model

    def complete(self, system_prompt: str, user_message: str) -> str:
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
                return res_data['message']['content']
        except HTTPError as e:
            raise LLMError(f"Ollama HTTP error {e.code}: {e.reason}") from e
        except URLError as e:
            raise LLMError(f"Ollama connection error: {e.reason}") from e
        except Exception as e:
            raise LLMError(f"Ollama completion failed: {str(e)}") from e

    def embed(self, text: str) -> list[float]:
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
