# Retrieval Embeddings module (Phase 2)
import os
from openai import OpenAI

def embed_text(text: str) -> list[float]:
    """
    Call OpenAI text-embedding-3-small API to get text embedding vector.
    Raises an exception if the API call fails — do not catch silently.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

class EmbeddingsManager:
    def __init__(self):
        pass

    async def get_embedding(self, text: str) -> list[float]:
        """Call OpenAI text-embedding-3-small API to get text embedding vector."""
        # Wrap sync function in an async wrapper to support async calls if needed
        return embed_text(text)
