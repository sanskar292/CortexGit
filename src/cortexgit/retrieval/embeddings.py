# Retrieval Embeddings module (Phase 2)
import os
import asyncio
from cortexgit.llm_providers import EmbeddingProvider

def embed_text(text: str, embedding_provider: EmbeddingProvider = None) -> list[float]:
    """
    Call EmbeddingProvider API to get text embedding vector.
    Raises an exception if the API call fails — do not catch silently.
    """
    if embedding_provider is None:
        from cortexgit.llm_providers.provider_factory import create_embedding_provider
        embedding_provider = create_embedding_provider(
            os.getenv("CORTEXGIT_EMBEDDING_PROVIDER") or "openai"
        )
    return embedding_provider.embed(text)

class EmbeddingsManager:
    def __init__(self, embedding_provider: EmbeddingProvider = None):
        self.embedding_provider = embedding_provider

    async def get_embedding(self, text: str) -> list[float]:
        """Call embedding provider to get text embedding vector."""
        if self.embedding_provider is None:
            from cortexgit.llm_providers.provider_factory import create_embedding_provider
            self.embedding_provider = create_embedding_provider(
                os.getenv("CORTEXGIT_EMBEDDING_PROVIDER") or "openai"
            )
        # Wrap sync embed function in an async wrapper to support async calls
        return await asyncio.to_thread(self.embedding_provider.embed, text)
