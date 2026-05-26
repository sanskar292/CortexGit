from abc import ABC, abstractmethod

class LLMError(Exception):
    """Raised when an LLM provider encounters an error during completion."""
    pass

class EmbeddingError(Exception):
    """Raised when an embedding provider encounters an error during embedding generation."""
    pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """
        Generates a completion string given a system prompt and a user message.
        
        Args:
            system_prompt: The system instructions.
            user_message: The user prompt or message.
            
        Returns:
            The generated response string.
            
        Raises:
            LLMError: If the completion fails.
        """
        pass

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generates an embedding vector for the given text.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding vector.
            
        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        pass
