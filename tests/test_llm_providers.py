import pytest
from cortexgit.llm_providers import (
    LLMProvider,
    EmbeddingProvider,
    LLMError,
    EmbeddingError,
)

def test_custom_exceptions():
    """Ensure LLMError and EmbeddingError can be raised and are subclasses of Exception."""
    with pytest.raises(LLMError) as exc_info:
        raise LLMError("Test LLM error message")
    assert str(exc_info.value) == "Test LLM error message"
    assert issubclass(LLMError, Exception)

    with pytest.raises(EmbeddingError) as exc_info:
        raise EmbeddingError("Test embedding error message")
    assert str(exc_info.value) == "Test embedding error message"
    assert issubclass(EmbeddingError, Exception)


def test_cannot_instantiate_llm_provider():
    """Ensure LLMProvider is an abstract class and cannot be directly instantiated."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore


def test_concrete_llm_provider_missing_implementation():
    """Ensure subclassing LLMProvider without implementing complete() raises TypeError."""
    class IncompleteLLMProvider(LLMProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteLLMProvider()  # type: ignore


def test_valid_llm_provider_implementation():
    """Ensure a concrete subclass of LLMProvider can be instantiated and behaves correctly."""
    class DummyLLMProvider(LLMProvider):
        def complete(self, system_prompt: str, user_message: str) -> str:
            return f"System: {system_prompt} | User: {user_message}"

    provider = DummyLLMProvider()
    response = provider.complete("Test system prompt", "Test user message")
    assert response == "System: Test system prompt | User: Test user message"


def test_cannot_instantiate_embedding_provider():
    """Ensure EmbeddingProvider is an abstract class and cannot be directly instantiated."""
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore


def test_concrete_embedding_provider_missing_implementation():
    """Ensure subclassing EmbeddingProvider without implementing embed() raises TypeError."""
    class IncompleteEmbeddingProvider(EmbeddingProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteEmbeddingProvider()  # type: ignore


def test_valid_embedding_provider_implementation():
    """Ensure a concrete subclass of EmbeddingProvider can be instantiated and behaves correctly."""
    class DummyEmbeddingProvider(EmbeddingProvider):
        def embed(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    provider = DummyEmbeddingProvider()
    response = provider.embed("hello")
    assert response == [0.1, 0.2, 0.3]
