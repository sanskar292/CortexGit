import pytest
from cortexgit.llm.summarizer import summarize
from cortexgit.llm.entity_extractor import extract_entities
from cortexgit.llm_providers import LLMProvider
from cortexgit.core.write_back_gate import ValidationError

class DummyLLMProvider(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.called_with = []

    def complete(self, system_prompt: str, user_message: str) -> str:
        self.called_with.append((system_prompt, user_message))
        return self.response_text

@pytest.mark.anyio
async def test_summarize_with_custom_llm_provider():
    # Valid output matching schema
    response_json = '{"summary": "Test summary", "entities_mentioned": ["x", "y"], "event_range": [1, 5]}'
    provider = DummyLLMProvider(response_json)
    
    events = [{"event_id": 1, "session_id": "s1", "agent_id": "a1", "event_type": "info", "payload": {}}]
    result = await summarize(events, llm_provider=provider)
    
    assert result["summary"] == "Test summary"
    assert len(provider.called_with) == 1
    assert "You are a memory summarizer" in provider.called_with[0][0]

@pytest.mark.anyio
async def test_summarize_with_invalid_custom_llm_provider_output():
    # Invalid output (missing required fields)
    response_json = '{"summary": "Test summary"}'
    provider = DummyLLMProvider(response_json)
    
    events = [{"event_id": 1, "session_id": "s1", "agent_id": "a1", "event_type": "info", "payload": {}}]
    with pytest.raises(ValidationError):
        await summarize(events, llm_provider=provider)

@pytest.mark.anyio
async def test_extract_entities_with_custom_llm_provider():
    response_json = '{"updates": [{"key": "user.name", "value": "Alice"}]}'
    provider = DummyLLMProvider(response_json)
    
    event = {"event_id": 1, "session_id": "s1", "agent_id": "a1", "event_type": "info", "payload": {}}
    result = await extract_entities(event, llm_provider=provider)
    
    assert result["updates"] == [{"key": "user.name", "value": "Alice"}]
    assert len(provider.called_with) == 1
    assert "You are an entity extractor" in provider.called_with[0][0]

@pytest.mark.anyio
async def test_extract_entities_with_invalid_custom_llm_provider_output():
    # Invalid structure
    response_json = '{"updates": "not a list"}'
    provider = DummyLLMProvider(response_json)
    
    event = {"event_id": 1, "session_id": "s1", "agent_id": "a1", "event_type": "info", "payload": {}}
    with pytest.raises(ValidationError):
        await extract_entities(event, llm_provider=provider)

# ----------------- Custom Embedding Provider Tests -----------------

from cortexgit.retrieval.semantic_recall import semantic_recall
from cortexgit.llm_providers import EmbeddingProvider

class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, response_vector: list[float]):
        self.response_vector = response_vector
        self.called_with = []

    def embed(self, text: str) -> list[float]:
        self.called_with.append(text)
        return self.response_vector

@pytest.mark.anyio
async def test_semantic_recall_with_custom_embedding_provider():
    provider = DummyEmbeddingProvider([0.9, 0.8, 0.7])
    from unittest.mock import AsyncMock, MagicMock
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute.return_value = mock_result
    
    result = await semantic_recall("goal text", mock_session, top_n=3, embedding_provider=provider)
    assert result == []
    assert provider.called_with == ["goal text"]


