import pytest
from unittest.mock import AsyncMock, patch
from cortexgit.llm.summarizer import summarize
from cortexgit.core.write_back_gate import ValidationError

@pytest.fixture
def mock_events():
    return [
        {"event_id": "1", "event_type": "user", "payload": {"text": "hello"}},
        {"event_id": "2", "event_type": "agent", "payload": {"text": "hi"}}
    ]

@pytest.mark.integration
@pytest.mark.anyio
@patch("cortexgit.llm.summarizer.AsyncAnthropic")
async def test_valid_llm_output_passes(mock_anthropic_class, mock_events):
    """Valid LLM output passes write-back gate validation and is returned."""
    # Setup mock Anthropic client and response
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='{"summary": "Agent greeted the user.", "entities_mentioned": ["user", "agent"], "event_range": [1, 2]}')
    ]
    mock_client.messages.create.return_value = mock_message

    result = await summarize(mock_events)
    
    assert result == {
        "summary": "Agent greeted the user.",
        "entities_mentioned": ["user", "agent"],
        "event_range": [1, 2]
    }
    
    # Assert model and parameters were passed correctly
    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert "You are a memory summarizer" in kwargs["system"]


@pytest.mark.integration
@pytest.mark.anyio
@patch("cortexgit.llm.summarizer.AsyncAnthropic")
async def test_missing_field_raises_validation_error(mock_anthropic_class, mock_events):
    """LLM output with missing field raises ValidationError from the write-back gate."""
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    # Missing event_range
    mock_message.content = [
        AsyncMock(text='{"summary": "Agent greeted the user.", "entities_mentioned": ["user", "agent"]}')
    ]
    mock_client.messages.create.return_value = mock_message

    with pytest.raises(ValidationError) as exc_info:
        await summarize(mock_events)
        
    assert "required" in str(exc_info.value).lower() or "validation failed" in str(exc_info.value).lower()


@pytest.mark.integration
@pytest.mark.anyio
@patch("cortexgit.llm.summarizer.AsyncAnthropic")
async def test_extra_field_raises_validation_error(mock_anthropic_class, mock_events):
    """LLM output with extra field not in schema raises ValidationError."""
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    # Extra field extra_stuff
    mock_message.content = [
        AsyncMock(text='{"summary": "Agent greeted the user.", "entities_mentioned": ["user"], "event_range": [1, 2], "extra_stuff": true}')
    ]
    mock_client.messages.create.return_value = mock_message

    with pytest.raises(ValidationError) as exc_info:
        await summarize(mock_events)
        
    assert "additional properties" in str(exc_info.value).lower() or "validation failed" in str(exc_info.value).lower()
