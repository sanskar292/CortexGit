import pytest
from unittest.mock import patch, AsyncMock
from cortexgit.core.write_back_gate import ValidationError
from cortexgit.llm.entity_extractor import extract_reg_entities

@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    """Ensure pytest-asyncio runs tests correctly."""
    return "asyncio"


@pytest.mark.anyio
@patch("cortexgit.llm.entity_extractor.AsyncAnthropic")
async def test_valid_reg_entity_output_passes(mock_anthropic_class):
    """Valid REG entity output from the LLM passes successfully."""
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='''
        {
          "updates": [
            {
              "entity_name": "CortexGit",
              "entity_type": "project",
              "properties": {
                "description": "SDK",
                "status": "active"
              },
              "connected_to": [
                {"target_entity": "Sanskar", "relation_type": "created_by"}
              ]
            }
          ]
        }
        ''')
    ]
    mock_client.messages.create.return_value = mock_message

    result = await extract_reg_entities({"text": "dummy event"})
    assert "updates" in result
    assert len(result["updates"]) == 1
    assert result["updates"][0]["entity_name"] == "CortexGit"
    assert result["updates"][0]["entity_type"] == "project"
    assert result["updates"][0]["connected_to"][0]["target_entity"] == "Sanskar"


@pytest.mark.anyio
@patch("cortexgit.llm.entity_extractor.AsyncAnthropic")
async def test_missing_connected_to_fails(mock_anthropic_class):
    """REG entity output missing connected_to fails validation with ValidationError."""
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='''
        {
          "updates": [
            {
              "entity_name": "CortexGit",
              "entity_type": "project",
              "properties": {
                "description": "SDK"
              }
            }
          ]
        }
        ''')
    ]
    mock_client.messages.create.return_value = mock_message

    with pytest.raises(ValidationError) as exc_info:
        await extract_reg_entities({"text": "dummy event"})
    assert "connected_to" in str(exc_info.value).lower() or "validation failed" in str(exc_info.value).lower()


@pytest.mark.anyio
@patch("cortexgit.llm.entity_extractor.AsyncAnthropic")
async def test_invalid_entity_type_fails(mock_anthropic_class):
    """REG entity output with an invalid entity_type enum value fails validation."""
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='''
        {
          "updates": [
            {
              "entity_name": "CortexGit",
              "entity_type": "invalid_type",
              "connected_to": []
            }
          ]
        }
        ''')
    ]
    mock_client.messages.create.return_value = mock_message

    with pytest.raises(ValidationError) as exc_info:
        await extract_reg_entities({"text": "dummy event"})
    assert "enum" in str(exc_info.value).lower() or "invalid_type" in str(exc_info.value).lower() or "validation failed" in str(exc_info.value).lower()


@pytest.mark.anyio
@patch("cortexgit.llm.entity_extractor.AsyncAnthropic")
async def test_empty_updates_array_passes(mock_anthropic_class):
    """An empty updates array in the LLM response is valid and passes cleanly."""
    mock_client = AsyncMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_message = AsyncMock()
    mock_message.content = [
        AsyncMock(text='{"updates": []}')
    ]
    mock_client.messages.create.return_value = mock_message

    result = await extract_reg_entities({"text": "dummy event"})
    assert result == {"updates": []}
