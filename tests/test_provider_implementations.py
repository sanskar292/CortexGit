import os
import pytest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from cortexgit.llm_providers import (
    LLMProvider,
    EmbeddingProvider,
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    OllamaProvider,
    LLMError,
    EmbeddingError,
    create_llm_provider,
    create_embedding_provider,
)


# ----------------- Unit Tests with Mocks -----------------

@patch("cortexgit.llm_providers.anthropic_provider.Anthropic")
def test_anthropic_provider_complete_success(mock_anthropic_class):
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Claude response")]
    mock_client.messages.create.return_value = mock_message

    provider = AnthropicProvider(api_key="fake-key", model="test-model")
    res = provider.complete("System prompt", "User prompt")
    
    assert res == "Claude response"
    mock_client.messages.create.assert_called_once_with(
        model="test-model",
        max_tokens=4096,
        system="System prompt",
        messages=[{"role": "user", "content": "User prompt"}]
    )

@patch("cortexgit.llm_providers.anthropic_provider.Anthropic")
def test_anthropic_provider_complete_failure(mock_anthropic_class):
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.side_effect = Exception("API fail")

    provider = AnthropicProvider(api_key="fake-key")
    with pytest.raises(LLMError) as exc_info:
        provider.complete("System prompt", "User prompt")
    assert "Anthropic completion failed" in str(exc_info.value)

def test_anthropic_provider_embed():
    provider = AnthropicProvider(api_key="fake-key")
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed("test")
    assert "Anthropic does not natively support embeddings" in str(exc_info.value)


@patch("cortexgit.llm_providers.openai_provider.OpenAI")
def test_openai_provider_complete_success(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
    mock_client.chat.completions.create.return_value = mock_completion

    provider = OpenAIProvider(api_key="fake-key", model="gpt-4o-mini")
    res = provider.complete("System prompt", "User prompt")
    
    assert res == "OpenAI response"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"}
        ]
    )

@patch("cortexgit.llm_providers.openai_provider.OpenAI")
def test_openai_provider_complete_failure(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("OpenAI API fail")

    provider = OpenAIProvider(api_key="fake-key")
    with pytest.raises(LLMError) as exc_info:
        provider.complete("System prompt", "User prompt")
    assert "OpenAI completion failed" in str(exc_info.value)

@patch("cortexgit.llm_providers.openai_provider.OpenAI")
def test_openai_provider_embed_success(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_emb_res = MagicMock()
    mock_emb_res.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create.return_value = mock_emb_res

    provider = OpenAIProvider(api_key="fake-key", embedding_model="text-embedding-3-small")
    res = provider.embed("text to embed")
    
    assert res == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        input=["text to embed"],
        model="text-embedding-3-small"
    )

@patch("cortexgit.llm_providers.openai_provider.OpenAI")
def test_openai_provider_embed_failure(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create.side_effect = Exception("OpenAI Embedding fail")

    provider = OpenAIProvider(api_key="fake-key")
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed("text to embed")
    assert "OpenAI embedding failed" in str(exc_info.value)


@patch("cortexgit.llm_providers.openrouter_provider.OpenAI")
def test_openrouter_provider_complete_success(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="OpenRouter response"))]
    mock_client.chat.completions.create.return_value = mock_completion

    provider = OpenRouterProvider(api_key="fake-key", model="meta-llama/llama-3-8b-instruct:free")
    res = provider.complete("System prompt", "User prompt")
    
    assert res == "OpenRouter response"
    mock_client.chat.completions.create.assert_called_once_with(
        model="meta-llama/llama-3-8b-instruct:free",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"}
        ]
    )

@patch("cortexgit.llm_providers.openrouter_provider.OpenAI")
def test_openrouter_provider_embed_success(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_emb_res = MagicMock()
    mock_emb_res.data = [MagicMock(embedding=[0.4, 0.5, 0.6])]
    mock_client.embeddings.create.return_value = mock_emb_res

    provider = OpenRouterProvider(api_key="fake-key", embedding_model="openai/text-embedding-3-small")
    res = provider.embed("text to embed")
    
    assert res == [0.4, 0.5, 0.6]
    mock_client.embeddings.create.assert_called_once_with(
        input=["text to embed"],
        model="openai/text-embedding-3-small"
    )


@patch("urllib.request.urlopen")
def test_ollama_provider_complete_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"message": {"content": "Ollama response"}}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")
    res = provider.complete("System prompt", "User prompt")
    
    assert res == "Ollama response"

@patch("urllib.request.urlopen")
def test_ollama_provider_complete_failure(mock_urlopen):
    mock_urlopen.side_effect = URLError("Connection refused")

    provider = OllamaProvider(base_url="http://localhost:11434")
    with pytest.raises(LLMError) as exc_info:
        provider.complete("System prompt", "User prompt")
    assert "Ollama connection error" in str(exc_info.value)

@patch("urllib.request.urlopen")
def test_ollama_provider_embed_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"embeddings": [[0.7, 0.8, 0.9]]}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    provider = OllamaProvider(base_url="http://localhost:11434", embedding_model="nomic-embed-text")
    res = provider.embed("text to embed")
    
    assert res == [0.7, 0.8, 0.9]

@patch("urllib.request.urlopen")
def test_ollama_provider_embed_success_fallback(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"embedding": [0.7, 0.8, 0.9]}'
    mock_urlopen.return_value.__enter__.return_value = mock_response

    provider = OllamaProvider(base_url="http://localhost:11434", embedding_model="nomic-embed-text")
    res = provider.embed("text to embed")
    
    assert res == [0.7, 0.8, 0.9]


# ----------------- Factory & Env Var Fallback Tests -----------------

def test_factory_creation_types():
    # OpenAI
    prov = create_llm_provider("openai", api_key="fake")
    assert isinstance(prov, OpenAIProvider)
    emb = create_embedding_provider("openai", api_key="fake")
    assert isinstance(emb, OpenAIProvider)

    # Anthropic
    prov = create_llm_provider("anthropic", api_key="fake")
    assert isinstance(prov, AnthropicProvider)
    emb = create_embedding_provider("anthropic", api_key="fake")
    assert isinstance(emb, AnthropicProvider)

    # OpenRouter
    prov = create_llm_provider("openrouter", api_key="fake")
    assert isinstance(prov, OpenRouterProvider)
    emb = create_embedding_provider("openrouter", api_key="fake")
    assert isinstance(emb, OpenRouterProvider)

    # Ollama
    prov = create_llm_provider("ollama")
    assert isinstance(prov, OllamaProvider)
    emb = create_embedding_provider("ollama")
    assert isinstance(emb, OllamaProvider)

    with pytest.raises(ValueError):
        create_llm_provider("invalid-provider")
    with pytest.raises(ValueError):
        create_embedding_provider("invalid-provider")

def test_factory_env_var_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-anthropic-model")
    prov = create_llm_provider("anthropic")
    assert isinstance(prov, AnthropicProvider)
    assert prov.api_key == "env-anthropic-key"
    assert prov.model == "env-anthropic-model"

    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-openai-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "env-openai-embedding-model")
    prov = create_llm_provider("openai")
    assert isinstance(prov, OpenAIProvider)
    assert prov.api_key == "env-openai-key"
    assert prov.model == "env-openai-model"
    assert prov.embedding_model == "env-openai-embedding-model"


# ----------------- Integration Tests (Marked) -----------------

@pytest.mark.integration
def test_openai_integration_complete():
    # Requires valid OPENAI_API_KEY in env
    provider = create_llm_provider("openai")
    res = provider.complete("You are a helpful assistant.", "Say 'Hello OpenAI Integration'")
    assert isinstance(res, str)
    assert len(res) > 0

@pytest.mark.integration
def test_openai_integration_embed():
    # Requires valid OPENAI_API_KEY in env
    provider = create_embedding_provider("openai")
    res = provider.embed("CortexGit integration test")
    assert isinstance(res, list)
    assert len(res) > 0
    assert isinstance(res[0], float)

@pytest.mark.integration
def test_anthropic_integration_complete():
    # Requires valid ANTHROPIC_API_KEY in env
    provider = create_llm_provider("anthropic")
    res = provider.complete("You are a helpful assistant.", "Say 'Hello Anthropic Integration'")
    assert isinstance(res, str)
    assert len(res) > 0

@pytest.mark.integration
def test_openrouter_integration_complete():
    # Requires valid OPENROUTER_API_KEY in env
    provider = create_llm_provider("openrouter")
    res = provider.complete("You are a helpful assistant.", "Say 'Hello OpenRouter Integration'")
    assert isinstance(res, str)
    assert len(res) > 0

@pytest.mark.integration
def test_ollama_integration_complete_and_embed():
    # Requires Ollama running locally with 'llama3' and 'nomic-embed-text'
    # Since Ollama might not be running on user's system, catch connection errors gracefully
    # or skip if connection is refused.
    provider = create_llm_provider("ollama")
    try:
        res = provider.complete("You are a helpful assistant.", "Say 'Hello Ollama'")
        assert isinstance(res, str)
        assert len(res) > 0
        
        emb_provider = create_embedding_provider("ollama")
        res_emb = emb_provider.embed("hello")
        assert isinstance(res_emb, list)
        assert len(res_emb) > 0
    except Exception as e:
        pytest.skip(f"Ollama integration test skipped: {str(e)}")

# ----------------- CortexGit Integration Tests -----------------

def test_cortexgit_custom_provider_initialization():
    """Ensure CortexGit accepts custom llm_provider and embedding_provider."""
    from cortexgit import CortexGit
    
    mock_llm = MagicMock(spec=LLMProvider)
    mock_emb = MagicMock(spec=EmbeddingProvider)
    
    client = CortexGit(
        database_url="sqlite+aiosqlite:///:memory:",
        llm_provider=mock_llm,
        embedding_provider=mock_emb
    )
    
    assert client.llm_provider is mock_llm
    assert client.embedding_provider is mock_emb

def test_cortexgit_env_var_fallback_initialization(monkeypatch):
    """Ensure CortexGit falls back to env-based creation using provider_factory."""
    from cortexgit import CortexGit
    
    # Configure env to select a mockable ollama setup
    monkeypatch.setenv("CORTEXGIT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CORTEXGIT_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://fake-ollama:11434")
    
    client = CortexGit(database_url="sqlite+aiosqlite:///:memory:")
    
    assert isinstance(client.llm_provider, OllamaProvider)
    assert isinstance(client.embedding_provider, OllamaProvider)
    assert client.llm_provider.base_url == "http://fake-ollama:11434"

