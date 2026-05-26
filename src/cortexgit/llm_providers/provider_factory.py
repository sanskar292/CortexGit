import os
from cortexgit.llm_providers.base import LLMProvider, EmbeddingProvider
from cortexgit.llm_providers.anthropic_provider import AnthropicProvider
from cortexgit.llm_providers.openai_provider import OpenAIProvider
from cortexgit.llm_providers.openrouter_provider import OpenRouterProvider
from cortexgit.llm_providers.ollama_provider import OllamaProvider

def create_llm_provider(name: str, **kwargs) -> LLMProvider:
    """
    Creates and returns an LLMProvider instance based on the provider name.
    
    Args:
        name: Name of the provider (anthropic, openai, openrouter, ollama).
        **kwargs: Optional configuration parameters to override defaults or environment variables.
    
    Returns:
        An instantiated LLMProvider.
        
    Raises:
        ValueError: If the provider name is unknown.
    """
    name_lower = name.lower()
    
    if name_lower == "anthropic":
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        model = kwargs.get("model") or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
        return AnthropicProvider(api_key=api_key, model=model)
        
    elif name_lower == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        model = kwargs.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        embedding_model = kwargs.get("embedding_model") or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        return OpenAIProvider(api_key=api_key, model=model, embedding_model=embedding_model)
        
    elif name_lower == "openrouter":
        api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        model = kwargs.get("model") or os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3-8b-instruct:free"
        embedding_model = kwargs.get("embedding_model") or os.getenv("OPENROUTER_EMBEDDING_MODEL") or "openai/text-embedding-3-small"
        return OpenRouterProvider(api_key=api_key, model=model, embedding_model=embedding_model)
        
    elif name_lower == "ollama":
        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model = kwargs.get("model") or os.getenv("OLLAMA_MODEL") or "llama3"
        embedding_model = kwargs.get("embedding_model") or os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text"
        return OllamaProvider(base_url=base_url, model=model, embedding_model=embedding_model)
        
    else:
        raise ValueError(f"Unknown LLM provider: {name}")

def create_embedding_provider(name: str, **kwargs) -> EmbeddingProvider:
    """
    Creates and returns an EmbeddingProvider instance based on the provider name.
    
    Args:
        name: Name of the provider (anthropic, openai, openrouter, ollama).
        **kwargs: Optional configuration parameters to override defaults or environment variables.
        
    Returns:
        An instantiated EmbeddingProvider.
        
    Raises:
        ValueError: If the provider name is unknown.
    """
    name_lower = name.lower()
    
    if name_lower == "anthropic":
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        model = kwargs.get("model") or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
        return AnthropicProvider(api_key=api_key, model=model)
        
    elif name_lower == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        model = kwargs.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        embedding_model = kwargs.get("embedding_model") or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        return OpenAIProvider(api_key=api_key, model=model, embedding_model=embedding_model)
        
    elif name_lower == "openrouter":
        api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        model = kwargs.get("model") or os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3-8b-instruct:free"
        embedding_model = kwargs.get("embedding_model") or os.getenv("OPENROUTER_EMBEDDING_MODEL") or "openai/text-embedding-3-small"
        return OpenRouterProvider(api_key=api_key, model=model, embedding_model=embedding_model)
        
    elif name_lower == "ollama":
        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model = kwargs.get("model") or os.getenv("OLLAMA_MODEL") or "llama3"
        embedding_model = kwargs.get("embedding_model") or os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text"
        return OllamaProvider(base_url=base_url, model=model, embedding_model=embedding_model)
        
    else:
        raise ValueError(f"Unknown Embedding provider: {name}")
