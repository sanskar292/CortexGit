import os
from anthropic import Anthropic, AnthropicError
from cortexgit.llm_providers.base import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError

class AnthropicProvider(LLMProvider, EmbeddingProvider):
    """Anthropic provider implementing LLMProvider and EmbeddingProvider."""

    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        # Initialize client. If api_key is None, anthropic client will look for ANTHROPIC_API_KEY env var.
        self.client = Anthropic(api_key=self.api_key)

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            return message.content[0].text
        except AnthropicError as e:
            raise LLMError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {str(e)}") from e

    def embed(self, text: str) -> list[float]:
        raise EmbeddingError("Anthropic does not natively support embeddings.")
