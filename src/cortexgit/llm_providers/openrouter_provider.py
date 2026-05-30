import os
from openai import OpenAI, OpenAIError
from cortexgit.llm_providers.base import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError

class OpenRouterProvider(LLMProvider, EmbeddingProvider):
    """OpenRouter provider implementing LLMProvider and EmbeddingProvider."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        embedding_model: str = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
        self.embedding_model = embedding_model or os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/sanskar292/CortexGit",
                "X-Title": "CortexGit"
            }
        )

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            if not response.choices:
                raise LLMError("OpenRouter returned an empty choices array.")
            content = response.choices[0].message.content
            if content is None:
                raise LLMError("OpenRouter returned an empty content message.")
            return content
        except OpenAIError as e:
            raise LLMError(f"OpenRouter API error: {str(e)}") from e
        except Exception as e:
            raise LLMError(f"OpenRouter completion failed: {str(e)}") from e

    def embed(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except OpenAIError as e:
            raise EmbeddingError(f"OpenRouter Embedding API error: {str(e)}") from e
        except Exception as e:
            raise EmbeddingError(f"OpenRouter embedding failed: {str(e)}") from e
