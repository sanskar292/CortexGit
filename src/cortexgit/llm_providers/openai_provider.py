import os
from openai import OpenAI, OpenAIError
from cortexgit.llm_providers.base import LLMProvider, EmbeddingProvider, LLMError, EmbeddingError

class OpenAIProvider(LLMProvider, EmbeddingProvider):
    """OpenAI provider implementing LLMProvider and EmbeddingProvider."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=self.api_key)

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMError("OpenAI returned an empty content message.")
            return content
        except OpenAIError as e:
            raise LLMError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {str(e)}") from e

    def embed(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except OpenAIError as e:
            raise EmbeddingError(f"OpenAI Embedding API error: {str(e)}") from e
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {str(e)}") from e
