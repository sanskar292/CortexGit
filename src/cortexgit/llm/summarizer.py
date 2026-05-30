import os
import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.core.write_back_gate import WriteBackGate, ValidationError
from cortexgit.db.models import SnapshotStore
from cortexgit.llm_providers import LLMProvider, EmbeddingProvider

async def summarize(events: list[dict], llm_provider: LLMProvider = None) -> dict:
    """
    Summarizes a sequence of events from an agent session using the LLM provider.
    
    Uses exact system prompt from ARCHITECTURE.md.
    Enforces validation through the WriteBackGate.
    Raises ValidationError if validation fails.
    No retry or prompt modification logic on failure.
    """
    # 1. Format the events into a JSON string
    events_str = json.dumps(events, indent=2)

    # 2. Define the exact system prompt from ARCHITECTURE.md
    system_prompt = (
        "You are a memory summarizer for an AI agent system.\n"
        "You will receive a sequence of events from an agent session.\n"
        "Summarize what happened, what decisions were made, and what facts were established.\n"
        "Be specific. Do not generalize. Do not invent connections not present in the events.\n"
        "Return only valid JSON matching this schema exactly:\n"
        '{ "summary": string, "entities_mentioned": string[], "event_range": [int, int] }\n'
        "No preamble. No explanation. No markdown. Raw JSON only."
    )

    # 3. Initialize LLM provider if not provided
    if llm_provider is None:
        from cortexgit.llm_providers.provider_factory import create_llm_provider
        llm_provider = create_llm_provider(
            os.getenv("CORTEXGIT_LLM_PROVIDER") or (
                "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
            )
        )

    # Call LLM API using the provider complete() method run in a separate thread to avoid blocking
    user_message = f"Here is the sequence of events:\n{events_str}"
    response_text = await asyncio.to_thread(llm_provider.complete, system_prompt, user_message)
    response_text = response_text.strip()

    # 4. Parse the response text as JSON
    # Handle optional markdown code block wrapping from the model output defensively
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        parsed_output = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValidationError(f"LLM returned invalid JSON: {e}. Raw response: {response_text!r}") from e

    # 5. Pass output through WriteBackGate with schema_name="snapshot"
    gate = WriteBackGate()
    validated_output = gate.validate(parsed_output, "snapshot")

    return validated_output


class Summarizer:
    def __init__(self, llm_provider: LLMProvider = None):
        self.llm_provider = llm_provider

    async def summarize_session(self, events: list[dict]) -> dict:
        """Call LLM provider to generate structured snapshot summary.
        
        CRITICAL: Output must go through write-back gate before saving to snapshot store.
        """
        return await summarize(events, self.llm_provider)


async def write_snapshot(
    session_id: str,
    validated_output: dict,
    db: AsyncSession = None,
    embedding_provider: EmbeddingProvider = None
) -> SnapshotStore:
    """
    Embeds the summary text using embedding_provider and writes the snapshot to the snapshot store.
    Immutable after write. Sets event_range from validated_output['event_range'].
    """
    if db is None:
        from cortexgit.db.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            return await _write_snapshot_with_db(session_id, validated_output, session, embedding_provider)
    else:
        return await _write_snapshot_with_db(session_id, validated_output, db, embedding_provider)


async def _write_snapshot_with_db(
    session_id: str,
    validated_output: dict,
    db: AsyncSession,
    embedding_provider: EmbeddingProvider = None
) -> SnapshotStore:
    # 1. Initialize fallback embedding provider if not provided
    if embedding_provider is None:
        from cortexgit.llm_providers.provider_factory import create_embedding_provider
        embedding_provider = create_embedding_provider(
            os.getenv("CORTEXGIT_EMBEDDING_PROVIDER") or "openai"
        )

    # 2. Embed the summary using embedding_provider in a separate thread
    embedding = await asyncio.to_thread(embedding_provider.embed, validated_output["summary"])

    # 3. Extract lower and upper limits of the event range.
    lower, upper = validated_output["event_range"]
    event_range_val = (lower, upper)

    # 4. Create and add SnapshotStore entry
    snapshot = SnapshotStore(
        session_id=session_id,
        event_range=event_range_val,
        summary=validated_output["summary"],
        entities_mentioned=validated_output["entities_mentioned"],
        embedding=embedding
    )

    db.add(snapshot)
    await db.commit()
    await db.refresh(snapshot)
    return snapshot
