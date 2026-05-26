import os
import json
from anthropic import AsyncAnthropic
from cortexgit.core.write_back_gate import WriteBackGate, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from cortexgit.db.models import SnapshotStore
from cortexgit.retrieval.embeddings import embed_text


async def summarize(events: list[dict]) -> dict:
    """
    Summarizes a sequence of events from an agent session using the Anthropic API.
    
    Uses exact system prompt from ARCHITECTURE.md.
    Enforces validation through the WriteBackGate.
    Raises ValidationError if validation fails.
    No retry or prompt modification logic on failure.
    """
    # 1. Initialize the Anthropic asynchronous client
    client = AsyncAnthropic()

    # 2. Format the events into a JSON string
    events_str = json.dumps(events, indent=2)

    # 3. Define the exact system prompt from ARCHITECTURE.md
    system_prompt = (
        "You are a memory summarizer for an AI agent system.\n"
        "You will receive a sequence of events from an agent session.\n"
        "Summarize what happened, what decisions were made, and what facts were established.\n"
        "Be specific. Do not generalize. Do not invent connections not present in the events.\n"
        "Return only valid JSON matching this schema exactly:\n"
        '{ "summary": string, "entities_mentioned": string[], "event_range": [int, int] }\n'
        "No preamble. No explanation. No markdown. Raw JSON only."
    )

    # 4. Call Anthropic API with the specified model
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Here is the sequence of events:\n{events_str}"
            }
        ]
    )

    # 5. Parse the response text as JSON
    response_text = response.content[0].text.strip()
    
    # Handle optional markdown code block wrapping from the model output defensively
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    parsed_output = json.loads(response_text)

    # 6. Pass output through WriteBackGate with schema_name="snapshot"
    gate = WriteBackGate()
    validated_output = gate.validate(parsed_output, "snapshot")

    return validated_output


class Summarizer:
    def __init__(self):
        pass

    async def summarize_session(self, events: list[dict]) -> dict:
        """Call Anthropic API to generate structured snapshot summary.
        
        CRITICAL: Output must go through write-back gate before saving to snapshot store.
        """
        return await summarize(events)


async def write_snapshot(session_id: str, validated_output: dict, db: AsyncSession = None) -> SnapshotStore:
    """
    Embeds the summary text using embed_text() and writes the snapshot to the snapshot store.
    Immutable after write. Sets event_range from validated_output['event_range'].
    """
    if db is None:
        from cortexgit.db.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            return await _write_snapshot_with_db(session_id, validated_output, session)
    else:
        return await _write_snapshot_with_db(session_id, validated_output, db)


async def _write_snapshot_with_db(session_id: str, validated_output: dict, db: AsyncSession) -> SnapshotStore:
    # 1. Embed the summary using embed_text()
    embedding = embed_text(validated_output["summary"])

    # 2. Extract lower and upper limits of the event range.
    # asyncpg treats a (lower, upper) Python tuple as an inclusive range [lower, upper],
    # which PostgreSQL INT4RANGE canonicalizes to [lower, upper+1). So we pass the raw
    # values from the LLM output without adding 1.
    lower, upper = validated_output["event_range"]
    event_range_val = (lower, upper)

    # 3. Create and add SnapshotStore entry
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
