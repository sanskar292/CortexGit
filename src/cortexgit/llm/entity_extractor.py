# LLM Entity Extractor module (Phase 3)
import os
import json
from anthropic import AsyncAnthropic
from cortexgit.core.write_back_gate import WriteBackGate, ValidationError

async def extract_entities(event: dict) -> dict:
    """Call Anthropic API to extract entity updates from event.
    
    CRITICAL: Output must go through write-back gate before saving to entity registry.
    Raises ValidationError if validation fails.
    No retry or prompt modification logic on failure.
    """
    # 1. Initialize the Anthropic asynchronous client
    client = AsyncAnthropic()

    # 2. Format the event into a JSON string
    event_str = json.dumps(event, indent=2)

    # 3. Define the exact system prompt from ARCHITECTURE.md
    system_prompt = (
        "You are an entity extractor for an AI agent memory system.\n"
        "You will receive a single agent event.\n"
        "Extract any named entities, decisions, goals, or facts that should be remembered.\n"
        "Keys must be lowercase with dots and underscores only. Example: project.current_goal\n"
        "Return only valid JSON matching this schema exactly:\n"
        '{ "updates": [{ "key": string, "value": any }] }\n'
        "No preamble. No explanation. No markdown. Raw JSON only.\n"
        "If nothing should be extracted, return: { \"updates\": [] }"
    )

    # 4. Call Anthropic API with the specified model
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Here is the event:\n{event_str}"
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

    # 6. Pass output through WriteBackGate with schema_name="entity_extraction"
    gate = WriteBackGate()
    validated_output = gate.validate(parsed_output, "entity_extraction")

    return validated_output


class EntityExtractor:
    def __init__(self):
        pass

    async def extract_entities(self, event: dict) -> dict:
        """Call Anthropic API to extract entity updates from event.
        
        CRITICAL: Output must go through write-back gate before saving to entity registry.
        """
        return await extract_entities(event)
