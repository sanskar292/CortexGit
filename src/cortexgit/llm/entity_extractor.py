import os
import json
import asyncio
from cortexgit.core.write_back_gate import WriteBackGate, ValidationError
from cortexgit.llm_providers import LLMProvider

async def extract_entities(event: dict, llm_provider: LLMProvider = None) -> dict:
    """Call LLM API using unified LLM provider to extract entity updates from event.
    
    CRITICAL: Output must go through write-back gate before saving to entity registry.
    Raises ValidationError if validation fails.
    No retry or prompt modification logic on failure.
    """
    # 1. Format the event into a JSON string
    event_str = json.dumps(event, indent=2)

    # 2. Define the exact system prompt from ARCHITECTURE.md
    system_prompt = (
        "You are an entity extractor for an AI agent memory system.\n"
        "You will receive a single agent event.\n"
        "Extract any named entities, decisions, goals, or facts that should be remembered.\n"
        "Keys must be lowercase letters, digits, dots, and underscores ONLY. No hyphens, no spaces.\n"
        "  BAD key:  'user-name', 'gemma-session-1'\n"
        "  GOOD key: 'user.name', 'session.id'\n"
        "Return only valid JSON matching this schema exactly:\n"
        '{ "updates": [{ "key": string, "value": any }] }\n'
        "No preamble. No explanation. No markdown. Raw JSON only.\n"
        "If nothing should be extracted, return: { \"updates\": [] }"
    )

    # 3. Initialize LLM provider if not provided
    if llm_provider is None:
        from cortexgit.llm_providers.provider_factory import create_llm_provider
        llm_provider = create_llm_provider(
            os.getenv("CORTEXGIT_LLM_PROVIDER") or (
                "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
            )
        )

    # 4. Call LLM API using complete() run in a separate thread
    user_message = f"Here is the event:\n{event_str}"
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

    # 5. Pass output through WriteBackGate with schema_name="entity_extraction"
    gate = WriteBackGate()
    validated_output = gate.validate(parsed_output, "entity_extraction")


    return validated_output


async def extract_reg_entities(event: dict, llm_provider: LLMProvider = None) -> dict:
    """Extract relational entities from event for the Relational Entity Graph (REG).
    
    CRITICAL: Output must go through write-back gate before returning.
    Raises ValidationError if validation fails.
    No retry or prompt modification logic on failure.
    """
    event_str = json.dumps(event, indent=2)

    system_prompt = (
        "You are extracting entities for a relational entity graph.\n"
        "From the given event, extract:\n"
        "- Named entities (projects, concepts, people)\n"
        "- Their types (project | concept | person)\n"
        "- Relationships between them\n\n"
        "Return only valid JSON matching this schema exactly:\n"
        '{ "updates": [ { "entity_name": string, "entity_type": enum, "description": string, "properties": {"status": string}, "connected_to": [{"target_entity": string, "relation_type": string}] } ] }\n'
        "No preamble. No explanation. Raw JSON only.\n"
        'If nothing to extract, return: { "updates": [] }'
    )

    # Initialize LLM provider if not provided
    if llm_provider is None:
        from cortexgit.llm_providers.provider_factory import create_llm_provider
        llm_provider = create_llm_provider(
            os.getenv("CORTEXGIT_LLM_PROVIDER") or (
                "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
            )
        )

    # Call LLM API using complete() run in a separate thread
    user_message = f"Here is the event:\n{event_str}"
    response_text = await asyncio.to_thread(llm_provider.complete, system_prompt, user_message)
    response_text = response_text.strip()

    # Parse response text as JSON
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

    # Validate each update using validate_entity_extraction
    from cortexgit.graph.entity_node import validate_entity_extraction

    validated_updates = []
    for update in parsed_output.get("updates", []):
        # Defensively move top-level description or status into properties if they exist
        if "description" in update and ("properties" not in update or "description" not in update["properties"]):
            properties = update.setdefault("properties", {})
            properties["description"] = update.pop("description")
            
        if "status" in update and ("properties" not in update or "status" not in update["properties"]):
            properties = update.setdefault("properties", {})
            properties["status"] = update.pop("status")

        validated_update = validate_entity_extraction(update)
        validated_updates.append(validated_update)

    return {"updates": validated_updates}


class EntityExtractor:
    def __init__(self, llm_provider: LLMProvider = None):
        self.llm_provider = llm_provider

    async def extract_entities(self, event: dict) -> dict:
        """Call LLM provider to extract entity updates from event.
        
        CRITICAL: Output must go through write-back gate before saving to entity registry.
        """
        return await extract_entities(event, self.llm_provider)

    async def extract_reg_entities(self, event: dict) -> dict:
        """Call LLM provider to extract relational entities from event for the Relational Entity Graph (REG)."""
        return await extract_reg_entities(event, self.llm_provider)

