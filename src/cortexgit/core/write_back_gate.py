# Core Write-Back Gate module (Phase 1)
# Enforces validation schemas on LLM outputs before storing.
import os
import json
import jsonschema

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class WriteBackGate:
    def __init__(self):
        # Resolve the schemas directory relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.schemas_dir = os.path.join(base_dir, "schemas")

    def validate(self, output: dict, schema_name: str) -> dict:
        """Validate output against schema_name. Raise ValidationError on failure.
        
        CRITICAL: No retry logic is allowed here. If validation fails, reject immediately.
        """
        # 1. Load the corresponding schema file
        schema_file_name = f"{schema_name}.json" if not schema_name.endswith(".json") else schema_name
        schema_path = os.path.join(self.schemas_dir, schema_file_name)
        
        if not os.path.exists(schema_path):
            # Fall back to schema_name + "_schema.json" if it exists
            alternative_name = f"{schema_name}_schema.json"
            alternative_path = os.path.join(self.schemas_dir, alternative_name)
            if os.path.exists(alternative_path):
                schema_path = alternative_path
            else:
                raise FileNotFoundError(f"Schema configuration file '{schema_file_name}' not found.")

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception as e:
            raise ValidationError(f"Failed to load or parse schema file: {e}")

        # 2. Perform JSONSchema validation
        try:
            jsonschema.validate(instance=output, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            # Raise ValidationError wrapping the original message
            raise ValidationError(f"Schema validation failed: {e.message}") from e

        return output
