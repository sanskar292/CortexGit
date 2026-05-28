import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from cortexgit.core.write_back_gate import WriteBackGate, ValidationError

# Resolve the schemas directory relative to this file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMAS_DIR = os.path.join(base_dir, "schemas")
SCHEMA_PATH = os.path.join(SCHEMAS_DIR, "reg_entity_schema.json")

try:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        REG_ENTITY_SCHEMA = json.load(f)
except Exception:
    # Fallback inline schema
    REG_ENTITY_SCHEMA = {
        "type": "object",
        "required": ["entity_name", "entity_type", "connected_to"],
        "additionalProperties": False,
        "properties": {
            "entity_name": {
                "type": "string",
                "minLength": 1
            },
            "entity_type": {
                "type": "string",
                "enum": ["project", "concept", "person"]
            },
            "properties": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "description": {
                        "type": "string",
                        "minLength": 1
                    },
                    "status": {
                        "type": "string",
                        "minLength": 1
                    }
                }
            },
            "connected_to": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["target_entity", "relation_type"],
                    "additionalProperties": false,
                    "properties": {
                        "target_entity": {
                            "type": "string",
                            "minLength": 1
                        },
                        "relation_type": {
                            "type": "string",
                            "minLength": 1
                        }
                    }
                }
            }
        }
    }


@dataclass
class EntityNode:
    entity_name: str
    entity_type: str  # project | concept | person
    node_id: Optional[UUID] = None
    description: Optional[str] = None
    status: Optional[str] = None
    degree_centrality: float = 0.0
    hit_frequency: int = 0
    last_hit: Optional[datetime] = None
    ttl_expiry: Optional[datetime] = None


def validate_entity_extraction(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate LLM output dict against the REG_ENTITY_SCHEMA.
    Returns the validated dict or raises ValidationError on failure.
    """
    gate = WriteBackGate()
    return gate.validate(llm_output, "reg_entity_schema")
