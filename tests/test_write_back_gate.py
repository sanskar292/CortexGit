import pytest
from cortexgit.core.write_back_gate import WriteBackGate, ValidationError

@pytest.fixture
def gate():
    return WriteBackGate()

def test_valid_snapshot_passes(gate):
    """Valid snapshot output passes."""
    valid_snapshot = {
        "summary": "This is a valid summary of events that has at least ten characters.",
        "entities_mentioned": ["agent_alpha", "project_goal"],
        "event_range": [1, 50]
    }
    result = gate.validate(valid_snapshot, "snapshot_schema")
    assert result == valid_snapshot

def test_valid_entity_extraction_passes(gate):
    """Valid entity extraction output passes."""
    valid_extraction = {
        "updates": [
            {"key": "agent.current_task", "value": "debugging tests"},
            {"key": "system.status", "value": {"healthy": True}}
        ]
    }
    result = gate.validate(valid_extraction, "entity_extraction_schema")
    assert result == valid_extraction

def test_missing_required_field_fails(gate):
    """Missing required field raises ValidationError."""
    invalid_snapshot = {
        "summary": "Summary is long enough, but missing entities_mentioned and event_range."
    }
    with pytest.raises(ValidationError) as exc_info:
        gate.validate(invalid_snapshot, "snapshot_schema")
    assert "required" in str(exc_info.value).lower()

def test_wrong_type_fails(gate):
    """Wrong type raises ValidationError."""
    invalid_snapshot = {
        "summary": 1234567890,  # Should be string
        "entities_mentioned": ["alpha"],
        "event_range": [1, 5]
    }
    with pytest.raises(ValidationError) as exc_info:
        gate.validate(invalid_snapshot, "snapshot_schema")
    assert "is not of type 'string'" in str(exc_info.value)

def test_extra_field_fails(gate):
    """Extra field not in schema raises ValidationError (since additionalProperties: false)."""
    invalid_snapshot = {
        "summary": "This is a valid summary of events.",
        "entities_mentioned": ["alpha"],
        "event_range": [1, 5],
        "unsupported_extra_field": True  # Not in schema
    }
    with pytest.raises(ValidationError) as exc_info:
        gate.validate(invalid_snapshot, "snapshot_schema")
    assert "Additional properties are not allowed" in str(exc_info.value)

def test_empty_summary_fails(gate):
    """Empty summary string raises ValidationError (minLength is 10)."""
    invalid_snapshot = {
        "summary": "",  # Empty
        "entities_mentioned": ["alpha"],
        "event_range": [1, 5]
    }
    with pytest.raises(ValidationError) as exc_info:
        gate.validate(invalid_snapshot, "snapshot_schema")
    assert "is too short" in str(exc_info.value)

def test_empty_updates_passes(gate):
    """Empty updates array passes (valid case)."""
    valid_extraction = {
        "updates": []  # Empty array is valid
    }
    result = gate.validate(valid_extraction, "entity_extraction_schema")
    assert result == valid_extraction
