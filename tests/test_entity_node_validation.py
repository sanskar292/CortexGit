import pytest
from cortexgit.core.write_back_gate import ValidationError
from cortexgit.graph.entity_node import validate_entity_extraction

def test_valid_entity_passes():
    """Valid entity representation passes validation."""
    valid_entity = {
      "entity_name": "CortexGit",
      "entity_type": "project",
      "properties": {
        "description": "Deterministic memory graph library",
        "status": "active"
      },
      "connected_to": [
        {"target_entity": "Sanskar", "relation_type": "created_by"},
        {"target_entity": "Antigravity", "relation_type": "developed_by"}
      ]
    }
    result = validate_entity_extraction(valid_entity)
    assert result == valid_entity


def test_valid_entity_without_optional_properties_passes():
    """Valid entity representation without optional properties dictionary passes validation."""
    valid_entity = {
      "entity_name": "Antigravity",
      "entity_type": "person",
      "connected_to": []
    }
    result = validate_entity_extraction(valid_entity)
    assert result == valid_entity


def test_missing_entity_name_fails():
    """Missing required entity_name field raises ValidationError."""
    invalid_entity = {
      "entity_type": "project",
      "connected_to": []
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    assert "entity_name" in str(exc_info.value).lower()
    assert "required" in str(exc_info.value).lower()


def test_invalid_entity_type_fails():
    """Invalid entity_type enum raises ValidationError."""
    invalid_entity = {
      "entity_name": "InvalidType",
      "entity_type": "robot",  # Must be project, concept, or person
      "connected_to": []
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    assert "enum" in str(exc_info.value).lower() or "robot" in str(exc_info.value).lower()


def test_extra_root_field_fails():
    """Extra fields in root dictionary raise ValidationError."""
    invalid_entity = {
      "entity_name": "CortexGit",
      "entity_type": "project",
      "connected_to": [],
      "unexpected_extra_field": "hello"
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    assert "additional properties" in str(exc_info.value).lower() or "unexpected_extra_field" in str(exc_info.value).lower()


def test_extra_property_field_fails():
    """Extra fields in optional properties dictionary raise ValidationError."""
    invalid_entity = {
      "entity_name": "CortexGit",
      "entity_type": "project",
      "properties": {
        "description": "Deterministic memory graph library",
        "status": "active",
        "extra_prop": "should fail"
      },
      "connected_to": []
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    assert "additional properties" in str(exc_info.value).lower() or "extra_prop" in str(exc_info.value).lower()


def test_extra_connection_field_fails():
    """Extra fields in connected_to item dictionary raise ValidationError."""
    invalid_entity = {
      "entity_name": "CortexGit",
      "entity_type": "project",
      "connected_to": [
        {"target_entity": "Sanskar", "relation_type": "created_by", "unsupported": 123}
      ]
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    assert "additional properties" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()


def test_empty_entity_name_fails():
    """Empty entity_name string raises ValidationError (minLength 1)."""
    invalid_entity = {
      "entity_name": "",
      "entity_type": "project",
      "connected_to": []
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    error_msg = str(exc_info.value).lower()
    assert "too short" in error_msg or "length" in error_msg or "minlength" in error_msg or "non-empty" in error_msg


def test_empty_description_fails():
    """Empty description string raises ValidationError if properties dictionary is provided (minLength 1)."""
    invalid_entity = {
      "entity_name": "CortexGit",
      "entity_type": "project",
      "properties": {
        "description": "",
        "status": "active"
      },
      "connected_to": []
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    error_msg = str(exc_info.value).lower()
    assert "too short" in error_msg or "length" in error_msg or "minlength" in error_msg or "non-empty" in error_msg


def test_empty_target_entity_fails():
    """Empty target_entity string in connection raises ValidationError (minLength 1)."""
    invalid_entity = {
      "entity_name": "CortexGit",
      "entity_type": "project",
      "connected_to": [
        {"target_entity": "", "relation_type": "created_by"}
      ]
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_entity_extraction(invalid_entity)
    error_msg = str(exc_info.value).lower()
    assert "too short" in error_msg or "length" in error_msg or "minlength" in error_msg or "non-empty" in error_msg

