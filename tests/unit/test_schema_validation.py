"""
Schema Validation Tests

Ensures CLI output conforms to documented JSON schema.
This catches breaking changes to output format.
"""

import json
from pathlib import Path

import pytest

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


SCHEMA_PATH = Path(__file__).parent.parent / "fixtures" / "schemas" / "tag_result.schema.json"


@pytest.fixture
def tag_result_schema() -> dict:
    """Load the tag result JSON schema."""
    if not SCHEMA_PATH.exists():
        pytest.skip(f"Schema file not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture
def valid_single_result() -> dict:
    """A valid single-image result."""
    return {
        "file": "/path/to/image.jpg",
        "results": {
            "ram_plus": {
                "tags": [
                    {"label": "dog", "confidence": 0.95},
                    {"label": "outdoor", "confidence": 0.87},
                ],
                "model": "RAM++",
                "version": "1.3.0",
                "inference_time_ms": 142.5,
            }
        }
    }


@pytest.fixture
def valid_batch_result() -> list:
    """A valid batch result (array)."""
    return [
        {
            "file": "/path/to/image1.jpg",
            "results": {
                "ram_plus": {
                    "tags": [{"label": "cat", "confidence": 0.92}],
                    "model": "RAM++",
                    "version": "1.3.0",
                }
            }
        },
        {
            "file": "/path/to/image2.jpg",
            "results": {
                "ram_plus": {
                    "tags": [{"label": "dog", "confidence": 0.88}],
                    "model": "RAM++",
                    "version": "1.3.0",
                }
            }
        }
    ]


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestSchemaValidation:
    """Tests that validate output against JSON schema."""

    def test_valid_single_result_passes(self, tag_result_schema, valid_single_result):
        """Valid single result should pass schema validation."""
        jsonschema.validate(valid_single_result, tag_result_schema)

    def test_valid_batch_result_passes(self, tag_result_schema, valid_batch_result):
        """Valid batch result should pass schema validation."""
        jsonschema.validate(valid_batch_result, tag_result_schema)

    def test_missing_file_fails(self, tag_result_schema):
        """Result without 'file' field should fail."""
        invalid = {
            "results": {"ram_plus": {"tags": [], "model": "x", "version": "1"}}
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, tag_result_schema)

    def test_missing_results_fails(self, tag_result_schema):
        """Result without 'results' field should fail."""
        invalid = {"file": "/path/to/image.jpg"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, tag_result_schema)

    def test_invalid_confidence_range_fails(self, tag_result_schema):
        """Confidence outside 0-1 range should fail."""
        invalid = {
            "file": "/path/to/image.jpg",
            "results": {
                "plugin": {
                    "tags": [{"label": "test", "confidence": 1.5}],  # Invalid!
                    "model": "x",
                    "version": "1",
                }
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, tag_result_schema)

    def test_negative_confidence_fails(self, tag_result_schema):
        """Negative confidence should fail."""
        invalid = {
            "file": "/path/to/image.jpg",
            "results": {
                "plugin": {
                    "tags": [{"label": "test", "confidence": -0.5}],  # Invalid!
                    "model": "x",
                    "version": "1",
                }
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, tag_result_schema)

    def test_empty_label_fails(self, tag_result_schema):
        """Empty tag label should fail."""
        invalid = {
            "file": "/path/to/image.jpg",
            "results": {
                "plugin": {
                    "tags": [{"label": "", "confidence": 0.5}],  # Invalid!
                    "model": "x",
                    "version": "1",
                }
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, tag_result_schema)

    def test_missing_tag_label_fails(self, tag_result_schema):
        """Tag without label should fail."""
        invalid = {
            "file": "/path/to/image.jpg",
            "results": {
                "plugin": {
                    "tags": [{"confidence": 0.5}],  # Missing label!
                    "model": "x",
                    "version": "1",
                }
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, tag_result_schema)

    def test_null_confidence_allowed(self, tag_result_schema):
        """Null confidence should be allowed (some plugins don't provide it)."""
        valid = {
            "file": "/path/to/image.jpg",
            "results": {
                "plugin": {
                    "tags": [{"label": "test", "confidence": None}],
                    "model": "x",
                    "version": "1",
                }
            }
        }
        jsonschema.validate(valid, tag_result_schema)

    def test_confidence_omitted_allowed(self, tag_result_schema):
        """Omitted confidence should be allowed."""
        valid = {
            "file": "/path/to/image.jpg",
            "results": {
                "plugin": {
                    "tags": [{"label": "test"}],  # No confidence
                    "model": "x",
                    "version": "1",
                }
            }
        }
        jsonschema.validate(valid, tag_result_schema)

    def test_error_field_allowed(self, tag_result_schema):
        """Error field should be allowed for failed processing."""
        valid = {
            "file": "/path/to/corrupt.jpg",
            "results": {},
            "error": "Failed to read image: corrupt header"
        }
        jsonschema.validate(valid, tag_result_schema)

    def test_plugin_error_allowed(self, tag_result_schema):
        """Plugin-level error should be allowed."""
        valid = {
            "file": "/path/to/image.jpg",
            "results": {
                "failed_plugin": {
                    "tags": [],
                    "model": "x",
                    "version": "1",
                    "error": "CUDA out of memory"
                }
            }
        }
        jsonschema.validate(valid, tag_result_schema)


class TestSchemaEvolution:
    """Tests to catch unintentional schema changes."""

    def test_required_fields_documented(self, tag_result_schema):
        """Ensure required fields are explicitly documented."""
        single_def = tag_result_schema["definitions"]["singleResult"]
        assert "required" in single_def
        assert "file" in single_def["required"]
        assert "results" in single_def["required"]

    def test_tag_label_is_required(self, tag_result_schema):
        """Tag label must be required."""
        tag_def = tag_result_schema["definitions"]["tag"]
        assert "required" in tag_def
        assert "label" in tag_def["required"]

    def test_confidence_bounds_documented(self, tag_result_schema):
        """Confidence bounds should be 0-1."""
        tag_def = tag_result_schema["definitions"]["tag"]
        conf_props = tag_def["properties"]["confidence"]

        assert conf_props.get("minimum") == 0
        assert conf_props.get("maximum") == 1
