"""Tests for visual_buffet.services.xmp_handler module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


class TestXMPSidecarPath:
    """Test XMP sidecar path generation."""

    def test_get_sidecar_path_jpg(self, temp_dir):
        """Test sidecar path for JPEG."""
        from visual_buffet.services.xmp_handler import get_sidecar_path

        image_path = temp_dir / "photo.jpg"
        sidecar = get_sidecar_path(image_path)
        assert sidecar == temp_dir / "photo.jpg.xmp"

    def test_get_sidecar_path_raw(self, temp_dir):
        """Test sidecar path for RAW."""
        from visual_buffet.services.xmp_handler import get_sidecar_path

        image_path = temp_dir / "photo.ARW"
        sidecar = get_sidecar_path(image_path)
        assert sidecar == temp_dir / "photo.ARW.xmp"

    def test_has_xmp_sidecar_false(self, temp_dir):
        """Test has_xmp_sidecar returns False when no sidecar."""
        from visual_buffet.services.xmp_handler import has_xmp_sidecar

        image_path = temp_dir / "photo.jpg"
        image_path.touch()
        assert has_xmp_sidecar(image_path) is False

    def test_has_xmp_sidecar_true(self, temp_dir):
        """Test has_xmp_sidecar returns True when sidecar exists."""
        from visual_buffet.services.xmp_handler import get_sidecar_path, has_xmp_sidecar

        image_path = temp_dir / "photo.jpg"
        image_path.touch()
        get_sidecar_path(image_path).touch()
        assert has_xmp_sidecar(image_path) is True


class TestXMPTagData:
    """Test XMPTagData dataclass."""

    def test_xmp_tag_data_defaults(self):
        """Test XMPTagData default values."""
        from visual_buffet.services.xmp_handler import XMPTagData

        data = XMPTagData()
        assert data.tags == []
        assert data.plugins_used == []
        assert data.tagged_at != ""  # Auto-generated
        assert data.schema_version >= 1
        assert data.threshold == 0.0
        assert data.size_used == "original"
        assert data.inference_time_ms == 0.0

    def test_xmp_tag_data_with_values(self):
        """Test XMPTagData with custom values."""
        from visual_buffet.services.xmp_handler import MLTag, XMPTagData

        tags = [MLTag(label="dog", confidence=0.95)]
        data = XMPTagData(
            tags=tags,
            plugins_used=["ram_plus"],
            threshold=0.5,
            size_used="small",
            inference_time_ms=142.5,
        )
        assert len(data.tags) == 1
        assert data.tags[0].label == "dog"
        assert data.plugins_used == ["ram_plus"]
        assert data.threshold == 0.5


class TestMLTag:
    """Test MLTag dataclass."""

    def test_ml_tag_minimal(self):
        """Test MLTag with minimal data."""
        from visual_buffet.services.xmp_handler import MLTag

        tag = MLTag(label="cat")
        assert tag.label == "cat"
        assert tag.confidence is None
        assert tag.plugin is None

    def test_ml_tag_full(self):
        """Test MLTag with all fields."""
        from visual_buffet.services.xmp_handler import MLTag

        tag = MLTag(label="dog", confidence=0.95, plugin="ram_plus")
        assert tag.label == "dog"
        assert tag.confidence == 0.95
        assert tag.plugin == "ram_plus"


class TestXMPHandler:
    """Test XMPHandler class."""

    def test_handler_init_no_exiftool(self):
        """Test handler initializes without exiftool."""
        from visual_buffet.services.xmp_handler import XMPHandler

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()
            assert handler.exiftool_available is False

    def test_handler_init_with_exiftool(self):
        """Test handler initializes with exiftool."""
        from visual_buffet.services.xmp_handler import XMPHandler

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=True):
            handler = XMPHandler()
            assert handler.exiftool_available is True

    def test_has_tags_no_files(self, temp_dir, test_image):
        """Test has_tags returns False when no tag files."""
        from visual_buffet.services.xmp_handler import XMPHandler

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()
            assert handler.has_tags(test_image) is False

    def test_has_tags_with_json(self, temp_dir, test_image):
        """Test has_tags returns True when JSON exists."""
        from visual_buffet.services.xmp_handler import XMPHandler

        # Create JSON tags file
        json_path = test_image.parent / f"{test_image.stem}_tags.json"
        json_path.write_text('{"results": {}}')

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()
            assert handler.has_tags(test_image) is True

    def test_write_tags_json_fallback(self, temp_dir, test_image):
        """Test write_tags falls back to JSON when no exiftool."""
        from visual_buffet.services.xmp_handler import XMPHandler

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()
            success = handler.write_tags(
                test_image,
                [{"label": "dog", "confidence": 0.95, "plugin": "ram_plus"}],
                plugins_used=["ram_plus"],
                threshold=0.5,
            )

            assert success is True

            # Check JSON file was created
            json_path = test_image.parent / f"{test_image.stem}_tags.json"
            assert json_path.exists()

            # Verify content
            data = json.loads(json_path.read_text())
            assert "file" in data
            assert "results" in data
            assert data["threshold"] == 0.5

    def test_read_tags_from_json(self, temp_dir, test_image):
        """Test read_tags from JSON file."""
        from visual_buffet.services.xmp_handler import XMPHandler

        # Create JSON tags file
        json_path = test_image.parent / f"{test_image.stem}_tags.json"
        json_path.write_text(json.dumps({
            "file": str(test_image),
            "tagged_at": "2025-12-23T00:00:00Z",
            "results": {
                "ram_plus": {
                    "tags": [{"label": "dog", "confidence": 0.95}],
                }
            }
        }))

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()
            data = handler.read_tags(test_image)

            assert data is not None
            assert len(data.tags) == 1
            assert data.tags[0].label == "dog"
            assert data.plugins_used == ["ram_plus"]

    def test_clear_tags_json(self, temp_dir, test_image):
        """Test clear_tags removes JSON file."""
        from visual_buffet.services.xmp_handler import XMPHandler

        # Create JSON tags file
        json_path = test_image.parent / f"{test_image.stem}_tags.json"
        json_path.write_text('{"results": {}}')

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()
            success = handler.clear_tags(test_image)

            assert success is True
            assert not json_path.exists()

    def test_write_tags_roundtrip(self, temp_dir, test_image):
        """Test write then read tags."""
        from visual_buffet.services.xmp_handler import XMPHandler

        with patch("visual_buffet.services.xmp_handler._check_exiftool", return_value=False):
            handler = XMPHandler()

            # Write tags
            tags = [
                {"label": "dog", "confidence": 0.95, "plugin": "ram_plus"},
                {"label": "outdoor", "confidence": 0.87, "plugin": "ram_plus"},
            ]
            handler.write_tags(
                test_image,
                tags,
                plugins_used=["ram_plus"],
                threshold=0.5,
                size_used="small",
                inference_time_ms=150.0,
            )

            # Read tags back
            data = handler.read_tags(test_image)
            assert data is not None
            assert len(data.tags) == 2
            assert {t.label for t in data.tags} == {"dog", "outdoor"}


class TestXMPIntegration:
    """Test XMP integration with exiftool (mocked)."""

    def test_read_xmp_sidecar_no_file(self, temp_dir):
        """Test read_xmp_sidecar returns None when no sidecar."""
        from visual_buffet.services.xmp_handler import read_xmp_sidecar

        image_path = temp_dir / "photo.jpg"
        assert read_xmp_sidecar(image_path) is None

    def test_read_xmp_sidecar_no_exiftool(self, temp_dir, test_image):
        """Test read_xmp_sidecar returns None when no exiftool."""
        from visual_buffet.services.xmp_handler import get_sidecar_path, read_xmp_sidecar

        # Create empty sidecar
        get_sidecar_path(test_image).touch()

        with patch("visual_buffet.services.xmp_handler._get_exiftool_path", return_value=None):
            result = read_xmp_sidecar(test_image)
            assert result is None

    def test_write_xmp_sidecar_no_exiftool(self, test_image):
        """Test write_xmp_sidecar raises when no exiftool."""
        from visual_buffet.services.xmp_handler import XMPError, XMPTagData, write_xmp_sidecar

        with patch("visual_buffet.services.xmp_handler._get_exiftool_path", return_value=None):
            with pytest.raises(XMPError, match="exiftool not found"):
                write_xmp_sidecar(test_image, XMPTagData())

    def test_clear_xmp_tags_no_sidecar(self, temp_dir):
        """Test clear_xmp_tags succeeds when no sidecar exists."""
        from visual_buffet.services.xmp_handler import clear_xmp_tags

        image_path = temp_dir / "photo.jpg"
        image_path.touch()
        result = clear_xmp_tags(image_path)
        assert result is True


class TestExiftoolDetection:
    """Test exiftool detection."""

    def test_check_exiftool_not_found(self):
        """Test _check_exiftool returns False when not found."""
        from visual_buffet.services.xmp_handler import _check_exiftool

        with patch("visual_buffet.services.xmp_handler._get_exiftool_path", return_value=None):
            assert _check_exiftool() is False

    def test_check_exiftool_found(self):
        """Test _check_exiftool returns True when found."""
        from visual_buffet.services.xmp_handler import _check_exiftool

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("visual_buffet.services.xmp_handler._get_exiftool_path", return_value="/usr/bin/exiftool"):
            with patch("subprocess.run", return_value=mock_result):
                assert _check_exiftool() is True
