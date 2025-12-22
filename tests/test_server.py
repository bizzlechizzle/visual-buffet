"""Tests for FastAPI server."""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Import app for testing
from visual_buffet.gui.server import (
    _is_raw_filename,
    _is_supported_filename,
    app,
    create_app,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_raw_filename_arw(self):
        """Test ARW files are detected as RAW."""
        assert _is_raw_filename("photo.ARW") is True
        assert _is_raw_filename("photo.arw") is True

    def test_is_raw_filename_cr2(self):
        """Test CR2 files are detected as RAW."""
        assert _is_raw_filename("photo.CR2") is True
        assert _is_raw_filename("photo.cr2") is True

    def test_is_raw_filename_cr3(self):
        """Test CR3 files are detected as RAW."""
        assert _is_raw_filename("photo.CR3") is True
        assert _is_raw_filename("photo.cr3") is True

    def test_is_raw_filename_nef(self):
        """Test NEF files are detected as RAW."""
        assert _is_raw_filename("photo.NEF") is True
        assert _is_raw_filename("photo.nef") is True

    def test_is_raw_filename_dng(self):
        """Test DNG files are detected as RAW."""
        assert _is_raw_filename("photo.DNG") is True
        assert _is_raw_filename("photo.dng") is True

    def test_is_raw_filename_jpeg_not_raw(self):
        """Test JPEG files are not detected as RAW."""
        assert _is_raw_filename("photo.jpg") is False
        assert _is_raw_filename("photo.jpeg") is False

    def test_is_raw_filename_png_not_raw(self):
        """Test PNG files are not detected as RAW."""
        assert _is_raw_filename("photo.png") is False

    def test_is_raw_filename_empty(self):
        """Test empty filename returns False."""
        assert _is_raw_filename("") is False
        assert _is_raw_filename(None) is False

    def test_is_supported_filename_jpeg(self):
        """Test JPEG files are supported."""
        assert _is_supported_filename("photo.jpg") is True
        assert _is_supported_filename("photo.jpeg") is True
        assert _is_supported_filename("photo.JPG") is True

    def test_is_supported_filename_png(self):
        """Test PNG files are supported."""
        assert _is_supported_filename("photo.png") is True
        assert _is_supported_filename("photo.PNG") is True

    def test_is_supported_filename_webp(self):
        """Test WebP files are supported."""
        assert _is_supported_filename("photo.webp") is True
        assert _is_supported_filename("photo.WEBP") is True

    def test_is_supported_filename_heic(self):
        """Test HEIC files are supported."""
        assert _is_supported_filename("photo.heic") is True
        assert _is_supported_filename("photo.HEIC") is True

    def test_is_supported_filename_raw(self):
        """Test RAW files are supported."""
        assert _is_supported_filename("photo.arw") is True
        assert _is_supported_filename("photo.cr2") is True
        assert _is_supported_filename("photo.nef") is True

    def test_is_supported_filename_unsupported(self):
        """Test unsupported formats return False."""
        assert _is_supported_filename("document.pdf") is False
        assert _is_supported_filename("video.mp4") is False
        assert _is_supported_filename("file.txt") is False

    def test_is_supported_filename_empty(self):
        """Test empty filename returns False."""
        assert _is_supported_filename("") is False
        assert _is_supported_filename(None) is False


class TestAppCreation:
    """Tests for app creation and configuration."""

    def test_create_app_returns_fastapi(self):
        """Test create_app returns a FastAPI instance."""
        from fastapi import FastAPI

        test_app = create_app()
        assert isinstance(test_app, FastAPI)

    def test_app_has_cors_middleware(self):
        """Test app has CORS middleware configured."""
        middlewares = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middlewares

    def test_app_title(self):
        """Test app has correct title."""
        assert app.title == "Visual Buffet"

    def test_app_version(self):
        """Test app has version set."""
        assert app.version == "0.1.0"


class TestAPIRoutes:
    """Tests for API route availability."""

    def test_root_route_exists(self):
        """Test root route is defined."""
        routes = [r.path for r in app.routes]
        assert "/" in routes

    def test_status_route_exists(self):
        """Test /api/status route is defined."""
        routes = [r.path for r in app.routes]
        assert "/api/status" in routes

    def test_plugins_route_exists(self):
        """Test /api/plugins route is defined."""
        routes = [r.path for r in app.routes]
        assert "/api/plugins" in routes

    def test_config_route_exists(self):
        """Test /api/config route is defined."""
        routes = [r.path for r in app.routes]
        assert "/api/config" in routes

    def test_upload_route_exists(self):
        """Test /api/upload route is defined."""
        routes = [r.path for r in app.routes]
        assert "/api/upload" in routes

    def test_settings_route_exists(self):
        """Test /api/settings route is defined."""
        routes = [r.path for r in app.routes]
        assert "/api/settings" in routes


class TestAPIEndpoints:
    """Integration tests for API endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_root_returns_html(self, client):
        """Test root returns HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_status_returns_json(self, client):
        """Test status returns JSON with hardware info."""
        with patch("visual_buffet.gui.server.detect_hardware") as mock_hw:
            mock_profile = MagicMock()
            mock_profile.cpu_model = "Test CPU"
            mock_profile.cpu_cores = 8
            mock_profile.ram_total_gb = 32.0
            mock_profile.ram_available_gb = 16.0
            mock_profile.gpu_type = "cuda"
            mock_profile.gpu_name = "Test GPU"
            mock_profile.gpu_vram_gb = 8.0
            mock_hw.return_value = mock_profile

            with patch("visual_buffet.gui.server.discover_plugins", return_value=[]):
                with patch(
                    "visual_buffet.gui.server.get_recommended_batch_size",
                    return_value=4,
                ):
                    response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "hardware" in data
        assert "plugins" in data

    def test_plugins_returns_list(self, client):
        """Test plugins endpoint returns list."""
        with patch("visual_buffet.gui.server.discover_plugins", return_value=[]):
            response = client.get("/api/plugins")

        assert response.status_code == 200
        data = response.json()
        assert "plugins" in data
        assert isinstance(data["plugins"], list)

    def test_config_returns_dict(self, client):
        """Test config endpoint returns config dict."""
        with patch("visual_buffet.gui.server.load_config", return_value={}):
            response = client.get("/api/config")

        assert response.status_code == 200
        assert isinstance(response.json(), dict)

    def test_upload_requires_file(self, client):
        """Test upload endpoint requires file."""
        response = client.post("/api/upload")
        assert response.status_code == 422  # Validation error

    def test_upload_valid_jpeg(self, client):
        """Test uploading valid JPEG image."""
        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        response = client.post(
            "/api/upload",
            files={"file": ("test.jpg", buffer, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "filename" in data
        assert data["filename"] == "test.jpg"

    def test_upload_valid_png(self, client):
        """Test uploading valid PNG image."""
        img = Image.new("RGBA", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/api/upload",
            files={"file": ("test.png", buffer, "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data

    def test_upload_invalid_file(self, client):
        """Test uploading invalid file type."""
        buffer = io.BytesIO(b"not an image")

        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", buffer, "text/plain")},
        )

        assert response.status_code == 400

    def test_settings_get(self, client):
        """Test getting GUI settings."""
        with patch("visual_buffet.gui.server.load_config", return_value={}):
            response = client.get("/api/settings")

        assert response.status_code == 200
        data = response.json()
        assert "plugin_settings" in data

    def test_settings_post(self, client):
        """Test saving GUI settings."""
        with patch("visual_buffet.utils.config.load_config", return_value={}):
            with patch("visual_buffet.utils.config.save_config"):
                response = client.post(
                    "/api/settings",
                    json={"plugin_settings": {"test": {"enabled": True}}},
                )

        assert response.status_code == 200
        assert response.json()["status"] == "saved"
