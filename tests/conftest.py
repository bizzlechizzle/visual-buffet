"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_image(temp_dir):
    """Create a temporary test image (JPEG)."""
    image_path = temp_dir / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_path, "JPEG")
    return image_path


@pytest.fixture
def test_png(temp_dir):
    """Create a temporary test image (PNG)."""
    image_path = temp_dir / "test_image.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(image_path, "PNG")
    return image_path


@pytest.fixture
def test_images(temp_dir):
    """Create multiple test images."""
    images = []
    for i in range(5):
        image_path = temp_dir / f"test_{i}.jpg"
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        img.save(image_path, "JPEG")
        images.append(image_path)
    return images


@pytest.fixture
def mock_plugin_dir(temp_dir):
    """Create a mock plugin directory structure."""
    plugin_dir = temp_dir / "mock_plugin"
    plugin_dir.mkdir()

    # Create plugin.toml
    (plugin_dir / "plugin.toml").write_text(
        """
[plugin]
name = "mock_plugin"
version = "1.0.0"
description = "A mock plugin for testing"
entry_point = "MockPlugin"

[plugin.dependencies]

[plugin.hardware]
gpu_recommended = false
min_ram_gb = 2
"""
    )

    # Create __init__.py with MockPlugin class
    (plugin_dir / "__init__.py").write_text(
        '''
"""Mock plugin for testing."""

from pathlib import Path
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult


class MockPlugin(PluginBase):
    """Mock plugin implementation."""

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="mock_plugin",
            version="1.0.0",
            description="A mock plugin for testing",
        )

    def is_available(self) -> bool:
        return True

    def tag(self, image_path: Path) -> TagResult:
        return TagResult(
            tags=[
                Tag(label="test_tag_1", confidence=0.95),
                Tag(label="test_tag_2", confidence=0.80),
            ],
            model="mock_model",
            version="1.0.0",
            inference_time_ms=10.0,
        )
'''
    )

    # Create models directory
    (plugin_dir / "models").mkdir()

    return plugin_dir
