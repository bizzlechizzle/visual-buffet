"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli():
    """Get the actual CLI command for testing."""
    from visual_buffet.cli import main
    return main


@pytest.fixture
def cli_isolated(runner):
    """Create an isolated CLI runner with a temporary filesystem."""
    with runner.isolated_filesystem():
        yield runner


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


@pytest.fixture
def mock_tagging_engine():
    """Create a mock tagging engine."""
    engine = MagicMock()
    engine.plugins = {"mock_plugin": MagicMock()}
    engine.plugins["mock_plugin"].is_available.return_value = True
    engine.tag_batch.return_value = [
        {
            "file": "test.jpg",
            "results": {
                "mock_plugin": {
                    "tags": [{"label": "test", "confidence": 0.95}],
                    "model": "mock",
                    "version": "1.0.0",
                    "inference_time_ms": 10.0,
                }
            },
        }
    ]
    return engine


@pytest.fixture
def mock_hardware_profile():
    """Create a mock hardware profile."""
    from visual_buffet.core.hardware import HardwareProfile

    return HardwareProfile(
        cpu_model="Test CPU",
        cpu_cores=8,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        gpu_type="cuda",
        gpu_name="Test GPU",
        gpu_vram_gb=8.0,
    )


@pytest.fixture
def large_test_image(temp_dir):
    """Create a larger test image for processing tests."""
    image_path = temp_dir / "large_test.jpg"
    img = Image.new("RGB", (1920, 1080), color="green")
    img.save(image_path, "JPEG", quality=95)
    return image_path


@pytest.fixture
def corrupted_file(temp_dir):
    """Create a file that looks like an image but isn't."""
    file_path = temp_dir / "corrupted.jpg"
    file_path.write_bytes(b"not a valid image file")
    return file_path


@pytest.fixture
def empty_file(temp_dir):
    """Create an empty file."""
    file_path = temp_dir / "empty.jpg"
    file_path.touch()
    return file_path
