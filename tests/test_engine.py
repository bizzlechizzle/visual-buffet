"""Tests for tagging engine."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from visual_buffet.core.engine import TaggingEngine
from visual_buffet.exceptions import PluginError
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult


class MockPlugin(PluginBase):
    """Mock plugin for testing."""

    def __init__(self, plugin_dir: Path, name: str = "mock", available: bool = True):
        super().__init__(plugin_dir)
        self._name = name
        self._available = available

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name=self._name,
            version="1.0.0",
            description=f"Mock {self._name} plugin",
        )

    def is_available(self) -> bool:
        return self._available

    def tag(self, image_path: Path) -> TagResult:
        return TagResult(
            tags=[
                Tag(label=f"{self._name}_tag1", confidence=0.95),
                Tag(label=f"{self._name}_tag2", confidence=0.80),
                Tag(label=f"{self._name}_tag3", confidence=0.65),
                Tag(label=f"{self._name}_tag4", confidence=0.40),
            ],
            model=f"{self._name}_model",
            version="1.0.0",
            inference_time_ms=100.0,
        )


class TestTaggingEngineInit:
    """Tests for TaggingEngine initialization."""

    def test_engine_init_no_plugins(self):
        """Test engine initialization with no plugins."""
        with patch("visual_buffet.core.engine.load_all_plugins", return_value=[]):
            engine = TaggingEngine()
            assert engine.plugins == {}

    def test_engine_init_with_plugins(self):
        """Test engine initialization with plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_plugins = [
                MockPlugin(Path(tmpdir), "plugin1"),
                MockPlugin(Path(tmpdir), "plugin2"),
            ]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()

            assert len(engine.plugins) == 2
            assert "plugin1" in engine.plugins
            assert "plugin2" in engine.plugins


class TestTagImage:
    """Tests for single image tagging."""

    def test_tag_image_single_plugin(self):
        """Test tagging with single plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image)

            assert "results" in result
            assert "mock" in result["results"]
            assert len(result["results"]["mock"]["tags"]) > 0

    def test_tag_image_multiple_plugins(self):
        """Test tagging with multiple plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [
                MockPlugin(Path(tmpdir), "plugin1"),
                MockPlugin(Path(tmpdir), "plugin2"),
            ]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image)

            assert "plugin1" in result["results"]
            assert "plugin2" in result["results"]

    def test_tag_image_specific_plugins(self):
        """Test tagging with specific plugins only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [
                MockPlugin(Path(tmpdir), "plugin1"),
                MockPlugin(Path(tmpdir), "plugin2"),
            ]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image, plugin_names=["plugin1"])

            assert "plugin1" in result["results"]
            assert "plugin2" not in result["results"]

    def test_tag_image_threshold_filter(self):
        """Test tagging with confidence threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image, threshold=0.7)

            # Only tags >= 0.7 confidence should be included
            tags = result["results"]["mock"]["tags"]
            assert all(t["confidence"] >= 0.7 for t in tags)

    def test_tag_image_limit(self):
        """Test tagging with tag limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image, limit=2)

            assert len(result["results"]["mock"]["tags"]) <= 2

    def test_tag_image_skips_unavailable(self):
        """Test tagging skips unavailable plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [
                MockPlugin(Path(tmpdir), "available", available=True),
                MockPlugin(Path(tmpdir), "unavailable", available=False),
            ]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image)

            assert "available" in result["results"]
            assert "unavailable" not in result["results"]


class TestTagBatch:
    """Tests for batch image tagging."""

    def test_tag_batch_single_image(self):
        """Test batch tagging with single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                results = engine.tag_batch([test_image])

            assert len(results) == 1
            assert results[0]["file"] == str(test_image)
            assert "results" in results[0]

    def test_tag_batch_multiple_images(self):
        """Test batch tagging with multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = []
            for i in range(3):
                test_image = Path(tmpdir) / f"test{i}.jpg"
                img = Image.new("RGB", (100, 100))
                img.save(test_image, "JPEG")
                images.append(test_image)

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                results = engine.tag_batch(images)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["file"] == str(images[i])

    def test_tag_batch_handles_errors(self):
        """Test batch tagging handles individual image errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one valid and one non-existent image
            valid_image = Path(tmpdir) / "valid.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(valid_image, "JPEG")

            invalid_image = Path(tmpdir) / "invalid.jpg"

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                results = engine.tag_batch([valid_image, invalid_image])

            assert len(results) == 2
            # Valid image should have results
            assert "results" in results[0]
            # Invalid image should have error
            assert "error" in results[1]

    def test_tag_batch_empty_list(self):
        """Test batch tagging with empty list."""
        with patch("visual_buffet.core.engine.load_all_plugins", return_value=[]):
            engine = TaggingEngine()
            results = engine.tag_batch([])

        assert results == []


class TestTaggingEnginePluginErrors:
    """Tests for handling plugin errors."""

    def test_plugin_error_recorded(self):
        """Test plugin errors are recorded in results."""

        class ErrorPlugin(PluginBase):
            def get_info(self):
                return PluginInfo(name="error", version="1.0.0", description="")

            def is_available(self):
                return True

            def tag(self, image_path):
                raise PluginError("Test error")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [ErrorPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image)

            assert "error" in result["results"]
            assert "error" in result["results"]["error"]
            assert "Test error" in result["results"]["error"]["error"]


class TestTaggingEngineOutput:
    """Tests for output format."""

    def test_output_structure(self):
        """Test output has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                result = engine.tag_image(test_image)

            # Check plugin result structure
            plugin_result = result["results"]["mock"]
            assert "tags" in plugin_result
            assert "model" in plugin_result
            assert "version" in plugin_result
            assert "inference_time_ms" in plugin_result

            # Check tag structure
            tag = plugin_result["tags"][0]
            assert "label" in tag
            assert "confidence" in tag

    def test_batch_output_structure(self):
        """Test batch output has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            mock_plugins = [MockPlugin(Path(tmpdir))]

            with patch("visual_buffet.core.engine.load_all_plugins", return_value=mock_plugins):
                engine = TaggingEngine()
                results = engine.tag_batch([test_image])

            assert len(results) == 1
            result = results[0]
            assert "file" in result
            assert "results" in result
            assert "mock" in result["results"]
