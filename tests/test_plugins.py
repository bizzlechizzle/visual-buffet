"""Tests for plugin system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imlage.exceptions import PluginError, PluginNotFoundError
from imlage.plugins.base import PluginBase
from imlage.plugins.loader import (
    discover_plugins,
    get_plugins_dir,
    load_all_plugins,
    load_plugin,
)
from imlage.plugins.schemas import PluginInfo, Tag, TagResult


class MockPlugin(PluginBase):
    """Mock plugin for testing."""

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
            tags=[Tag(label="mock_tag", confidence=0.99)],
            model="mock_model",
            version="1.0.0",
            inference_time_ms=1.0,
        )


class TestPluginBase:
    """Tests for PluginBase abstract class."""

    def test_plugin_base_init(self):
        """Test PluginBase initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = MockPlugin(Path(tmpdir))
            assert plugin.plugin_dir == Path(tmpdir)

    def test_plugin_base_model_path(self):
        """Test model path is in plugin directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = MockPlugin(Path(tmpdir))
            model_path = plugin.get_model_path()

            assert model_path == Path(tmpdir) / "models"

    def test_plugin_base_setup_default(self):
        """Test default setup returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = MockPlugin(Path(tmpdir))
            assert plugin.setup() is True


class TestGetPluginsDir:
    """Tests for plugin directory resolution."""

    def test_get_plugins_dir_exists(self):
        """Test get_plugins_dir returns Path."""
        plugins_dir = get_plugins_dir()
        assert isinstance(plugins_dir, Path)
        assert "plugins" in str(plugins_dir)


class TestDiscoverPlugins:
    """Tests for plugin discovery."""

    def test_discover_no_plugins(self):
        """Test discovery with no plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("imlage.plugins.loader.get_plugins_dir", return_value=Path(tmpdir)):
                plugins = discover_plugins()
            assert plugins == []

    def test_discover_valid_plugin(self):
        """Test discovery finds valid plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create plugin directory structure
            plugin_dir = Path(tmpdir) / "test_plugin"
            plugin_dir.mkdir()

            # Create plugin.toml
            (plugin_dir / "plugin.toml").write_text(
                """
[plugin]
name = "test_plugin"
version = "1.0.0"
description = "Test plugin"
entry_point = "TestPlugin"
"""
            )

            # Create __init__.py
            (plugin_dir / "__init__.py").write_text("")

            with patch("imlage.plugins.loader.get_plugins_dir", return_value=Path(tmpdir)):
                plugins = discover_plugins()

            assert len(plugins) == 1
            assert plugins[0]["plugin"]["name"] == "test_plugin"

    def test_discover_ignores_invalid(self):
        """Test discovery ignores directories without plugin.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid plugin directory (no plugin.toml)
            invalid_dir = Path(tmpdir) / "invalid_plugin"
            invalid_dir.mkdir()
            (invalid_dir / "__init__.py").write_text("")

            with patch("imlage.plugins.loader.get_plugins_dir", return_value=Path(tmpdir)):
                plugins = discover_plugins()

            assert plugins == []


class TestLoadPlugin:
    """Tests for plugin loading."""

    def test_load_plugin_nonexistent(self):
        """Test loading non-existent plugin raises error."""
        with pytest.raises(PluginNotFoundError):
            load_plugin(Path("/nonexistent/plugin"))

    def test_load_plugin_no_toml(self):
        """Test loading plugin without plugin.toml raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "__init__.py").write_text("")

            with pytest.raises(PluginError, match="plugin.toml"):
                load_plugin(plugin_dir)

    def test_load_plugin_no_init(self):
        """Test loading plugin without __init__.py raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "plugin.toml").write_text(
                """
[plugin]
name = "test"
version = "1.0.0"
entry_point = "TestPlugin"
"""
            )

            with pytest.raises(PluginError, match="__init__.py"):
                load_plugin(plugin_dir)

    def test_load_plugin_missing_entry_point(self):
        """Test loading plugin with missing entry point raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "plugin.toml").write_text(
                """
[plugin]
name = "test"
version = "1.0.0"
entry_point = "NonExistentClass"
"""
            )
            (plugin_dir / "__init__.py").write_text("# Empty plugin")

            with pytest.raises(PluginError, match="not found"):
                load_plugin(plugin_dir)


class TestLoadAllPlugins:
    """Tests for loading all plugins."""

    def test_load_all_empty(self):
        """Test loading all with no plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("imlage.plugins.loader.get_plugins_dir", return_value=Path(tmpdir)):
                plugins = load_all_plugins()

            assert plugins == []

    def test_load_all_skips_errors(self):
        """Test loading all skips plugins with errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create broken plugin
            broken_dir = Path(tmpdir) / "broken"
            broken_dir.mkdir()
            (broken_dir / "plugin.toml").write_text("[plugin]\nname = 'broken'")
            # Missing __init__.py

            with patch("imlage.plugins.loader.get_plugins_dir", return_value=Path(tmpdir)):
                plugins = load_all_plugins()

            assert plugins == []


class TestPluginInfo:
    """Tests for PluginInfo from plugins."""

    def test_mock_plugin_info(self):
        """Test getting info from mock plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = MockPlugin(Path(tmpdir))
            info = plugin.get_info()

            assert info.name == "mock_plugin"
            assert info.version == "1.0.0"
            assert info.description == "A mock plugin for testing"


class TestPluginTagging:
    """Tests for plugin tagging functionality."""

    def test_mock_plugin_tag(self):
        """Test tagging with mock plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = MockPlugin(Path(tmpdir))

            # Create test image
            from PIL import Image

            test_image = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100))
            img.save(test_image, "JPEG")

            result = plugin.tag(test_image)

            assert result.model == "mock_model"
            assert len(result.tags) == 1
            assert result.tags[0].label == "mock_tag"
            assert result.tags[0].confidence == 0.99


class TestPluginAvailability:
    """Tests for plugin availability checks."""

    def test_mock_plugin_available(self):
        """Test mock plugin availability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = MockPlugin(Path(tmpdir))
            assert plugin.is_available() is True

    def test_plugin_not_available_no_models(self):
        """Test plugin not available when models missing."""

        class UnavailablePlugin(PluginBase):
            def get_info(self) -> PluginInfo:
                return PluginInfo(name="test", version="1.0.0", description="")

            def is_available(self) -> bool:
                return (self.get_model_path() / "model.pth").exists()

            def tag(self, image_path: Path) -> TagResult:
                raise NotImplementedError

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = UnavailablePlugin(Path(tmpdir))
            assert plugin.is_available() is False
