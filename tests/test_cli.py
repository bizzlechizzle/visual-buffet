"""CLI tests."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from visual_buffet.cli import main


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def test_image():
    """Create temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color="red")
        img.save(f.name, "JPEG")
        yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)


class TestMainCommand:
    """Tests for main CLI entry point."""

    def test_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Visual Buffet" in result.output

    def test_debug_flag(self, runner):
        """Test --debug flag is accepted."""
        result = runner.invoke(main, ["--debug", "--help"])
        assert result.exit_code == 0

    def test_command_groups(self, runner):
        """Test all command groups are present."""
        result = runner.invoke(main, ["--help"])
        assert "tag" in result.output
        assert "plugins" in result.output
        assert "hardware" in result.output
        assert "config" in result.output


class TestTagCommand:
    """Tests for tag command."""

    def test_tag_help(self, runner):
        """Test tag command help."""
        result = runner.invoke(main, ["tag", "--help"])
        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--plugin" in result.output
        assert "--output" in result.output
        assert "--threshold" in result.output
        assert "--limit" in result.output

    def test_tag_no_path(self, runner):
        """Test tag command requires path."""
        result = runner.invoke(main, ["tag"])
        assert result.exit_code != 0

    def test_tag_no_images_found(self, runner):
        """Test tag command with path that has no images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["tag", tmpdir])
            assert result.exit_code != 0
            assert "No images found" in result.output

    @patch("visual_buffet.cli.TaggingEngine")
    def test_tag_no_plugins_available(self, mock_engine_class, runner, test_image):
        """Test tag command with no plugins."""
        mock_engine = MagicMock()
        mock_engine.plugins = {}
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(main, ["tag", str(test_image)])
        assert result.exit_code != 0
        assert "No plugins" in result.output

    @patch("visual_buffet.cli.TaggingEngine")
    def test_tag_success(self, mock_engine_class, runner, test_image):
        """Test successful tagging."""
        mock_plugin = MagicMock()
        mock_plugin.is_available.return_value = True

        mock_engine = MagicMock()
        mock_engine.plugins = {"mock_plugin": mock_plugin}
        mock_engine.tag_batch.return_value = [
            {
                "file": str(test_image),
                "results": {
                    "mock_plugin": {
                        "tags": [{"label": "dog", "confidence": 0.95}],
                        "model": "mock",
                        "version": "1.0.0",
                        "inference_time_ms": 100,
                    }
                },
            }
        ]
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(main, ["tag", str(test_image)])
        assert result.exit_code == 0

    @patch("visual_buffet.cli.TaggingEngine")
    def test_tag_output_file(self, mock_engine_class, runner, test_image):
        """Test tagging with output file."""
        mock_plugin = MagicMock()
        mock_plugin.is_available.return_value = True

        mock_engine = MagicMock()
        mock_engine.plugins = {"mock_plugin": mock_plugin}
        mock_engine.tag_batch.return_value = [
            {
                "file": str(test_image),
                "results": {"mock_plugin": {"tags": [], "model": "mock", "version": "1.0.0", "inference_time_ms": 50}},
            }
        ]
        mock_engine_class.return_value = mock_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.json"
            result = runner.invoke(main, ["tag", str(test_image), "-o", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()

            # Verify JSON is valid
            with open(output_file) as f:
                data = json.load(f)
            assert isinstance(data, list)


class TestPluginsCommand:
    """Tests for plugins command group."""

    def test_plugins_help(self, runner):
        """Test plugins command help."""
        result = runner.invoke(main, ["plugins", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "setup" in result.output
        assert "info" in result.output

    @patch("visual_buffet.cli.discover_plugins")
    def test_plugins_list_empty(self, mock_discover, runner):
        """Test plugins list with no plugins."""
        mock_discover.return_value = []
        result = runner.invoke(main, ["plugins", "list"])
        assert result.exit_code == 0
        assert "No plugins found" in result.output

    @patch("visual_buffet.cli.discover_plugins")
    @patch("visual_buffet.cli.load_plugin")
    def test_plugins_list_with_plugins(self, mock_load, mock_discover, runner):
        """Test plugins list with available plugins."""
        mock_discover.return_value = [
            {
                "plugin": {
                    "name": "test_plugin",
                    "version": "1.0.0",
                    "description": "A test plugin",
                },
                "_path": "/fake/path",
            }
        ]

        mock_plugin = MagicMock()
        mock_plugin.is_available.return_value = True
        mock_load.return_value = mock_plugin

        result = runner.invoke(main, ["plugins", "list"])
        assert result.exit_code == 0
        assert "test_plugin" in result.output

    def test_plugins_setup_not_found(self, runner):
        """Test plugins setup with non-existent plugin."""
        with patch("visual_buffet.cli.get_plugins_dir") as mock_dir:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_dir.return_value = Path(tmpdir)
                result = runner.invoke(main, ["plugins", "setup", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_plugins_info_not_found(self, runner):
        """Test plugins info with non-existent plugin."""
        with patch("visual_buffet.cli.get_plugins_dir") as mock_dir:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_dir.return_value = Path(tmpdir)
                result = runner.invoke(main, ["plugins", "info", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output


class TestHardwareCommand:
    """Tests for hardware command."""

    def test_hardware_help(self, runner):
        """Test hardware command help."""
        result = runner.invoke(main, ["hardware", "--help"])
        assert result.exit_code == 0
        assert "--refresh" in result.output

    @patch("visual_buffet.cli.detect_hardware")
    @patch("visual_buffet.cli.get_recommended_batch_size")
    def test_hardware_display(self, mock_batch, mock_detect, runner):
        """Test hardware detection display."""
        from visual_buffet.plugins.schemas import HardwareProfile

        mock_detect.return_value = HardwareProfile(
            cpu_model="Test CPU",
            cpu_cores=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
            gpu_type="cuda",
            gpu_name="Test GPU",
            gpu_vram_gb=8.0,
        )
        mock_batch.return_value = 4

        result = runner.invoke(main, ["hardware"])
        assert result.exit_code == 0
        assert "Test CPU" in result.output
        assert "8" in result.output
        assert "GPU" in result.output


class TestConfigCommand:
    """Tests for config command group."""

    def test_config_help(self, runner):
        """Test config command help."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "set" in result.output
        assert "get" in result.output

    @patch("visual_buffet.cli.load_config")
    @patch("visual_buffet.utils.config.get_config_path")
    def test_config_show(self, mock_path, mock_load, runner):
        """Test config show command."""
        mock_path.return_value = Path("/fake/config.toml")
        mock_load.return_value = {"general": {"threshold": 0.5}}

        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "threshold" in result.output

    @patch("visual_buffet.cli.load_config")
    @patch("visual_buffet.cli.save_config")
    def test_config_set(self, mock_save, mock_load, runner):
        """Test config set command."""
        mock_load.return_value = {"general": {"threshold": 0.5}}

        result = runner.invoke(main, ["config", "set", "general.threshold", "0.7"])
        assert result.exit_code == 0
        assert "Set" in result.output
        mock_save.assert_called_once()

    @patch("visual_buffet.cli.load_config")
    def test_config_get(self, mock_load, runner):
        """Test config get command."""
        mock_load.return_value = {"general": {"threshold": 0.5}}

        result = runner.invoke(main, ["config", "get", "general.threshold"])
        assert result.exit_code == 0
        assert "0.5" in result.output

    @patch("visual_buffet.cli.load_config")
    def test_config_get_missing_key(self, mock_load, runner):
        """Test config get with missing key."""
        mock_load.return_value = {"general": {"threshold": 0.5}}

        result = runner.invoke(main, ["config", "get", "nonexistent.key"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestCLIEdgeCases:
    """Tests for CLI edge cases."""

    def test_keyboard_interrupt(self, runner, test_image):
        """Test KeyboardInterrupt handling."""
        with patch("visual_buffet.cli.TaggingEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.plugins = {"mock": MagicMock()}
            mock_engine.plugins["mock"].is_available.return_value = True
            mock_engine.tag_batch.side_effect = KeyboardInterrupt

            mock_engine_class.return_value = mock_engine

            result = runner.invoke(main, ["tag", str(test_image)])
            assert "Interrupted" in result.output

    def test_visual_buffet_error_handling(self, runner, test_image):
        """Test VisualBuffetError handling."""
        from visual_buffet.exceptions import VisualBuffetError

        with patch("visual_buffet.cli.TaggingEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.plugins = {"mock": MagicMock()}
            mock_engine.plugins["mock"].is_available.return_value = True
            mock_engine.tag_batch.side_effect = VisualBuffetError("Test error")

            mock_engine_class.return_value = mock_engine

            result = runner.invoke(main, ["tag", str(test_image)])
            assert result.exit_code != 0
            assert "Error" in result.output
