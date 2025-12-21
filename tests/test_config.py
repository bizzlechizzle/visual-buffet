"""Tests for configuration utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from imlage.utils.config import (
    DEFAULT_CONFIG,
    get_config_path,
    get_value,
    load_config,
    save_config,
    set_value,
)


class TestConfigPath:
    """Tests for config path resolution."""

    def test_get_config_path_default(self):
        """Test default config path is in user config directory."""
        path = get_config_path()
        assert "imlage" in str(path)
        assert path.name == "config.toml"


class TestDefaultConfig:
    """Tests for default configuration."""

    def test_default_config_structure(self):
        """Test default config has expected structure."""
        assert "general" in DEFAULT_CONFIG
        assert "plugins" in DEFAULT_CONFIG

    def test_default_config_general_values(self):
        """Test default config general values."""
        assert DEFAULT_CONFIG["general"]["default_threshold"] == 0.5
        assert DEFAULT_CONFIG["general"]["default_limit"] == 50
        assert DEFAULT_CONFIG["general"]["default_format"] == "json"

    def test_default_config_plugins_values(self):
        """Test default config plugins values."""
        assert "enabled" in DEFAULT_CONFIG["plugins"]
        assert isinstance(DEFAULT_CONFIG["plugins"]["enabled"], list)


class TestLoadConfig:
    """Tests for loading configuration."""

    def test_load_config_missing_file(self):
        """Test loading config when file doesn't exist returns defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.toml"

            with patch("imlage.utils.config.get_config_path", return_value=config_path):
                config = load_config()

            # Should match default config structure
            assert "general" in config
            assert "plugins" in config

    def test_load_config_existing_file(self):
        """Test loading config from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[general]
default_threshold = 0.7
default_limit = 100
default_format = "json"

[plugins]
enabled = ["ram_plus"]
"""
            )

            with patch("imlage.utils.config.get_config_path", return_value=config_path):
                config = load_config()

            assert config["general"]["default_threshold"] == 0.7
            assert config["general"]["default_limit"] == 100
            assert config["plugins"]["enabled"] == ["ram_plus"]


class TestSaveConfig:
    """Tests for saving configuration."""

    def test_save_config_creates_file(self):
        """Test saving config creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.toml"

            with patch("imlage.utils.config.get_config_path", return_value=config_path):
                config = {"general": {"threshold": 0.8}}
                save_config(config)

            assert config_path.exists()

    def test_save_config_content(self):
        """Test saved config content is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            with patch("imlage.utils.config.get_config_path", return_value=config_path):
                config = {
                    "general": {"default_threshold": 0.75},
                    "plugins": {"enabled": ["ram_plus", "wd14"]},
                }
                save_config(config)

            content = config_path.read_text()
            assert "0.75" in content
            assert "ram_plus" in content


class TestGetValue:
    """Tests for getting nested config values."""

    def test_get_value_simple_key(self):
        """Test getting simple key."""
        config = {"threshold": 0.5}
        value = get_value(config, "threshold")
        assert value == 0.5

    def test_get_value_nested_key(self):
        """Test getting nested key with dot notation."""
        config = {
            "general": {
                "settings": {
                    "threshold": 0.75
                }
            }
        }
        value = get_value(config, "general.settings.threshold")
        assert value == 0.75

    def test_get_value_missing_key(self):
        """Test getting missing key returns None."""
        config = {"general": {"threshold": 0.5}}
        value = get_value(config, "general.missing")
        assert value is None

    def test_get_value_missing_nested(self):
        """Test getting missing nested path returns None."""
        config = {"general": {"threshold": 0.5}}
        value = get_value(config, "missing.path.key")
        assert value is None

    def test_get_value_with_default(self):
        """Test getting missing key with default."""
        config = {"general": {"threshold": 0.5}}
        value = get_value(config, "general.missing", default=0.0)
        assert value == 0.0


class TestSetValue:
    """Tests for setting nested config values."""

    def test_set_value_simple_key(self):
        """Test setting simple key."""
        config = {}
        set_value(config, "threshold", 0.8)
        assert config["threshold"] == 0.8

    def test_set_value_nested_key(self):
        """Test setting nested key with dot notation."""
        config = {}
        set_value(config, "general.settings.threshold", 0.75)
        assert config["general"]["settings"]["threshold"] == 0.75

    def test_set_value_overwrite(self):
        """Test overwriting existing value."""
        config = {"general": {"threshold": 0.5}}
        set_value(config, "general.threshold", 0.9)
        assert config["general"]["threshold"] == 0.9

    def test_set_value_preserves_siblings(self):
        """Test setting value preserves sibling keys."""
        config = {
            "general": {
                "threshold": 0.5,
                "limit": 50,
            }
        }
        set_value(config, "general.threshold", 0.9)
        assert config["general"]["threshold"] == 0.9
        assert config["general"]["limit"] == 50

    def test_set_value_list(self):
        """Test setting list value."""
        config = {}
        set_value(config, "plugins.enabled", ["ram_plus", "wd14"])
        assert config["plugins"]["enabled"] == ["ram_plus", "wd14"]


class TestConfigRoundTrip:
    """Tests for config save/load round trip."""

    def test_config_round_trip(self):
        """Test saving and loading config preserves values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            original = {
                "general": {
                    "default_threshold": 0.65,
                    "default_limit": 75,
                    "default_format": "json",
                },
                "plugins": {
                    "enabled": ["ram_plus"],
                },
            }

            with patch("imlage.utils.config.get_config_path", return_value=config_path):
                save_config(original)
                loaded = load_config()

            assert loaded["general"]["default_threshold"] == 0.65
            assert loaded["general"]["default_limit"] == 75
            assert loaded["plugins"]["enabled"] == ["ram_plus"]
