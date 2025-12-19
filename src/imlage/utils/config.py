"""Configuration file management.

Config is stored in TOML format at:
- macOS/Linux: ~/.config/imlage/config.toml
- Windows: %APPDATA%\\imlage\\config.toml

Usage:
    config = load_config()
    threshold = get_value(config, "plugins.ram_plus.threshold", 0.5)
"""

import platform
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

import tomli_w

from ..exceptions import ConfigError


def get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if platform.system() == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    else:
        base = Path.home() / ".config"
    return base / "imlage"


def get_config_path() -> Path:
    """Get path to config file."""
    return get_config_dir() / "config.toml"


DEFAULT_CONFIG = {
    "general": {
        "default_format": "json",
        "default_threshold": 0.5,
        "default_limit": 50,
    },
    "plugins": {
        "enabled": ["ram_plus"],
    },
}


def load_config() -> dict:
    """Load configuration from file.

    Creates default config if file doesn't exist.

    Returns:
        Dict with configuration values

    Raises:
        ConfigError: If config file is malformed
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise ConfigError(f"Failed to load config: {e}")


def save_config(config: dict) -> None:
    """Save configuration to file.

    Args:
        config: Dict with configuration values
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def get_value(config: dict, key: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Config dict
        key: Dot-separated key (e.g., "plugins.ram_plus.threshold")
        default: Default value if key not found

    Returns:
        Config value or default

    Example:
        >>> config = {"plugins": {"ram_plus": {"threshold": 0.7}}}
        >>> get_value(config, "plugins.ram_plus.threshold")
        0.7
    """
    parts = key.split(".")
    current = config

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def set_value(config: dict, key: str, value: Any) -> None:
    """Set a nested config value using dot notation.

    Args:
        config: Config dict (modified in place)
        key: Dot-separated key
        value: Value to set

    Example:
        >>> config = {}
        >>> set_value(config, "plugins.ram_plus.threshold", 0.7)
        >>> config
        {'plugins': {'ram_plus': {'threshold': 0.7}}}
    """
    parts = key.split(".")
    current = config

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value
