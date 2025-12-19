"""Plugin discovery and loading.

Scans the plugins/ directory for valid plugins and loads them.
Each plugin must have a plugin.toml file with metadata.
"""

import importlib.util
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from ..exceptions import PluginError, PluginNotFoundError
from .base import PluginBase


def get_plugins_dir() -> Path:
    """Get the plugins directory path.

    Returns:
        Path to the plugins/ directory at project root
    """
    # plugins/ is at project root
    # This file is at src/imlage/plugins/loader.py
    # Go up: loader.py -> plugins/ -> imlage/ -> src/ -> project_root
    src_dir = Path(__file__).parent.parent.parent
    project_root = src_dir.parent
    return project_root / "plugins"


def discover_plugins() -> list[dict]:
    """Discover all plugins in the plugins directory.

    Returns:
        List of dicts with plugin metadata from plugin.toml
    """
    plugins_dir = get_plugins_dir()

    if not plugins_dir.exists():
        return []

    plugins = []

    for plugin_dir in plugins_dir.iterdir():
        if not plugin_dir.is_dir():
            continue

        # Skip hidden directories and __pycache__
        if plugin_dir.name.startswith(".") or plugin_dir.name == "__pycache__":
            continue

        toml_path = plugin_dir / "plugin.toml"
        if not toml_path.exists():
            continue

        try:
            with open(toml_path, "rb") as f:
                metadata = tomllib.load(f)

            metadata["_path"] = plugin_dir
            plugins.append(metadata)
        except Exception:
            # Skip invalid plugins
            continue

    return plugins


def load_plugin(plugin_dir: Path) -> PluginBase:
    """Load a plugin class from its directory.

    Args:
        plugin_dir: Path to plugin directory

    Returns:
        Instantiated plugin object

    Raises:
        PluginError: If plugin cannot be loaded
    """
    toml_path = plugin_dir / "plugin.toml"

    if not toml_path.exists():
        raise PluginNotFoundError(f"No plugin.toml in {plugin_dir}")

    # Read metadata
    try:
        with open(toml_path, "rb") as f:
            metadata = tomllib.load(f)
    except Exception as e:
        raise PluginError(f"Invalid plugin.toml: {e}")

    plugin_section = metadata.get("plugin", {})
    entry_point = plugin_section.get("entry_point")

    if not entry_point:
        raise PluginError("plugin.toml missing 'entry_point'")

    # Import the plugin module
    init_path = plugin_dir / "__init__.py"
    if not init_path.exists():
        raise PluginError(f"No __init__.py in {plugin_dir}")

    try:
        # Dynamic import
        module_name = f"imlage_plugin_{plugin_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, init_path)
        if spec is None or spec.loader is None:
            raise PluginError(f"Cannot load module from {init_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the plugin class
        plugin_class = getattr(module, entry_point)

        if not issubclass(plugin_class, PluginBase):
            raise PluginError(f"{entry_point} does not inherit from PluginBase")

        # Instantiate and return
        return plugin_class(plugin_dir)

    except AttributeError:
        raise PluginError(f"Plugin class '{entry_point}' not found in {init_path}")
    except PluginError:
        raise
    except Exception as e:
        raise PluginError(f"Failed to load plugin: {e}")


def load_all_plugins() -> list[PluginBase]:
    """Load all discovered plugins.

    Returns:
        List of loaded plugin instances
    """
    plugins = []

    for metadata in discover_plugins():
        try:
            plugin = load_plugin(metadata["_path"])
            plugins.append(plugin)
        except PluginError:
            # Skip plugins that fail to load
            continue

    return plugins
