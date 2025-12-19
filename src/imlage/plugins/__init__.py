"""Plugin system for IMLAGE.

This module provides the plugin infrastructure:
- PluginBase: Abstract base class all plugins must inherit
- Schemas: Data structures for plugin communication
- Loader: Plugin discovery and loading
"""

from .base import PluginBase
from .loader import discover_plugins, get_plugins_dir, load_all_plugins, load_plugin
from .schemas import HardwareProfile, PluginInfo, Tag, TagResult

__all__ = [
    "PluginBase",
    "PluginInfo",
    "Tag",
    "TagResult",
    "HardwareProfile",
    "discover_plugins",
    "load_plugin",
    "load_all_plugins",
    "get_plugins_dir",
]
