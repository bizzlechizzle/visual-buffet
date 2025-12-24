"""Plugin system for Visual Buffet.

This module provides the plugin infrastructure:
- PluginBase: Abstract base class all plugins must inherit
- Schemas: Data structures for plugin communication
- Loader: Plugin discovery and loading
"""

from .base import PluginBase
from .loader import discover_plugins, get_plugins_dir, load_all_plugins, load_plugin
from .schemas import HardwareProfile, ImageSize, PluginInfo, Tag, TagResult

__all__ = [
    "PluginBase",
    "PluginInfo",
    "Tag",
    "TagResult",
    "HardwareProfile",
    "ImageSize",
    "discover_plugins",
    "load_plugin",
    "load_all_plugins",
    "get_plugins_dir",
]
