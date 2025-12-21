"""Abstract base class for all IMLAGE plugins.

Every plugin MUST inherit from PluginBase and implement all abstract methods.
This ensures consistent behavior across all plugins.

Example:
    class MyPlugin(PluginBase):
        def get_info(self):
            return PluginInfo(name="my_plugin", ...)

        def is_available(self):
            return self.get_model_path().exists()

        def tag(self, image_path):
            # Do ML inference
            return TagResult(...)
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import PluginInfo, TagResult


class PluginBase(ABC):
    """Base class that all plugins must inherit from.

    Plugins are discovered automatically from the plugins/ directory.
    Each plugin must have a plugin.toml file with metadata.

    Lifecycle:
        1. Plugin is discovered by loader
        2. Plugin class is instantiated
        3. is_available() is checked
        4. If not available, setup() can be called
        5. tag() is called for each image
    """

    def __init__(self, plugin_dir: Path):
        """Initialize plugin with its directory path.

        Args:
            plugin_dir: Path to the plugin's directory (e.g., plugins/ram_plus/)
        """
        self._plugin_dir = plugin_dir

    @property
    def plugin_dir(self) -> Path:
        """Get the plugin's directory path."""
        return self._plugin_dir

    def get_model_path(self) -> Path:
        """Get path to plugin's models directory.

        Returns:
            Path to plugins/<name>/models/
        """
        return self._plugin_dir / "models"

    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Return plugin metadata.

        Returns:
            PluginInfo with name, version, description, requirements
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if plugin is ready to use.

        Should verify:
            - Model files exist
            - Dependencies are installed
            - Hardware requirements are met

        Returns:
            True if plugin can run, False otherwise
        """
        pass

    @abstractmethod
    def tag(self, image_path: Path) -> TagResult:
        """Tag an image and return results.

        This is the main method called by the engine.

        Args:
            image_path: Path to image file

        Returns:
            TagResult with list of tags and metadata

        Raises:
            ImageError: If image cannot be loaded
            PluginError: If inference fails
        """
        pass

    def setup(self) -> bool:
        """Download models and prepare plugin for use.

        Called when is_available() returns False.
        Should download model files, create directories, etc.

        Returns:
            True if setup succeeded, False otherwise
        """
        # Default implementation does nothing
        return True
