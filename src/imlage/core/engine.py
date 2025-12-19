"""Core tagging engine.

Orchestrates the tagging process:
1. Load plugins
2. Validate images
3. Run plugins on images
4. Aggregate results
"""

from pathlib import Path
from typing import Any

from ..exceptions import ImageError
from ..plugins.base import PluginBase
from ..plugins.loader import load_all_plugins
from ..utils.image import validate_image


class TaggingEngine:
    """Main engine for processing images through plugins."""

    def __init__(self, plugins: list[PluginBase] | None = None):
        """Initialize engine with plugins.

        Args:
            plugins: List of plugins to use. If None, loads all available.
        """
        if plugins is None:
            plugins = load_all_plugins()

        self._plugins = {p.get_info().name: p for p in plugins}

    @property
    def plugins(self) -> dict[str, PluginBase]:
        """Get dict of loaded plugins by name."""
        return self._plugins

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def tag_image(
        self,
        image_path: Path,
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Tag a single image.

        Args:
            image_path: Path to image file
            plugin_names: Plugins to use (None = all)
            threshold: Minimum confidence to include
            limit: Maximum tags per plugin

        Returns:
            Dict matching output contract:
            {
                "file": "path/to/image.jpg",
                "results": {
                    "plugin_name": {
                        "tags": [...],
                        "model": "...",
                        ...
                    }
                }
            }
        """
        # Validate image
        validate_image(image_path)

        # Select plugins
        if plugin_names:
            plugins = {
                name: p
                for name, p in self._plugins.items()
                if name in plugin_names and p.is_available()
            }
        else:
            plugins = {
                name: p for name, p in self._plugins.items() if p.is_available()
            }

        # Run each plugin
        results: dict[str, Any] = {}
        for name, plugin in plugins.items():
            try:
                result = plugin.tag(image_path)

                # Apply threshold filter
                filtered_tags = [t for t in result.tags if t.confidence >= threshold]

                # Sort by confidence (highest first)
                filtered_tags.sort(key=lambda t: t.confidence, reverse=True)

                # Apply limit
                if limit:
                    filtered_tags = filtered_tags[:limit]

                result.tags = filtered_tags
                results[name] = result.to_dict()

            except Exception as e:
                results[name] = {"error": str(e)}

        return {"file": str(image_path), "results": results}

    def tag_batch(
        self,
        image_paths: list[Path],
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Tag multiple images.

        Args:
            image_paths: List of image paths
            plugin_names: Plugins to use
            threshold: Minimum confidence
            limit: Maximum tags per plugin

        Returns:
            List of result dicts, one per image
        """
        results = []

        for path in image_paths:
            try:
                result = self.tag_image(path, plugin_names, threshold, limit)
                results.append(result)
            except ImageError as e:
                results.append({"file": str(path), "error": str(e)})

        return results
