"""Core tagging engine.

Orchestrates the tagging process:
1. Load plugins
2. Validate images
3. Run plugins on images (at appropriate resolution for quality setting)
4. Aggregate results (merge if HIGH quality mode)
"""

from pathlib import Path
from typing import Any

from ..exceptions import ImageError
from ..plugins.base import PluginBase
from ..plugins.loader import load_all_plugins
from ..plugins.schemas import Tag, TagQuality, merge_tags
from ..utils.image import (
    generate_thumbnail,
    get_thumbnail_path,
    validate_image,
    THUMBNAIL_FORMAT,
)


class TaggingEngine:
    """Main engine for processing images through plugins.

    Supports per-plugin quality settings that determine which resolution
    thumbnails are used for tagging:

    - QUICK: 480px thumbnail (fast, ~66% tag coverage)
    - STANDARD: 1080px preview (balanced, ~87% tag coverage)
    - HIGH: Multiple resolutions merged (~98% tag coverage)

    Example:
        >>> engine = TaggingEngine()
        >>> result = engine.tag_image(
        ...     Path("photo.jpg"),
        ...     plugin_configs={
        ...         "ram_plus": {"quality": "quick"},
        ...         "florence_2": {"quality": "high"},
        ...     }
        ... )
    """

    def __init__(
        self,
        plugins: list[PluginBase] | None = None,
        thumbnail_dir: Path | None = None,
    ):
        """Initialize engine with plugins.

        Args:
            plugins: List of plugins to use. If None, loads all available.
            thumbnail_dir: Directory for cached thumbnails. If None, uses temp.
        """
        if plugins is None:
            plugins = load_all_plugins()

        self._plugins = {p.get_info().name: p for p in plugins}
        self._thumbnail_dir = thumbnail_dir or Path.home() / ".imlage" / "cache" / "thumbnails"
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    @property
    def plugins(self) -> dict[str, PluginBase]:
        """Get dict of loaded plugins by name."""
        return self._plugins

    @property
    def thumbnail_dir(self) -> Path:
        """Get thumbnail cache directory."""
        return self._thumbnail_dir

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def _get_or_create_thumbnail(
        self,
        image_path: Path,
        resolution: int,
    ) -> Path:
        """Get or create a thumbnail at the specified resolution.

        Args:
            image_path: Original image path
            resolution: Desired resolution (max long side)

        Returns:
            Path to thumbnail
        """
        # Use image filename as ID
        image_id = image_path.stem
        thumb_path = get_thumbnail_path(
            self._thumbnail_dir, image_id, resolution, THUMBNAIL_FORMAT
        )

        if not thumb_path.exists():
            generate_thumbnail(image_path, thumb_path, resolution)

        return thumb_path

    def _tag_at_resolution(
        self,
        plugin: PluginBase,
        image_path: Path,
        resolution: int,
    ) -> list[Tag]:
        """Run plugin on image at specific resolution.

        Args:
            plugin: Plugin to use
            image_path: Original image path
            resolution: Resolution to tag at

        Returns:
            List of Tag objects
        """
        thumb_path = self._get_or_create_thumbnail(image_path, resolution)
        result = plugin.tag(thumb_path)
        return result.tags

    def _tag_with_quality(
        self,
        plugin: PluginBase,
        image_path: Path,
        quality: TagQuality,
    ) -> tuple[list[Tag], list[int]]:
        """Tag image using quality-based resolution selection.

        Args:
            plugin: Plugin to use
            image_path: Original image path
            quality: Quality level (QUICK, STANDARD, HIGH)

        Returns:
            Tuple of (tags, resolutions_used)
        """
        resolutions = quality.resolutions
        all_tags: list[Tag] = []

        for resolution in resolutions:
            tags = self._tag_at_resolution(plugin, image_path, resolution)
            all_tags.extend(tags)

        return all_tags, resolutions

    def tag_image(
        self,
        image_path: Path,
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
        plugin_configs: dict[str, Any] | None = None,
        default_quality: TagQuality | str = TagQuality.STANDARD,
        use_thumbnails: bool = True,
    ) -> dict[str, Any]:
        """Tag a single image.

        Args:
            image_path: Path to image file
            plugin_names: Plugins to use (None = all)
            threshold: Default minimum confidence to include
            limit: Default maximum tags per plugin
            plugin_configs: Per-plugin settings dict:
                {
                    "plugin_name": {
                        "threshold": 0.5,
                        "limit": 50,
                        "quality": "high"  # quick | standard | high
                    }
                }
            default_quality: Default quality level for plugins without config
            use_thumbnails: If True, use quality-based thumbnails. If False,
                           use original image (legacy behavior).

        Returns:
            Dict matching output contract:
            {
                "file": "path/to/image.jpg",
                "results": {
                    "plugin_name": {
                        "tags": [...],
                        "model": "...",
                        "quality": "standard",
                        "resolutions": [1080],
                        ...
                    }
                }
            }
        """
        # Validate image
        validate_image(image_path)

        # Normalize default quality
        if isinstance(default_quality, str):
            default_quality = TagQuality(default_quality)

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
                # Get per-plugin config or use defaults
                config = plugin_configs.get(name, {}) if plugin_configs else {}

                # Handle both dict and Pydantic model
                if hasattr(config, 'threshold'):
                    plugin_threshold = config.threshold
                    plugin_limit = config.limit
                    quality_str = getattr(config, 'quality', None)
                else:
                    plugin_threshold = config.get("threshold", threshold)
                    plugin_limit = config.get("limit", limit)
                    quality_str = config.get("quality")

                # Parse quality setting
                if quality_str:
                    quality = TagQuality(quality_str) if isinstance(quality_str, str) else quality_str
                else:
                    quality = default_quality

                # Tag using quality-based resolution or original
                if use_thumbnails:
                    all_tags, resolutions_used = self._tag_with_quality(
                        plugin, image_path, quality
                    )

                    # For HIGH mode with multiple resolutions, merge tags
                    if len(resolutions_used) > 1:
                        merged = merge_tags(all_tags)
                        # Convert back to Tag objects for filtering
                        filtered_tags = [
                            Tag(label=t.label, confidence=t.confidence)
                            for t in merged
                        ]
                    else:
                        filtered_tags = all_tags
                else:
                    # Legacy: tag original image directly
                    result = plugin.tag(image_path)
                    filtered_tags = result.tags
                    resolutions_used = []
                    quality = TagQuality.STANDARD

                # Apply threshold filter (only for tags with confidence scores)
                filtered_tags = [
                    t for t in filtered_tags
                    if t.confidence is None or t.confidence >= plugin_threshold
                ]

                # Sort by confidence (highest first) if confidence is available
                # Tags without confidence keep their original order (relevance order)
                if filtered_tags and filtered_tags[0].confidence is not None:
                    filtered_tags.sort(key=lambda t: t.confidence or 0, reverse=True)

                # Apply limit
                if plugin_limit:
                    filtered_tags = filtered_tags[:plugin_limit]

                # Build result dict
                plugin_info = plugin.get_info()
                results[name] = {
                    "tags": [t.to_dict() for t in filtered_tags],
                    "model": plugin_info.name,
                    "version": plugin_info.version,
                    "quality": quality.value,
                    "resolutions": resolutions_used,
                }

            except Exception as e:
                results[name] = {"error": str(e)}

        return {"file": str(image_path), "results": results}

    def tag_batch(
        self,
        image_paths: list[Path],
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
        plugin_configs: dict[str, Any] | None = None,
        default_quality: TagQuality | str = TagQuality.STANDARD,
        use_thumbnails: bool = True,
    ) -> list[dict[str, Any]]:
        """Tag multiple images.

        Args:
            image_paths: List of image paths
            plugin_names: Plugins to use
            threshold: Minimum confidence
            limit: Maximum tags per plugin
            plugin_configs: Per-plugin settings (threshold, limit, quality)
            default_quality: Default quality for plugins without config
            use_thumbnails: If True, use quality-based thumbnails

        Returns:
            List of result dicts, one per image
        """
        results = []

        for path in image_paths:
            try:
                result = self.tag_image(
                    path,
                    plugin_names,
                    threshold,
                    limit,
                    plugin_configs,
                    default_quality,
                    use_thumbnails,
                )
                results.append(result)
            except ImageError as e:
                results.append({"file": str(path), "error": str(e)})

        return results
