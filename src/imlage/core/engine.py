"""Core tagging engine.

Orchestrates the tagging process:
1. Load plugins
2. Validate images
3. Run plugins on images (at appropriate resolution for quality setting)
4. Aggregate results (merge if HIGH quality mode)
5. Save thumbnails and tags alongside source images
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..exceptions import ImageError
from ..plugins.base import PluginBase
from ..plugins.loader import load_all_plugins
from ..plugins.schemas import Tag, TagQuality, merge_tags
from ..utils.image import (
    THUMBNAIL_FORMAT,
    generate_thumbnail,
    validate_image,
)

# Folder name for imlage data stored alongside images
IMLAGE_FOLDER = "imlage"


class TaggingEngine:
    """Main engine for processing images through plugins.

    Supports per-plugin quality settings that determine which resolution
    thumbnails are used for tagging:

    - QUICK: 480px thumbnail (fast, ~66% tag coverage)
    - STANDARD: 1080px preview (balanced, ~87% tag coverage)
    - HIGH: Multiple resolutions merged (~98% tag coverage)

    Thumbnails and tags are saved in an 'imlage/' folder next to each image:

        /photos/vacation/
        ├── beach.jpg
        └── imlage/
            ├── beach_480.webp
            ├── beach_1080.webp
            ├── beach_2048.webp
            └── beach_tags.json

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
    ):
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

    def _get_imlage_dir(self, image_path: Path) -> Path:
        """Get the imlage folder for an image (next to the image).

        Args:
            image_path: Path to the source image

        Returns:
            Path to imlage folder (e.g., /photos/imlage/)
        """
        imlage_dir = image_path.parent / IMLAGE_FOLDER
        imlage_dir.mkdir(parents=True, exist_ok=True)
        return imlage_dir

    def _get_thumbnail_path(
        self,
        image_path: Path,
        resolution: int,
    ) -> Path:
        """Get the path for a thumbnail.

        Args:
            image_path: Original image path
            resolution: Thumbnail resolution

        Returns:
            Path where thumbnail should be stored
        """
        imlage_dir = self._get_imlage_dir(image_path)
        return imlage_dir / f"{image_path.stem}_{resolution}.{THUMBNAIL_FORMAT}"

    def _get_tags_path(self, image_path: Path) -> Path:
        """Get the path for tags JSON file.

        Args:
            image_path: Original image path

        Returns:
            Path where tags should be stored
        """
        imlage_dir = self._get_imlage_dir(image_path)
        return imlage_dir / f"{image_path.stem}_tags.json"

    def _get_or_create_thumbnail(
        self,
        image_path: Path,
        resolution: int,
    ) -> Path:
        """Get or create a thumbnail at the specified resolution.

        Thumbnails are stored in an 'imlage/' folder next to the source image.

        Args:
            image_path: Original image path
            resolution: Desired resolution (max long side)

        Returns:
            Path to thumbnail
        """
        thumb_path = self._get_thumbnail_path(image_path, resolution)

        if not thumb_path.exists():
            generate_thumbnail(image_path, thumb_path, resolution)

        return thumb_path

    def _save_tags(
        self,
        image_path: Path,
        results: dict[str, Any],
    ) -> Path:
        """Save tagging results to JSON file next to the image.

        Args:
            image_path: Original image path
            results: Tagging results dict

        Returns:
            Path to saved tags file
        """
        tags_path = self._get_tags_path(image_path)

        # Build the tags document
        tags_doc = {
            "file": str(image_path),
            "filename": image_path.name,
            "tagged_at": datetime.now().isoformat(),
            "results": results,
        }

        with open(tags_path, "w") as f:
            json.dump(tags_doc, f, indent=2)

        return tags_path

    def load_tags(self, image_path: Path) -> dict[str, Any] | None:
        """Load existing tags for an image if available.

        Args:
            image_path: Original image path

        Returns:
            Tags dict if found, None otherwise
        """
        tags_path = self._get_tags_path(image_path)

        if not tags_path.exists():
            return None

        with open(tags_path) as f:
            return json.load(f)

    def _tag_at_resolution(
        self,
        plugin: PluginBase,
        image_path: Path,
        resolution: int,
    ) -> tuple[list[Tag], float]:
        """Run plugin on image at specific resolution.

        Args:
            plugin: Plugin to use
            image_path: Original image path
            resolution: Resolution to tag at (0 = use original)

        Returns:
            Tuple of (tags, inference_time_ms)
        """
        if resolution == 0:
            # Use original image directly
            result = plugin.tag(image_path)
        else:
            thumb_path = self._get_or_create_thumbnail(image_path, resolution)
            result = plugin.tag(thumb_path)
        return result.tags, result.inference_time_ms or 0.0

    def _tag_with_quality(
        self,
        plugin: PluginBase,
        image_path: Path,
        quality: TagQuality,
    ) -> tuple[list[Tag], list[int], float]:
        """Tag image using quality-based resolution selection.

        Args:
            plugin: Plugin to use
            image_path: Original image path
            quality: Quality level (QUICK, STANDARD, HIGH)

        Returns:
            Tuple of (tags, resolutions_used, total_inference_time_ms)
        """
        resolutions = quality.resolutions
        all_tags: list[Tag] = []
        total_time_ms = 0.0

        for resolution in resolutions:
            tags, time_ms = self._tag_at_resolution(plugin, image_path, resolution)
            all_tags.extend(tags)
            total_time_ms += time_ms

        return all_tags, resolutions, total_time_ms

    def tag_image(
        self,
        image_path: Path,
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
        plugin_configs: dict[str, Any] | None = None,
        default_quality: TagQuality | str = TagQuality.STANDARD,
        use_thumbnails: bool = True,
        save_tags: bool = True,
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
            save_tags: If True, save tags to JSON file next to image.

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

        Side effects:
            - Creates thumbnails in {image_dir}/imlage/
            - Saves tags to {image_dir}/imlage/{image_stem}_tags.json
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
                # Get plugin's recommended threshold (from PluginInfo)
                plugin_info = plugin.get_info()
                recommended = plugin_info.recommended_threshold

                # Get per-plugin config or use defaults
                config = plugin_configs.get(name, {}) if plugin_configs else {}

                # Handle both dict and Pydantic model
                if hasattr(config, 'threshold'):
                    # Pydantic model - check if threshold is None (use recommended)
                    explicit_threshold = config.threshold
                    if explicit_threshold is not None:
                        plugin_threshold = explicit_threshold
                    else:
                        # Use plugin's recommended threshold
                        plugin_threshold = recommended
                    plugin_limit = config.limit
                    quality_str = getattr(config, 'quality', None)
                else:
                    # Dict config - use plugin's recommended threshold if no explicit threshold
                    # This is critical for SigLIP which needs 0.01, not 0.5
                    explicit_threshold = config.get("threshold")
                    if explicit_threshold is not None:
                        plugin_threshold = explicit_threshold
                    elif threshold > 0 and recommended > 0:
                        # User set a global threshold but plugin has a recommendation
                        # Use the plugin's recommendation (it knows its output range)
                        plugin_threshold = recommended
                    else:
                        plugin_threshold = threshold if threshold > 0 else recommended
                    plugin_limit = config.get("limit", limit)
                    quality_str = config.get("quality")

                # Parse quality setting
                if quality_str:
                    quality = TagQuality(quality_str) if isinstance(quality_str, str) else quality_str
                else:
                    quality = default_quality

                # Tag using quality-based resolution or original
                if use_thumbnails:
                    all_tags, resolutions_used, inference_time_ms = self._tag_with_quality(
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
                    inference_time_ms = result.inference_time_ms or 0.0

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

                # Build result dict (plugin_info already fetched at loop start)
                results[name] = {
                    "tags": [t.to_dict() for t in filtered_tags],
                    "model": plugin_info.name,
                    "version": plugin_info.version,
                    "quality": quality.value,
                    "resolutions": resolutions_used,
                    "inference_time_ms": round(inference_time_ms, 2),
                }

            except Exception as e:
                results[name] = {"error": str(e)}

        # Save tags to file
        if save_tags and results:
            self._save_tags(image_path, results)

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
        save_tags: bool = True,
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
            save_tags: If True, save tags to JSON files next to images

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
                    save_tags,
                )
                results.append(result)
            except ImageError as e:
                results.append({"file": str(path), "error": str(e)})

        return results

    def generate_thumbnails(self, image_path: Path) -> dict[str, Path]:
        """Generate all standard thumbnails for an image.

        Creates thumbnails at 480px, 1080px, and 2048px in the
        imlage/ folder next to the image.

        Args:
            image_path: Path to source image

        Returns:
            Dict mapping resolution to thumbnail path
        """
        from ..plugins.schemas import THUMBNAIL_SIZES

        validate_image(image_path)
        thumbnails = {}

        for name, resolution in THUMBNAIL_SIZES.items():
            thumb_path = self._get_or_create_thumbnail(image_path, resolution)
            thumbnails[name] = thumb_path

        return thumbnails
