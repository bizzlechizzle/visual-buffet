"""Core tagging engine.

Orchestrates the tagging process:
1. Load plugins
2. Validate images
3. Run plugins on images (at specified size)
4. Save tags alongside source images
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..exceptions import ImageError
from ..plugins.base import PluginBase
from ..plugins.loader import load_all_plugins
from ..plugins.schemas import ImageSize, Tag
from ..utils.image import is_raw_image, load_image, validate_image

logger = logging.getLogger(__name__)


class TaggingEngine:
    """Main engine for processing images through plugins.

    Supports configurable image size for tagging:
    - LITTLE: 480px (fastest)
    - SMALL: 1080px (balanced)
    - LARGE: 2048px (high detail)
    - HUGE: 4096px (maximum detail)
    - ORIGINAL: native resolution (default)

    Tags are saved as JSON files directly next to each image:

        /photos/vacation/
        ├── beach.jpg
        └── beach_tags.json

    Example:
        >>> engine = TaggingEngine()
        >>> result = engine.tag_image(
        ...     Path("photo.jpg"),
        ...     size="small",  # Use 1080px
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

        # Inject discovery plugins into SigLIP
        self._setup_siglip_discovery()

    def _setup_siglip_discovery(self) -> None:
        """Give SigLIP references to discovery plugins.

        This enables SigLIP's discovery mode, where it first runs discovery
        plugins to find candidate tags, then scores them with real confidence.

        Available discovery sources:
        - RAM++: General image tagging (4500+ categories)
        - Florence-2: Detailed captioning converted to tags
        - YOLO: Object detection (80 COCO classes)
        - PaddleOCR: Text detection (words found in image)
        - EasyOCR: Scene text detection (80+ languages)
        """
        siglip = self._plugins.get("siglip")
        if siglip and hasattr(siglip, "set_discovery_plugins"):
            discovery_plugins = {}
            if ram := self._plugins.get("ram_plus"):
                discovery_plugins["ram_plus"] = ram
            if florence := self._plugins.get("florence_2"):
                discovery_plugins["florence_2"] = florence
            if yolo := self._plugins.get("yolo"):
                discovery_plugins["yolo"] = yolo
            if paddle_ocr := self._plugins.get("paddle_ocr"):
                discovery_plugins["paddle_ocr"] = paddle_ocr
            if easyocr := self._plugins.get("easyocr"):
                discovery_plugins["easyocr"] = easyocr
            siglip.set_discovery_plugins(discovery_plugins)

    @property
    def plugins(self) -> dict[str, PluginBase]:
        """Get dict of loaded plugins by name."""
        return self._plugins

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def _resolve_threshold(
        self,
        config: Any,
        recommended: float,
        default: float,
    ) -> float:
        """Resolve the threshold to use for a plugin.

        Priority:
        1. Explicit threshold in config (if not None)
        2. Plugin's recommended threshold (if > 0)
        3. Default threshold passed to tag_image

        Args:
            config: Plugin config (dict or Pydantic model)
            recommended: Plugin's recommended threshold from PluginInfo
            default: Default threshold passed to tag_image()

        Returns:
            Threshold to use for filtering tags
        """
        # Extract explicit threshold from config
        if hasattr(config, "threshold"):
            explicit = config.threshold
        elif isinstance(config, dict):
            explicit = config.get("threshold")
        else:
            explicit = None

        # Use explicit if set, otherwise fall back to recommended or default
        if explicit is not None:
            return explicit
        if recommended > 0:
            return recommended
        return default

    def _get_tags_path(self, image_path: Path) -> Path:
        """Get the path for tags JSON file (stored next to the image).

        Args:
            image_path: Original image path

        Returns:
            Path where tags should be stored (e.g., /photos/beach_tags.json)
        """
        return image_path.parent / f"{image_path.stem}_tags.json"

    def _create_temp_from_raw(self, image_path: Path) -> Path:
        """Convert a RAW image to a temporary JPEG file.

        This is needed when plugins don't support RAW formats directly.
        Without this:
        - YOLO/Ultralytics fails with "unsupported format" error
        - PIL-based plugins silently read tiny embedded thumbnail

        Args:
            image_path: Path to RAW image file

        Returns:
            Path to temporary JPEG file (caller must delete)
        """
        # Load RAW properly with rawpy (via load_image)
        img = load_image(image_path)

        # Create temp file with .jpg extension
        fd, temp_path_str = tempfile.mkstemp(suffix=".jpg")

        try:
            temp_path = Path(temp_path_str)
            # Save at high quality to preserve detail
            img.save(temp_path, "JPEG", quality=95)
            return temp_path
        finally:
            # Close file descriptor (file handle opened by mkstemp)
            os.close(fd)

    def _create_resized_temp(self, image_path: Path, max_size: int) -> Path:
        """Create a temporary resized image for tagging.

        Args:
            image_path: Original image path
            max_size: Maximum size for longest side

        Returns:
            Path to temporary resized image (caller must delete)
        """
        img = load_image(image_path)

        # Calculate new dimensions maintaining aspect ratio
        if img.width >= img.height:
            if img.width <= max_size:
                # Image is smaller than target, use as-is
                new_width, new_height = img.width, img.height
            else:
                ratio = max_size / img.width
                new_width = max_size
                new_height = int(img.height * ratio)
        else:
            if img.height <= max_size:
                new_width, new_height = img.width, img.height
            else:
                ratio = max_size / img.height
                new_height = max_size
                new_width = int(img.width * ratio)

        # Resize if needed
        if new_width != img.width or new_height != img.height:
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Create temp file
        fd, temp_path_str = tempfile.mkstemp(suffix=".jpg")

        try:
            temp_path = Path(temp_path_str)
            img.save(temp_path, "JPEG", quality=95)
            return temp_path
        finally:
            os.close(fd)

    def _save_tags(
        self,
        image_path: Path,
        results: dict[str, Any],
    ) -> Path:
        """Save tagging results to JSON file directly next to the image.

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
            "tagged_at": datetime.now(timezone.utc).isoformat(),
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

    def _tag_at_size(
        self,
        plugin: PluginBase,
        image_path: Path,
        size: ImageSize,
    ) -> tuple[list[Tag], float, dict | None]:
        """Run plugin on image at specified size.

        Args:
            plugin: Plugin to use
            image_path: Original image path
            size: Size to tag at (ORIGINAL = use as-is)

        Returns:
            Tuple of (tags, inference_time_ms, metadata)
        """
        resolution = size.resolution

        if resolution == 0:
            # Use original image directly
            # For RAW files, convert to temp JPEG first
            if is_raw_image(image_path):
                temp_path = self._create_temp_from_raw(image_path)
                try:
                    result = plugin.tag(temp_path)
                finally:
                    temp_path.unlink(missing_ok=True)
            else:
                result = plugin.tag(image_path)
        else:
            # Resize to temp file, tag, delete
            temp_path = self._create_resized_temp(image_path, resolution)
            try:
                result = plugin.tag(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

        return result.tags, result.inference_time_ms or 0.0, result.metadata

    def tag_image(
        self,
        image_path: Path,
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        plugin_configs: dict[str, Any] | None = None,
        size: ImageSize | str = ImageSize.ORIGINAL,
        save_tags: bool = True,
    ) -> dict[str, Any]:
        """Tag a single image.

        Args:
            image_path: Path to image file
            plugin_names: Plugins to use (None = all available)
            threshold: Default minimum confidence to include
            plugin_configs: Per-plugin settings dict:
                {
                    "plugin_name": {
                        "threshold": 0.5,
                    }
                }
            size: Image size for tagging (little/small/large/huge/original)
            save_tags: If True, save tags to JSON file next to image.

        Returns:
            Dict matching output contract:
            {
                "file": "path/to/image.jpg",
                "results": {
                    "plugin_name": {
                        "tags": [...],
                        "model": "...",
                        "version": "...",
                        "size": "original",
                        "inference_time_ms": 142.5,
                        ...
                    }
                }
            }

        Side effects:
            - Saves tags to {image_dir}/{image_stem}_tags.json
        """
        # Validate image
        validate_image(image_path)

        # Normalize size
        if isinstance(size, str):
            size = ImageSize(size)

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
                # Get plugin info and config
                plugin_info = plugin.get_info()
                config = plugin_configs.get(name, {}) if plugin_configs else {}

                # Resolve threshold (explicit > recommended > default)
                plugin_threshold = self._resolve_threshold(
                    config, plugin_info.recommended_threshold, threshold
                )

                # Configure plugin with any extra settings (e.g., discovery mode for SigLIP)
                if hasattr(plugin, 'configure'):
                    if hasattr(config, 'model_dump'):
                        plugin.configure(**config.model_dump())
                    elif isinstance(config, dict):
                        plugin.configure(**config)

                # Tag at specified size
                tags, inference_time_ms, plugin_metadata = self._tag_at_size(
                    plugin, image_path, size
                )

                # Apply threshold filter (only for tags with confidence scores)
                filtered_tags = [
                    t for t in tags
                    if t.confidence is None or t.confidence >= plugin_threshold
                ]

                # Sort by confidence (highest first) if confidence is available
                if filtered_tags and filtered_tags[0].confidence is not None:
                    filtered_tags.sort(key=lambda t: t.confidence or 0, reverse=True)

                # Build result dict
                result_dict = {
                    "tags": [t.to_dict() for t in filtered_tags],
                    "model": plugin_info.name,
                    "version": plugin_info.version,
                    "size": size.value,
                    "inference_time_ms": round(inference_time_ms, 2),
                }

                # Include metadata if present (e.g., discovery mode results)
                if plugin_metadata:
                    result_dict["metadata"] = plugin_metadata

                results[name] = result_dict

            except Exception as e:
                logger.exception("Plugin %s failed on %s", name, image_path)
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
        plugin_configs: dict[str, Any] | None = None,
        size: ImageSize | str = ImageSize.ORIGINAL,
        save_tags: bool = True,
        on_progress: Callable[[Path, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Tag multiple images.

        Args:
            image_paths: List of image paths
            plugin_names: Plugins to use
            threshold: Minimum confidence
            plugin_configs: Per-plugin settings
            size: Image size for tagging
            save_tags: If True, save tags to JSON files next to images
            on_progress: Optional callback called after each file completes.
                Receives (image_path, result_dict) where result_dict contains
                the tagging results including inference_time_ms for ETA calc.

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
                    plugin_configs,
                    size,
                    save_tags,
                )
                results.append(result)
            except ImageError as e:
                result = {"file": str(path), "error": str(e)}
                results.append(result)

            # Call progress callback if provided
            if on_progress:
                on_progress(path, result)

        return results
