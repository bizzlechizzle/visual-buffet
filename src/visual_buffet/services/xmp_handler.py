"""XMP sidecar handling for visual-buffet.

Integrates with the wake-n-blake → shoemaker → visual-buffet pipeline by
reading existing XMP sidecars and adding ML tag metadata.

Pipeline Order:
    1. wake-n-blake (import) - Creates XMP with provenance
    2. shoemaker (thumbnails) - Adds thumbnail metadata
    3. visual-buffet (ML tags) - Adds ML tagging results (this module)
"""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visual_buffet.constants import (
    XMP_NAMESPACE_PREFIX,
    XMP_SCHEMA_VERSION,
)
from visual_buffet.exceptions import VisualBuffetError

logger = logging.getLogger(__name__)


class XMPError(VisualBuffetError):
    """Error during XMP operations."""

    pass


@dataclass
class MLTag:
    """A single ML-generated tag."""

    label: str
    confidence: float | None = None
    plugin: str | None = None


@dataclass
class XMPTagData:
    """ML tag data to write to XMP."""

    tags: list[MLTag] = field(default_factory=list)
    plugins_used: list[str] = field(default_factory=list)
    tagged_at: str = ""
    schema_version: int = XMP_SCHEMA_VERSION
    threshold: float = 0.0
    size_used: str = "original"
    inference_time_ms: float = 0.0

    def __post_init__(self):
        if not self.tagged_at:
            self.tagged_at = datetime.now(UTC).isoformat()


def _get_exiftool_path() -> str | None:
    """Find exiftool in PATH."""
    return shutil.which("exiftool")


def _check_exiftool() -> bool:
    """Check if exiftool is available."""
    path = _get_exiftool_path()
    if not path:
        return False
    try:
        result = subprocess.run(
            [path, "-ver"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def get_sidecar_path(image_path: Path) -> Path:
    """Get the XMP sidecar path for an image.

    Args:
        image_path: Path to the image file

    Returns:
        Path to the XMP sidecar file (e.g., image.jpg.xmp)
    """
    return image_path.with_suffix(image_path.suffix + ".xmp")


def has_xmp_sidecar(image_path: Path) -> bool:
    """Check if an XMP sidecar exists for an image.

    Args:
        image_path: Path to the image file

    Returns:
        True if sidecar exists
    """
    return get_sidecar_path(image_path).exists()


def read_xmp_sidecar(image_path: Path) -> dict[str, Any] | None:
    """Read XMP sidecar data for an image.

    Uses exiftool to read XMP metadata while preserving all namespaces.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary of XMP data, or None if no sidecar exists
    """
    sidecar_path = get_sidecar_path(image_path)

    if not sidecar_path.exists():
        return None

    exiftool = _get_exiftool_path()
    if not exiftool:
        logger.warning("exiftool not found, cannot read XMP sidecar")
        return None

    try:
        result = subprocess.run(
            [
                exiftool,
                "-json",
                "-struct",
                "-XMP:all",
                str(sidecar_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"exiftool failed: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        return data[0] if data else None

    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read XMP sidecar: {e}")
        return None


def read_visual_buffet_tags(image_path: Path) -> XMPTagData | None:
    """Read visual-buffet tag data from XMP sidecar.

    Args:
        image_path: Path to the image file

    Returns:
        XMPTagData if visual-buffet data exists, None otherwise
    """
    xmp_data = read_xmp_sidecar(image_path)
    if not xmp_data:
        return None

    # Look for visual-buffet namespace data
    vb_data = xmp_data.get(f"XMP-{XMP_NAMESPACE_PREFIX}")
    if not vb_data:
        # Try dc:subject for standard tags
        subjects = xmp_data.get("Subject", [])
        if subjects:
            return XMPTagData(
                tags=[MLTag(label=s) for s in subjects],
            )
        return None

    # Parse visual-buffet structured data
    tags = []
    raw_tags = vb_data.get("Tags", [])
    for tag in raw_tags:
        if isinstance(tag, dict):
            tags.append(
                MLTag(
                    label=tag.get("Label", ""),
                    confidence=tag.get("Confidence"),
                    plugin=tag.get("Plugin"),
                )
            )
        else:
            tags.append(MLTag(label=str(tag)))

    return XMPTagData(
        tags=tags,
        plugins_used=vb_data.get("PluginsUsed", []),
        tagged_at=vb_data.get("TaggedAt", ""),
        schema_version=vb_data.get("SchemaVersion", 1),
        threshold=vb_data.get("Threshold", 0.0),
        size_used=vb_data.get("SizeUsed", "original"),
        inference_time_ms=vb_data.get("InferenceTimeMs", 0.0),
    )


def write_xmp_sidecar(
    image_path: Path,
    tag_data: XMPTagData,
    *,
    write_dc_subject: bool = True,
    append_custody_event: bool = True,
) -> bool:
    """Write visual-buffet tag data to XMP sidecar.

    Uses exiftool to write XMP metadata while preserving existing namespaces
    (wake-n-blake, shoemaker, etc.).

    Args:
        image_path: Path to the image file
        tag_data: Tag data to write
        write_dc_subject: Also write tags to dc:subject for compatibility
        append_custody_event: Add custody event to wnb:CustodyChain

    Returns:
        True if successful
    """
    exiftool = _get_exiftool_path()
    if not exiftool:
        raise XMPError("exiftool not found. Install with: brew install exiftool")

    sidecar_path = get_sidecar_path(image_path)

    # Build exiftool arguments
    args = [
        exiftool,
        "-overwrite_original",
    ]

    # Visual-buffet namespace tags (structured)
    vb_prefix = f"-XMP-{XMP_NAMESPACE_PREFIX}"
    args.extend([
        f"{vb_prefix}:SchemaVersion={tag_data.schema_version}",
        f"{vb_prefix}:TaggedAt={tag_data.tagged_at}",
        f"{vb_prefix}:Threshold={tag_data.threshold}",
        f"{vb_prefix}:SizeUsed={tag_data.size_used}",
        f"{vb_prefix}:InferenceTimeMs={tag_data.inference_time_ms}",
    ])

    # Add plugins used
    for plugin in tag_data.plugins_used:
        args.append(f"{vb_prefix}:PluginsUsed+={plugin}")

    # Add tags as structured data
    for tag in tag_data.tags:
        tag_struct = f"{{Label={tag.label}"
        if tag.confidence is not None:
            tag_struct += f",Confidence={tag.confidence:.4f}"
        if tag.plugin:
            tag_struct += f",Plugin={tag.plugin}"
        tag_struct += "}"
        args.append(f"{vb_prefix}:Tags+={tag_struct}")

    # Also write to dc:subject for compatibility with other tools
    if write_dc_subject:
        # Get unique labels sorted by confidence
        unique_labels = []
        seen = set()
        sorted_tags = sorted(
            tag_data.tags,
            key=lambda t: t.confidence or 0.0,
            reverse=True,
        )
        for tag in sorted_tags:
            if tag.label not in seen:
                unique_labels.append(tag.label)
                seen.add(tag.label)

        for label in unique_labels:
            args.append(f"-XMP-dc:Subject+={label}")

    # Add custody event if wake-n-blake chain exists
    if append_custody_event and has_xmp_sidecar(image_path):
        existing = read_xmp_sidecar(image_path)
        if existing and "CustodyChain" in str(existing):
            # Increment event count
            args.append("-XMP-wnb:EventCount+=1")
            args.append(f"-XMP-wnb:SidecarUpdated={datetime.now(UTC).isoformat()}")

    # Target the sidecar file
    args.append(str(sidecar_path))

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(f"exiftool failed: {result.stderr}")
            return False

        return True

    except subprocess.SubprocessError as e:
        logger.error(f"Failed to write XMP sidecar: {e}")
        return False


def clear_xmp_tags(image_path: Path) -> bool:
    """Clear visual-buffet tags from XMP sidecar.

    Args:
        image_path: Path to the image file

    Returns:
        True if successful
    """
    exiftool = _get_exiftool_path()
    if not exiftool:
        raise XMPError("exiftool not found")

    sidecar_path = get_sidecar_path(image_path)
    if not sidecar_path.exists():
        return True  # Nothing to clear

    try:
        result = subprocess.run(
            [
                exiftool,
                "-overwrite_original",
                f"-XMP-{XMP_NAMESPACE_PREFIX}:all=",
                str(sidecar_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        return result.returncode == 0

    except subprocess.SubprocessError as e:
        logger.error(f"Failed to clear XMP tags: {e}")
        return False


class XMPHandler:
    """High-level XMP sidecar handler for visual-buffet.

    Provides methods for reading/writing ML tag data to XMP sidecars
    while integrating with the wake-n-blake/shoemaker pipeline.
    """

    def __init__(self, *, use_json_fallback: bool = True):
        """Initialize XMP handler.

        Args:
            use_json_fallback: Fall back to JSON files if exiftool unavailable
        """
        self._exiftool_available = _check_exiftool()
        self._use_json_fallback = use_json_fallback

        if not self._exiftool_available:
            logger.warning(
                "exiftool not found. XMP integration limited. "
                "Install with: brew install exiftool"
            )

    @property
    def exiftool_available(self) -> bool:
        """Check if exiftool is available."""
        return self._exiftool_available

    def read_tags(self, image_path: Path) -> XMPTagData | None:
        """Read existing ML tags from XMP or JSON.

        Args:
            image_path: Path to the image file

        Returns:
            XMPTagData if tags exist, None otherwise
        """
        image_path = Path(image_path)

        # Try XMP first
        if self._exiftool_available and has_xmp_sidecar(image_path):
            data = read_visual_buffet_tags(image_path)
            if data:
                return data

        # Fall back to JSON
        if self._use_json_fallback:
            json_path = image_path.parent / f"{image_path.stem}_tags.json"
            if json_path.exists():
                return self._read_json_tags(json_path)

        return None

    def write_tags(
        self,
        image_path: Path,
        tags: list[dict[str, Any]],
        *,
        plugins_used: list[str] | None = None,
        threshold: float = 0.0,
        size_used: str = "original",
        inference_time_ms: float = 0.0,
        write_json: bool = True,
    ) -> bool:
        """Write ML tags to XMP sidecar and/or JSON.

        Args:
            image_path: Path to the image file
            tags: List of tag dicts with 'label' and optional 'confidence', 'plugin'
            plugins_used: List of plugins that generated the tags
            threshold: Threshold used for filtering
            size_used: Image size preset used
            inference_time_ms: Total inference time
            write_json: Also write JSON file for compatibility

        Returns:
            True if at least one write method succeeded
        """
        image_path = Path(image_path)
        plugins_used = plugins_used or []

        # Convert to XMPTagData
        tag_data = XMPTagData(
            tags=[
                MLTag(
                    label=t.get("label", ""),
                    confidence=t.get("confidence"),
                    plugin=t.get("plugin"),
                )
                for t in tags
            ],
            plugins_used=plugins_used,
            threshold=threshold,
            size_used=size_used,
            inference_time_ms=inference_time_ms,
        )

        success = False

        # Write to XMP if exiftool available
        if self._exiftool_available:
            try:
                if write_xmp_sidecar(image_path, tag_data):
                    success = True
                    logger.debug(f"Wrote XMP sidecar for {image_path}")
            except XMPError as e:
                logger.warning(f"Failed to write XMP: {e}")

        # Write JSON file
        if write_json or (self._use_json_fallback and not success):
            json_path = image_path.parent / f"{image_path.stem}_tags.json"
            if self._write_json_tags(json_path, image_path, tags, plugins_used, threshold, size_used, inference_time_ms):
                success = True

        return success

    def _read_json_tags(self, json_path: Path) -> XMPTagData | None:
        """Read tags from JSON file."""
        try:
            data = json.loads(json_path.read_text())
            results = data.get("results", {})

            tags = []
            plugins = []

            for plugin_name, plugin_data in results.items():
                plugins.append(plugin_name)
                for tag in plugin_data.get("tags", []):
                    tags.append(
                        MLTag(
                            label=tag.get("label", ""),
                            confidence=tag.get("confidence"),
                            plugin=plugin_name,
                        )
                    )

            return XMPTagData(
                tags=tags,
                plugins_used=plugins,
                tagged_at=data.get("tagged_at", ""),
            )

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read JSON tags: {e}")
            return None

    def _write_json_tags(
        self,
        json_path: Path,
        image_path: Path,
        tags: list[dict[str, Any]],
        plugins_used: list[str],
        threshold: float,
        size_used: str,
        inference_time_ms: float,
    ) -> bool:
        """Write tags to JSON file."""
        try:
            # Group tags by plugin
            results = {}
            for plugin in plugins_used:
                plugin_tags = [t for t in tags if t.get("plugin") == plugin]
                results[plugin] = {
                    "tags": plugin_tags,
                    "model": plugin,
                }

            data = {
                "file": str(image_path),
                "filename": image_path.name,
                "tagged_at": datetime.now(UTC).isoformat(),
                "threshold": threshold,
                "size": size_used,
                "inference_time_ms": inference_time_ms,
                "results": results,
            }

            json_path.write_text(json.dumps(data, indent=2))
            return True

        except OSError as e:
            logger.error(f"Failed to write JSON tags: {e}")
            return False

    def has_tags(self, image_path: Path) -> bool:
        """Check if image has existing ML tags.

        Args:
            image_path: Path to the image file

        Returns:
            True if tags exist in XMP or JSON
        """
        image_path = Path(image_path)

        # Check XMP
        if self._exiftool_available and has_xmp_sidecar(image_path):
            data = read_visual_buffet_tags(image_path)
            if data and data.tags:
                return True

        # Check JSON
        if self._use_json_fallback:
            json_path = image_path.parent / f"{image_path.stem}_tags.json"
            if json_path.exists():
                return True

        return False

    def clear_tags(self, image_path: Path) -> bool:
        """Clear ML tags from both XMP and JSON.

        Args:
            image_path: Path to the image file

        Returns:
            True if successful
        """
        image_path = Path(image_path)
        success = True

        # Clear XMP
        if self._exiftool_available and has_xmp_sidecar(image_path):
            if not clear_xmp_tags(image_path):
                success = False

        # Clear JSON
        json_path = image_path.parent / f"{image_path.stem}_tags.json"
        if json_path.exists():
            try:
                json_path.unlink()
            except OSError:
                success = False

        return success
