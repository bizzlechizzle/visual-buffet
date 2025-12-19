"""Data schemas for the plugin system.

These dataclasses define the structure of data exchanged between
the core engine and plugins. All plugins must use these schemas.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TagQuality(str, Enum):
    """Tag quality levels determining which resolution(s) to use.

    Each plugin can have its own quality setting. Higher quality
    uses more resolutions and merges results for maximum tag coverage.

    Attributes:
        QUICK: 1080px preview only. Fast, ~87% tag coverage.
        STANDARD: 480px + 2048px merged. Balanced, ~92% coverage.
        HIGH: All standard thumbnails (480 + 1080 + 2048) merged. ~96% coverage.
        MAX: All resolutions (480 + 1080 + 2048 + 4096 + original). 100% coverage.

    Resolution mapping (0 = original image):
        QUICK    -> [1080]
        STANDARD -> [480, 2048]
        HIGH     -> [480, 1080, 2048]
        MAX      -> [480, 1080, 2048, 4096, 0]

    Example:
        >>> quality = TagQuality.QUICK
        >>> quality.resolutions
        [1080]
    """

    QUICK = "quick"
    STANDARD = "standard"
    HIGH = "high"
    MAX = "max"

    @property
    def resolutions(self) -> list[int]:
        """Get resolution(s) for this quality level.

        Note: 0 indicates the original image should be used.
        """
        return QUALITY_RESOLUTIONS[self]


# Resolution mapping for each quality level
# Note: 0 = use original image (optimized but full resolution)
QUALITY_RESOLUTIONS: dict[TagQuality, list[int]] = {
    TagQuality.QUICK: [1080],
    TagQuality.STANDARD: [480, 2048],
    TagQuality.HIGH: [480, 1080, 2048],
    TagQuality.MAX: [480, 1080, 2048, 4096, 0],
}

# Standard thumbnail sizes generated for all images
THUMBNAIL_SIZES: dict[str, int] = {
    "grid": 480,      # Grid view thumbnails
    "preview": 1080,  # Lightbox preview
    "zoom": 2048,     # Lightbox zoom / full-res preview
    "ultra": 4096,    # Ultra quality for MAX mode tagging
}


@dataclass
class Tag:
    """A single tag with optional confidence score.

    Attributes:
        label: The tag text (e.g., "dog", "outdoor")
        confidence: How confident the model is (0.0 to 1.0), or None if not provided

    Example:
        >>> tag = Tag(label="cat", confidence=0.95)
        >>> tag.label
        'cat'
        >>> tag_no_conf = Tag(label="dog", confidence=None)
    """

    label: str
    confidence: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"label": self.label}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass
class TagResult:
    """Result from a plugin's tag() method.

    Attributes:
        tags: List of Tag objects
        model: Model name/identifier used
        version: Plugin version
        inference_time_ms: How long inference took in milliseconds

    Example:
        >>> result = TagResult(
        ...     tags=[Tag("dog", 0.9)],
        ...     model="ram_plus_swin_large",
        ...     version="1.0.0",
        ...     inference_time_ms=142.5
        ... )
    """

    tags: list[Tag]
    model: str
    version: str
    inference_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary matching output contract."""
        return {
            "tags": [t.to_dict() for t in self.tags],
            "model": self.model,
            "version": self.version,
            "inference_time_ms": self.inference_time_ms,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MergedTag:
    """A tag with metadata from multi-resolution merging.

    Used when HIGH quality mode merges tags from multiple resolutions.

    Attributes:
        label: The tag text
        confidence: Highest confidence score across resolutions (or None)
        sources: Number of resolutions that found this tag
        min_resolution: Smallest resolution where tag was found
    """

    label: str
    confidence: float | None = None
    sources: int = 1
    min_resolution: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"label": self.label}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.sources > 1:
            result["sources"] = self.sources
        if self.min_resolution is not None:
            result["min_resolution"] = self.min_resolution
        return result


def merge_tags(all_tags: list[Tag], resolutions_used: list[int] | None = None) -> list[MergedTag]:
    """Merge tags from multiple resolutions into deduplicated list.

    Tags found at multiple resolutions get higher implicit confidence.
    Keeps the highest confidence score when duplicates have scores.

    Args:
        all_tags: List of Tag objects from all resolutions
        resolutions_used: List of resolutions for min_resolution tracking

    Returns:
        List of MergedTag objects, sorted by sources (desc), confidence (desc)

    Example:
        >>> tags = [Tag("cat", 0.9), Tag("cat", 0.8), Tag("dog", 0.7)]
        >>> merged = merge_tags(tags)
        >>> merged[0].label
        'cat'
        >>> merged[0].sources
        2
    """
    merged: dict[str, MergedTag] = {}

    for tag in all_tags:
        label = tag.label.lower().strip()
        if not label:
            continue

        if label not in merged:
            merged[label] = MergedTag(
                label=label,
                confidence=tag.confidence,
                sources=1,
                min_resolution=None,
            )
        else:
            merged[label].sources += 1
            # Keep highest confidence
            if tag.confidence is not None:
                if merged[label].confidence is None:
                    merged[label].confidence = tag.confidence
                else:
                    merged[label].confidence = max(merged[label].confidence, tag.confidence)

    # Sort by sources (more = better), then confidence, then label
    result = sorted(
        merged.values(),
        key=lambda t: (-t.sources, -(t.confidence or 0), t.label),
    )

    return result


@dataclass
class PluginInfo:
    """Metadata about a plugin.

    Attributes:
        name: Unique plugin identifier (e.g., "ram_plus")
        version: Semantic version (e.g., "1.0.0")
        description: Human-readable description
        hardware_reqs: Dict of requirements {"gpu": bool, "min_ram_gb": int}
    """

    name: str
    version: str
    description: str
    hardware_reqs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HardwareProfile:
    """Detected hardware capabilities.

    Cached to ~/.imlage/hardware.json after first detection.

    Attributes:
        cpu_model: CPU name (e.g., "Apple M2 Pro")
        cpu_cores: Number of CPU cores
        ram_total_gb: Total RAM in gigabytes
        ram_available_gb: Currently available RAM
        gpu_type: "cuda", "mps", or None
        gpu_name: GPU name if available
        gpu_vram_gb: GPU memory in GB if available
    """

    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_type: str | None = None
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        """Create HardwareProfile from dictionary."""
        return cls(**data)
