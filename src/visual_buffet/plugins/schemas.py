"""Data schemas for the plugin system.

These dataclasses define the structure of data exchanged between
the core engine and plugins. All plugins must use these schemas.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def boost_confidence(
    raw_confidence: float,
    sources: int,
    boost_per_source: float = 0.15,
) -> float:
    """Boost confidence in log-odds space for multi-resolution agreement.

    When a tag is detected at multiple resolutions, this indicates scale-invariant
    recognition - the model sees it regardless of image scale. This function
    boosts the confidence score to reflect this additional evidence.

    The boost is applied in log-odds (logit) space, which:
    - Keeps scores bounded between 0 and 1
    - Provides diminishing returns at extremes (0.99 can't boost much)
    - Is mathematically equivalent to treating each resolution as Bayesian evidence

    Args:
        raw_confidence: Original sigmoid probability (0-1) from the model
        sources: Number of resolutions that detected this tag
        boost_per_source: Log-odds boost per additional source (default 0.15)

    Returns:
        Boosted confidence (0-1), never exceeds 1.0

    Example:
        >>> boost_confidence(0.70, sources=1)
        0.7
        >>> boost_confidence(0.70, sources=5)
        0.8175  # ~82%
        >>> boost_confidence(0.95, sources=5)
        0.9726  # Only +2% boost at high confidence
    """
    if sources <= 1 or raw_confidence <= 0 or raw_confidence >= 1:
        return raw_confidence

    # Convert to log-odds (logit)
    log_odds = math.log(raw_confidence / (1 - raw_confidence))

    # Add boost per additional source
    boosted_log_odds = log_odds + (sources - 1) * boost_per_source

    # Convert back to probability (sigmoid)
    return 1 / (1 + math.exp(-boosted_log_odds))


class TagQuality(str, Enum):
    """Tag quality levels determining which resolution(s) to use.

    Each plugin can have its own quality setting. Higher quality
    uses more resolutions and merges results for maximum tag coverage.

    Attributes:
        QUICK: 1080px preview only. Fast, ~87% tag coverage.
        STANDARD: 480px + 2048px merged. Balanced, ~92% coverage.
        MAX: All resolutions (480 + 1080 + 2048 + 4096 + original). 100% coverage.

    Resolution mapping (0 = original image):
        QUICK    -> [1080]
        STANDARD -> [480, 2048]
        MAX      -> [480, 1080, 2048, 4096, 0]

    Example:
        >>> quality = TagQuality.QUICK
        >>> quality.resolutions
        [1080]
    """

    QUICK = "quick"
    STANDARD = "standard"
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
        metadata: Optional plugin-specific metadata (e.g., discovery mode info)

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
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary matching output contract."""
        result = {
            "tags": [t.to_dict() for t in self.tags],
            "model": self.model,
            "version": self.version,
            "inference_time_ms": self.inference_time_ms,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MergedTag:
    """A tag with metadata from multi-resolution merging.

    Used when multi-resolution quality modes merge tags from different resolutions.
    Confidence is boosted using log-odds when a tag is found at multiple resolutions.

    Attributes:
        label: The tag text
        raw_confidence: Highest raw confidence score across resolutions (or None)
        boosted_confidence: Confidence after multi-resolution boost (or None)
        sources: Number of resolutions that found this tag
        max_sources: Total number of resolutions used (for display as "3/5")
        min_resolution: Smallest resolution where tag was found

    Example:
        >>> tag = MergedTag("sunset", raw_confidence=0.85, sources=5, max_sources=5)
        >>> tag.confidence  # Returns boosted if available
        0.92
    """

    label: str
    raw_confidence: float | None = None
    boosted_confidence: float | None = None
    sources: int = 1
    max_sources: int = 1
    min_resolution: int | None = None

    @property
    def confidence(self) -> float | None:
        """Return boosted confidence if available, else raw."""
        return self.boosted_confidence if self.boosted_confidence is not None else self.raw_confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"label": self.label}
        if self.boosted_confidence is not None:
            result["confidence"] = round(self.boosted_confidence, 4)
            result["raw_confidence"] = round(self.raw_confidence, 4)
        elif self.raw_confidence is not None:
            result["confidence"] = round(self.raw_confidence, 4)
        if self.sources > 1 or self.max_sources > 1:
            result["sources"] = self.sources
            result["max_sources"] = self.max_sources
        if self.min_resolution is not None:
            result["min_resolution"] = self.min_resolution
        return result


def merge_tags(
    all_tags: list[Tag],
    resolutions_used: list[int] | None = None,
    boost_per_source: float = 0.15,
) -> list[MergedTag]:
    """Merge tags from multiple resolutions with boosted confidence.

    Tags found at multiple resolutions get their confidence boosted using
    log-odds space calculation. This reflects the additional evidence that
    scale-invariant detection provides.

    Args:
        all_tags: List of Tag objects from all resolutions
        resolutions_used: List of resolutions used (for max_sources calculation)
        boost_per_source: Log-odds boost per additional source (default 0.15)

    Returns:
        List of MergedTag objects, sorted by boosted confidence (desc)

    Example:
        >>> tags = [Tag("cat", 0.9), Tag("cat", 0.8), Tag("dog", 0.7)]
        >>> merged = merge_tags(tags, resolutions_used=[480, 1080])
        >>> merged[0].label
        'cat'
        >>> merged[0].sources
        2
        >>> merged[0].boosted_confidence > merged[0].raw_confidence
        True
    """
    merged: dict[str, MergedTag] = {}
    max_sources = len(resolutions_used) if resolutions_used else 1

    for tag in all_tags:
        label = tag.label.lower().strip()
        if not label:
            continue

        if label not in merged:
            merged[label] = MergedTag(
                label=label,
                raw_confidence=tag.confidence,
                sources=1,
                max_sources=max_sources,
                min_resolution=None,
            )
        else:
            merged[label].sources += 1
            # Keep highest raw confidence
            if tag.confidence is not None:
                if merged[label].raw_confidence is None:
                    merged[label].raw_confidence = tag.confidence
                else:
                    merged[label].raw_confidence = max(
                        merged[label].raw_confidence, tag.confidence
                    )

    # Calculate boosted confidence for all tags
    for tag in merged.values():
        if tag.raw_confidence is not None:
            tag.boosted_confidence = boost_confidence(
                tag.raw_confidence,
                tag.sources,
                boost_per_source,
            )

    # Sort by boosted confidence (desc), then sources (desc), then label
    result = sorted(
        merged.values(),
        key=lambda t: (-(t.boosted_confidence or t.raw_confidence or 0), -t.sources, t.label),
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
        provides_confidence: Whether this plugin returns confidence scores
        recommended_threshold: Default confidence threshold for this plugin.
            SigLIP uses sigmoid (0.01-0.05), others use 0.0.
    """

    name: str
    version: str
    description: str
    hardware_reqs: dict[str, Any] = field(default_factory=dict)
    provides_confidence: bool = False
    recommended_threshold: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HardwareProfile:
    """Detected hardware capabilities.

    Cached to ~/.visual-buffet/hardware.json after first detection.

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
