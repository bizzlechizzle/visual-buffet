"""Data schemas for the plugin system.

These dataclasses define the structure of data exchanged between
the core engine and plugins. All plugins must use these schemas.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ImageSize(str, Enum):
    """Image size options for tagging.

    Controls the resolution used for ML inference:
    - LITTLE: ~480px (fastest, good for batch processing)
    - SMALL: ~1080px (balanced speed/quality)
    - LARGE: ~2048px (high detail)
    - HUGE: ~4096px (maximum detail)
    - ORIGINAL: native resolution (default, no resize)

    Example:
        >>> size = ImageSize.SMALL
        >>> size.resolution
        1080
    """

    LITTLE = "little"
    SMALL = "small"
    LARGE = "large"
    HUGE = "huge"
    ORIGINAL = "original"

    @property
    def resolution(self) -> int:
        """Get resolution in pixels (0 = original, no resize)."""
        return SIZE_RESOLUTIONS[self]


# Resolution mapping for each size level
# Note: 0 = use original image (no resize)
SIZE_RESOLUTIONS: dict[ImageSize, int] = {
    ImageSize.LITTLE: 480,
    ImageSize.SMALL: 1080,
    ImageSize.LARGE: 2048,
    ImageSize.HUGE: 4096,
    ImageSize.ORIGINAL: 0,
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
