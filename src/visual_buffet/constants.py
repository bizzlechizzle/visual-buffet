"""Application-wide constants.

Centralizes magic numbers and strings to avoid hardcoding throughout the codebase.
"""

from enum import Enum
from typing import Final

# =============================================================================
# VERSION AND METADATA
# =============================================================================

APP_NAME: Final[str] = "visual-buffet"

# =============================================================================
# CLI DEFAULTS
# =============================================================================

DEFAULT_THRESHOLD: Final[float] = 0.5
MIN_THRESHOLD: Final[float] = 0.0
MAX_THRESHOLD: Final[float] = 1.0

DEFAULT_GUI_HOST: Final[str] = "127.0.0.1"
DEFAULT_GUI_PORT: Final[int] = 8420

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

# Image sizes for ML inference
class ImageSize(str, Enum):
    """Image size presets for ML processing."""

    LITTLE = "little"    # 480px - fastest
    SMALL = "small"      # 1080px - balanced
    LARGE = "large"      # 2048px - high detail
    HUGE = "huge"        # 4096px - maximum detail
    ORIGINAL = "original"  # native resolution

# Resolution mappings (max dimension in pixels)
IMAGE_SIZE_RESOLUTIONS: Final[dict[str, int | None]] = {
    "little": 480,
    "small": 1080,
    "large": 2048,
    "huge": 4096,
    "original": None,  # No resize
}

# JPEG quality for conversions
JPEG_QUALITY_HIGH: Final[int] = 95
JPEG_QUALITY_MEDIUM: Final[int] = 85
JPEG_QUALITY_LOW: Final[int] = 70

# =============================================================================
# GUI / SERVER
# =============================================================================

# Thumbnail dimensions
THUMBNAIL_SIZE: Final[int] = 200
PREVIEW_SIZE: Final[int] = 1920

# Session limits
MAX_SESSIONS: Final[int] = 100
MAX_FILE_SIZE_MB: Final[int] = 200  # 200 MB per file
MAX_TOTAL_CACHE_MB: Final[int] = 2048  # 2 GB total cache

# Batch processing
BATCH_TIMEOUT_MS: Final[int] = 5000  # 5 seconds

# =============================================================================
# HARDWARE
# =============================================================================

# Minimum requirements
MIN_RAM_GB: Final[int] = 4
RECOMMENDED_RAM_GB: Final[int] = 8
MIN_VRAM_GB: Final[int] = 4
RECOMMENDED_VRAM_GB: Final[int] = 8

# Batch size defaults by VRAM
BATCH_SIZE_LOW_VRAM: Final[int] = 1   # < 4GB VRAM
BATCH_SIZE_MED_VRAM: Final[int] = 2   # 4-8GB VRAM
BATCH_SIZE_HIGH_VRAM: Final[int] = 4  # > 8GB VRAM
BATCH_SIZE_CPU_ONLY: Final[int] = 1   # No GPU

# =============================================================================
# FILE HANDLING
# =============================================================================

# Supported image formats
STANDARD_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif",
})

HEIC_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".heic", ".heif",
})

RAW_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".arw",   # Sony
    ".cr2", ".cr3",  # Canon
    ".nef",   # Nikon
    ".dng",   # Adobe
    ".orf",   # Olympus
    ".rw2",   # Panasonic
    ".raf",   # Fujifilm
    ".pef",   # Pentax
    ".srw",   # Samsung
})

ALL_SUPPORTED_EXTENSIONS: Final[frozenset[str]] = (
    STANDARD_IMAGE_EXTENSIONS | HEIC_EXTENSIONS | RAW_EXTENSIONS
)

# =============================================================================
# PLUGIN SYSTEM
# =============================================================================

PLUGIN_TOML_FILENAME: Final[str] = "plugin.toml"
PLUGIN_MODELS_DIR: Final[str] = "models"

# Default plugin thresholds (per plugin type)
DEFAULT_TAGGING_THRESHOLD: Final[float] = 0.5
DEFAULT_OCR_THRESHOLD: Final[float] = 0.3
DEFAULT_DETECTION_THRESHOLD: Final[float] = 0.5

# Tag limits
MAX_TAGS_PER_PLUGIN: Final[int] = 100
MAX_TAG_LABEL_LENGTH: Final[int] = 256

# =============================================================================
# XMP METADATA
# =============================================================================

# XMP namespace for visual-buffet
XMP_NAMESPACE_URI: Final[str] = "http://visual-buffet.dev/xmp/1.0/"
XMP_NAMESPACE_PREFIX: Final[str] = "vbuffet"

# Schema version
XMP_SCHEMA_VERSION: Final[int] = 1

# =============================================================================
# OUTPUT FORMATS
# =============================================================================

class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    # Future: CSV, XML, etc.

# =============================================================================
# ERROR CODES
# =============================================================================

class ExitCode(int, Enum):
    """CLI exit codes."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    FILE_NOT_FOUND = 2
    INVALID_INPUT = 3
    PLUGIN_ERROR = 4
    NO_PLUGINS = 5
    KEYBOARD_INTERRUPT = 130
