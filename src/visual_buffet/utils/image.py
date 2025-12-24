"""Image loading and validation utilities.

Handles:
- Validating image formats
- Loading images with Pillow
- Loading RAW images with rawpy
- Loading HEIC/HEIF images with pillow-heif
- Expanding globs/folders to file lists
"""

from pathlib import Path

import rawpy
from PIL import Image

# Register HEIC/HEIF support with Pillow
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC files won't be readable

from ..exceptions import ImageError

# RAW image extensions (camera-specific formats)
RAW_EXTENSIONS = {
    ".arw",  # Sony
    ".cr2",  # Canon
    ".cr3",  # Canon (newer)
    ".nef",  # Nikon
    ".dng",  # Adobe Digital Negative
    ".orf",  # Olympus
    ".rw2",  # Panasonic
    ".raf",  # Fujifilm
    ".pef",  # Pentax
    ".srw",  # Samsung
}

# Standard image extensions (Pillow-native)
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}

# All supported extensions
SUPPORTED_EXTENSIONS = STANDARD_EXTENSIONS | RAW_EXTENSIONS


def is_supported_image(path: Path) -> bool:
    """Check if file is a supported image format.

    Args:
        path: Path to check

    Returns:
        True if supported image format
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_raw_image(path: Path) -> bool:
    """Check if file is a RAW image format.

    Args:
        path: Path to check

    Returns:
        True if RAW format requiring rawpy
    """
    return path.suffix.lower() in RAW_EXTENSIONS


def validate_image(path: Path) -> None:
    """Validate that an image file exists and is readable.

    Args:
        path: Path to image file

    Raises:
        ImageError: If file doesn't exist, isn't an image, or can't be read
    """
    if not path.exists():
        raise ImageError(f"Image not found: {path}")

    if not path.is_file():
        raise ImageError(f"Not a file: {path}")

    if not is_supported_image(path):
        raise ImageError(
            f"Unsupported format: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Try to open to verify it's a valid image
    try:
        if is_raw_image(path):
            # Validate RAW files with rawpy
            with rawpy.imread(str(path)) as raw:
                # Just opening validates the file
                pass
        else:
            # Validate standard images with Pillow
            with Image.open(path) as img:
                img.verify()
    except Exception as e:
        raise ImageError(f"Cannot read image {path}: {e}")


def load_image(path: Path) -> Image.Image:
    """Load an image file.

    Args:
        path: Path to image file

    Returns:
        PIL Image object

    Raises:
        ImageError: If image cannot be loaded
    """
    validate_image(path)

    try:
        if is_raw_image(path):
            # Load RAW files with rawpy, convert to PIL Image
            with rawpy.imread(str(path)) as raw:
                # Postprocess RAW to RGB array
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=8,
                )
            img = Image.fromarray(rgb)
        else:
            img = Image.open(path)

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise ImageError(f"Failed to load image {path}: {e}")


def expand_paths(paths: list[str], recursive: bool = False) -> list[Path]:
    """Expand paths to list of image files.

    Handles:
    - Single files
    - Directories (lists contents)
    - Glob patterns (*.jpg)

    Args:
        paths: List of path strings
        recursive: If True, search directories recursively

    Returns:
        List of Path objects to image files
    """
    result = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_file():
            if is_supported_image(path):
                result.append(path)
        elif path.is_dir():
            if recursive:
                for ext in SUPPORTED_EXTENSIONS:
                    result.extend(path.rglob(f"*{ext}"))
                    # Also match uppercase extensions
                    result.extend(path.rglob(f"*{ext.upper()}"))
            else:
                for ext in SUPPORTED_EXTENSIONS:
                    result.extend(path.glob(f"*{ext}"))
                    result.extend(path.glob(f"*{ext.upper()}"))
        elif "*" in path_str or "?" in path_str:
            # Glob pattern
            parent = path.parent if path.parent.exists() else Path(".")
            pattern = path.name
            result.extend(p for p in parent.glob(pattern) if is_supported_image(p))

    # Remove duplicates and sort
    return sorted(set(result))
