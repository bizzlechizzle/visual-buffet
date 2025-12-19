"""Image loading and validation utilities.

Handles:
- Validating image formats
- Loading images with Pillow
- Expanding globs/folders to file lists
"""

from pathlib import Path

from PIL import Image

from ..exceptions import ImageError

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def is_supported_image(path: Path) -> bool:
    """Check if file is a supported image format.

    Args:
        path: Path to check

    Returns:
        True if supported image format
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


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
