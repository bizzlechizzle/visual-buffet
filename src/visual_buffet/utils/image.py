"""Image loading and validation utilities.

Handles:
- Validating image formats
- Loading images with Pillow
- Loading RAW images with rawpy
- Loading HEIC/HEIF images with pillow-heif
- Expanding globs/folders to file lists
- Generating thumbnails at standard sizes
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
from ..plugins.schemas import THUMBNAIL_SIZES

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

# Default thumbnail format and quality
THUMBNAIL_FORMAT = "webp"
THUMBNAIL_QUALITY = 80


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


def get_resized_dimensions(width: int, height: int, max_long_side: int) -> tuple[int, int]:
    """Calculate new dimensions maintaining aspect ratio.

    Args:
        width: Original width
        height: Original height
        max_long_side: Maximum size for the longest side

    Returns:
        Tuple of (new_width, new_height)

    Example:
        >>> get_resized_dimensions(6000, 4000, 1080)
        (1080, 720)
    """
    if width >= height:
        if width <= max_long_side:
            return width, height
        ratio = max_long_side / width
    else:
        if height <= max_long_side:
            return width, height
        ratio = max_long_side / height
    return int(width * ratio), int(height * ratio)


def generate_thumbnail(
    source_path: Path,
    output_path: Path,
    max_long_side: int,
    quality: int = THUMBNAIL_QUALITY,
) -> Path:
    """Generate a thumbnail at the specified size.

    Args:
        source_path: Path to source image
        output_path: Path for output thumbnail
        max_long_side: Maximum size for longest side in pixels
        quality: JPEG/WebP quality (1-100)

    Returns:
        Path to generated thumbnail

    Raises:
        ImageError: If thumbnail generation fails
    """
    try:
        # Load image (RAW or standard)
        if is_raw_image(source_path):
            with rawpy.imread(str(source_path)) as raw:
                # Use half_size=True for faster thumbnail generation from RAW
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,
                    no_auto_bright=False,
                    output_bps=8,
                )
            img = Image.fromarray(rgb)
        else:
            img = Image.open(source_path)

        # Convert to RGB if needed
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Calculate new size
        new_width, new_height = get_resized_dimensions(
            img.width, img.height, max_long_side
        )

        # Skip if would upscale
        if new_width >= img.width and new_height >= img.height:
            # Just copy/convert the original
            resized = img
        else:
            resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        ext = output_path.suffix.lower()
        if ext == ".webp":
            resized.save(output_path, "WEBP", quality=quality, method=4)
        elif ext in (".jpg", ".jpeg"):
            resized.save(output_path, "JPEG", quality=quality, optimize=True)
        else:
            resized.save(output_path, quality=quality)

        # Close non-context-managed image
        if not is_raw_image(source_path):
            img.close()

        return output_path

    except Exception as e:
        raise ImageError(f"Failed to generate thumbnail: {e}")


def generate_all_thumbnails(
    source_path: Path,
    output_dir: Path,
    image_id: str | None = None,
    format: str = THUMBNAIL_FORMAT,
) -> dict[str, Path]:
    """Generate all standard thumbnails for an image.

    Creates thumbnails at all sizes defined in THUMBNAIL_SIZES:
    - grid (480px): For grid view
    - preview (1080px): For lightbox preview
    - zoom (2048px): For lightbox zoom

    Args:
        source_path: Path to source image
        output_dir: Directory for output thumbnails
        image_id: Optional ID for filename (defaults to source stem)
        format: Output format (webp, jpg)

    Returns:
        Dict mapping size name to thumbnail path:
        {"grid": Path, "preview": Path, "zoom": Path}

    Example:
        >>> thumbs = generate_all_thumbnails(Path("photo.jpg"), Path("cache/"))
        >>> thumbs["grid"]
        Path("cache/photo_480.webp")
    """
    if image_id is None:
        image_id = source_path.stem

    ext = f".{format}"
    thumbnails = {}

    for name, size in THUMBNAIL_SIZES.items():
        output_path = output_dir / f"{image_id}_{size}{ext}"

        if not output_path.exists():
            generate_thumbnail(source_path, output_path, size)

        thumbnails[name] = output_path

    return thumbnails


def get_thumbnail_path(
    output_dir: Path,
    image_id: str,
    resolution: int,
    format: str = THUMBNAIL_FORMAT,
) -> Path:
    """Get expected path for a thumbnail.

    Args:
        output_dir: Thumbnail directory
        image_id: Image identifier
        resolution: Thumbnail resolution (480, 1080, 2048)
        format: File format

    Returns:
        Expected thumbnail path
    """
    return output_dir / f"{image_id}_{resolution}.{format}"


def get_thumbnail_for_quality(
    output_dir: Path,
    image_id: str,
    resolution: int,
    source_path: Path | None = None,
    format: str = THUMBNAIL_FORMAT,
) -> Path:
    """Get or generate thumbnail for a specific resolution.

    If thumbnail doesn't exist and source_path is provided,
    generates it on demand.

    Args:
        output_dir: Thumbnail directory
        image_id: Image identifier
        resolution: Desired resolution
        source_path: Optional source for generation
        format: File format

    Returns:
        Path to thumbnail

    Raises:
        ImageError: If thumbnail doesn't exist and can't be generated
    """
    thumb_path = get_thumbnail_path(output_dir, image_id, resolution, format)

    if thumb_path.exists():
        return thumb_path

    if source_path is None:
        raise ImageError(f"Thumbnail not found and no source provided: {thumb_path}")

    return generate_thumbnail(source_path, thumb_path, resolution)
