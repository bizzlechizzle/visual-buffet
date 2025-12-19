#!/usr/bin/env python3
"""
Generate test image variants at different resolutions and formats.
Requires: Pillow, pillow-avif-plugin

Usage: python scripts/generate_variants.py images/ test-results/variants/
"""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    import pillow_avif  # noqa: F401 - registers AVIF support
except ImportError:
    print("Required: pip install Pillow pillow-avif-plugin")
    sys.exit(1)


# Test matrix configuration
RESOLUTIONS = [480, 1080, 2048, 4096]  # Max long side in pixels
FORMATS = {
    "jpg": {"ext": ".jpg", "save_args": lambda q: {"quality": q, "optimize": True}},
    "webp": {"ext": ".webp", "save_args": lambda q: {"quality": q, "method": 4}},
    "avif": {"ext": ".avif", "save_args": lambda q: {"quality": q, "speed": 6}},
}
QUALITY_LEVELS = [95, 80, 60]
QUALITY_TEST_RESOLUTION = 2048  # Only test all quality levels at this resolution
DEFAULT_QUALITY = 80  # For other resolutions


def get_resized_dimensions(width: int, height: int, max_long_side: int) -> tuple[int, int]:
    """Calculate new dimensions maintaining aspect ratio."""
    if width >= height:
        if width <= max_long_side:
            return width, height
        ratio = max_long_side / width
    else:
        if height <= max_long_side:
            return width, height
        ratio = max_long_side / height
    return int(width * ratio), int(height * ratio)


def generate_variants(source_dir: Path, output_dir: Path) -> dict:
    """Generate all test variants for images in source directory."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Find source images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
    source_images = [
        f for f in source_dir.iterdir()
        if f.suffix.lower() in image_extensions and not f.name.startswith(".")
    ]

    if not source_images:
        print(f"No images found in {source_dir}")
        return {}

    print(f"Found {len(source_images)} source images")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "config": {
            "resolutions": RESOLUTIONS,
            "formats": list(FORMATS.keys()),
            "quality_levels": QUALITY_LEVELS,
            "quality_test_resolution": QUALITY_TEST_RESOLUTION,
            "default_quality": DEFAULT_QUALITY,
        },
        "images": {},
    }

    for source_path in sorted(source_images):
        image_name = source_path.stem
        print(f"\nProcessing: {source_path.name}")

        # Create output directory for this image
        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Load source image
        try:
            img = Image.open(source_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        except Exception as e:
            print(f"  Error loading {source_path}: {e}")
            continue

        orig_width, orig_height = img.size
        variants = []

        # Copy original
        orig_dest = image_output_dir / f"{image_name}-original{source_path.suffix.lower()}"
        if not orig_dest.exists():
            img.save(orig_dest, quality=95)
        variants.append({
            "path": str(orig_dest.relative_to(output_dir)),
            "resolution": "original",
            "format": source_path.suffix.lower().strip("."),
            "quality": "original",
            "width": orig_width,
            "height": orig_height,
            "size_kb": round(orig_dest.stat().st_size / 1024, 1),
        })
        print(f"  original: {orig_width}x{orig_height}")

        # Generate resolution variants
        for max_res in RESOLUTIONS:
            new_width, new_height = get_resized_dimensions(orig_width, orig_height, max_res)

            # Skip if would upscale
            if new_width >= orig_width and new_height >= orig_height and max_res >= max(orig_width, orig_height):
                print(f"  {max_res}px: skipped (would upscale)")
                continue

            # Resize image
            resized = img.resize((new_width, new_height), Image.LANCZOS)

            # Determine quality levels to test
            if max_res == QUALITY_TEST_RESOLUTION:
                qualities = QUALITY_LEVELS
            else:
                qualities = [DEFAULT_QUALITY]

            for fmt_name, fmt_config in FORMATS.items():
                for quality in qualities:
                    # Build filename
                    if len(qualities) > 1:
                        filename = f"{image_name}-{max_res}px-{fmt_name}-q{quality}{fmt_config['ext']}"
                    else:
                        filename = f"{image_name}-{max_res}px-{fmt_name}{fmt_config['ext']}"

                    dest_path = image_output_dir / filename

                    if not dest_path.exists():
                        save_args = fmt_config["save_args"](quality)
                        try:
                            resized.save(dest_path, **save_args)
                        except Exception as e:
                            print(f"  Error saving {filename}: {e}")
                            continue

                    variants.append({
                        "path": str(dest_path.relative_to(output_dir)),
                        "resolution": max_res,
                        "format": fmt_name,
                        "quality": quality,
                        "width": new_width,
                        "height": new_height,
                        "size_kb": round(dest_path.stat().st_size / 1024, 1),
                    })

            print(f"  {max_res}px: {new_width}x{new_height} ({len(qualities)} quality levels × {len(FORMATS)} formats)")

        manifest["images"][image_name] = {
            "source": str(source_path),
            "original_size": [orig_width, orig_height],
            "variants": variants,
        }

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Summary
    total_variants = sum(len(img["variants"]) for img in manifest["images"].values())
    print(f"\nGenerated {total_variants} variants for {len(manifest['images'])} images")

    return manifest


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_variants.py <source_dir> <output_dir>")
        print("Example: python generate_variants.py images/ test-results/variants/")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(1)

    generate_variants(source_dir, output_dir)


if __name__ == "__main__":
    main()
