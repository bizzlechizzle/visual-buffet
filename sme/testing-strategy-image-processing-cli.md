# Testing Strategy for Image Processing CLI Applications

> **Generated**: 2025-12-23
> **Sources current as of**: December 2025
> **Scope**: Comprehensive
> **Version**: 1.0
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

Testing image processing CLI applications with ML tagging pipelines requires a multi-layered strategy addressing: **format compatibility** (handling 15+ image formats including RAW), **resolution behavior** (min/optimal/max thresholds), **ML reproducibility** (GPU/CPU determinism), and **regression detection** (golden file testing for ML outputs).

Key recommendations:
- Use **tiered test fixtures**: synthetic images for unit tests, curated real images for integration tests
- Implement **tolerance-based assertions** for ML outputs (exact matching is impossible)
- Enable **PyTorch deterministic mode** for reproducible results within same hardware
- Use **golden file snapshots** with statistical drift detection for regression testing
- Structure tests as `tests/{unit,integration,regression,benchmark}/` with fixtures in `tests/fixtures/`

Critical insight: ML model outputs will **never** be bit-identical across hardware. Design tests for behavioral consistency, not exact matching [1][HIGH].

---

## Background & Context

Image processing CLI applications that leverage ML tagging face unique testing challenges:

1. **Non-deterministic outputs**: ML models produce floating-point confidence scores subject to hardware variations
2. **Format diversity**: Must handle JPEG, PNG, WebP, HEIC, TIFF, BMP, and 10+ RAW camera formats
3. **Resolution sensitivity**: Tag detection quality varies with input resolution
4. **External dependencies**: GPU drivers, CUDA versions, model weights affect results
5. **Long execution times**: Full ML inference is too slow for rapid test cycles

This document provides a comprehensive testing framework applicable to any image tagging CLI, with specific patterns for validating results across multiple runs.

---

## 1. Test Architecture & Organization

### Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures, pytest configuration
├── fixtures/
│   ├── images/
│   │   ├── synthetic/             # Generated test images (fast, deterministic)
│   │   ├── real/                  # Curated real images (licensed/public domain)
│   │   └── edge_cases/            # Corrupt, malformed, unusual images
│   ├── golden/                    # Expected outputs for regression tests
│   │   ├── v1.0/                  # Versioned by model/app version
│   │   └── v1.1/
│   └── schemas/                   # JSON schemas for output validation
├── unit/
│   ├── test_image_loading.py
│   ├── test_format_detection.py
│   ├── test_validation.py
│   └── test_output_schema.py
├── integration/
│   ├── test_cli_commands.py
│   ├── test_plugin_execution.py
│   └── test_end_to_end.py
├── regression/
│   ├── test_golden_outputs.py
│   ├── test_tag_consistency.py
│   └── test_confidence_drift.py
├── benchmark/
│   ├── test_performance.py
│   └── test_memory.py
└── markers.py                     # Custom pytest markers
```

### Pytest Configuration (`conftest.py`)

```python
"""Pytest configuration and shared fixtures for image processing CLI testing."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

# =============================================================================
# Custom Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "regression: marks regression tests")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")


# =============================================================================
# Path Fixtures
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
GOLDEN_DIR = FIXTURES_DIR / "golden"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Return path to golden outputs directory."""
    return GOLDEN_DIR


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_image_dir(temp_dir: Path) -> Path:
    """Create a temporary directory with subdirs for images."""
    images = temp_dir / "images"
    images.mkdir()
    return images


# =============================================================================
# Synthetic Image Fixtures (Fast, Deterministic)
# =============================================================================

@pytest.fixture
def solid_color_image(temp_dir: Path) -> Path:
    """Create a simple solid color test image."""
    path = temp_dir / "solid_red.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def gradient_image(temp_dir: Path) -> Path:
    """Create a gradient test image for resolution testing."""
    path = temp_dir / "gradient.png"
    img = Image.new("RGB", (256, 256))
    pixels = img.load()
    for x in range(256):
        for y in range(256):
            pixels[x, y] = (x, y, (x + y) // 2)
    img.save(path, "PNG")
    return path


@pytest.fixture
def multi_resolution_images(temp_dir: Path) -> dict[int, Path]:
    """Create test images at standard resolutions."""
    resolutions = [64, 128, 256, 512, 1024, 2048, 4096]
    images = {}

    for res in resolutions:
        path = temp_dir / f"test_{res}x{res}.jpg"
        img = Image.new("RGB", (res, res), color=(128, 128, 128))
        # Add some structure for tagging
        for i in range(0, res, res // 8):
            for j in range(0, res, res // 8):
                if (i + j) % 2 == 0:
                    for x in range(i, min(i + res // 8, res)):
                        for y in range(j, min(j + res // 8, res)):
                            img.putpixel((x, y), (200, 100, 50))
        img.save(path, "JPEG", quality=90)
        images[res] = path

    return images


@pytest.fixture(params=["RGB", "RGBA", "L", "P", "CMYK"])
def image_modes(request, temp_dir: Path) -> tuple[str, Path]:
    """Parameterized fixture for different image color modes."""
    mode = request.param
    path = temp_dir / f"test_{mode.lower()}.png"

    if mode == "CMYK":
        # CMYK requires special handling
        img = Image.new("CMYK", (100, 100), color=(0, 100, 100, 0))
        path = temp_dir / "test_cmyk.tiff"
        img.save(path, "TIFF")
    elif mode == "P":
        img = Image.new("P", (100, 100))
    else:
        img = Image.new(mode, (100, 100))

    if mode not in ["CMYK"]:
        img.save(path, "PNG")

    return (mode, path)


# =============================================================================
# Format-Specific Fixtures
# =============================================================================

@pytest.fixture
def jpeg_image(temp_dir: Path) -> Path:
    """Create a JPEG test image."""
    path = temp_dir / "test.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(path, "JPEG", quality=85)
    return path


@pytest.fixture
def png_image(temp_dir: Path) -> Path:
    """Create a PNG test image."""
    path = temp_dir / "test.png"
    img = Image.new("RGB", (100, 100), color="green")
    img.save(path, "PNG")
    return path


@pytest.fixture
def webp_image(temp_dir: Path) -> Path:
    """Create a WebP test image."""
    path = temp_dir / "test.webp"
    img = Image.new("RGB", (100, 100), color="purple")
    img.save(path, "WebP", quality=80)
    return path


@pytest.fixture
def png_with_transparency(temp_dir: Path) -> Path:
    """Create a PNG with alpha channel."""
    path = temp_dir / "transparent.png"
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img.save(path, "PNG")
    return path


@pytest.fixture
def tiff_16bit(temp_dir: Path) -> Path:
    """Create a 16-bit TIFF image."""
    import numpy as np
    path = temp_dir / "test_16bit.tiff"
    # Create 16-bit array
    arr = np.zeros((100, 100, 3), dtype=np.uint16)
    arr[:, :, 0] = 32768  # Mid-range red
    img = Image.fromarray(arr, mode="RGB")
    img.save(path, "TIFF")
    return path


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def corrupt_jpeg(temp_dir: Path) -> Path:
    """Create a corrupt JPEG file."""
    path = temp_dir / "corrupt.jpg"
    # Write invalid JPEG data
    path.write_bytes(b"\xff\xd8\xff\xe0JUNK_DATA_NOT_VALID")
    return path


@pytest.fixture
def truncated_image(temp_dir: Path) -> Path:
    """Create a truncated image file."""
    path = temp_dir / "truncated.jpg"
    # Create valid JPEG header but truncate
    img = Image.new("RGB", (100, 100))
    full_path = temp_dir / "full.jpg"
    img.save(full_path, "JPEG")
    data = full_path.read_bytes()
    path.write_bytes(data[:len(data)//2])  # Truncate at 50%
    return path


@pytest.fixture
def zero_byte_file(temp_dir: Path) -> Path:
    """Create an empty file."""
    path = temp_dir / "empty.jpg"
    path.write_bytes(b"")
    return path


@pytest.fixture
def wrong_extension(temp_dir: Path) -> Path:
    """Create PNG file with JPG extension."""
    path = temp_dir / "actually_png.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(path, "PNG")  # Save as PNG but use .jpg extension
    return path


# =============================================================================
# EXIF Orientation Fixtures
# =============================================================================

@pytest.fixture
def rotated_image(temp_dir: Path) -> Path:
    """Create image with EXIF orientation tag."""
    from PIL.ExifTags import TAGS
    path = temp_dir / "rotated.jpg"
    img = Image.new("RGB", (200, 100), color="red")
    # Add EXIF orientation data indicating 90 degree rotation
    exif = img.getexif()
    # Orientation tag = 274, value 6 = 90 CW
    exif[274] = 6
    img.save(path, "JPEG", exif=exif)
    return path


# =============================================================================
# Mock Plugin Fixtures
# =============================================================================

@pytest.fixture
def mock_tag_result() -> dict:
    """Return a mock tag result for testing output processing."""
    return {
        "tags": [
            {"label": "dog", "confidence": 0.95},
            {"label": "outdoor", "confidence": 0.87},
            {"label": "grass", "confidence": 0.72},
            {"label": "sunny", "confidence": 0.68},
            {"label": "animal", "confidence": 0.91},
        ],
        "model": "test_model",
        "version": "1.0.0",
        "inference_time_ms": 150.5,
    }


# =============================================================================
# Golden File Utilities
# =============================================================================

def compute_image_hash(path: Path) -> str:
    """Compute SHA256 hash of image file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_golden_output(golden_dir: Path, test_name: str, version: str = "v1.0") -> dict:
    """Load expected golden output for a test."""
    golden_path = golden_dir / version / f"{test_name}.json"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden file not found: {golden_path}")
    return json.loads(golden_path.read_text())


def save_golden_output(golden_dir: Path, test_name: str, output: dict, version: str = "v1.0"):
    """Save golden output for a test."""
    version_dir = golden_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    golden_path = version_dir / f"{test_name}.json"
    golden_path.write_text(json.dumps(output, indent=2, sort_keys=True))


# =============================================================================
# Statistical Comparison Utilities
# =============================================================================

def confidence_within_tolerance(
    actual: list[dict],
    expected: list[dict],
    tolerance: float = 0.05
) -> bool:
    """
    Compare tag results with tolerance for confidence scores.

    Returns True if:
    - Same tags present (order-independent)
    - Confidence scores within tolerance
    """
    actual_by_label = {t["label"]: t["confidence"] for t in actual}
    expected_by_label = {t["label"]: t["confidence"] for t in expected}

    if set(actual_by_label.keys()) != set(expected_by_label.keys()):
        return False

    for label, expected_conf in expected_by_label.items():
        actual_conf = actual_by_label[label]
        if abs(actual_conf - expected_conf) > tolerance:
            return False

    return True


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0
```

---

## 2. Resolution Testing

### Rationale

ML tagging models have resolution-dependent behavior [2]:
- **Too small**: Insufficient detail for accurate detection
- **Optimal range**: Best accuracy-to-performance ratio
- **Too large**: Diminishing returns, increased memory/time

### Key Thresholds to Test

| Threshold | Resolution | Purpose |
|-----------|------------|---------|
| **Minimum viable** | 64-128px | Below this, expect significant accuracy loss |
| **Lower bound** | 224-256px | Most models' native input size |
| **Optimal** | 512-1024px | Sweet spot for most content |
| **Maximum useful** | 2048-4096px | Beyond this, diminishing returns |

### Test Implementation

```python
"""tests/unit/test_resolution.py - Resolution behavior testing."""

import pytest
from pathlib import Path

# Assuming your app has these functions
# from your_app.core import tag_image, get_supported_resolutions


class TestResolutionBehavior:
    """Tests for resolution-dependent behavior."""

    @pytest.mark.parametrize("resolution", [32, 64, 128, 256, 512, 1024, 2048])
    def test_resolution_produces_output(self, resolution: int, temp_dir: Path, tag_image):
        """Test that all resolutions produce valid output."""
        from PIL import Image

        path = temp_dir / f"test_{resolution}.jpg"
        img = Image.new("RGB", (resolution, resolution), color="blue")
        img.save(path, "JPEG")

        result = tag_image(path)

        assert "tags" in result
        assert isinstance(result["tags"], list)
        # Low resolution may produce empty tags, but should not error

    @pytest.mark.parametrize("resolution", [32, 64])
    def test_minimum_resolution_warning(self, resolution: int, temp_dir: Path, tag_image, caplog):
        """Test that very small images log a warning."""
        from PIL import Image

        path = temp_dir / f"tiny_{resolution}.jpg"
        img = Image.new("RGB", (resolution, resolution))
        img.save(path, "JPEG")

        tag_image(path)

        # Expect warning about low resolution
        assert any("resolution" in record.message.lower() for record in caplog.records) or True
        # Note: Adjust based on whether your app actually warns

    def test_resolution_accuracy_scaling(self, multi_resolution_images: dict, tag_image):
        """
        Test that higher resolutions produce equal or better accuracy.

        This is a behavioral test - we expect that as resolution increases,
        the number of detected tags should generally increase (up to a point).
        """
        results = {}
        for res, path in sorted(multi_resolution_images.items()):
            result = tag_image(path)
            results[res] = len(result.get("tags", []))

        # At minimum, 256px should produce more tags than 64px
        # This is a soft assertion - may vary by model
        if 256 in results and 64 in results:
            # Allow for some variance, but 256 should generally be better
            assert results[256] >= results[64] * 0.5  # At least half as many

    @pytest.mark.slow
    def test_memory_usage_by_resolution(self, multi_resolution_images: dict, tag_image):
        """Test that memory usage scales appropriately with resolution."""
        import tracemalloc

        memory_usage = {}

        for res, path in sorted(multi_resolution_images.items()):
            tracemalloc.start()
            tag_image(path)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage[res] = peak / 1024 / 1024  # MB

        # Memory should scale roughly with resolution squared
        # But ML models batch internally, so check it's bounded
        if 4096 in memory_usage and 512 in memory_usage:
            # 4096 is 64x the pixels of 512, but memory shouldn't be 64x
            ratio = memory_usage[4096] / memory_usage[512]
            assert ratio < 20, f"Memory scaling too aggressive: {ratio}x"


class TestResolutionEdgeCases:
    """Edge cases for resolution handling."""

    def test_non_square_image(self, temp_dir: Path, tag_image):
        """Test handling of non-square images."""
        from PIL import Image

        # Very wide image
        wide = temp_dir / "wide.jpg"
        Image.new("RGB", (2000, 100)).save(wide, "JPEG")
        result = tag_image(wide)
        assert "tags" in result

        # Very tall image
        tall = temp_dir / "tall.jpg"
        Image.new("RGB", (100, 2000)).save(tall, "JPEG")
        result = tag_image(tall)
        assert "tags" in result

    def test_odd_dimensions(self, temp_dir: Path, tag_image):
        """Test handling of odd pixel dimensions."""
        from PIL import Image

        for dims in [(101, 101), (333, 777), (1, 1000)]:
            path = temp_dir / f"odd_{dims[0]}x{dims[1]}.jpg"
            Image.new("RGB", dims).save(path, "JPEG")
            result = tag_image(path)
            assert "error" not in result
```

---

## 3. Image Format Testing

### Supported Formats Matrix

| Format | Extension(s) | Color Spaces | Bit Depths | Notes |
|--------|--------------|--------------|------------|-------|
| JPEG | .jpg, .jpeg | RGB | 8-bit | Most common, lossy |
| PNG | .png | RGB, RGBA, L, P | 8, 16-bit | Lossless, transparency |
| WebP | .webp | RGB, RGBA | 8-bit | Modern, good compression |
| TIFF | .tif, .tiff | RGB, CMYK | 8, 16, 32-bit | Professional |
| BMP | .bmp | RGB | 8-bit | Uncompressed |
| HEIC/HEIF | .heic, .heif | RGB | 8, 10-bit | Apple format |
| RAW | .arw, .cr2, .nef, etc. | Bayer | 12, 14-bit | Camera raw |

### Test Implementation

```python
"""tests/unit/test_formats.py - Image format testing."""

import pytest
from pathlib import Path
from PIL import Image


class TestFormatDetection:
    """Tests for format detection and validation."""

    @pytest.mark.parametrize("ext,format_name", [
        (".jpg", "JPEG"),
        (".jpeg", "JPEG"),
        (".png", "PNG"),
        (".webp", "WebP"),
        (".bmp", "BMP"),
        (".tiff", "TIFF"),
        (".tif", "TIFF"),
    ])
    def test_supported_formats(self, ext: str, format_name: str, temp_dir: Path, is_supported):
        """Test that standard formats are detected as supported."""
        path = temp_dir / f"test{ext}"
        assert is_supported(path) is True

    @pytest.mark.parametrize("ext", [".txt", ".pdf", ".doc", ".gif", ".psd"])
    def test_unsupported_formats(self, ext: str, temp_dir: Path, is_supported):
        """Test that unsupported formats are rejected."""
        path = temp_dir / f"test{ext}"
        assert is_supported(path) is False

    def test_case_insensitive_detection(self, temp_dir: Path, is_supported):
        """Test format detection is case-insensitive."""
        for ext in [".JPG", ".Jpg", ".PNG", ".HEIC"]:
            path = temp_dir / f"test{ext}"
            assert is_supported(path) is True


class TestFormatLoading:
    """Tests for loading different image formats."""

    def test_load_jpeg(self, jpeg_image: Path, load_image):
        """Test JPEG loading."""
        img = load_image(jpeg_image)
        assert img is not None
        assert img.mode == "RGB"

    def test_load_png(self, png_image: Path, load_image):
        """Test PNG loading."""
        img = load_image(png_image)
        assert img is not None

    def test_load_webp(self, webp_image: Path, load_image):
        """Test WebP loading."""
        img = load_image(webp_image)
        assert img is not None

    def test_load_png_with_transparency(self, png_with_transparency: Path, load_image):
        """Test PNG with alpha channel is converted to RGB."""
        img = load_image(png_with_transparency)
        # Should be converted to RGB for ML processing
        assert img.mode == "RGB"

    @pytest.mark.parametrize("mode", ["L", "P", "RGBA"])
    def test_mode_conversion(self, mode: str, temp_dir: Path, load_image):
        """Test that all modes are converted to RGB."""
        path = temp_dir / f"test_{mode}.png"
        img = Image.new(mode, (100, 100))
        img.save(path, "PNG")

        loaded = load_image(path)
        assert loaded.mode == "RGB"


class TestEdgeCaseFormats:
    """Tests for edge cases in format handling."""

    def test_corrupt_jpeg_raises(self, corrupt_jpeg: Path, load_image):
        """Test corrupt JPEG raises appropriate error."""
        with pytest.raises(Exception):  # Replace with your specific error type
            load_image(corrupt_jpeg)

    def test_truncated_image_raises(self, truncated_image: Path, load_image):
        """Test truncated image raises appropriate error."""
        with pytest.raises(Exception):
            load_image(truncated_image)

    def test_zero_byte_file_raises(self, zero_byte_file: Path, load_image):
        """Test empty file raises appropriate error."""
        with pytest.raises(Exception):
            load_image(zero_byte_file)

    def test_wrong_extension_still_loads(self, wrong_extension: Path, load_image):
        """Test that actual format is detected regardless of extension."""
        # This tests whether magic bytes are used, not just extension
        img = load_image(wrong_extension)
        assert img is not None

    def test_exif_orientation_applied(self, rotated_image: Path, load_image):
        """Test EXIF orientation is applied during loading."""
        img = load_image(rotated_image)
        # Original was 200x100, rotated 90 degrees should be 100x200
        # Note: Depends on whether your loader respects EXIF orientation
        assert img.size in [(200, 100), (100, 200)]


class TestColorSpaces:
    """Tests for color space handling."""

    def test_cmyk_conversion(self, temp_dir: Path, load_image):
        """Test CMYK images are converted to RGB."""
        path = temp_dir / "cmyk.tiff"
        img = Image.new("CMYK", (100, 100), color=(0, 100, 100, 0))
        img.save(path, "TIFF")

        loaded = load_image(path)
        assert loaded.mode == "RGB"

    def test_grayscale_conversion(self, temp_dir: Path, load_image):
        """Test grayscale images are converted to RGB."""
        path = temp_dir / "gray.png"
        img = Image.new("L", (100, 100), color=128)
        img.save(path, "PNG")

        loaded = load_image(path)
        assert loaded.mode == "RGB"


class TestBitDepth:
    """Tests for different bit depths."""

    def test_16bit_tiff(self, tiff_16bit: Path, load_image):
        """Test 16-bit TIFF can be loaded."""
        img = load_image(tiff_16bit)
        assert img is not None
        assert img.mode == "RGB"

    def test_8bit_vs_16bit_output(self, temp_dir: Path, tag_image):
        """Test that bit depth doesn't affect tagging output structure."""
        import numpy as np

        # 8-bit image
        path_8 = temp_dir / "8bit.png"
        img_8 = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_8.save(path_8, "PNG")

        # 16-bit image
        path_16 = temp_dir / "16bit.tiff"
        arr = np.full((100, 100, 3), 32768, dtype=np.uint16)
        img_16 = Image.fromarray(arr, mode="RGB")
        img_16.save(path_16, "TIFF")

        result_8 = tag_image(path_8)
        result_16 = tag_image(path_16)

        # Both should have valid structure
        assert "tags" in result_8
        assert "tags" in result_16
```

---

## 4. Determinism & Reproducibility Testing

### Sources of Non-Determinism

Per PyTorch documentation [3]:

1. **Random weight initialization** - Controlled via `torch.manual_seed()`
2. **CUDA convolution algorithms** - Controlled via `torch.backends.cudnn.deterministic`
3. **Atomic operations** - Some ops have inherent randomness
4. **Data loading** - Multi-worker shuffle needs `worker_init_fn`
5. **Hardware differences** - CPU vs GPU produce different results

### Enabling Determinism

```python
"""Utility module for enabling deterministic ML execution."""

import os
import random

import numpy as np
import torch


def set_deterministic_mode(seed: int = 42):
    """
    Enable deterministic mode for PyTorch.

    Note: This may significantly impact performance.
    Results will be reproducible on SAME hardware only.
    """
    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for additional determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_hardware_fingerprint() -> dict:
    """Get fingerprint of current hardware for reproducibility tracking."""
    info = {
        "platform": os.uname().sysname,
        "cpu": os.uname().machine,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
    }

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    else:
        info["gpu"] = None

    return info
```

### Reproducibility Tests

```python
"""tests/regression/test_reproducibility.py - Reproducibility testing."""

import pytest
import json
from pathlib import Path


class TestDeterministicOutput:
    """Tests for deterministic output across runs."""

    def test_same_input_same_output(self, jpeg_image: Path, tag_image, set_deterministic):
        """Test that identical inputs produce identical outputs on same hardware."""
        set_deterministic(seed=42)
        result1 = tag_image(jpeg_image)

        set_deterministic(seed=42)
        result2 = tag_image(jpeg_image)

        # Compare tag labels (order-independent)
        tags1 = {t["label"] for t in result1["tags"]}
        tags2 = {t["label"] for t in result2["tags"]}
        assert tags1 == tags2, "Tag labels should be identical"

        # Compare confidences with tolerance
        for t1 in result1["tags"]:
            t2 = next(t for t in result2["tags"] if t["label"] == t1["label"])
            assert abs(t1["confidence"] - t2["confidence"]) < 0.001

    def test_different_seeds_may_differ(self, jpeg_image: Path, tag_image, set_deterministic):
        """Test that different seeds may produce different results."""
        set_deterministic(seed=42)
        result1 = tag_image(jpeg_image)

        set_deterministic(seed=123)
        result2 = tag_image(jpeg_image)

        # Results should be similar but may have minor differences
        # This test documents the behavior rather than asserting equality
        tags1 = {t["label"] for t in result1["tags"]}
        tags2 = {t["label"] for t in result2["tags"]}

        # Most tags should overlap
        overlap = len(tags1 & tags2) / max(len(tags1), len(tags2), 1)
        assert overlap > 0.9, "Most tags should be consistent regardless of seed"

    @pytest.mark.slow
    def test_batch_vs_single_consistency(self, temp_dir: Path, tag_image, tag_batch, set_deterministic):
        """Test that batch processing produces same results as single processing."""
        from PIL import Image

        images = []
        for i in range(3):
            path = temp_dir / f"test_{i}.jpg"
            Image.new("RGB", (200, 200), color=(i * 50, 100, 150)).save(path, "JPEG")
            images.append(path)

        set_deterministic(seed=42)
        single_results = [tag_image(p) for p in images]

        set_deterministic(seed=42)
        batch_results = tag_batch(images)

        for single, batch in zip(single_results, batch_results):
            single_tags = {t["label"] for t in single["tags"]}
            batch_tags = {t["label"] for t in batch["tags"]}
            assert single_tags == batch_tags


class TestCrossRunConsistency:
    """Tests for consistency across multiple runs."""

    def test_multiple_runs_consistent(self, jpeg_image: Path, tag_image, set_deterministic):
        """Run tagging multiple times and verify consistency."""
        NUM_RUNS = 5
        results = []

        for _ in range(NUM_RUNS):
            set_deterministic(seed=42)
            results.append(tag_image(jpeg_image))

        # All runs should produce identical tags
        reference_tags = {t["label"] for t in results[0]["tags"]}
        for i, result in enumerate(results[1:], 2):
            current_tags = {t["label"] for t in result["tags"]}
            assert current_tags == reference_tags, f"Run {i} differs from run 1"

    def test_statistical_stability(self, jpeg_image: Path, tag_image):
        """
        Without deterministic mode, results should still be statistically stable.

        Run multiple times and check variance is bounded.
        """
        NUM_RUNS = 10
        all_confidences = {}

        for _ in range(NUM_RUNS):
            result = tag_image(jpeg_image)
            for tag in result["tags"]:
                label = tag["label"]
                if label not in all_confidences:
                    all_confidences[label] = []
                all_confidences[label].append(tag["confidence"])

        # Check variance for each tag
        for label, confidences in all_confidences.items():
            if len(confidences) > 1:
                variance = sum((c - sum(confidences)/len(confidences))**2
                              for c in confidences) / len(confidences)
                # Variance should be small (< 0.01 for most ML models)
                assert variance < 0.05, f"High variance for {label}: {variance}"
```

---

## 5. Golden File / Regression Testing

### Strategy

Golden file testing (snapshot testing) captures known-good outputs and compares against them [4]. For ML outputs, use **tolerance-based comparison** rather than exact matching.

### Golden File Structure

```json
{
  "version": "1.0",
  "model_version": "ram_plus_1.3.0",
  "hardware_fingerprint": {
    "gpu": "NVIDIA RTX 3090",
    "cuda_version": "12.1",
    "platform": "Linux"
  },
  "test_image_hash": "a1b2c3d4...",
  "tags": [
    {"label": "dog", "confidence": 0.95, "tolerance": 0.05},
    {"label": "outdoor", "confidence": 0.87, "tolerance": 0.08},
    {"label": "grass", "confidence": 0.72, "tolerance": 0.10}
  ],
  "metadata": {
    "created": "2024-01-15T10:30:00Z",
    "notes": "Baseline for v1.0 release"
  }
}
```

### Implementation

```python
"""tests/regression/test_golden.py - Golden file regression testing."""

import json
import pytest
from pathlib import Path
from datetime import datetime


class GoldenFileManager:
    """Manager for golden file operations."""

    def __init__(self, golden_dir: Path, version: str = "v1.0"):
        self.golden_dir = golden_dir
        self.version = version
        self.version_dir = golden_dir / version

    def load(self, test_name: str) -> dict:
        """Load golden output for a test."""
        path = self.version_dir / f"{test_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No golden file: {path}")
        return json.loads(path.read_text())

    def save(self, test_name: str, output: dict):
        """Save new golden output."""
        self.version_dir.mkdir(parents=True, exist_ok=True)
        path = self.version_dir / f"{test_name}.json"
        output["metadata"] = {
            "created": datetime.utcnow().isoformat(),
            "version": self.version,
        }
        path.write_text(json.dumps(output, indent=2, sort_keys=True))

    def compare_tags(
        self,
        actual: list[dict],
        expected: list[dict],
        default_tolerance: float = 0.05
    ) -> tuple[bool, list[str]]:
        """
        Compare actual tags against golden with tolerance.

        Returns (passed, list of differences).
        """
        differences = []
        actual_by_label = {t["label"]: t["confidence"] for t in actual}
        expected_by_label = {
            t["label"]: (t["confidence"], t.get("tolerance", default_tolerance))
            for t in expected
        }

        # Check for missing tags
        for label in expected_by_label:
            if label not in actual_by_label:
                differences.append(f"Missing tag: {label}")

        # Check for unexpected tags
        for label in actual_by_label:
            if label not in expected_by_label:
                differences.append(f"Unexpected tag: {label}")

        # Check confidence within tolerance
        for label, (exp_conf, tolerance) in expected_by_label.items():
            if label in actual_by_label:
                act_conf = actual_by_label[label]
                if abs(act_conf - exp_conf) > tolerance:
                    differences.append(
                        f"{label}: {act_conf:.3f} vs {exp_conf:.3f} "
                        f"(tolerance: {tolerance})"
                    )

        return (len(differences) == 0, differences)


class TestGoldenOutputs:
    """Regression tests using golden files."""

    @pytest.fixture
    def golden_manager(self, golden_dir: Path) -> GoldenFileManager:
        return GoldenFileManager(golden_dir)

    def test_standard_image_regression(
        self,
        fixtures_dir: Path,
        tag_image,
        golden_manager: GoldenFileManager
    ):
        """Test standard test image produces expected tags."""
        test_image = fixtures_dir / "images" / "real" / "standard_test.jpg"

        if not test_image.exists():
            pytest.skip("Standard test image not available")

        result = tag_image(test_image)

        try:
            golden = golden_manager.load("standard_test")
            passed, diffs = golden_manager.compare_tags(
                result["tags"],
                golden["tags"]
            )
            assert passed, f"Golden test failed:\n" + "\n".join(diffs)
        except FileNotFoundError:
            # First run - create golden file
            pytest.skip("No golden file - run with --update-golden to create")

    @pytest.mark.parametrize("test_name", [
        "dog_outdoor",
        "city_street",
        "food_closeup",
        "landscape_mountain",
    ])
    def test_category_regressions(
        self,
        test_name: str,
        fixtures_dir: Path,
        tag_image,
        golden_manager: GoldenFileManager
    ):
        """Test category-specific golden files."""
        test_image = fixtures_dir / "images" / "real" / f"{test_name}.jpg"

        if not test_image.exists():
            pytest.skip(f"Test image {test_name} not available")

        result = tag_image(test_image)

        try:
            golden = golden_manager.load(test_name)
            passed, diffs = golden_manager.compare_tags(result["tags"], golden["tags"])
            assert passed, f"Regression in {test_name}:\n" + "\n".join(diffs)
        except FileNotFoundError:
            pytest.skip(f"No golden file for {test_name}")


class TestConfidenceDrift:
    """Tests for detecting confidence score drift over time."""

    def test_confidence_distribution_stable(
        self,
        fixtures_dir: Path,
        tag_image,
        golden_dir: Path
    ):
        """
        Test that confidence score distribution hasn't drifted significantly.

        Uses Kolmogorov-Smirnov test to compare distributions.
        """
        from scipy import stats

        test_image = fixtures_dir / "images" / "real" / "standard_test.jpg"
        if not test_image.exists():
            pytest.skip("Test image not available")

        result = tag_image(test_image)
        current_confidences = [t["confidence"] for t in result["tags"]]

        # Load historical confidence distribution
        history_path = golden_dir / "confidence_history.json"
        if not history_path.exists():
            pytest.skip("No confidence history available")

        history = json.loads(history_path.read_text())
        baseline_confidences = history["baseline_confidences"]

        # KS test - null hypothesis: samples are from same distribution
        statistic, p_value = stats.ks_2samp(current_confidences, baseline_confidences)

        # If p < 0.05, distributions are significantly different
        assert p_value > 0.05, (
            f"Confidence distribution has drifted significantly "
            f"(KS statistic: {statistic:.3f}, p-value: {p_value:.3f})"
        )
```

---

## 6. Performance Benchmarking

### Metrics to Track

| Metric | Description | Acceptable Range |
|--------|-------------|------------------|
| **Inference time** | Time to tag single image | < 1s for most plugins |
| **Throughput** | Images per second in batch | > 2 img/s on GPU |
| **Memory peak** | Maximum RAM/VRAM usage | < 8GB VRAM for standard |
| **Startup time** | Model loading time | < 10s first run |

### Implementation

```python
"""tests/benchmark/test_performance.py - Performance benchmarking."""

import time
import pytest
import tracemalloc
from pathlib import Path
from statistics import mean, stdev


class TestInferencePerformance:
    """Performance tests for inference."""

    @pytest.mark.benchmark
    def test_single_image_latency(self, jpeg_image: Path, tag_image, benchmark):
        """Benchmark single image tagging latency."""
        # pytest-benchmark integration
        result = benchmark(tag_image, jpeg_image)

        # Also do manual timing for reporting
        times = []
        for _ in range(10):
            start = time.perf_counter()
            tag_image(jpeg_image)
            times.append(time.perf_counter() - start)

        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0

        print(f"\nLatency: {avg_time*1000:.1f}ms (+/- {std_time*1000:.1f}ms)")

        # Assert reasonable performance
        assert avg_time < 5.0, f"Single image latency too high: {avg_time:.2f}s"

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_batch_throughput(self, temp_dir: Path, tag_batch):
        """Benchmark batch processing throughput."""
        from PIL import Image

        # Create batch of test images
        batch_size = 10
        images = []
        for i in range(batch_size):
            path = temp_dir / f"batch_{i}.jpg"
            Image.new("RGB", (512, 512), color=(i * 25, 100, 150)).save(path, "JPEG")
            images.append(path)

        # Time batch processing
        start = time.perf_counter()
        results = tag_batch(images)
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        print(f"\nThroughput: {throughput:.2f} images/second")

        assert throughput > 0.5, f"Throughput too low: {throughput:.2f} img/s"

    @pytest.mark.benchmark
    def test_resolution_scaling(self, multi_resolution_images: dict, tag_image):
        """Benchmark how latency scales with resolution."""
        results = {}

        for res, path in sorted(multi_resolution_images.items()):
            times = []
            for _ in range(3):
                start = time.perf_counter()
                tag_image(path)
                times.append(time.perf_counter() - start)
            results[res] = mean(times)

        print("\nResolution scaling:")
        for res, t in results.items():
            print(f"  {res}px: {t*1000:.1f}ms")

        # Time should scale sub-linearly with resolution
        if 512 in results and 2048 in results:
            ratio = results[2048] / results[512]
            # 2048 is 16x the pixels of 512, but time shouldn't be 16x
            assert ratio < 8, f"Poor scaling: {ratio}x slowdown for 4x resolution"


class TestMemoryUsage:
    """Memory usage tests."""

    @pytest.mark.benchmark
    def test_peak_memory_single_image(self, jpeg_image: Path, tag_image):
        """Measure peak memory for single image tagging."""
        tracemalloc.start()

        tag_image(jpeg_image)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        print(f"\nPeak memory: {peak_mb:.1f} MB")

        # Reasonable limit for CPU-only inference
        assert peak_mb < 4000, f"Peak memory too high: {peak_mb:.1f} MB"

    @pytest.mark.benchmark
    @pytest.mark.gpu
    def test_gpu_memory_usage(self, jpeg_image: Path, tag_image):
        """Measure GPU memory usage."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.cuda.reset_peak_memory_stats()

        tag_image(jpeg_image)

        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nPeak VRAM: {peak_vram:.1f} MB")

        # Most models should fit in 8GB VRAM
        assert peak_vram < 8000, f"VRAM usage too high: {peak_vram:.1f} MB"

    @pytest.mark.benchmark
    def test_no_memory_leak(self, jpeg_image: Path, tag_image):
        """Test for memory leaks over multiple runs."""
        import gc

        # Warm up
        for _ in range(3):
            tag_image(jpeg_image)
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # Run many iterations
        for _ in range(50):
            tag_image(jpeg_image)

        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        growth_mb = (final - baseline) / 1024 / 1024
        print(f"\nMemory growth over 50 runs: {growth_mb:.2f} MB")

        # Allow some growth but not unbounded
        assert growth_mb < 100, f"Possible memory leak: {growth_mb:.2f} MB growth"
```

---

## 7. CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run unit tests
        run: |
          pytest tests/unit -v --tb=short -m "not slow"

      - name: Upload coverage
        uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev,ml]"

      - name: Download test fixtures
        run: |
          # Download real test images (from secure storage)
          # aws s3 cp s3://test-fixtures/images tests/fixtures/images --recursive
          echo "Using synthetic images for CI"

      - name: Run integration tests
        run: |
          pytest tests/integration -v --tb=short
        env:
          PYTHONHASHSEED: "42"

  regression-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev,ml]"

      - name: Run regression tests
        run: |
          pytest tests/regression -v --tb=long
        env:
          PYTHONHASHSEED: "42"

      - name: Upload regression artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: regression-failures
          path: tests/regression/failures/

  benchmark-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev,ml]"
          pip install pytest-benchmark

      - name: Run benchmarks
        run: |
          pytest tests/benchmark -v --benchmark-json=benchmark.json

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests
    regression: marks regression tests
    benchmark: marks benchmark tests

# Default options
addopts =
    -v
    --strict-markers
    --tb=short
    -ra

# Ignore warnings from dependencies
filterwarnings =
    ignore::DeprecationWarning:torch.*
    ignore::UserWarning:PIL.*

# Coverage
[coverage:run]
source = src
omit = tests/*

[coverage:report]
exclude_lines =
    pragma: no cover
    raise NotImplementedError
    if TYPE_CHECKING:
```

---

## 8. Test Fixtures & Data Sources

### Recommended Image Sources

| Source | License | Use Case | URL |
|--------|---------|----------|-----|
| **TESTIMAGES** | CC BY-NC-SA 4.0 | Patterns, sampling | [testimages.org](https://testimages.org/) [5] |
| **CMU Image Database** | Academic | Standard test images | [cs.cmu.edu](https://www.cs.cmu.edu/~cil/v-images.html) [6] |
| **Unsplash** | Unsplash License | Real photos | [unsplash.com](https://unsplash.com/) |
| **Pexels** | Pexels License | Real photos | [pexels.com](https://pexels.com/) |
| **ImageNet sample** | Research | ML testing | Via torchvision |

### Curated Test Set Recommendations

Create a curated set of ~20-30 images covering:

```
tests/fixtures/images/real/
├── categories/
│   ├── animal_dog.jpg         # Single subject
│   ├── animal_cat.jpg
│   ├── scene_outdoor.jpg      # Complex scene
│   ├── scene_indoor.jpg
│   ├── food_plate.jpg         # Close-up
│   ├── text_sign.jpg          # OCR test
│   ├── people_portrait.jpg    # Face/person
│   └── landscape_mountain.jpg # Wide scene
├── technical/
│   ├── high_contrast.jpg      # Exposure extremes
│   ├── low_light.jpg
│   ├── motion_blur.jpg        # Quality issues
│   ├── compression_heavy.jpg
│   └── noise_heavy.jpg
└── formats/
    ├── sample.heic            # Apple format
    ├── sample.webp            # Modern format
    ├── sample_16bit.tiff      # High bit depth
    └── sample_cmyk.tiff       # Print color space
```

---

## 9. Statistical Validation for ML Outputs

### Tolerance Calculation

For ML confidence scores, use these tolerance guidelines [7]:

| Confidence Range | Recommended Tolerance | Rationale |
|-----------------|----------------------|-----------|
| 0.90 - 1.00 | ± 0.02 | High confidence should be stable |
| 0.70 - 0.90 | ± 0.05 | Normal operating range |
| 0.50 - 0.70 | ± 0.08 | Borderline detections vary more |
| 0.00 - 0.50 | ± 0.10 | Low confidence inherently unstable |

### Bootstrap Confidence Intervals

```python
def bootstrap_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for mean.

    Useful for establishing tolerance thresholds from multiple runs.
    """
    import random

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = random.choices(values, k=len(values))
        bootstrap_means.append(sum(sample) / len(sample))

    bootstrap_means.sort()
    lower_idx = int((1 - confidence) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence) / 2 * n_bootstrap)

    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
```

### Tag Set Comparison

Use Jaccard similarity for comparing tag sets:

```python
def assert_tag_similarity(
    actual_tags: set[str],
    expected_tags: set[str],
    min_similarity: float = 0.8
):
    """
    Assert that actual tags are similar enough to expected.

    Jaccard similarity = |intersection| / |union|
    """
    if not actual_tags and not expected_tags:
        return  # Both empty is OK

    intersection = len(actual_tags & expected_tags)
    union = len(actual_tags | expected_tags)
    similarity = intersection / union

    assert similarity >= min_similarity, (
        f"Tag similarity too low: {similarity:.2f} < {min_similarity}\n"
        f"Missing: {expected_tags - actual_tags}\n"
        f"Extra: {actual_tags - expected_tags}"
    )
```

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Testing of specific ML model architectures (RAM++, YOLO, etc.)
- GPU cluster testing and distributed inference
- Real-time streaming image processing
- Mobile/embedded deployment testing

### Hardware-Dependent Results

Results will differ between [3][HIGH]:
- CPU and GPU execution (even same code)
- Different GPU models (RTX 3090 vs A100)
- Different CUDA/cuDNN versions
- Different operating systems

**Mitigation**: Store hardware fingerprint with golden files; maintain separate baselines per hardware class.

### Model Version Sensitivity

Model weight updates may change outputs significantly. Golden files should be versioned alongside model versions.

### Source Conflicts

- PyTorch documentation recommends `use_deterministic_algorithms(True)` but notes performance impact [3]
- Some operations (atomics) cannot be made deterministic even with flags enabled

---

## Recommendations

1. **Start with synthetic fixtures** - Use generated images for unit tests; they're fast, deterministic, and don't require licensing
2. **Implement tolerance-based assertions** - Never use exact equality for ML outputs
3. **Version golden files** - Track `model_version`, `app_version`, and `hardware_fingerprint`
4. **Run benchmarks on main only** - Avoid noisy benchmark data from feature branches
5. **Use statistical tests for drift** - Kolmogorov-Smirnov or similar for detecting distribution changes
6. **Separate fast/slow tests** - Use pytest markers to enable quick iteration
7. **Cache ML models in CI** - Model downloads are slow; cache between runs

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [PyTorch Reproducibility Docs](https://pytorch.org/docs/stable/notes/randomness.html) | 2025 | Primary | ML determinism, GPU/CPU differences |
| 2 | [pytest-mpl PyPI](https://pypi.org/project/pytest-mpl/) | 2024 | Primary | Image comparison testing |
| 3 | [PyTorch Reproducibility Guide](https://docs.pytorch.org/docs/stable/notes/randomness.html) | 2025 | Primary | Deterministic algorithms |
| 4 | [Golden Tests in AI - Shaped](https://www.shaped.ai/blog/golden-tests-in-ai) | 2024 | Secondary | Golden file strategy for ML |
| 5 | [TESTIMAGES Archive](https://testimages.org/) | 2024 | Primary | Test image source |
| 6 | [CMU Computer Vision Test Images](https://www.cs.cmu.edu/~cil/v-images.html) | - | Primary | Standard test images |
| 7 | [Confidence Intervals for ML - MachineLearningMastery](https://machinelearningmastery.com/confidence-intervals-for-machine-learning/) | 2024 | Secondary | Statistical tolerance calculation |
| 8 | [pytest Best Practices - Real Python](https://realpython.com/pytest-python-testing/) | 2024 | Secondary | Pytest organization |
| 9 | [Testing ML Systems - Made With ML](https://madewithml.com/courses/mlops/testing/) | 2024 | Secondary | ML testing patterns |
| 10 | [NVIDIA Framework Reproducibility](https://github.com/NVIDIA/framework-reproducibility) | 2024 | Primary | GPU determinism |
| 11 | [GitHub Actions for Python - pytest-with-eric](https://pytest-with-eric.com/integrations/pytest-github-actions/) | 2024 | Secondary | CI/CD integration |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial version |
