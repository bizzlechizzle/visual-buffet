"""Tests for image utilities."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from visual_buffet.exceptions import ImageError
from visual_buffet.utils.image import (
    SUPPORTED_EXTENSIONS,
    expand_paths,
    is_supported_image,
    load_image,
    validate_image,
)


class TestSupportedFormats:
    """Tests for supported format constants."""

    def test_supported_extensions_includes_common(self):
        """Test common image extensions are supported."""
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_includes_heic(self):
        """Test HEIC/HEIF extensions are supported."""
        assert ".heic" in SUPPORTED_EXTENSIONS
        assert ".heif" in SUPPORTED_EXTENSIONS


class TestIsSupportedImage:
    """Tests for format validation."""

    def test_supported_formats(self):
        """Test supported formats return True."""
        assert is_supported_image(Path("image.jpg")) is True
        assert is_supported_image(Path("image.jpeg")) is True
        assert is_supported_image(Path("image.png")) is True
        assert is_supported_image(Path("image.webp")) is True
        assert is_supported_image(Path("image.heic")) is True
        assert is_supported_image(Path("image.tif")) is True
        assert is_supported_image(Path("image.tiff")) is True
        assert is_supported_image(Path("image.bmp")) is True

    def test_supported_raw_formats(self):
        """Test RAW camera formats return True."""
        assert is_supported_image(Path("image.arw")) is True
        assert is_supported_image(Path("image.cr2")) is True
        assert is_supported_image(Path("image.cr3")) is True
        assert is_supported_image(Path("image.nef")) is True
        assert is_supported_image(Path("image.dng")) is True
        assert is_supported_image(Path("image.orf")) is True
        assert is_supported_image(Path("image.rw2")) is True
        assert is_supported_image(Path("image.raf")) is True
        assert is_supported_image(Path("image.pef")) is True
        assert is_supported_image(Path("image.srw")) is True

    def test_unsupported_formats(self):
        """Test unsupported formats return False."""
        assert is_supported_image(Path("image.txt")) is False
        assert is_supported_image(Path("image.pdf")) is False
        assert is_supported_image(Path("image.gif")) is False
        assert is_supported_image(Path("image.psd")) is False

    def test_case_insensitive(self):
        """Test format check is case insensitive."""
        assert is_supported_image(Path("image.JPG")) is True
        assert is_supported_image(Path("image.PNG")) is True
        assert is_supported_image(Path("image.HEIC")) is True


class TestValidateImage:
    """Tests for image validation."""

    def test_validate_valid_image(self):
        """Test validation passes for valid image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a minimal valid JPEG
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name, "JPEG")

            path = Path(f.name)
            # validate_image raises on error, returns None on success
            validate_image(path)

            path.unlink()

    def test_validate_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        path = Path("/nonexistent/image.jpg")
        with pytest.raises(ImageError, match="not found"):
            validate_image(path)

    def test_validate_unsupported_format(self):
        """Test validation fails for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            path = Path(f.name)

            with pytest.raises(ImageError, match="Unsupported"):
                validate_image(path)

            path.unlink()

    def test_validate_corrupted_image(self):
        """Test validation fails for corrupted image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not a valid jpeg")
            path = Path(f.name)

            with pytest.raises(ImageError, match="Cannot read"):
                validate_image(path)

            path.unlink()


class TestLoadImage:
    """Tests for image loading."""

    def test_load_valid_image(self):
        """Test loading valid image returns PIL Image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (200, 200), color="blue")
            img.save(f.name, "PNG")

            path = Path(f.name)
            loaded = load_image(path)

            assert isinstance(loaded, Image.Image)
            assert loaded.size == (200, 200)
            assert loaded.mode == "RGB"

            path.unlink()

    def test_load_converts_to_rgb(self):
        """Test loading image converts to RGB mode."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create RGBA image
            img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
            img.save(f.name, "PNG")

            path = Path(f.name)
            loaded = load_image(path)

            assert loaded.mode == "RGB"

            path.unlink()

    def test_load_nonexistent_raises(self):
        """Test loading non-existent file raises error."""
        path = Path("/nonexistent/image.jpg")
        with pytest.raises(ImageError):
            load_image(path)


class TestExpandPaths:
    """Tests for path expansion."""

    def test_expand_single_file(self):
        """Test expanding single file path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name, "JPEG")

            path = Path(f.name)
            paths = expand_paths([str(path)])

            assert len(paths) == 1
            assert paths[0] == path

            path.unlink()

    def test_expand_directory(self):
        """Test expanding directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img = Image.new("RGB", (100, 100), color="red")
                img.save(Path(tmpdir) / f"image{i}.jpg", "JPEG")

            # Create non-image file
            (Path(tmpdir) / "readme.txt").write_text("test")

            paths = expand_paths([tmpdir])

            assert len(paths) == 3
            assert all(p.suffix == ".jpg" for p in paths)

    def test_expand_directory_recursive(self):
        """Test expanding directory recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create images in root
            img = Image.new("RGB", (100, 100))
            img.save(tmppath / "root.jpg", "JPEG")

            # Create subdirectory with images
            subdir = tmppath / "subdir"
            subdir.mkdir()
            img.save(subdir / "nested.jpg", "JPEG")

            paths_non_recursive = expand_paths([tmpdir], recursive=False)
            paths_recursive = expand_paths([tmpdir], recursive=True)

            assert len(paths_non_recursive) == 1
            assert len(paths_recursive) == 2

    def test_expand_glob_pattern(self):
        """Test expanding glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create various images
            for ext in [".jpg", ".png", ".webp"]:
                img = Image.new("RGB", (100, 100))
                img.save(tmppath / f"image{ext}")

            # Test glob for just JPGs
            paths = expand_paths([str(tmppath / "*.jpg")])

            assert len(paths) == 1
            assert paths[0].suffix == ".jpg"

    def test_expand_multiple_paths(self):
        """Test expanding multiple paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create images
            for i in range(2):
                img = Image.new("RGB", (100, 100))
                img.save(tmppath / f"img{i}.jpg", "JPEG")

            paths = expand_paths([
                str(tmppath / "img0.jpg"),
                str(tmppath / "img1.jpg"),
            ])

            assert len(paths) == 2

    def test_expand_filters_unsupported(self):
        """Test expansion filters out unsupported formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create mixed files
            img = Image.new("RGB", (100, 100))
            img.save(tmppath / "image.jpg", "JPEG")
            (tmppath / "document.txt").write_text("test")
            (tmppath / "data.json").write_text("{}")

            paths = expand_paths([tmpdir])

            assert len(paths) == 1
            assert paths[0].name == "image.jpg"

    def test_expand_empty_input(self):
        """Test expanding empty input returns empty list."""
        paths = expand_paths([])
        assert paths == []

    def test_expand_nonexistent_path(self):
        """Test expanding non-existent path returns empty list."""
        paths = expand_paths(["/nonexistent/path"])
        assert paths == []


class TestImageSizes:
    """Tests for various image sizes."""

    def test_small_image(self):
        """Test loading small image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (10, 10))
            img.save(f.name, "JPEG")

            path = Path(f.name)
            loaded = load_image(path)
            assert loaded.size == (10, 10)

            path.unlink()

    def test_large_image(self):
        """Test loading large image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (4000, 3000))
            img.save(f.name, "JPEG")

            path = Path(f.name)
            loaded = load_image(path)
            assert loaded.size == (4000, 3000)

            path.unlink()
