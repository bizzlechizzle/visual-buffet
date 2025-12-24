"""Tests for visual_buffet.constants module."""

import pytest


class TestConstants:
    """Test application constants."""

    def test_threshold_bounds(self):
        """Test threshold bounds are valid."""
        from visual_buffet.constants import (
            DEFAULT_THRESHOLD,
            MAX_THRESHOLD,
            MIN_THRESHOLD,
        )

        assert MIN_THRESHOLD == 0.0
        assert MAX_THRESHOLD == 1.0
        assert MIN_THRESHOLD <= DEFAULT_THRESHOLD <= MAX_THRESHOLD

    def test_gui_defaults(self):
        """Test GUI default values."""
        from visual_buffet.constants import DEFAULT_GUI_HOST, DEFAULT_GUI_PORT

        assert DEFAULT_GUI_HOST == "127.0.0.1"
        assert 1 <= DEFAULT_GUI_PORT <= 65535

    def test_image_size_enum(self):
        """Test ImageSize enum values."""
        from visual_buffet.constants import ImageSize

        assert ImageSize.LITTLE.value == "little"
        assert ImageSize.SMALL.value == "small"
        assert ImageSize.LARGE.value == "large"
        assert ImageSize.HUGE.value == "huge"
        assert ImageSize.ORIGINAL.value == "original"

    def test_image_size_resolutions(self):
        """Test image size resolution mappings."""
        from visual_buffet.constants import IMAGE_SIZE_RESOLUTIONS

        assert IMAGE_SIZE_RESOLUTIONS["little"] == 480
        assert IMAGE_SIZE_RESOLUTIONS["small"] == 1080
        assert IMAGE_SIZE_RESOLUTIONS["large"] == 2048
        assert IMAGE_SIZE_RESOLUTIONS["huge"] == 4096
        assert IMAGE_SIZE_RESOLUTIONS["original"] is None

    def test_exit_codes(self):
        """Test exit code enum values."""
        from visual_buffet.constants import ExitCode

        assert ExitCode.SUCCESS == 0
        assert ExitCode.GENERAL_ERROR == 1
        assert ExitCode.FILE_NOT_FOUND == 2
        assert ExitCode.INVALID_INPUT == 3
        assert ExitCode.PLUGIN_ERROR == 4
        assert ExitCode.NO_PLUGINS == 5
        assert ExitCode.KEYBOARD_INTERRUPT == 130

    def test_output_format_enum(self):
        """Test OutputFormat enum values."""
        from visual_buffet.constants import OutputFormat

        assert OutputFormat.JSON.value == "json"

    def test_file_extensions_are_frozensets(self):
        """Test file extension sets are immutable."""
        from visual_buffet.constants import (
            ALL_SUPPORTED_EXTENSIONS,
            HEIC_EXTENSIONS,
            RAW_EXTENSIONS,
            STANDARD_IMAGE_EXTENSIONS,
        )

        assert isinstance(STANDARD_IMAGE_EXTENSIONS, frozenset)
        assert isinstance(HEIC_EXTENSIONS, frozenset)
        assert isinstance(RAW_EXTENSIONS, frozenset)
        assert isinstance(ALL_SUPPORTED_EXTENSIONS, frozenset)

    def test_all_extensions_includes_all_types(self):
        """Test ALL_SUPPORTED_EXTENSIONS contains all types."""
        from visual_buffet.constants import (
            ALL_SUPPORTED_EXTENSIONS,
            HEIC_EXTENSIONS,
            RAW_EXTENSIONS,
            STANDARD_IMAGE_EXTENSIONS,
        )

        expected = STANDARD_IMAGE_EXTENSIONS | HEIC_EXTENSIONS | RAW_EXTENSIONS
        assert ALL_SUPPORTED_EXTENSIONS == expected

    def test_extensions_are_lowercase(self):
        """Test all extensions are lowercase and start with dot."""
        from visual_buffet.constants import ALL_SUPPORTED_EXTENSIONS

        for ext in ALL_SUPPORTED_EXTENSIONS:
            assert ext.startswith(".")
            assert ext == ext.lower()

    def test_xmp_constants(self):
        """Test XMP namespace constants."""
        from visual_buffet.constants import (
            XMP_NAMESPACE_PREFIX,
            XMP_NAMESPACE_URI,
            XMP_SCHEMA_VERSION,
        )

        assert XMP_NAMESPACE_URI == "http://visual-buffet.dev/xmp/1.0/"
        assert XMP_NAMESPACE_PREFIX == "vbuffet"
        assert XMP_SCHEMA_VERSION >= 1

    def test_jpeg_quality_levels(self):
        """Test JPEG quality constants are valid."""
        from visual_buffet.constants import (
            JPEG_QUALITY_HIGH,
            JPEG_QUALITY_LOW,
            JPEG_QUALITY_MEDIUM,
        )

        assert 1 <= JPEG_QUALITY_LOW <= 100
        assert 1 <= JPEG_QUALITY_MEDIUM <= 100
        assert 1 <= JPEG_QUALITY_HIGH <= 100
        assert JPEG_QUALITY_LOW < JPEG_QUALITY_MEDIUM < JPEG_QUALITY_HIGH

    def test_batch_size_constants(self):
        """Test batch size constants are reasonable."""
        from visual_buffet.constants import (
            BATCH_SIZE_CPU_ONLY,
            BATCH_SIZE_HIGH_VRAM,
            BATCH_SIZE_LOW_VRAM,
            BATCH_SIZE_MED_VRAM,
        )

        assert BATCH_SIZE_CPU_ONLY >= 1
        assert BATCH_SIZE_LOW_VRAM >= 1
        assert BATCH_SIZE_MED_VRAM >= BATCH_SIZE_LOW_VRAM
        assert BATCH_SIZE_HIGH_VRAM >= BATCH_SIZE_MED_VRAM

    def test_session_limits(self):
        """Test session limit constants."""
        from visual_buffet.constants import (
            MAX_FILE_SIZE_MB,
            MAX_SESSIONS,
            MAX_TOTAL_CACHE_MB,
        )

        assert MAX_SESSIONS >= 1
        assert MAX_FILE_SIZE_MB >= 1
        assert MAX_TOTAL_CACHE_MB >= MAX_FILE_SIZE_MB
