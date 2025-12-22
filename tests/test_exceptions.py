"""Tests for custom exceptions."""

import pytest

from visual_buffet.exceptions import (
    ConfigError,
    HardwareDetectionError,
    ImageError,
    ModelNotFoundError,
    PluginError,
    PluginNotFoundError,
    VisualBuffetError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_visual_buffet_error_is_exception(self):
        """Test VisualBuffetError inherits from Exception."""
        assert issubclass(VisualBuffetError, Exception)

    def test_plugin_error_inherits_from_base(self):
        """Test PluginError inherits from VisualBuffetError."""
        assert issubclass(PluginError, VisualBuffetError)

    def test_plugin_not_found_inherits_from_plugin_error(self):
        """Test PluginNotFoundError inherits from PluginError."""
        assert issubclass(PluginNotFoundError, PluginError)
        assert issubclass(PluginNotFoundError, VisualBuffetError)

    def test_model_not_found_inherits_from_plugin_error(self):
        """Test ModelNotFoundError inherits from PluginError."""
        assert issubclass(ModelNotFoundError, PluginError)
        assert issubclass(ModelNotFoundError, VisualBuffetError)

    def test_config_error_inherits_from_base(self):
        """Test ConfigError inherits from VisualBuffetError."""
        assert issubclass(ConfigError, VisualBuffetError)

    def test_hardware_detection_error_inherits_from_base(self):
        """Test HardwareDetectionError inherits from VisualBuffetError."""
        assert issubclass(HardwareDetectionError, VisualBuffetError)

    def test_image_error_inherits_from_base(self):
        """Test ImageError inherits from VisualBuffetError."""
        assert issubclass(ImageError, VisualBuffetError)


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_visual_buffet_error_message(self):
        """Test VisualBuffetError stores message correctly."""
        error = VisualBuffetError("Test error message")
        assert str(error) == "Test error message"

    def test_plugin_error_message(self):
        """Test PluginError stores message correctly."""
        error = PluginError("Plugin failed to load")
        assert str(error) == "Plugin failed to load"

    def test_plugin_not_found_error_message(self):
        """Test PluginNotFoundError stores message correctly."""
        error = PluginNotFoundError("Plugin 'xyz' not found")
        assert str(error) == "Plugin 'xyz' not found"

    def test_model_not_found_error_message(self):
        """Test ModelNotFoundError stores message correctly."""
        error = ModelNotFoundError("Model file missing")
        assert str(error) == "Model file missing"

    def test_config_error_message(self):
        """Test ConfigError stores message correctly."""
        error = ConfigError("Invalid configuration")
        assert str(error) == "Invalid configuration"

    def test_hardware_detection_error_message(self):
        """Test HardwareDetectionError stores message correctly."""
        error = HardwareDetectionError("GPU detection failed")
        assert str(error) == "GPU detection failed"

    def test_image_error_message(self):
        """Test ImageError stores message correctly."""
        error = ImageError("Image corrupted")
        assert str(error) == "Image corrupted"


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    def test_catch_visual_buffet_error(self):
        """Test catching VisualBuffetError catches all derived exceptions."""
        exceptions = [
            PluginError("plugin"),
            PluginNotFoundError("not found"),
            ModelNotFoundError("model"),
            ConfigError("config"),
            HardwareDetectionError("hardware"),
            ImageError("image"),
        ]

        for exc in exceptions:
            with pytest.raises(VisualBuffetError):
                raise exc

    def test_catch_plugin_error(self):
        """Test catching PluginError catches derived plugin exceptions."""
        with pytest.raises(PluginError):
            raise PluginNotFoundError("not found")

        with pytest.raises(PluginError):
            raise ModelNotFoundError("model")

    def test_specific_catch(self):
        """Test catching specific exception types."""
        with pytest.raises(PluginNotFoundError):
            raise PluginNotFoundError("specific")

        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("specific")

    def test_exception_not_caught_by_sibling(self):
        """Test that sibling exceptions don't catch each other."""
        with pytest.raises(ConfigError):
            try:
                raise ConfigError("config error")
            except ImageError:
                pytest.fail("ImageError should not catch ConfigError")


class TestExceptionInstances:
    """Tests for exception instantiation."""

    def test_empty_message(self):
        """Test exceptions can be created without message."""
        error = VisualBuffetError()
        assert str(error) == ""

    def test_exception_with_args(self):
        """Test exceptions preserve args tuple."""
        error = PluginError("error", "extra", 123)
        assert error.args == ("error", "extra", 123)
