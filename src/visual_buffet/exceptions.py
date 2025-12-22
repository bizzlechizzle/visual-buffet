"""Custom exceptions for Visual Buffet.

Each exception type represents a category of error.
Catch specific exceptions to handle errors appropriately.
"""


class VisualBuffetError(Exception):
    """Base exception for all Visual Buffet errors."""

    pass


class PluginError(VisualBuffetError):
    """Raised when a plugin fails to load or execute."""

    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin doesn't exist."""

    pass


class ModelNotFoundError(PluginError):
    """Raised when a plugin's model file is missing."""

    pass


class ConfigError(VisualBuffetError):
    """Raised when configuration is invalid or missing."""

    pass


class HardwareDetectionError(VisualBuffetError):
    """Raised when hardware detection fails."""

    pass


class ImageError(VisualBuffetError):
    """Raised when an image cannot be loaded or is invalid."""

    pass
