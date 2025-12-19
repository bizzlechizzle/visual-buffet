"""Custom exceptions for IMLAGE.

Each exception type represents a category of error.
Catch specific exceptions to handle errors appropriately.
"""


class ImlageError(Exception):
    """Base exception for all IMLAGE errors."""

    pass


class PluginError(ImlageError):
    """Raised when a plugin fails to load or execute."""

    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin doesn't exist."""

    pass


class ModelNotFoundError(PluginError):
    """Raised when a plugin's model file is missing."""

    pass


class ConfigError(ImlageError):
    """Raised when configuration is invalid or missing."""

    pass


class HardwareDetectionError(ImlageError):
    """Raised when hardware detection fails."""

    pass


class ImageError(ImlageError):
    """Raised when an image cannot be loaded or is invalid."""

    pass
