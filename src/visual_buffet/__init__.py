"""Visual Buffet - Compare visual tagging results from local ML tools."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


# Lazy import to avoid loading vocablearn unless needed
def get_vocab_integration():
    """Get VocabIntegration class for vocabulary learning.

    Returns:
        VocabIntegration class
    """
    from visual_buffet.vocab_integration import VocabIntegration
    return VocabIntegration


def _get_version() -> str:
    """Get version from package metadata or VERSION file."""
    # Try installed package metadata first (works when installed)
    try:
        return version("visual-buffet")
    except PackageNotFoundError:
        pass

    # Fall back to VERSION file (works in development)
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()

    return "0.0.0"


__version__ = _get_version()
