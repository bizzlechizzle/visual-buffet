"""Core IMLAGE logic.

This module contains:
- TaggingEngine: Main processing engine
- Hardware detection
"""

from .engine import TaggingEngine
from .hardware import detect_hardware, get_recommended_batch_size

__all__ = [
    "TaggingEngine",
    "detect_hardware",
    "get_recommended_batch_size",
]
