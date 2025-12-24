"""Services module for visual-buffet.

Contains service layer components including:
- XMP metadata handling
- Tag persistence
"""

from visual_buffet.services.xmp_handler import (
    XMPHandler,
    read_xmp_sidecar,
    write_xmp_sidecar,
    clear_xmp_tags,
    has_xmp_sidecar,
)

__all__ = [
    "XMPHandler",
    "read_xmp_sidecar",
    "write_xmp_sidecar",
    "clear_xmp_tags",
    "has_xmp_sidecar",
]
