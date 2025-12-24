"""Services module for visual-buffet.

Contains service layer components including:
- XMP metadata handling
- Tag persistence
- OCR verification
"""

from visual_buffet.services.ocr_verification import (
    OCRSource,
    OCRVerificationResult,
    OCRVerificationService,
    VerificationTier,
    VerifiedText,
)
from visual_buffet.services.xmp_handler import (
    XMPHandler,
    clear_xmp_tags,
    has_xmp_sidecar,
    read_xmp_sidecar,
    write_xmp_sidecar,
)

__all__ = [
    # XMP handling
    "XMPHandler",
    "read_xmp_sidecar",
    "write_xmp_sidecar",
    "clear_xmp_tags",
    "has_xmp_sidecar",
    # OCR verification
    "OCRVerificationService",
    "OCRVerificationResult",
    "VerifiedText",
    "VerificationTier",
    "OCRSource",
]
