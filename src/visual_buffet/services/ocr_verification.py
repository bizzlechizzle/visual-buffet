"""OCR Verification Service.

Trust but verify: Uses multiple OCR engines + SigLIP validation
to produce confidence-tiered text extraction for auto-tagging.

Architecture:
    PaddleOCR (primary, max recall)
        → docTR (cross-verification)
        → SigLIP (semantic validation)
        → Confidence tiers (VERIFIED/LIKELY/UNVERIFIED/REJECTED)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..plugins.base import PluginBase

logger = logging.getLogger(__name__)


class VerificationTier(str, Enum):
    """Confidence tiers for auto-tagging decisions.

    VERIFIED: Both OCR engines agree OR high OCR + SigLIP validation
              → Auto-tag, prime search results
    LIKELY:   High OCR confidence OR single verification signal
              → Auto-tag, secondary search results
    UNVERIFIED: Low confidence, no verification
              → Store but flag for manual review
    REJECTED: Very low confidence, no verification
              → Don't store (noise)
    """

    VERIFIED = "verified"
    LIKELY = "likely"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"

    @property
    def auto_tag(self) -> bool:
        """Whether this tier should be auto-tagged."""
        return self in (VerificationTier.VERIFIED, VerificationTier.LIKELY)

    @property
    def searchable(self) -> bool:
        """Whether this tier should be searchable."""
        return self != VerificationTier.REJECTED


@dataclass
class OCRSource:
    """Result from a single OCR engine."""

    engine: str
    text: str
    confidence: float
    bbox: list[list[float]] = field(default_factory=list)


@dataclass
class VerifiedText:
    """A verified OCR text result with confidence tier."""

    text: str
    normalized: str
    tier: VerificationTier
    verification_score: float

    # Source signals
    paddle_ocr: OCRSource | None = None
    doctr: OCRSource | None = None
    siglip_score: float | None = None

    # Derived flags
    auto_tag: bool = False
    searchable: bool = True

    def __post_init__(self):
        """Set derived flags from tier."""
        self.auto_tag = self.tier.auto_tag
        self.searchable = self.tier.searchable

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "text": self.text,
            "normalized": self.normalized,
            "tier": self.tier.value,
            "verification_score": round(self.verification_score, 4),
            "auto_tag": self.auto_tag,
            "searchable": self.searchable,
            "sources": {},
        }

        if self.paddle_ocr:
            result["sources"]["paddle_ocr"] = {
                "confidence": round(self.paddle_ocr.confidence, 4),
                "bbox": self.paddle_ocr.bbox,
            }

        if self.doctr:
            result["sources"]["doctr"] = {
                "confidence": round(self.doctr.confidence, 4),
                "bbox": self.doctr.bbox,
            }

        if self.siglip_score is not None:
            result["sources"]["siglip"] = {
                "score": round(self.siglip_score, 4),
            }

        return result


@dataclass
class OCRVerificationResult:
    """Complete OCR verification result for an image."""

    image_path: str
    texts: list[VerifiedText]
    total_time_ms: float

    # Tier counts
    verified_count: int = 0
    likely_count: int = 0
    unverified_count: int = 0
    rejected_count: int = 0

    def __post_init__(self):
        """Compute tier counts."""
        for text in self.texts:
            if text.tier == VerificationTier.VERIFIED:
                self.verified_count += 1
            elif text.tier == VerificationTier.LIKELY:
                self.likely_count += 1
            elif text.tier == VerificationTier.UNVERIFIED:
                self.unverified_count += 1
            elif text.tier == VerificationTier.REJECTED:
                self.rejected_count += 1

    @property
    def auto_tag_texts(self) -> list[VerifiedText]:
        """Get texts that should be auto-tagged."""
        return [t for t in self.texts if t.auto_tag]

    @property
    def searchable_texts(self) -> list[VerifiedText]:
        """Get texts that should be searchable."""
        return [t for t in self.texts if t.searchable]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "texts": [t.to_dict() for t in self.texts],
            "total_time_ms": round(self.total_time_ms, 2),
            "summary": {
                "total": len(self.texts),
                "verified": self.verified_count,
                "likely": self.likely_count,
                "unverified": self.unverified_count,
                "rejected": self.rejected_count,
                "auto_tagged": self.verified_count + self.likely_count,
            },
        }


class OCRVerificationService:
    """Service for verified OCR extraction.

    Uses PaddleOCR for maximum recall, docTR for cross-verification,
    and SigLIP for semantic validation. Produces confidence-tiered
    results suitable for auto-tagging.

    Example:
        >>> service = OCRVerificationService(plugins)
        >>> result = service.verify_image(Path("photo.jpg"))
        >>> for text in result.auto_tag_texts:
        ...     print(f"{text.text} [{text.tier.value}]")
    """

    # Weights for verification score computation
    PADDLE_WEIGHT = 0.5
    DOCTR_WEIGHT = 0.3
    SIGLIP_WEIGHT = 0.2

    # Tier thresholds
    VERIFIED_SCORE = 0.7
    LIKELY_SCORE = 0.5

    # High confidence threshold (for alt rules)
    HIGH_CONF = 0.9
    MEDIUM_CONF = 0.7

    # SigLIP validation threshold
    SIGLIP_VALID = 0.01

    # Rejection threshold
    REJECT_CONF = 0.3
    REJECT_SIGLIP = 0.001

    # SigLIP prompt template (best performing from testing)
    SIGLIP_TEMPLATE = "A sign reading {}."

    def __init__(
        self,
        plugins: dict[str, PluginBase],
        paddle_threshold: float = 0.3,
        doctr_threshold: float = 0.3,
    ):
        """Initialize the OCR verification service.

        Args:
            plugins: Dict of loaded plugins by name.
                     Required: paddle_ocr
                     Optional: doctr, siglip
            paddle_threshold: Confidence threshold for PaddleOCR
            doctr_threshold: Confidence threshold for docTR
        """
        self._plugins = plugins
        self._paddle_threshold = paddle_threshold
        self._doctr_threshold = doctr_threshold

        # Validate required plugins
        if "paddle_ocr" not in plugins:
            raise ValueError("OCRVerificationService requires paddle_ocr plugin")

        # Check optional plugins
        self._has_doctr = "doctr" in plugins and plugins["doctr"].is_available()
        self._has_siglip = "siglip" in plugins and plugins["siglip"].is_available()

        # Lazy-loaded SigLIP components
        self._siglip_model = None
        self._siglip_processor = None
        self._siglip_device = None

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def texts_match(t1: str, t2: str, threshold: float = 0.8) -> bool:
        """Check if two texts match (fuzzy)."""
        n1 = OCRVerificationService.normalize_text(t1)
        n2 = OCRVerificationService.normalize_text(t2)

        if not n1 or not n2:
            return False
        if n1 == n2:
            return True
        if n1 in n2 or n2 in n1:
            return True
        if len(n1) > 3 and len(n2) > 3:
            ratio = SequenceMatcher(None, n1, n2).ratio()
            return ratio > threshold
        return False

    def _run_paddle_ocr(self, image_path: Path) -> list[OCRSource]:
        """Run PaddleOCR on image."""
        results = []
        plugin = self._plugins["paddle_ocr"]

        try:
            plugin.configure(threshold=self._paddle_threshold)
            tag_result = plugin.tag(image_path)

            for tag in tag_result.tags:
                if tag.confidence and tag.confidence >= self._paddle_threshold:
                    results.append(
                        OCRSource(
                            engine="paddle_ocr",
                            text=tag.label,
                            confidence=tag.confidence,
                        )
                    )
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")

        return results

    def _run_doctr(self, image_path: Path) -> list[OCRSource]:
        """Run docTR on image."""
        if not self._has_doctr:
            return []

        results = []
        plugin = self._plugins["doctr"]

        try:
            plugin.configure(threshold=self._doctr_threshold)
            tag_result = plugin.tag(image_path)

            for tag in tag_result.tags:
                if tag.confidence and tag.confidence >= self._doctr_threshold:
                    results.append(
                        OCRSource(
                            engine="doctr",
                            text=tag.label,
                            confidence=tag.confidence,
                        )
                    )
        except Exception as e:
            logger.warning(f"docTR failed: {e}")

        return results

    def _load_siglip(self):
        """Lazy-load SigLIP model for validation."""
        if self._siglip_model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            if torch.cuda.is_available():
                self._siglip_device = torch.device("cuda")
                dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._siglip_device = torch.device("mps")
                dtype = torch.float32
            else:
                self._siglip_device = torch.device("cpu")
                dtype = torch.float32

            model_id = "google/siglip-so400m-patch14-384"
            self._siglip_processor = AutoProcessor.from_pretrained(model_id)
            self._siglip_model = AutoModel.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(self._siglip_device)
            self._siglip_model.eval()

        except Exception as e:
            logger.warning(f"Failed to load SigLIP for validation: {e}")
            self._has_siglip = False

    def _run_siglip_validation(
        self, image_path: Path, texts: list[str]
    ) -> dict[str, float]:
        """Run SigLIP validation on extracted texts.

        Args:
            image_path: Path to image
            texts: List of texts to validate

        Returns:
            Dict mapping normalized text to SigLIP score
        """
        if not self._has_siglip or not texts:
            return {}

        self._load_siglip()
        if self._siglip_model is None:
            return {}

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # Create prompts
            prompts = [self.SIGLIP_TEMPLATE.format(t.lower()) for t in texts]

            # Process inputs
            inputs = self._siglip_processor(
                text=prompts,
                images=image,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(self._siglip_device)

            # Run inference
            with torch.no_grad():
                outputs = self._siglip_model(**inputs)
                logits = outputs.logits_per_image
                probs = torch.sigmoid(logits).squeeze(0).cpu().float().numpy()

            # Build result dict
            results = {}
            for text, score in zip(texts, probs, strict=True):
                normalized = self.normalize_text(text)
                results[normalized] = float(score)

            return results

        except Exception as e:
            logger.warning(f"SigLIP validation failed: {e}")
            return {}

    def _compute_verification_score(
        self,
        paddle_conf: float,
        doctr_found: bool,
        siglip_score: float,
    ) -> float:
        """Compute composite verification score."""
        return (
            self.PADDLE_WEIGHT * paddle_conf
            + self.DOCTR_WEIGHT * (1.0 if doctr_found else 0.0)
            + self.SIGLIP_WEIGHT * min(siglip_score * 10, 1.0)
        )

    def _determine_tier(
        self,
        paddle_conf: float,
        doctr_found: bool,
        siglip_score: float,
        verification_score: float,
    ) -> VerificationTier:
        """Determine confidence tier for a text result."""
        # VERIFIED conditions
        if verification_score >= self.VERIFIED_SCORE:
            return VerificationTier.VERIFIED
        if paddle_conf >= self.HIGH_CONF and doctr_found:
            return VerificationTier.VERIFIED
        if paddle_conf >= 0.8 and siglip_score >= self.SIGLIP_VALID:
            return VerificationTier.VERIFIED

        # LIKELY conditions
        if verification_score >= self.LIKELY_SCORE:
            return VerificationTier.LIKELY
        if paddle_conf >= self.MEDIUM_CONF:
            return VerificationTier.LIKELY
        if doctr_found:
            return VerificationTier.LIKELY

        # REJECTED conditions
        if (
            paddle_conf < self.REJECT_CONF
            and not doctr_found
            and siglip_score < self.REJECT_SIGLIP
        ):
            return VerificationTier.REJECTED

        return VerificationTier.UNVERIFIED

    def verify_image(
        self,
        image_path: Path,
        run_siglip: bool = True,
    ) -> OCRVerificationResult:
        """Run verified OCR on an image.

        Args:
            image_path: Path to image file
            run_siglip: Whether to run SigLIP validation
                       (can disable for speed)

        Returns:
            OCRVerificationResult with confidence-tiered texts
        """
        import time

        start_time = time.perf_counter()

        # 1. Run PaddleOCR (primary)
        paddle_results = self._run_paddle_ocr(image_path)

        if not paddle_results:
            return OCRVerificationResult(
                image_path=str(image_path),
                texts=[],
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # 2. Run docTR (cross-verification)
        doctr_results = self._run_doctr(image_path)

        # 3. Run SigLIP validation
        siglip_scores: dict[str, float] = {}
        if run_siglip and self._has_siglip:
            texts_to_validate = [r.text for r in paddle_results]
            siglip_scores = self._run_siglip_validation(image_path, texts_to_validate)

        # 4. Compute verification for each PaddleOCR result
        verified_texts = []

        for paddle in paddle_results:
            normalized = self.normalize_text(paddle.text)

            # Find docTR match
            doctr_match = None
            for doctr in doctr_results:
                if self.texts_match(paddle.text, doctr.text):
                    doctr_match = doctr
                    break

            doctr_found = doctr_match is not None
            siglip_score = siglip_scores.get(normalized, 0.0)

            # Compute scores and tier
            verification_score = self._compute_verification_score(
                paddle.confidence, doctr_found, siglip_score
            )

            tier = self._determine_tier(
                paddle.confidence, doctr_found, siglip_score, verification_score
            )

            verified_texts.append(
                VerifiedText(
                    text=paddle.text,
                    normalized=normalized,
                    tier=tier,
                    verification_score=verification_score,
                    paddle_ocr=paddle,
                    doctr=doctr_match,
                    siglip_score=siglip_score if run_siglip else None,
                )
            )

        total_time = (time.perf_counter() - start_time) * 1000

        return OCRVerificationResult(
            image_path=str(image_path),
            texts=verified_texts,
            total_time_ms=total_time,
        )

    def verify_batch(
        self,
        image_paths: list[Path],
        run_siglip: bool = True,
        on_progress: callable = None,
    ) -> list[OCRVerificationResult]:
        """Run verified OCR on multiple images.

        Args:
            image_paths: List of image paths
            run_siglip: Whether to run SigLIP validation
            on_progress: Optional callback(image_path, result)

        Returns:
            List of OCRVerificationResult
        """
        results = []

        for path in image_paths:
            try:
                result = self.verify_image(path, run_siglip)
                results.append(result)

                if on_progress:
                    on_progress(path, result)

            except Exception as e:
                logger.exception(f"Verification failed for {path}: {e}")

        return results
