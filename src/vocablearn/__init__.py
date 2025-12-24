"""vocablearn - Vocabulary Learning Library for Image Tagging.

A library for tracking image tags, learning confidence priors from
historical data, and calibrating model confidence scores.

Features:
- Tag recording and tracking from multiple ML models
- Bayesian prior learning from feedback
- Isotonic regression calibration
- SigLIP-based scene classification
- Tag co-occurrence analysis (PMI)
- Image embedding similarity/duplicate detection
- OCR vocabulary tracking (per-app isolated)

Example:
    >>> from vocablearn import VocabLearn
    >>> vocab = VocabLearn("./vocab.db")
    >>> vocab.record_tags("img001.jpg", [{"label": "bar", "confidence": 0.95}], "ram_plus")
    >>> vocab.record_feedback("img001.jpg", "bar", correct=True)
    >>> calibrated = vocab.get_calibrated_confidence("bar", 0.8, "ram_plus")

Per-app isolation:
    >>> from vocablearn import AppConfig
    >>> config = AppConfig("abandoned-archive")
    >>> vocab = VocabLearn(config.vocab_db)
    >>> ocr = OCRStorage(config.ocr_db)
"""

__version__ = "0.1.11"

from vocablearn.api import VocabLearn
from vocablearn.app_config import AppConfig
from vocablearn.storage.ocr import OCRStorage, OCRText, OCRDetection
from vocablearn.models import (
    CalibrationPoint,
    ConfidenceTier,
    Tag,
    TagEvent,
    TagSource,
    VocabularyStats,
    calculate_unified_confidence,
    is_compound_tag,
    normalize_tag,
)

# Lazy imports for ML modules (require torch/transformers)
def get_scene_classifier():
    """Get SceneClassifier class (requires torch, transformers)."""
    from vocablearn.ml.classifier import SceneClassifier
    return SceneClassifier


def get_cooccurrence_booster():
    """Get CooccurrenceBooster class."""
    from vocablearn.ml.cooccurrence import CooccurrenceBooster
    return CooccurrenceBooster


def get_embedding_index():
    """Get EmbeddingIndex class (requires torch, transformers)."""
    from vocablearn.ml.embeddings import EmbeddingIndex
    return EmbeddingIndex


__all__ = [
    "VocabLearn",
    "AppConfig",
    "OCRStorage",
    "OCRText",
    "OCRDetection",
    "Tag",
    "TagEvent",
    "TagSource",
    "ConfidenceTier",
    "CalibrationPoint",
    "VocabularyStats",
    "calculate_unified_confidence",
    "is_compound_tag",
    "normalize_tag",
    # Lazy ML module getters
    "get_scene_classifier",
    "get_cooccurrence_booster",
    "get_embedding_index",
]
