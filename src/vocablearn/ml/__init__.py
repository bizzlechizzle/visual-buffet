"""Machine learning enhancements for vocablearn.

Provides:
- SigLIP-based scene classification (indoor/outdoor, etc.)
- Tag co-occurrence boosting via PMI
- SigLIP embeddings for duplicate detection
"""

from vocablearn.ml.classifier import SceneClassifier, classify_scene
from vocablearn.ml.cooccurrence import CooccurrenceBooster, calculate_pmi
from vocablearn.ml.embeddings import EmbeddingIndex, get_image_embedding

__all__ = [
    "SceneClassifier",
    "classify_scene",
    "CooccurrenceBooster",
    "calculate_pmi",
    "EmbeddingIndex",
    "get_image_embedding",
]
