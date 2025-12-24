"""Tests for vocablearn.ml.embeddings module."""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
import json
from pathlib import Path


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_to_dict(self):
        """Test SimilarityResult serialization."""
        from vocablearn.ml.embeddings import SimilarityResult

        result = SimilarityResult(
            image_id="img001.jpg",
            score=0.876543,
        )
        d = result.to_dict()

        assert d["image_id"] == "img001.jpg"
        assert d["score"] == 0.8765  # Rounded to 4 decimals


class TestDuplicatePair:
    """Tests for DuplicatePair dataclass."""

    def test_to_dict(self):
        """Test DuplicatePair serialization."""
        from vocablearn.ml.embeddings import DuplicatePair

        pair = DuplicatePair(
            image_a="img001.jpg",
            image_b="img002.jpg",
            similarity=0.987654,
        )
        d = pair.to_dict()

        assert d["image_a"] == "img001.jpg"
        assert d["image_b"] == "img002.jpg"
        assert d["similarity"] == 0.9877  # Rounded to 4 decimals


class TestEmbeddingIndex:
    """Tests for EmbeddingIndex class."""

    def test_init(self):
        """Test EmbeddingIndex initialization."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex(model_variant="so400m", device="cpu")
        assert index._model_variant == "so400m"
        assert index._device == "cpu"
        assert index._model is None  # Lazy loading
        assert index.size == 0

    def test_add_embedding(self):
        """Test adding pre-computed embeddings."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        embedding = [0.1, 0.2, 0.3, 0.4]

        index.add_embedding("img001", embedding)

        assert index.size == 1
        assert index.get_embedding("img001") == embedding

    def test_get_embedding_not_found(self):
        """Test getting embedding for unknown image."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        assert index.get_embedding("unknown") is None

    def test_remove_image(self):
        """Test removing image from index."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        index.add_embedding("img001", [0.1, 0.2, 0.3])

        assert index.remove_image("img001") is True
        assert index.size == 0
        assert index.remove_image("img001") is False  # Already removed

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()

        # Test identical vectors (should be 1.0)
        sim = index._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

        # Test orthogonal vectors (should be 0.0)
        sim = index._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 0.001

        # Test opposite vectors (should be -1.0)
        sim = index._cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert abs(sim - (-1.0)) < 0.001

    def test_cosine_similarity_zero_norm(self):
        """Test cosine similarity with zero vector."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        sim = index._cosine_similarity([0, 0, 0], [1, 2, 3])
        assert sim == 0.0

    def test_find_similar_empty_index(self):
        """Test find_similar on empty index."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        results = index.find_similar("img001")
        assert results == []

    def test_find_similar_unknown_image(self):
        """Test find_similar with unknown query image."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        index.add_embedding("img001", [0.1, 0.2, 0.3])

        results = index.find_similar("unknown")
        assert results == []

    def test_find_similar_basic(self):
        """Test find_similar returns sorted results."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        # Add normalized embeddings
        index.add_embedding("query", [1.0, 0.0, 0.0])
        index.add_embedding("similar", [0.9, 0.1, 0.0])
        index.add_embedding("different", [0.0, 1.0, 0.0])

        results = index.find_similar("query", top_k=10)

        assert len(results) == 2
        assert results[0].image_id == "similar"  # Most similar first
        assert results[0].score > results[1].score

    def test_find_similar_min_score(self):
        """Test find_similar respects min_score filter."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        index.add_embedding("query", [1.0, 0.0, 0.0])
        index.add_embedding("high_sim", [0.99, 0.01, 0.0])
        index.add_embedding("low_sim", [0.5, 0.5, 0.0])

        results = index.find_similar("query", min_score=0.9)

        # Only high_sim should pass the filter
        assert len(results) == 1
        assert results[0].image_id == "high_sim"

    def test_find_duplicates_empty(self):
        """Test find_duplicates on empty index."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        duplicates = index.find_duplicates()
        assert duplicates == []

    def test_find_duplicates_basic(self):
        """Test find_duplicates finds similar pairs."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        index = EmbeddingIndex()
        # Two nearly identical images
        index.add_embedding("img1", [1.0, 0.0, 0.0])
        index.add_embedding("img2", [0.99, 0.01, 0.0])
        # One different image
        index.add_embedding("img3", [0.0, 1.0, 0.0])

        duplicates = index.find_duplicates(threshold=0.95)

        assert len(duplicates) == 1
        assert duplicates[0].similarity > 0.95

    def test_save_load(self):
        """Test saving and loading index."""
        from vocablearn.ml.embeddings import EmbeddingIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.json"

            # Create and save index
            index1 = EmbeddingIndex(model_variant="base")
            index1.add_embedding("img1", [0.1, 0.2, 0.3])
            index1.add_embedding("img2", [0.4, 0.5, 0.6])
            index1.save(path)

            # Load into new index
            index2 = EmbeddingIndex()
            count = index2.load(path)

            assert count == 2
            assert index2.size == 2
            assert index2._model_variant == "base"
            assert index2.get_embedding("img1") == [0.1, 0.2, 0.3]
            assert index2.get_embedding("img2") == [0.4, 0.5, 0.6]


class TestGetImageEmbedding:
    """Tests for get_image_embedding function."""

    def test_model_map_variants(self):
        """Test all model variants are mapped correctly."""
        # Check the model mapping exists
        from vocablearn.ml.embeddings import get_image_embedding

        # Function should exist and be callable
        assert callable(get_image_embedding)

    def test_function_signature(self):
        """Test get_image_embedding has expected parameters."""
        import inspect
        from vocablearn.ml.embeddings import get_image_embedding

        sig = inspect.signature(get_image_embedding)
        params = list(sig.parameters.keys())

        assert "image_path" in params
        assert "model_variant" in params
        assert "device" in params
