"""Tests for vocablearn.ml.classifier module."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_to_dict(self):
        """Test ClassificationResult serialization."""
        from vocablearn.ml.classifier import ClassificationResult

        result = ClassificationResult(
            label="indoor",
            confidence=0.8567,
            all_scores={"indoor": 0.8567, "outdoor": 0.1234},
            method="siglip_zeroshot",
        )
        d = result.to_dict()

        assert d["label"] == "indoor"
        assert d["confidence"] == 0.8567
        assert d["all_scores"]["indoor"] == 0.8567
        assert d["all_scores"]["outdoor"] == 0.1234
        assert d["method"] == "siglip_zeroshot"

    def test_to_dict_rounding(self):
        """Test that confidence values are rounded."""
        from vocablearn.ml.classifier import ClassificationResult

        result = ClassificationResult(
            label="outdoor",
            confidence=0.123456789,
            all_scores={"outdoor": 0.123456789},
        )
        d = result.to_dict()

        assert d["confidence"] == 0.1235
        assert d["all_scores"]["outdoor"] == 0.1235


class TestSceneCategories:
    """Tests for pre-defined scene categories."""

    def test_indoor_outdoor_exists(self):
        """Test indoor_outdoor category exists with proper structure."""
        from vocablearn.ml.classifier import SCENE_CATEGORIES

        assert "indoor_outdoor" in SCENE_CATEGORIES
        assert "indoor" in SCENE_CATEGORIES["indoor_outdoor"]
        assert "outdoor" in SCENE_CATEGORIES["indoor_outdoor"]
        assert len(SCENE_CATEGORIES["indoor_outdoor"]["indoor"]) >= 2
        assert len(SCENE_CATEGORIES["indoor_outdoor"]["outdoor"]) >= 2

    def test_time_of_day_exists(self):
        """Test time_of_day category exists."""
        from vocablearn.ml.classifier import SCENE_CATEGORIES

        assert "time_of_day" in SCENE_CATEGORIES
        assert "daytime" in SCENE_CATEGORIES["time_of_day"]
        assert "nighttime" in SCENE_CATEGORIES["time_of_day"]

    def test_photo_type_exists(self):
        """Test photo_type category exists."""
        from vocablearn.ml.classifier import SCENE_CATEGORIES

        assert "photo_type" in SCENE_CATEGORIES
        assert "portrait" in SCENE_CATEGORIES["photo_type"]
        assert "landscape" in SCENE_CATEGORIES["photo_type"]


class TestSceneClassifier:
    """Tests for SceneClassifier class."""

    def test_init(self):
        """Test classifier initialization."""
        from vocablearn.ml.classifier import SceneClassifier

        classifier = SceneClassifier(model_variant="so400m", device="cpu")
        assert classifier._model_variant == "so400m"
        assert classifier._device == "cpu"
        assert classifier._model is None  # Lazy loading

    def test_model_variants(self):
        """Test different model variants are accepted."""
        from vocablearn.ml.classifier import SceneClassifier

        for variant in ["base", "so400m", "base-v2", "large-v2", "so400m-v2"]:
            classifier = SceneClassifier(model_variant=variant)
            assert classifier._model_variant == variant

    def test_lazy_model_loading(self):
        """Test model is loaded lazily on first use."""
        from vocablearn.ml.classifier import SceneClassifier

        classifier = SceneClassifier()
        assert classifier._model is None
        # Model should remain None until classify is called


class TestClassifySceneFunction:
    """Tests for classify_scene convenience function."""

    def test_function_exists(self):
        """Test that classify_scene function exists and is callable."""
        from vocablearn.ml.classifier import classify_scene

        assert callable(classify_scene)

    def test_valid_categories(self):
        """Test valid category names are accepted."""
        from vocablearn.ml.classifier import SCENE_CATEGORIES

        valid_categories = ["indoor_outdoor", "time_of_day", "photo_type"]
        for cat in valid_categories:
            assert cat in SCENE_CATEGORIES


class TestPromptEnsembling:
    """Tests for prompt ensembling logic."""

    def test_multiple_prompts_per_label(self):
        """Test each label has multiple prompts for ensembling."""
        from vocablearn.ml.classifier import SCENE_CATEGORIES

        for category_name, labels in SCENE_CATEGORIES.items():
            for label, prompts in labels.items():
                assert len(prompts) >= 2, f"{category_name}.{label} needs 2+ prompts"
                for prompt in prompts:
                    assert isinstance(prompt, str)
                    assert len(prompt) > 5  # Non-trivial prompts

    def test_prompts_are_non_empty_strings(self):
        """Test prompts are non-empty strings."""
        from vocablearn.ml.classifier import SCENE_CATEGORIES

        for category_name, labels in SCENE_CATEGORIES.items():
            for label, prompts in labels.items():
                for prompt in prompts:
                    assert isinstance(prompt, str)
                    assert len(prompt) >= 5  # Minimum reasonable length
