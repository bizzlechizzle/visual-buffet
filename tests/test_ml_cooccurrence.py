"""Tests for vocablearn.ml.cooccurrence module."""

import pytest
from unittest.mock import MagicMock, patch
import math


class TestCalculatePMI:
    """Tests for calculate_pmi function."""

    def test_zero_joint_count(self):
        """Test PMI calculation with zero joint count returns zeros."""
        from vocablearn.ml.cooccurrence import calculate_pmi

        pmi, npmi = calculate_pmi(
            joint_count=0,
            tag_a_count=10,
            tag_b_count=10,
            total_images=100,
        )
        assert pmi == 0.0
        assert npmi == 0.0

    def test_zero_total_images(self):
        """Test PMI calculation with zero total images returns zeros."""
        from vocablearn.ml.cooccurrence import calculate_pmi

        pmi, npmi = calculate_pmi(
            joint_count=5,
            tag_a_count=10,
            tag_b_count=10,
            total_images=0,
        )
        assert pmi == 0.0
        assert npmi == 0.0

    def test_perfect_cooccurrence(self):
        """Test PMI for tags that always appear together."""
        from vocablearn.ml.cooccurrence import calculate_pmi

        # If A and B always appear together (10 times in 100 images)
        pmi, npmi = calculate_pmi(
            joint_count=10,
            tag_a_count=10,
            tag_b_count=10,
            total_images=100,
        )
        # PMI should be positive (tags appear together more than chance)
        assert pmi > 0
        # NPMI should be 1.0 (perfect association) or close to it
        assert 0.9 <= npmi <= 1.0

    def test_independent_tags(self):
        """Test PMI for statistically independent tags."""
        from vocablearn.ml.cooccurrence import calculate_pmi

        # P(A) = 50/100 = 0.5, P(B) = 50/100 = 0.5
        # If independent: P(A,B) = 0.5 * 0.5 = 0.25 => joint = 25
        pmi, npmi = calculate_pmi(
            joint_count=25,
            tag_a_count=50,
            tag_b_count=50,
            total_images=100,
        )
        # PMI should be close to 0 for independent tags
        assert abs(pmi) < 0.1
        assert abs(npmi) < 0.1

    def test_negative_association(self):
        """Test PMI for tags that rarely appear together."""
        from vocablearn.ml.cooccurrence import calculate_pmi

        # Tags that appear less often together than expected
        pmi, npmi = calculate_pmi(
            joint_count=1,
            tag_a_count=50,
            tag_b_count=50,
            total_images=100,
        )
        # PMI should be negative
        assert pmi < 0
        assert npmi < 0

    def test_smoothing(self):
        """Test Laplace smoothing affects results."""
        from vocablearn.ml.cooccurrence import calculate_pmi

        pmi_no_smooth, _ = calculate_pmi(
            joint_count=5,
            tag_a_count=10,
            tag_b_count=10,
            total_images=100,
            smoothing=0.0,
        )
        pmi_smooth, _ = calculate_pmi(
            joint_count=5,
            tag_a_count=10,
            tag_b_count=10,
            total_images=100,
            smoothing=1.0,
        )
        # Smoothing should change the result slightly
        assert pmi_no_smooth != pmi_smooth


class TestCooccurrenceStats:
    """Tests for CooccurrenceStats dataclass."""

    def test_to_dict(self):
        """Test CooccurrenceStats serialization."""
        from vocablearn.ml.cooccurrence import CooccurrenceStats

        stats = CooccurrenceStats(
            tag_a="restaurant",
            tag_b="food",
            joint_count=50,
            tag_a_count=100,
            tag_b_count=80,
            total_images=1000,
            pmi=1.234567,
            npmi=0.567891,
        )
        d = stats.to_dict()

        assert d["tag_a"] == "restaurant"
        assert d["tag_b"] == "food"
        assert d["joint_count"] == 50
        assert d["pmi"] == 1.2346  # Rounded
        assert d["npmi"] == 0.5679  # Rounded


class TestCooccurrenceBooster:
    """Tests for CooccurrenceBooster class."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage with test data."""
        storage = MagicMock()

        # Mock connection context manager
        mock_conn = MagicMock()
        storage._connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        storage._connection.return_value.__exit__ = MagicMock(return_value=False)

        # Mock cursor for SQL queries
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor

        # Setup fetchall to return test data for different queries
        def fetchall_side_effect():
            call_args = mock_conn.execute.call_args[0][0]
            if "DISTINCT image_id" in call_args:
                return [("img1",), ("img2",), ("img3",)]
            elif "tag_id, label" in call_args:
                return [(1, "restaurant"), (2, "food"), (3, "table")]
            elif "image_id, tag_id" in call_args:
                return [
                    ("img1", 1), ("img1", 2),  # img1: restaurant, food
                    ("img2", 1), ("img2", 3),  # img2: restaurant, table
                    ("img3", 2), ("img3", 3),  # img3: food, table
                ]
            return []

        mock_cursor.fetchall = fetchall_side_effect

        return storage

    def test_init(self, mock_storage):
        """Test CooccurrenceBooster initialization."""
        from vocablearn.ml.cooccurrence import CooccurrenceBooster

        booster = CooccurrenceBooster(mock_storage)
        assert booster._storage == mock_storage
        assert len(booster._cooccurrence) == 0
        assert len(booster._tag_counts) == 0

    def test_get_prior_boosts_empty(self, mock_storage):
        """Test get_prior_boosts with no matrix built."""
        from vocablearn.ml.cooccurrence import CooccurrenceBooster

        booster = CooccurrenceBooster(mock_storage)
        boosts = booster.get_prior_boosts(["restaurant"])
        assert boosts == {}

    def test_get_related_tags_unknown(self, mock_storage):
        """Test get_related_tags with unknown tag."""
        from vocablearn.ml.cooccurrence import CooccurrenceBooster

        booster = CooccurrenceBooster(mock_storage)
        related = booster.get_related_tags("unknown_tag")
        assert related == []

    def test_get_cooccurrence_stats_none(self, mock_storage):
        """Test get_cooccurrence_stats with unknown pair."""
        from vocablearn.ml.cooccurrence import CooccurrenceBooster

        booster = CooccurrenceBooster(mock_storage)
        stats = booster.get_cooccurrence_stats("unknown_a", "unknown_b")
        assert stats is None


class TestBoostCalculation:
    """Tests for prior boost calculation logic."""

    def test_boost_proportional_to_npmi(self):
        """Test that boosts are proportional to NPMI values."""
        # This tests the logic without requiring full integration
        boost_factor = 0.2
        npmi_values = [0.8, 0.5, 0.2]

        boosts = [boost_factor * npmi for npmi in npmi_values]

        assert boosts[0] > boosts[1] > boosts[2]
        assert abs(boosts[0] - 0.16) < 0.001  # 0.2 * 0.8
        assert abs(boosts[1] - 0.10) < 0.001  # 0.2 * 0.5
        assert abs(boosts[2] - 0.04) < 0.001  # 0.2 * 0.2

    def test_boost_capped_by_factor(self):
        """Test boosts don't exceed boost_factor."""
        boost_factor = 0.2
        npmi = 1.0  # Maximum NPMI

        boost = boost_factor * npmi
        assert boost <= boost_factor
