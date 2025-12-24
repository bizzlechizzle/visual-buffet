"""Confidence calibration using isotonic regression.

Provides per-tag and global calibrators that learn from feedback
to map raw model confidence to calibrated confidence.
"""

import pickle
from abc import ABC, abstractmethod

# Numpy is optional - only needed for calibration fitting
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


class Calibrator(ABC):
    """Abstract base class for confidence calibrators."""

    @abstractmethod
    def fit(self, confidences: list[float], outcomes: list[bool]) -> None:
        """Fit calibrator from historical data.

        Args:
            confidences: Raw model confidence scores
            outcomes: Whether predictions were correct
        """
        pass

    @abstractmethod
    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence.

        Args:
            raw_confidence: Raw model confidence

        Returns:
            Calibrated confidence in [0, 1]
        """
        pass

    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize calibrator for storage."""
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: bytes) -> "Calibrator":
        """Deserialize calibrator from storage."""
        pass


class TagCalibrator(Calibrator):
    """Per-tag confidence calibrator using isotonic regression.

    Learns a mapping from raw confidence to calibrated confidence
    specific to a single tag.

    Example:
        >>> cal = TagCalibrator(tag_id=1, model="ram_plus")
        >>> cal.fit([0.5, 0.7, 0.9], [False, True, True])
        >>> cal.calibrate(0.8)
        0.85
    """

    def __init__(self, tag_id: int, model: str):
        """Initialize tag calibrator.

        Args:
            tag_id: Tag ID this calibrator is for
            model: Source model name
        """
        self.tag_id = tag_id
        self.model = model
        self._calibrator = None
        self.sample_count = 0
        self._is_fitted = False

    def fit(self, confidences: list[float], outcomes: list[bool]) -> None:
        """Fit calibrator from historical data.

        Uses isotonic regression to learn a monotonic mapping
        from raw confidence to actual accuracy.

        Args:
            confidences: Raw model confidence scores
            outcomes: Whether predictions were correct (True/False)
        """
        if len(confidences) < 10:
            # Not enough data for reliable calibration
            self._is_fitted = False
            return

        if not HAS_NUMPY:
            # numpy not available, fall back to no calibration
            self._is_fitted = False
            return

        try:
            from sklearn.isotonic import IsotonicRegression

            self._calibrator = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.01,  # Avoid exact 0
                y_max=0.99,  # Avoid exact 1
            )

            # Convert to numpy arrays
            X = np.array(confidences)
            y = np.array(outcomes, dtype=float)

            self._calibrator.fit(X, y)
            self.sample_count = len(confidences)
            self._is_fitted = True

        except ImportError:
            # sklearn not available, fall back to no calibration
            self._is_fitted = False

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence.

        If calibrator is not fitted, returns raw confidence unchanged.

        Args:
            raw_confidence: Raw model confidence

        Returns:
            Calibrated confidence in [0, 1]
        """
        if not self._is_fitted or self._calibrator is None:
            return raw_confidence

        # Clip input to valid range
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        result = self._calibrator.predict([raw_confidence])[0]
        return float(result)

    def serialize(self) -> bytes:
        """Serialize calibrator for storage."""
        return pickle.dumps({
            "tag_id": self.tag_id,
            "model": self.model,
            "calibrator": self._calibrator,
            "sample_count": self.sample_count,
            "is_fitted": self._is_fitted,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "TagCalibrator":
        """Deserialize calibrator from storage."""
        state = pickle.loads(data)
        cal = cls(tag_id=state["tag_id"], model=state["model"])
        cal._calibrator = state["calibrator"]
        cal.sample_count = state["sample_count"]
        cal._is_fitted = state["is_fitted"]
        return cal

    @property
    def is_fitted(self) -> bool:
        """Whether calibrator has been fitted."""
        return self._is_fitted


class GlobalCalibrator(Calibrator):
    """Global confidence calibrator for a model.

    Used when per-tag calibrators don't have enough data.
    Learns a mapping across all tags for a given model.

    Example:
        >>> cal = GlobalCalibrator(model="ram_plus")
        >>> cal.fit(all_confidences, all_outcomes)
        >>> cal.calibrate(0.8)
        0.78
    """

    def __init__(self, model: str):
        """Initialize global calibrator.

        Args:
            model: Source model name
        """
        self.model = model
        self._calibrator = None
        self.sample_count = 0
        self._is_fitted = False

    def fit(self, confidences: list[float], outcomes: list[bool]) -> None:
        """Fit calibrator from historical data across all tags.

        Args:
            confidences: Raw model confidence scores
            outcomes: Whether predictions were correct
        """
        if len(confidences) < 50:
            # Not enough data for reliable global calibration
            self._is_fitted = False
            return

        if not HAS_NUMPY:
            # numpy not available, fall back to no calibration
            self._is_fitted = False
            return

        try:
            from sklearn.isotonic import IsotonicRegression

            self._calibrator = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.01,
                y_max=0.99,
            )

            X = np.array(confidences)
            y = np.array(outcomes, dtype=float)

            self._calibrator.fit(X, y)
            self.sample_count = len(confidences)
            self._is_fitted = True

        except ImportError:
            self._is_fitted = False

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence.

        Args:
            raw_confidence: Raw model confidence

        Returns:
            Calibrated confidence in [0, 1]
        """
        if not self._is_fitted or self._calibrator is None:
            return raw_confidence

        raw_confidence = max(0.0, min(1.0, raw_confidence))
        result = self._calibrator.predict([raw_confidence])[0]
        return float(result)

    def serialize(self) -> bytes:
        """Serialize calibrator for storage."""
        return pickle.dumps({
            "model": self.model,
            "calibrator": self._calibrator,
            "sample_count": self.sample_count,
            "is_fitted": self._is_fitted,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "GlobalCalibrator":
        """Deserialize calibrator from storage."""
        state = pickle.loads(data)
        cal = cls(model=state["model"])
        cal._calibrator = state["calibrator"]
        cal.sample_count = state["sample_count"]
        cal._is_fitted = state["is_fitted"]
        return cal

    @property
    def is_fitted(self) -> bool:
        """Whether calibrator has been fitted."""
        return self._is_fitted


def compute_calibration_error(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error (ECE) and related metrics.

    Args:
        confidences: Predicted confidence scores
        outcomes: Actual outcomes (True/False)
        n_bins: Number of bins for ECE calculation

    Returns:
        Dict with ECE, MCE, and per-bin statistics
    """
    if len(confidences) == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    if not HAS_NUMPY:
        # Return empty result without numpy
        return {"ece": 0.0, "mce": 0.0, "bins": [], "error": "numpy not available"}

    confidences = np.array(confidences)
    outcomes = np.array(outcomes, dtype=float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    bins = []
    ece = 0.0
    mce = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not mask.any():
            continue

        bin_confidences = confidences[mask]
        bin_outcomes = outcomes[mask]

        avg_confidence = bin_confidences.mean()
        avg_accuracy = bin_outcomes.mean()
        bin_size = len(bin_confidences)

        calibration_error = abs(avg_accuracy - avg_confidence)

        ece += (bin_size / len(confidences)) * calibration_error
        mce = max(mce, calibration_error)

        bins.append({
            "bin": bin_idx,
            "lower": bin_boundaries[bin_idx],
            "upper": bin_boundaries[bin_idx + 1],
            "avg_confidence": float(avg_confidence),
            "avg_accuracy": float(avg_accuracy),
            "count": bin_size,
            "calibration_error": float(calibration_error),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins,
    }
