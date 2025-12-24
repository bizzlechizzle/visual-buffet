"""Learning algorithms for vocablearn.

Provides confidence calibration, prior calculation, and active learning.
"""

from vocablearn.learning.active import UncertaintySampler, select_for_review
from vocablearn.learning.calibration import Calibrator, GlobalCalibrator, TagCalibrator
from vocablearn.learning.priors import calculate_prior, update_all_priors

__all__ = [
    "Calibrator",
    "GlobalCalibrator",
    "TagCalibrator",
    "calculate_prior",
    "update_all_priors",
    "select_for_review",
    "UncertaintySampler",
]
