"""Data models for vocablearn.

Defines core entities for vocabulary tracking and learning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TagSource(str, Enum):
    """Source model that generated the tag."""

    RAM_PLUS = "ram_plus"
    FLORENCE_2 = "florence_2"
    SIGLIP = "siglip"
    HUMAN = "human"
    ENSEMBLE = "ensemble"


class ConfidenceTier(int, Enum):
    """Computed confidence tier for display.

    Tiers are ordered from highest to lowest confidence.
    """

    GOLD = 1  # Model agreement (both RAM++ and Florence-2)
    HIGH = 2  # RAM++ >= 0.7
    MEDIUM = 3  # RAM++ 0.5-0.7
    CONTEXTUAL = 4  # Florence-2 compound + SigLIP verified
    LOW = 5  # Low confidence / unverified

    @classmethod
    def from_scores(
        cls,
        ram_confidence: Optional[float],
        florence_found: bool,
        siglip_confidence: Optional[float] = None,
        is_compound: bool = False,
    ) -> "ConfidenceTier":
        """Determine tier from model scores.

        Args:
            ram_confidence: RAM++ confidence score (0-1)
            florence_found: Whether Florence-2 found this tag
            siglip_confidence: SigLIP verification score (0-1)
            is_compound: Whether tag is a compound phrase

        Returns:
            Appropriate confidence tier
        """
        # Tier 1: Model agreement
        if ram_confidence is not None and ram_confidence >= 0.5 and florence_found:
            return cls.GOLD

        # Tier 2: High RAM++ confidence
        if ram_confidence is not None and ram_confidence >= 0.7:
            return cls.HIGH

        # Tier 3: Medium RAM++ confidence
        if ram_confidence is not None and ram_confidence >= 0.5:
            return cls.MEDIUM

        # Tier 4: Florence-2 compound with SigLIP verification
        if florence_found and is_compound:
            if siglip_confidence is not None and siglip_confidence >= 0.10:
                return cls.CONTEXTUAL

        # Tier 5: Everything else
        return cls.LOW


@dataclass
class Tag:
    """A vocabulary entry with learned statistics.

    Attributes:
        tag_id: Unique identifier
        label: Original tag text
        normalized: Lowercase, trimmed version for matching
        is_compound: Whether tag contains underscore (compound phrase)
        total_occurrences: Total times this tag has been assigned
        confirmed_correct: Human-verified correct count
        confirmed_incorrect: Human-verified incorrect count
        prior_confidence: Learned confidence prior (0-1)
        confidence_lower: Lower bound of confidence interval
        confidence_upper: Upper bound of confidence interval
        ram_plus_hits: Times found by RAM++
        florence_2_hits: Times found by Florence-2
        siglip_verified: Times verified by SigLIP (>= 0.10)
        model_agreement_count: Times found by both RAM++ and Florence-2
        first_seen: First time this tag was recorded
        last_seen: Most recent time this tag was recorded
    """

    tag_id: int
    label: str
    normalized: str
    is_compound: bool = False

    # Aggregate statistics
    total_occurrences: int = 0
    confirmed_correct: int = 0
    confirmed_incorrect: int = 0

    # Computed confidence
    prior_confidence: float = 0.5
    confidence_lower: float = 0.0
    confidence_upper: float = 1.0

    # Model-specific statistics
    ram_plus_hits: int = 0
    florence_2_hits: int = 0
    siglip_verified: int = 0
    model_agreement_count: int = 0

    # Metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    @property
    def sample_size(self) -> int:
        """Total human feedback samples for this tag."""
        return self.confirmed_correct + self.confirmed_incorrect

    @property
    def accuracy_rate(self) -> float:
        """Historical accuracy rate from feedback.

        Returns 0.5 if no feedback exists (uninformative prior).
        """
        if self.sample_size == 0:
            return 0.5
        return self.confirmed_correct / self.sample_size

    @property
    def agreement_rate(self) -> float:
        """Rate of model agreement (both models found tag).

        Returns 0.0 if tag has never been seen.
        """
        if self.total_occurrences == 0:
            return 0.0
        return self.model_agreement_count / self.total_occurrences

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tag_id": self.tag_id,
            "label": self.label,
            "normalized": self.normalized,
            "is_compound": self.is_compound,
            "total_occurrences": self.total_occurrences,
            "confirmed_correct": self.confirmed_correct,
            "confirmed_incorrect": self.confirmed_incorrect,
            "prior_confidence": self.prior_confidence,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "ram_plus_hits": self.ram_plus_hits,
            "florence_2_hits": self.florence_2_hits,
            "siglip_verified": self.siglip_verified,
            "model_agreement_count": self.model_agreement_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class TagEvent:
    """A single tag assignment event.

    Records when a tag was assigned to an image, including
    source model, confidence scores, and any human feedback.

    Attributes:
        event_id: Unique identifier
        image_id: Image this tag was assigned to
        tag_id: Reference to vocabulary entry
        source: Source model that generated tag
        raw_confidence: Original model confidence
        unified_confidence: Calibrated/combined confidence
        confidence_tier: Computed tier (1-5)
        ram_confidence: RAM++ confidence if available
        florence_found: Whether Florence-2 found this tag
        siglip_confidence: SigLIP verification score if available
        human_verified: Whether human has reviewed this
        human_correct: Human verdict (True/False/None)
        verified_at: When human verification occurred
        verified_by: User who verified
        tagged_at: When tag was assigned
    """

    event_id: int
    image_id: str
    tag_id: int

    # Source and confidence
    source: TagSource
    raw_confidence: Optional[float]
    unified_confidence: float
    confidence_tier: ConfidenceTier

    # Multi-model data
    ram_confidence: Optional[float] = None
    florence_found: bool = False
    siglip_confidence: Optional[float] = None

    # Feedback
    human_verified: bool = False
    human_correct: Optional[bool] = None
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None

    # Metadata
    tagged_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "image_id": self.image_id,
            "tag_id": self.tag_id,
            "source": self.source.value,
            "raw_confidence": self.raw_confidence,
            "unified_confidence": self.unified_confidence,
            "confidence_tier": self.confidence_tier.value,
            "ram_confidence": self.ram_confidence,
            "florence_found": self.florence_found,
            "siglip_confidence": self.siglip_confidence,
            "human_verified": self.human_verified,
            "human_correct": self.human_correct,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verified_by": self.verified_by,
            "tagged_at": self.tagged_at.isoformat() if self.tagged_at else None,
        }


@dataclass
class CalibrationPoint:
    """A data point for calibration curve fitting.

    Used to train isotonic regression calibrators.

    Attributes:
        tag_id: Reference to vocabulary entry
        model: Source model ('ram_plus', 'florence_2', etc.)
        raw_confidence: Raw model confidence score
        was_correct: Whether tag was correct (from feedback)
        recorded_at: When this data point was recorded
    """

    tag_id: int
    model: str
    raw_confidence: float
    was_correct: bool
    recorded_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tag_id": self.tag_id,
            "model": self.model,
            "raw_confidence": self.raw_confidence,
            "was_correct": self.was_correct,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
        }


@dataclass
class VocabularyStats:
    """Overall vocabulary statistics.

    Summary statistics for the entire vocabulary database.
    """

    total_vocabulary: int = 0
    total_events: int = 0
    total_feedback: int = 0
    calibrated_tags: int = 0
    avg_prior_confidence: float = 0.5

    # Per-model statistics
    ram_plus_unique_tags: int = 0
    florence_2_unique_tags: int = 0
    model_agreement_tags: int = 0

    # Tier distribution
    tier_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_vocabulary": self.total_vocabulary,
            "total_events": self.total_events,
            "total_feedback": self.total_feedback,
            "calibrated_tags": self.calibrated_tags,
            "avg_prior_confidence": self.avg_prior_confidence,
            "ram_plus_unique_tags": self.ram_plus_unique_tags,
            "florence_2_unique_tags": self.florence_2_unique_tags,
            "model_agreement_tags": self.model_agreement_tags,
            "tier_counts": self.tier_counts,
        }


def normalize_tag(label: str) -> str:
    """Normalize a tag label for consistent matching.

    Args:
        label: Raw tag label

    Returns:
        Normalized label (lowercase, trimmed, spaces to underscores)
    """
    return label.lower().strip().replace(" ", "_")


def is_compound_tag(label: str) -> bool:
    """Check if a tag is a compound phrase.

    Args:
        label: Tag label

    Returns:
        True if tag contains underscore or multiple words
    """
    return "_" in label or " " in label


def calculate_unified_confidence(
    ram_confidence: Optional[float],
    florence_found: bool,
    siglip_confidence: Optional[float] = None,
    is_compound: bool = False,
    prior_confidence: float = 0.5,
) -> float:
    """Calculate unified confidence score.

    Combines multiple model signals into a single confidence score.

    Args:
        ram_confidence: RAM++ confidence (0-1)
        florence_found: Whether Florence-2 found tag
        siglip_confidence: SigLIP verification score (0-1)
        is_compound: Whether tag is compound phrase
        prior_confidence: Learned prior from vocabulary

    Returns:
        Unified confidence in [0, 1]
    """
    # Tier 1: Model agreement (gold standard)
    if ram_confidence is not None and ram_confidence >= 0.5 and florence_found:
        # Base high confidence, boost slightly by RAM++
        return min(1.0, 0.95 + (ram_confidence * 0.05))

    # Tier 2/3: RAM++ only
    if ram_confidence is not None and ram_confidence >= 0.5:
        # Map RAM++ 0.5-1.0 to our 0.6-0.9 range
        return 0.6 + (ram_confidence - 0.5) * 0.6

    # Tier 4: Florence-2 compound phrases
    if florence_found and is_compound:
        if siglip_confidence is not None and siglip_confidence >= 0.10:
            # SigLIP verified compound phrase
            return 0.70 + (siglip_confidence * 0.20)
        else:
            # Unverified compound phrase - use prior
            return max(0.40, prior_confidence * 0.8)

    # Florence-2 single word (no RAM++ confirmation)
    if florence_found:
        return max(0.30, prior_confidence * 0.6)

    # Fallback: use prior
    return prior_confidence * 0.5
