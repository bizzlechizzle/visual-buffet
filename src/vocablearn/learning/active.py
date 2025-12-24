"""Active learning for image tagging.

Provides sample selection strategies to prioritize human review
of the most informative images.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vocablearn.storage.sqlite import SQLiteStorage


@dataclass
class ReviewCandidate:
    """An image candidate for human review.

    Attributes:
        image_id: Image identifier
        uncertain_tags: List of tag labels that are uncertain
        uncertainty_score: Overall uncertainty score (higher = more uncertain)
        priority: Computed review priority
    """

    image_id: str
    uncertain_tags: list[str]
    uncertainty_score: float
    priority: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "uncertain_tags": self.uncertain_tags,
            "uncertainty_score": self.uncertainty_score,
            "priority": self.priority,
        }


class SamplingStrategy(ABC):
    """Abstract base class for active learning sampling strategies."""

    @abstractmethod
    def select(
        self,
        storage: SQLiteStorage,
        n: int,
    ) -> list[ReviewCandidate]:
        """Select images for review.

        Args:
            storage: Storage backend
            n: Number of images to select

        Returns:
            List of ReviewCandidate objects
        """
        pass


class UncertaintySampler(SamplingStrategy):
    """Uncertainty sampling strategy.

    Selects images where tag confidence is closest to the decision
    threshold (most uncertain predictions).

    This is the classic active learning approach that prioritizes
    samples near the decision boundary.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        uncertainty_range: float = 0.3,
    ):
        """Initialize uncertainty sampler.

        Args:
            threshold: Decision threshold (default 0.5)
            uncertainty_range: Range around threshold to consider uncertain
        """
        self.threshold = threshold
        self.uncertainty_range = uncertainty_range

    def select(
        self,
        storage: SQLiteStorage,
        n: int,
    ) -> list[ReviewCandidate]:
        """Select most uncertain images.

        Args:
            storage: Storage backend
            n: Number of images to select

        Returns:
            List of ReviewCandidate objects sorted by uncertainty
        """
        min_conf = self.threshold - self.uncertainty_range
        max_conf = self.threshold + self.uncertainty_range

        # Get unverified events in the uncertainty zone
        events = storage.get_unverified_events(
            min_confidence=min_conf,
            max_confidence=max_conf,
            limit=n * 10,  # Get more to group by image
        )

        # Group by image
        image_events = {}
        for event in events:
            if event.image_id not in image_events:
                image_events[event.image_id] = []
            image_events[event.image_id].append(event)

        # Calculate uncertainty score per image
        candidates = []
        for image_id, img_events in image_events.items():
            # Get tag labels
            uncertain_tags = []
            total_uncertainty = 0.0

            for event in img_events:
                tag = storage.get_tag_by_id(event.tag_id)
                if tag:
                    uncertain_tags.append(tag.label)
                # Uncertainty = distance from threshold
                uncertainty = 1.0 - abs(event.unified_confidence - self.threshold) / 0.5
                total_uncertainty += uncertainty

            avg_uncertainty = total_uncertainty / len(img_events) if img_events else 0

            candidates.append(ReviewCandidate(
                image_id=image_id,
                uncertain_tags=uncertain_tags,
                uncertainty_score=avg_uncertainty,
                priority=len(uncertain_tags),  # More uncertain tags = higher priority
            ))

        # Sort by uncertainty (highest first)
        candidates.sort(key=lambda c: (c.priority, c.uncertainty_score), reverse=True)

        return candidates[:n]


class DiversitySampler(SamplingStrategy):
    """Diversity sampling strategy.

    Selects images that cover a diverse range of tags,
    ensuring broad vocabulary coverage in the review set.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_confidence: float = 0.7,
    ):
        """Initialize diversity sampler.

        Args:
            min_confidence: Minimum confidence to consider
            max_confidence: Maximum confidence to consider
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def select(
        self,
        storage: SQLiteStorage,
        n: int,
    ) -> list[ReviewCandidate]:
        """Select diverse images.

        Uses a greedy algorithm to maximize tag coverage.

        Args:
            storage: Storage backend
            n: Number of images to select

        Returns:
            List of ReviewCandidate objects with diverse tag coverage
        """
        # Get unverified events
        events = storage.get_unverified_events(
            min_confidence=self.min_confidence,
            max_confidence=self.max_confidence,
            limit=10000,
        )

        # Group by image
        image_tags = {}
        for event in events:
            if event.image_id not in image_tags:
                image_tags[event.image_id] = set()
            image_tags[event.image_id].add(event.tag_id)

        # Greedy selection for maximum coverage
        selected = []
        covered_tags = set()

        while len(selected) < n and image_tags:
            # Find image that adds most new tags
            best_image = None
            best_new_tags = 0
            best_tags = set()

            for image_id, tags in image_tags.items():
                new_tags = len(tags - covered_tags)
                if new_tags > best_new_tags:
                    best_new_tags = new_tags
                    best_image = image_id
                    best_tags = tags

            if best_image is None:
                break

            # Get tag labels
            tag_labels = []
            for tag_id in best_tags:
                tag = storage.get_tag_by_id(tag_id)
                if tag:
                    tag_labels.append(tag.label)

            selected.append(ReviewCandidate(
                image_id=best_image,
                uncertain_tags=tag_labels,
                uncertainty_score=len(best_tags - covered_tags) / len(best_tags) if best_tags else 0,
                priority=best_new_tags,
            ))

            covered_tags.update(best_tags)
            del image_tags[best_image]

        return selected


class HighVolumeSampler(SamplingStrategy):
    """High volume tag sampling strategy.

    Prioritizes images containing high-volume uncertain tags,
    maximizing the impact of each review.
    """

    def __init__(
        self,
        min_occurrences: int = 50,
        confidence_threshold: float = 0.5,
    ):
        """Initialize high volume sampler.

        Args:
            min_occurrences: Minimum tag occurrences to consider high-volume
            confidence_threshold: Decision threshold
        """
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold

    def select(
        self,
        storage: SQLiteStorage,
        n: int,
    ) -> list[ReviewCandidate]:
        """Select images with high-volume uncertain tags.

        Args:
            storage: Storage backend
            n: Number of images to select

        Returns:
            List of ReviewCandidate objects prioritizing high-volume tags
        """
        # Get high-volume tags that need calibration
        tags_needing_cal = storage.get_tags_needing_calibration(
            min_occurrences=self.min_occurrences
        )
        high_volume_tag_ids = {t.tag_id for t in tags_needing_cal}

        if not high_volume_tag_ids:
            # Fall back to uncertainty sampling
            return UncertaintySampler().select(storage, n)

        # Get unverified events for high-volume tags
        events = storage.get_unverified_events(limit=10000)

        # Group by image, prioritizing high-volume tags
        image_scores = {}
        image_tags = {}

        for event in events:
            if event.image_id not in image_scores:
                image_scores[event.image_id] = 0
                image_tags[event.image_id] = []

            tag = storage.get_tag_by_id(event.tag_id)
            if tag:
                image_tags[event.image_id].append(tag.label)

            # Score based on tag volume
            if event.tag_id in high_volume_tag_ids:
                image_scores[event.image_id] += 10  # High priority
            else:
                image_scores[event.image_id] += 1

        # Sort by score
        sorted_images = sorted(
            image_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        candidates = []
        for image_id, score in sorted_images[:n]:
            candidates.append(ReviewCandidate(
                image_id=image_id,
                uncertain_tags=image_tags.get(image_id, []),
                uncertainty_score=score / 100.0,
                priority=score,
            ))

        return candidates


def select_for_review(
    storage: SQLiteStorage,
    n: int = 100,
    strategy: str = "uncertainty",
) -> list[ReviewCandidate]:
    """Select images for human review.

    Convenience function that selects the appropriate strategy.

    Args:
        storage: Storage backend
        n: Number of images to select
        strategy: Strategy name ('uncertainty', 'diversity', 'high_volume')

    Returns:
        List of ReviewCandidate objects
    """
    strategies = {
        "uncertainty": UncertaintySampler(),
        "diversity": DiversitySampler(),
        "high_volume": HighVolumeSampler(),
    }

    sampler = strategies.get(strategy, UncertaintySampler())
    return sampler.select(storage, n)
