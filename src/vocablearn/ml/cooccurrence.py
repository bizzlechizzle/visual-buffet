"""Tag co-occurrence analysis and PMI-based boosting.

Uses Pointwise Mutual Information (PMI) to identify tag co-occurrence
patterns and boost priors for related tags.

Example:
    >>> booster = CooccurrenceBooster(storage)
    >>> booster.build_cooccurrence_matrix()
    >>> related = booster.get_related_tags("restaurant")
    >>> # Returns: [("food", 0.85), ("table", 0.72), ("dining", 0.68), ...]
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CooccurrenceStats:
    """Statistics for tag co-occurrence."""

    tag_a: str
    tag_b: str
    joint_count: int  # Times both tags appear together
    tag_a_count: int  # Total occurrences of tag A
    tag_b_count: int  # Total occurrences of tag B
    total_images: int  # Total images in corpus
    pmi: float  # Pointwise Mutual Information
    npmi: float  # Normalized PMI (-1 to 1)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tag_a": self.tag_a,
            "tag_b": self.tag_b,
            "joint_count": self.joint_count,
            "tag_a_count": self.tag_a_count,
            "tag_b_count": self.tag_b_count,
            "total_images": self.total_images,
            "pmi": round(self.pmi, 4),
            "npmi": round(self.npmi, 4),
        }


def calculate_pmi(
    joint_count: int,
    tag_a_count: int,
    tag_b_count: int,
    total_images: int,
    smoothing: float = 0.0,
) -> tuple[float, float]:
    """Calculate PMI and NPMI for tag pair.

    PMI = log2(P(a,b) / (P(a) * P(b)))
    NPMI = PMI / -log2(P(a,b))  # Normalized to [-1, 1]

    Args:
        joint_count: Times both tags appear together
        tag_a_count: Total occurrences of tag A
        tag_b_count: Total occurrences of tag B
        total_images: Total images in corpus
        smoothing: Laplace smoothing (default 0)

    Returns:
        Tuple of (PMI, NPMI)
    """
    if total_images == 0 or joint_count == 0:
        return 0.0, 0.0

    # Apply smoothing
    joint = joint_count + smoothing
    count_a = tag_a_count + smoothing
    count_b = tag_b_count + smoothing
    total = total_images + smoothing * 4  # Adjust for smoothing

    # Calculate probabilities
    p_ab = joint / total
    p_a = count_a / total
    p_b = count_b / total

    # Avoid division by zero
    if p_a == 0 or p_b == 0:
        return 0.0, 0.0

    # PMI
    pmi = math.log2(p_ab / (p_a * p_b))

    # NPMI (normalized to [-1, 1])
    if p_ab > 0:
        npmi = pmi / -math.log2(p_ab)
    else:
        npmi = 0.0

    return pmi, npmi


class CooccurrenceBooster:
    """Tag co-occurrence analysis and prior boosting.

    Builds a co-occurrence matrix from vocabulary data and
    uses PMI to identify related tags for prior boosting.

    Example:
        >>> booster = CooccurrenceBooster(storage)
        >>> booster.build_cooccurrence_matrix()
        >>> # Get tags that often appear with "restaurant"
        >>> related = booster.get_related_tags("restaurant", top_k=10)
        >>> # Boost priors based on existing tags
        >>> boosts = booster.get_prior_boosts(["restaurant", "table"])
    """

    def __init__(self, storage):
        """Initialize the booster.

        Args:
            storage: SQLiteStorage instance
        """
        self._storage = storage
        self._cooccurrence: dict[tuple[int, int], int] = {}
        self._tag_counts: dict[int, int] = {}
        self._total_images = 0
        self._tag_id_to_label: dict[int, str] = {}
        self._label_to_tag_id: dict[str, int] = {}

    def build_cooccurrence_matrix(self, min_count: int = 2) -> int:
        """Build co-occurrence matrix from database.

        Args:
            min_count: Minimum joint count to include

        Returns:
            Number of co-occurrence pairs found
        """
        logger.info("Building co-occurrence matrix...")

        # Get all images and their tags
        with self._storage._connection() as conn:
            # Get unique images
            cursor = conn.execute("SELECT DISTINCT image_id FROM tag_events")
            image_ids = [row[0] for row in cursor.fetchall()]
            self._total_images = len(image_ids)

            # Get tag label mappings
            cursor = conn.execute("SELECT tag_id, label FROM vocabulary")
            for row in cursor.fetchall():
                self._tag_id_to_label[row[0]] = row[1]
                self._label_to_tag_id[row[1].lower()] = row[0]

            # Build image -> tags mapping
            image_tags: dict[str, set[int]] = defaultdict(set)
            cursor = conn.execute("SELECT image_id, tag_id FROM tag_events")
            for row in cursor.fetchall():
                image_tags[row[0]].add(row[1])

        # Count tag occurrences
        self._tag_counts.clear()
        for tag_set in image_tags.values():
            for tag_id in tag_set:
                self._tag_counts[tag_id] = self._tag_counts.get(tag_id, 0) + 1

        # Build co-occurrence matrix
        self._cooccurrence.clear()
        for tag_set in image_tags.values():
            tag_list = sorted(tag_set)
            for i, tag_a in enumerate(tag_list):
                for tag_b in tag_list[i+1:]:
                    key = (tag_a, tag_b)
                    self._cooccurrence[key] = self._cooccurrence.get(key, 0) + 1

        # Filter by min_count
        self._cooccurrence = {
            k: v for k, v in self._cooccurrence.items()
            if v >= min_count
        }

        logger.info(
            f"Built co-occurrence matrix: {len(self._cooccurrence)} pairs "
            f"from {self._total_images} images"
        )

        return len(self._cooccurrence)

    def get_related_tags(
        self,
        tag_label: str,
        top_k: int = 10,
        min_npmi: float = 0.1,
    ) -> list[tuple[str, float]]:
        """Get tags most related to the given tag.

        Args:
            tag_label: Tag to find related tags for
            top_k: Number of related tags to return
            min_npmi: Minimum NPMI threshold

        Returns:
            List of (tag_label, npmi) tuples, sorted by NPMI descending
        """
        tag_id = self._label_to_tag_id.get(tag_label.lower())
        if tag_id is None:
            return []

        tag_count = self._tag_counts.get(tag_id, 0)
        if tag_count == 0:
            return []

        # Find all co-occurrences with this tag
        related = []
        for (tag_a, tag_b), joint_count in self._cooccurrence.items():
            if tag_a == tag_id:
                other_id = tag_b
            elif tag_b == tag_id:
                other_id = tag_a
            else:
                continue

            other_count = self._tag_counts.get(other_id, 0)
            if other_count == 0:
                continue

            pmi, npmi = calculate_pmi(
                joint_count=joint_count,
                tag_a_count=tag_count,
                tag_b_count=other_count,
                total_images=self._total_images,
            )

            if npmi >= min_npmi:
                other_label = self._tag_id_to_label.get(other_id, f"tag_{other_id}")
                related.append((other_label, npmi))

        # Sort by NPMI descending
        related.sort(key=lambda x: x[1], reverse=True)

        return related[:top_k]

    def get_prior_boosts(
        self,
        existing_tags: list[str],
        boost_factor: float = 0.2,
        min_npmi: float = 0.2,
    ) -> dict[str, float]:
        """Get prior boosts for tags based on existing tags.

        Uses PMI to boost priors for tags that commonly co-occur
        with the existing tags.

        Args:
            existing_tags: Tags already assigned to the image
            boost_factor: Maximum boost to apply (0-1)
            min_npmi: Minimum NPMI to consider for boosting

        Returns:
            Dict mapping tag label to boost amount (add to prior)
        """
        boosts: dict[str, float] = defaultdict(float)

        for tag_label in existing_tags:
            related = self.get_related_tags(
                tag_label,
                top_k=20,
                min_npmi=min_npmi,
            )

            for related_label, npmi in related:
                # Skip if related tag is in existing tags
                if related_label.lower() in [t.lower() for t in existing_tags]:
                    continue

                # Boost proportional to NPMI
                boost = boost_factor * npmi
                boosts[related_label] = max(boosts[related_label], boost)

        return dict(boosts)

    def get_cooccurrence_stats(
        self,
        tag_a: str,
        tag_b: str,
    ) -> Optional[CooccurrenceStats]:
        """Get co-occurrence statistics for a tag pair.

        Args:
            tag_a: First tag label
            tag_b: Second tag label

        Returns:
            CooccurrenceStats or None if pair not found
        """
        id_a = self._label_to_tag_id.get(tag_a.lower())
        id_b = self._label_to_tag_id.get(tag_b.lower())

        if id_a is None or id_b is None:
            return None

        # Ensure consistent ordering
        if id_a > id_b:
            id_a, id_b = id_b, id_a
            tag_a, tag_b = tag_b, tag_a

        key = (id_a, id_b)
        joint_count = self._cooccurrence.get(key, 0)

        if joint_count == 0:
            return None

        count_a = self._tag_counts.get(id_a, 0)
        count_b = self._tag_counts.get(id_b, 0)

        pmi, npmi = calculate_pmi(
            joint_count=joint_count,
            tag_a_count=count_a,
            tag_b_count=count_b,
            total_images=self._total_images,
        )

        return CooccurrenceStats(
            tag_a=tag_a,
            tag_b=tag_b,
            joint_count=joint_count,
            tag_a_count=count_a,
            tag_b_count=count_b,
            total_images=self._total_images,
            pmi=pmi,
            npmi=npmi,
        )

    def export_top_pairs(
        self,
        top_k: int = 100,
        min_count: int = 5,
    ) -> list[dict]:
        """Export top co-occurring tag pairs.

        Args:
            top_k: Number of pairs to return
            min_count: Minimum joint count

        Returns:
            List of co-occurrence stats dicts, sorted by NPMI
        """
        pairs = []

        for (id_a, id_b), joint_count in self._cooccurrence.items():
            if joint_count < min_count:
                continue

            count_a = self._tag_counts.get(id_a, 0)
            count_b = self._tag_counts.get(id_b, 0)

            if count_a == 0 or count_b == 0:
                continue

            pmi, npmi = calculate_pmi(
                joint_count=joint_count,
                tag_a_count=count_a,
                tag_b_count=count_b,
                total_images=self._total_images,
            )

            label_a = self._tag_id_to_label.get(id_a, f"tag_{id_a}")
            label_b = self._tag_id_to_label.get(id_b, f"tag_{id_b}")

            pairs.append({
                "tag_a": label_a,
                "tag_b": label_b,
                "joint_count": joint_count,
                "pmi": round(pmi, 4),
                "npmi": round(npmi, 4),
            })

        # Sort by NPMI descending
        pairs.sort(key=lambda x: x["npmi"], reverse=True)

        return pairs[:top_k]
