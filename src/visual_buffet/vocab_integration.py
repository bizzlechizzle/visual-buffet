"""VocabLearn integration for Visual Buffet.

Integrates vocabulary learning into the tagging pipeline:
- Records tags from tagging results
- Provides calibrated confidence scores
- Tracks model agreement
- Supports active learning for human review
"""

import hashlib
from pathlib import Path
from typing import Any, Optional

from vocablearn import (
    ConfidenceTier,
    Tag,
    TagSource,
    VocabLearn,
    VocabularyStats,
    calculate_unified_confidence,
    is_compound_tag,
)


class VocabIntegration:
    """Integrates VocabLearn with Visual Buffet tagging pipeline.

    Example:
        >>> vocab = VocabIntegration("./vocabulary.db")
        >>> # After tagging an image
        >>> vocab.record_tagging_result(image_path, tagging_result)
        >>> # Get calibrated tags
        >>> calibrated = vocab.get_calibrated_tags(tagging_result)
        >>> # Record human feedback
        >>> vocab.record_feedback(image_path, "abandoned_bar", correct=True)
    """

    def __init__(self, db_path: str | Path = "vocabulary.db"):
        """Initialize vocabulary integration.

        Args:
            db_path: Path to vocabulary database
        """
        self._vocab = VocabLearn(db_path)

    @property
    def vocab(self) -> VocabLearn:
        """Access underlying VocabLearn instance."""
        return self._vocab

    def _image_id(self, image_path: Path | str) -> str:
        """Generate consistent image ID from path.

        Uses the file's absolute path hash for consistency.
        """
        path = Path(image_path).resolve()
        return hashlib.sha256(str(path).encode()).hexdigest()[:16]

    def record_tagging_result(
        self,
        image_path: Path | str,
        result: dict[str, Any],
    ) -> int:
        """Record tags from a tagging result.

        Parses the visual-buffet tagging output and records
        all tags with their model-specific confidence data.

        Args:
            image_path: Path to the tagged image
            result: Tagging result from TaggingEngine.tag_image()

        Returns:
            Number of tag events recorded
        """
        image_id = self._image_id(image_path)
        results = result.get("results", {})

        # Collect tags from each model
        ram_tags: dict[str, float] = {}  # label -> confidence
        florence_tags: set[str] = set()
        siglip_scores: dict[str, float] = {}  # label -> confidence

        # Extract RAM++ tags
        if "ram_plus" in results and "tags" in results["ram_plus"]:
            for tag in results["ram_plus"]["tags"]:
                label = tag.get("label", tag.get("name", ""))
                confidence = tag.get("confidence", 0.5)
                if label:
                    ram_tags[label] = confidence

        # Extract Florence-2 tags
        if "florence_2" in results and "tags" in results["florence_2"]:
            for tag in results["florence_2"]["tags"]:
                label = tag.get("label", tag.get("name", ""))
                if label:
                    florence_tags.add(label)

        # Extract SigLIP scores
        if "siglip" in results and "tags" in results["siglip"]:
            for tag in results["siglip"]["tags"]:
                label = tag.get("label", tag.get("name", ""))
                confidence = tag.get("confidence", 0.0)
                if label:
                    siglip_scores[label] = confidence

        # Build unified tag set with multi-model data
        all_labels = set(ram_tags.keys()) | florence_tags | set(siglip_scores.keys())
        tags_to_record = []

        for label in all_labels:
            tag_data = {
                "label": label,
                "ram_confidence": ram_tags.get(label),
                "florence_found": label in florence_tags,
                "siglip_confidence": siglip_scores.get(label),
            }

            # Set primary confidence based on available data
            if label in ram_tags:
                tag_data["confidence"] = ram_tags[label]
            elif label in siglip_scores:
                tag_data["confidence"] = siglip_scores[label]

            tags_to_record.append(tag_data)

        # Record to vocabulary
        if tags_to_record:
            event_ids = self._vocab.record_tags(
                image_id=image_id,
                tags=tags_to_record,
                source="ensemble",
            )
            return len(event_ids)

        return 0

    def get_calibrated_tags(
        self,
        result: dict[str, Any],
        min_confidence: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Get calibrated tags with unified confidence.

        Takes raw tagging results and returns a unified tag list
        with calibrated confidence scores and tier assignments.

        Args:
            result: Tagging result from TaggingEngine.tag_image()
            min_confidence: Minimum unified confidence to include

        Returns:
            List of calibrated tag dicts with:
            - label: Tag text
            - unified_confidence: Calibrated confidence (0-1)
            - confidence_tier: Tier name (GOLD, HIGH, MEDIUM, CONTEXTUAL, LOW)
            - sources: List of models that found this tag
            - ram_confidence: RAM++ score if available
            - florence_found: Whether Florence-2 found it
            - siglip_confidence: SigLIP verification score if available
        """
        results = result.get("results", {})

        # Collect data from each model
        ram_tags: dict[str, float] = {}
        florence_tags: set[str] = set()
        siglip_scores: dict[str, float] = {}

        if "ram_plus" in results and "tags" in results["ram_plus"]:
            for tag in results["ram_plus"]["tags"]:
                label = tag.get("label", tag.get("name", ""))
                if label:
                    ram_tags[label] = tag.get("confidence", 0.5)

        if "florence_2" in results and "tags" in results["florence_2"]:
            for tag in results["florence_2"]["tags"]:
                label = tag.get("label", tag.get("name", ""))
                if label:
                    florence_tags.add(label)

        if "siglip" in results and "tags" in results["siglip"]:
            for tag in results["siglip"]["tags"]:
                label = tag.get("label", tag.get("name", ""))
                if label:
                    siglip_scores[label] = tag.get("confidence", 0.0)

        # Build calibrated tag list
        calibrated_tags = []
        all_labels = set(ram_tags.keys()) | florence_tags | set(siglip_scores.keys())

        for label in all_labels:
            ram_conf = ram_tags.get(label)
            florence_found = label in florence_tags
            siglip_conf = siglip_scores.get(label)
            is_compound = is_compound_tag(label)

            # Get prior from vocabulary
            vocab_tag = self._vocab.get_tag(label)
            prior = vocab_tag.prior_confidence if vocab_tag else 0.5

            # Calculate unified confidence
            unified = calculate_unified_confidence(
                ram_confidence=ram_conf,
                florence_found=florence_found,
                siglip_confidence=siglip_conf,
                is_compound=is_compound,
                prior_confidence=prior,
            )

            if unified < min_confidence:
                continue

            # Determine tier
            tier = ConfidenceTier.from_scores(
                ram_confidence=ram_conf,
                florence_found=florence_found,
                siglip_confidence=siglip_conf,
                is_compound=is_compound,
            )

            # Build source list
            sources = []
            if ram_conf is not None:
                sources.append("ram_plus")
            if florence_found:
                sources.append("florence_2")
            if siglip_conf is not None:
                sources.append("siglip")

            calibrated_tags.append({
                "label": label,
                "unified_confidence": round(unified, 4),
                "confidence_tier": tier.name,
                "sources": sources,
                "ram_confidence": ram_conf,
                "florence_found": florence_found,
                "siglip_confidence": siglip_conf,
                "is_compound": is_compound,
                "prior_confidence": prior,
            })

        # Sort by unified confidence (highest first)
        calibrated_tags.sort(key=lambda t: t["unified_confidence"], reverse=True)

        return calibrated_tags

    def record_feedback(
        self,
        image_path: Path | str,
        tag_label: str,
        correct: bool,
        user: Optional[str] = None,
    ) -> None:
        """Record human feedback on a tag.

        Args:
            image_path: Path to the image
            tag_label: Tag that was verified
            correct: Whether the tag was correct
            user: Optional user identifier
        """
        image_id = self._image_id(image_path)
        self._vocab.record_feedback(
            image_id=image_id,
            tag_label=tag_label,
            correct=correct,
            verified_by=user,
        )

    def get_tag_info(self, label: str) -> Optional[dict[str, Any]]:
        """Get vocabulary information for a tag.

        Args:
            label: Tag label

        Returns:
            Dict with tag statistics or None if not found
        """
        tag = self._vocab.get_tag(label)
        if tag is None:
            return None
        return tag.to_dict()

    def get_statistics(self) -> dict[str, Any]:
        """Get overall vocabulary statistics."""
        return self._vocab.get_statistics()

    def update_learning(self, min_samples: int = 5) -> dict[str, int]:
        """Update priors and calibrators from feedback.

        Should be called periodically to learn from accumulated feedback.

        Args:
            min_samples: Minimum samples required for updates

        Returns:
            Dict with counts of updated priors and calibrators
        """
        priors_updated = self._vocab.update_priors(min_samples=min_samples)
        calibrators_rebuilt = self._vocab.rebuild_calibrators(min_samples=min_samples * 10)

        return {
            "priors_updated": priors_updated,
            "calibrators_rebuilt": calibrators_rebuilt,
        }

    def select_for_review(
        self,
        n: int = 10,
        strategy: str = "uncertainty",
    ) -> list[dict[str, Any]]:
        """Select images for human review.

        Args:
            n: Number of images to select
            strategy: Selection strategy (uncertainty, diversity, high_volume)

        Returns:
            List of review candidate dicts
        """
        return self._vocab.select_for_review(n=n, strategy=strategy)

    def export_vocabulary(self, path: Path | str) -> None:
        """Export vocabulary to JSON file."""
        self._vocab.export_vocabulary(path)

    def import_vocabulary(self, path: Path | str, merge: bool = True) -> int:
        """Import vocabulary from JSON file."""
        return self._vocab.import_vocabulary(path, merge=merge)
