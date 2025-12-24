"""High-level API for vocablearn.

Provides a simple interface for vocabulary tracking, learning,
and calibration.
"""

import json
from pathlib import Path

from vocablearn.learning.active import select_for_review
from vocablearn.learning.calibration import GlobalCalibrator, TagCalibrator
from vocablearn.learning.priors import update_all_priors
from vocablearn.models import (
    ConfidenceTier,
    Tag,
    TagEvent,
    TagSource,
    calculate_unified_confidence,
    is_compound_tag,
)
from vocablearn.storage.sqlite import SQLiteStorage


class VocabLearn:
    """High-level API for vocabulary tracking and learning.

    Provides a simple interface for:
    - Recording tag assignments
    - Recording human feedback
    - Getting calibrated confidence scores
    - Active learning sample selection
    - Vocabulary export/import

    Example:
        >>> vocab = VocabLearn("./vocab.db")
        >>> vocab.record_tags("img001.jpg", [{"label": "bar", "confidence": 0.95}], "ram_plus")
        >>> vocab.record_feedback("img001.jpg", "bar", correct=True)
        >>> conf = vocab.get_calibrated_confidence("bar", 0.8, "ram_plus")
    """

    def __init__(self, db_path: str | Path):
        """Initialize VocabLearn.

        Args:
            db_path: Path to SQLite database (created if not exists)
        """
        self.db_path = Path(db_path)
        self._storage = SQLiteStorage(db_path)
        self._tag_calibrators: dict[tuple[int, str], TagCalibrator] = {}
        self._global_calibrators: dict[str, GlobalCalibrator] = {}

    # =========================================================================
    # INGESTION
    # =========================================================================

    def record_tags(
        self,
        image_id: str,
        tags: list[dict],
        source: str,
    ) -> list[int]:
        """Record tag assignments for an image.

        Args:
            image_id: Unique image identifier
            tags: List of tag dicts with keys:
                - label: Tag text (required)
                - confidence: Raw model confidence (optional)
                - ram_confidence: RAM++ confidence (optional)
                - florence_found: Whether Florence-2 found it (optional)
                - siglip_confidence: SigLIP verification score (optional)
            source: Source model ('ram_plus', 'florence_2', 'ensemble')

        Returns:
            List of event_ids for recorded tags

        Example:
            >>> vocab.record_tags("img001.jpg", [
            ...     {"label": "bar", "confidence": 0.95},
            ...     {"label": "stool", "confidence": 0.88},
            ... ], "ram_plus")
        """
        event_ids = []
        source_enum = TagSource(source)

        for tag_data in tags:
            label = tag_data["label"]
            tag_id = self._storage.get_or_create_tag(label)

            # Extract confidence scores
            raw_conf = tag_data.get("confidence")
            ram_conf = tag_data.get("ram_confidence", raw_conf if source == "ram_plus" else None)
            florence_found = tag_data.get("florence_found", source == "florence_2")
            siglip_conf = tag_data.get("siglip_confidence")

            # Get prior for this tag
            tag = self._storage.get_tag_by_id(tag_id)
            prior = tag.prior_confidence if tag else 0.5

            # Calculate unified confidence
            unified_conf = calculate_unified_confidence(
                ram_confidence=ram_conf,
                florence_found=florence_found,
                siglip_confidence=siglip_conf,
                is_compound=is_compound_tag(label),
                prior_confidence=prior,
            )

            # Determine tier
            tier = ConfidenceTier.from_scores(
                ram_confidence=ram_conf,
                florence_found=florence_found,
                siglip_confidence=siglip_conf,
                is_compound=is_compound_tag(label),
            )

            # Record event
            event_id = self._storage.record_event(
                image_id=image_id,
                tag_id=tag_id,
                source=source_enum,
                unified_confidence=unified_conf,
                confidence_tier=tier,
                raw_confidence=raw_conf,
                ram_confidence=ram_conf,
                florence_found=florence_found,
                siglip_confidence=siglip_conf,
            )
            event_ids.append(event_id)

            # Update tag statistics
            self._storage.update_tag_statistics(
                tag_id=tag_id,
                increment_occurrences=1,
                increment_ram_plus=1 if source == "ram_plus" else 0,
                increment_florence_2=1 if source == "florence_2" or florence_found else 0,
                increment_siglip=1 if siglip_conf and siglip_conf >= 0.10 else 0,
                increment_agreement=1 if ram_conf and florence_found else 0,
            )

        return event_ids

    def record_feedback(
        self,
        image_id: str,
        tag_label: str,
        correct: bool,
        verified_by: str | None = None,
    ) -> None:
        """Record human feedback on a tag.

        Updates tag event, vocabulary statistics, and calibration data.

        Args:
            image_id: Image identifier
            tag_label: Tag that was verified
            correct: Whether tag was correct
            verified_by: Optional user identifier

        Example:
            >>> vocab.record_feedback("img001.jpg", "bar", correct=True)
        """
        # Get tag
        tag = self._storage.get_tag(tag_label)
        if tag is None:
            return

        # Get event
        event = self._storage.get_event(image_id, tag.tag_id)
        if event is None:
            return

        # Record feedback on event
        self._storage.record_feedback(
            image_id=image_id,
            tag_id=tag.tag_id,
            correct=correct,
            verified_by=verified_by,
        )

        # Update vocabulary statistics
        self._storage.update_tag_statistics(
            tag_id=tag.tag_id,
            increment_correct=1 if correct else 0,
            increment_incorrect=0 if correct else 1,
        )

        # Record calibration data point
        if event.raw_confidence is not None:
            self._storage.record_calibration_point(
                tag_id=tag.tag_id,
                model=event.source.value,
                raw_confidence=event.raw_confidence,
                was_correct=correct,
            )

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_tag(self, label: str) -> Tag | None:
        """Get vocabulary entry for a tag.

        Args:
            label: Tag label (case-insensitive)

        Returns:
            Tag object with learned statistics, or None if not found
        """
        return self._storage.get_tag(label)

    def get_calibrated_confidence(
        self,
        label: str,
        raw_confidence: float,
        source: str,
    ) -> float:
        """Get calibrated confidence for a tag.

        Uses isotonic regression if sufficient calibration data exists,
        otherwise returns prior-adjusted raw confidence.

        Args:
            label: Tag label
            raw_confidence: Raw model confidence
            source: Source model

        Returns:
            Calibrated confidence in [0, 1]
        """
        tag = self._storage.get_tag(label)
        if tag is None:
            return raw_confidence

        # Try tag-specific calibrator first
        key = (tag.tag_id, source)
        if key not in self._tag_calibrators:
            # Try to load from storage
            model_data = self._storage.get_calibration_model(tag.tag_id, source)
            if model_data:
                self._tag_calibrators[key] = TagCalibrator.deserialize(model_data)

        if key in self._tag_calibrators and self._tag_calibrators[key].is_fitted:
            return self._tag_calibrators[key].calibrate(raw_confidence)

        # Try global calibrator
        if source not in self._global_calibrators:
            model_data = self._storage.get_calibration_model(None, source)
            if model_data:
                self._global_calibrators[source] = GlobalCalibrator.deserialize(model_data)

        if source in self._global_calibrators and self._global_calibrators[source].is_fitted:
            return self._global_calibrators[source].calibrate(raw_confidence)

        # No calibration available - use prior-weighted confidence
        # Blend raw confidence with prior
        prior = tag.prior_confidence
        weight = min(1.0, tag.sample_size / 50.0)  # Trust prior more with more samples
        return raw_confidence * (1 - weight * 0.3) + prior * (weight * 0.3)

    def get_tags_for_image(self, image_id: str) -> list[TagEvent]:
        """Get all tags assigned to an image.

        Args:
            image_id: Image identifier

        Returns:
            List of TagEvent objects
        """
        return self._storage.get_events_for_image(image_id)

    def search_vocabulary(
        self,
        query: str | None = None,
        min_occurrences: int = 0,
        min_confidence: float = 0.0,
        is_compound: bool | None = None,
        limit: int = 100,
    ) -> list[Tag]:
        """Search vocabulary with filters.

        Args:
            query: Text search (prefix match)
            min_occurrences: Minimum total occurrences
            min_confidence: Minimum prior confidence
            is_compound: Filter compound tags only
            limit: Maximum results

        Returns:
            List of matching Tag objects
        """
        return self._storage.search_vocabulary(
            query=query,
            min_occurrences=min_occurrences,
            min_confidence=min_confidence,
            is_compound=is_compound,
            limit=limit,
        )

    # =========================================================================
    # LEARNING
    # =========================================================================

    def update_priors(self, min_samples: int = 5) -> int:
        """Update prior confidence for all tags.

        Recalculates prior_confidence based on feedback data
        using Bayesian updating.

        Args:
            min_samples: Minimum feedback samples required

        Returns:
            Number of tags updated
        """
        return update_all_priors(self._storage, min_samples=min_samples)

    def rebuild_calibrators(
        self,
        min_samples: int = 50,
        tags: list[str] | None = None,
    ) -> int:
        """Rebuild isotonic regression calibrators.

        Trains new calibrators from accumulated feedback data.

        Args:
            min_samples: Minimum calibration points required
            tags: Specific tags to rebuild (None = all eligible)

        Returns:
            Number of calibrators rebuilt
        """
        rebuilt = 0

        # Get tags that need calibration
        if tags:
            tag_objects = [self._storage.get_tag(t) for t in tags]
            tag_objects = [t for t in tag_objects if t is not None]
        else:
            tag_objects = self._storage.get_tags_needing_calibration(min_occurrences=min_samples)

        for tag in tag_objects:
            # Get calibration data for this tag
            cal_data = self._storage.get_calibration_data(tag_id=tag.tag_id)

            if len(cal_data) < min_samples:
                continue

            # Group by model
            model_data: dict[str, tuple[list[float], list[bool]]] = {}
            for point in cal_data:
                if point.model not in model_data:
                    model_data[point.model] = ([], [])
                model_data[point.model][0].append(point.raw_confidence)
                model_data[point.model][1].append(point.was_correct)

            # Train calibrator for each model
            for model, (confidences, outcomes) in model_data.items():
                if len(confidences) < min_samples:
                    continue

                calibrator = TagCalibrator(tag.tag_id, model)
                calibrator.fit(confidences, outcomes)

                if calibrator.is_fitted:
                    self._storage.save_calibration_model(
                        tag_id=tag.tag_id,
                        source_model=model,
                        model_data=calibrator.serialize(),
                        sample_count=len(confidences),
                    )
                    self._tag_calibrators[(tag.tag_id, model)] = calibrator
                    rebuilt += 1

        # Also rebuild global calibrators
        for source in ["ram_plus", "florence_2", "siglip"]:
            cal_data = self._storage.get_calibration_data(model=source)
            if len(cal_data) < min_samples * 2:
                continue

            confidences = [p.raw_confidence for p in cal_data]
            outcomes = [p.was_correct for p in cal_data]

            calibrator = GlobalCalibrator(source)
            calibrator.fit(confidences, outcomes)

            if calibrator.is_fitted:
                self._storage.save_calibration_model(
                    tag_id=None,
                    source_model=source,
                    model_data=calibrator.serialize(),
                    sample_count=len(confidences),
                )
                self._global_calibrators[source] = calibrator
                rebuilt += 1

        return rebuilt

    # =========================================================================
    # ACTIVE LEARNING
    # =========================================================================

    def select_for_review(
        self,
        n: int = 100,
        strategy: str = "uncertainty",
    ) -> list[dict]:
        """Select images for human review.

        Args:
            n: Number of images to select
            strategy: Selection strategy ('uncertainty', 'diversity', 'high_volume')

        Returns:
            List of dicts with image_id, uncertain_tags, priority
        """
        candidates = select_for_review(self._storage, n, strategy)
        return [c.to_dict() for c in candidates]

    # =========================================================================
    # EXPORT / IMPORT
    # =========================================================================

    def get_vocabulary_labels(
        self,
        min_occurrences: int = 1,
        min_confidence: float = 0.0,
        sources: list[str] | None = None,
    ) -> list[str]:
        """Get vocabulary labels for external use (e.g., SigLIP vocabulary).

        Returns a list of tag labels from the vocabulary database,
        filtered by occurrence count and confidence. Useful for feeding
        historical vocabulary to SigLIP for scoring.

        Args:
            min_occurrences: Minimum occurrences required (default 1)
            min_confidence: Minimum prior confidence (default 0.0)
            sources: Filter to specific sources (e.g., ['ram_plus', 'florence_2'])

        Returns:
            List of tag labels sorted by occurrence count

        Example:
            >>> vocab = VocabLearn("./vocab.db")
            >>> labels = vocab.get_vocabulary_labels(min_occurrences=3)
            >>> # Use these labels as SigLIP vocabulary
            >>> len(labels)
            1547
        """
        return self._storage.get_vocabulary_labels(
            min_occurrences=min_occurrences,
            min_confidence=min_confidence,
            sources=sources,
        )

    def export_vocabulary(self, path: str | Path) -> None:
        """Export vocabulary to JSON file.

        Exports all vocabulary entries with statistics.

        Args:
            path: Output file path
        """
        vocab = self._storage.export_vocabulary()

        with open(path, "w") as f:
            json.dump({
                "version": "1.0",
                "vocabulary": vocab,
                "statistics": self.get_statistics(),
            }, f, indent=2)

    def import_vocabulary(
        self,
        path: str | Path,
        merge: bool = True,
    ) -> int:
        """Import vocabulary from JSON file.

        Args:
            path: Path to exported vocabulary
            merge: If True, merge with existing. If False, replace.

        Returns:
            Number of tags imported
        """
        with open(path) as f:
            data = json.load(f)

        vocab = data.get("vocabulary", data)  # Support old format
        return self._storage.import_vocabulary(vocab, merge=merge)

    def export_calibration_curves(self, path: str | Path) -> None:
        """Export calibration curve data for analysis.

        CSV format with columns:
        tag, model, confidence_bucket, actual_accuracy, sample_count

        Args:
            path: Output file path
        """
        import csv

        cal_data = self._storage.get_calibration_data()

        # Group by tag, model, confidence bucket
        buckets: dict[tuple, dict] = {}
        for point in cal_data:
            tag = self._storage.get_tag_by_id(point.tag_id)
            label = tag.label if tag else f"tag_{point.tag_id}"
            bucket = round(point.raw_confidence, 1)
            key = (label, point.model, bucket)

            if key not in buckets:
                buckets[key] = {"correct": 0, "total": 0}
            buckets[key]["total"] += 1
            if point.was_correct:
                buckets[key]["correct"] += 1

        # Write CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "model", "confidence_bucket", "actual_accuracy", "sample_count"])

            for (label, model, bucket), stats in sorted(buckets.items()):
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                writer.writerow([label, model, bucket, round(accuracy, 3), stats["total"]])

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> dict:
        """Get overall system statistics.

        Returns:
            Dict with vocabulary statistics
        """
        stats = self._storage.get_statistics()
        return stats.to_dict()
