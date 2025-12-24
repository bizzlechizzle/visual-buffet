"""SQLite storage backend for vocablearn.

Implements vocabulary storage using SQLite database.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from vocablearn.models import (
    CalibrationPoint,
    ConfidenceTier,
    Tag,
    TagEvent,
    TagSource,
    VocabularyStats,
    is_compound_tag,
    normalize_tag,
)

# Schema version for migrations
SCHEMA_VERSION = 1

# SQL schema
SCHEMA_SQL = """
-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vocabulary table
CREATE TABLE IF NOT EXISTS vocabulary (
    tag_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    label               TEXT UNIQUE NOT NULL,
    normalized          TEXT NOT NULL,
    is_compound         BOOLEAN DEFAULT FALSE,

    -- Aggregate statistics
    total_occurrences   INTEGER DEFAULT 0,
    confirmed_correct   INTEGER DEFAULT 0,
    confirmed_incorrect INTEGER DEFAULT 0,

    -- Computed prior
    prior_confidence    REAL DEFAULT 0.5,
    confidence_lower    REAL DEFAULT 0.0,
    confidence_upper    REAL DEFAULT 1.0,

    -- Model-specific hit rates
    ram_plus_hits       INTEGER DEFAULT 0,
    florence_2_hits     INTEGER DEFAULT 0,
    siglip_verified     INTEGER DEFAULT 0,
    model_agreement     INTEGER DEFAULT 0,

    -- Metadata
    first_seen          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vocab_label ON vocabulary(label);
CREATE INDEX IF NOT EXISTS idx_vocab_normalized ON vocabulary(normalized);
CREATE INDEX IF NOT EXISTS idx_vocab_prior ON vocabulary(prior_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_vocab_occurrences ON vocabulary(total_occurrences DESC);

-- Tag events table
CREATE TABLE IF NOT EXISTS tag_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            TEXT NOT NULL,
    tag_id              INTEGER NOT NULL REFERENCES vocabulary(tag_id),

    -- Source and confidence
    source              TEXT NOT NULL,
    raw_confidence      REAL,
    unified_confidence  REAL NOT NULL,
    confidence_tier     INTEGER NOT NULL,

    -- Multi-model data
    ram_confidence      REAL,
    florence_found      BOOLEAN DEFAULT FALSE,
    siglip_confidence   REAL,

    -- Feedback
    human_verified      BOOLEAN DEFAULT FALSE,
    human_correct       BOOLEAN,
    verified_at         TIMESTAMP,
    verified_by         TEXT,

    -- Metadata
    tagged_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_image ON tag_events(image_id);
CREATE INDEX IF NOT EXISTS idx_events_tag ON tag_events(tag_id);
CREATE INDEX IF NOT EXISTS idx_events_confidence ON tag_events(unified_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_events_unverified ON tag_events(human_verified) WHERE human_verified = FALSE;
CREATE INDEX IF NOT EXISTS idx_events_source ON tag_events(source);

-- Calibration data table
CREATE TABLE IF NOT EXISTS calibration_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id              INTEGER NOT NULL REFERENCES vocabulary(tag_id),
    model               TEXT NOT NULL,
    raw_confidence      REAL NOT NULL,
    was_correct         BOOLEAN NOT NULL,
    recorded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calibration_tag ON calibration_data(tag_id);
CREATE INDEX IF NOT EXISTS idx_calibration_model ON calibration_data(model);
CREATE INDEX IF NOT EXISTS idx_calibration_tag_model ON calibration_data(tag_id, model);

-- Calibration models table
CREATE TABLE IF NOT EXISTS calibration_models (
    model_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id              INTEGER REFERENCES vocabulary(tag_id),
    source_model        TEXT NOT NULL,
    model_data          BLOB NOT NULL,
    sample_count        INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at          TIMESTAMP,

    UNIQUE(tag_id, source_model)
);

-- Feedback queue table
CREATE TABLE IF NOT EXISTS feedback_queue (
    queue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            TEXT NOT NULL,
    uncertainty_score   REAL NOT NULL,
    uncertain_tags      TEXT NOT NULL,
    priority            INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'pending',
    assigned_to         TEXT,
    assigned_at         TIMESTAMP,
    completed_at        TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON feedback_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_priority ON feedback_queue(priority DESC, uncertainty_score DESC);
"""


class SQLiteStorage:
    """SQLite storage backend for vocablearn.

    Provides persistent storage for vocabulary, tag events,
    calibration data, and feedback queue.

    Example:
        >>> storage = SQLiteStorage("./vocab.db")
        >>> tag_id = storage.get_or_create_tag("abandoned_bar")
        >>> storage.record_event(image_id="img001", tag_id=tag_id, ...)
    """

    def __init__(self, db_path: str | Path):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file (created if not exists)
        """
        self.db_path = Path(db_path)
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Ensure database schema exists and is up to date."""
        with self._connection() as conn:
            # Check if schema exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                # Create schema
                conn.executescript(SCHEMA_SQL)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    [SCHEMA_VERSION],
                )

    # =========================================================================
    # VOCABULARY OPERATIONS
    # =========================================================================

    def get_or_create_tag(self, label: str) -> int:
        """Get or create a vocabulary entry.

        Args:
            label: Tag label

        Returns:
            tag_id for the vocabulary entry
        """
        normalized = normalize_tag(label)
        is_compound = is_compound_tag(label)

        with self._connection() as conn:
            # Try to get existing
            cursor = conn.execute(
                "SELECT tag_id FROM vocabulary WHERE normalized = ?",
                [normalized],
            )
            row = cursor.fetchone()
            if row:
                return row["tag_id"]

            # Create new
            cursor = conn.execute(
                """
                INSERT INTO vocabulary (label, normalized, is_compound)
                VALUES (?, ?, ?)
                """,
                [label, normalized, is_compound],
            )
            return cursor.lastrowid

    def get_tag(self, label: str) -> Optional[Tag]:
        """Get vocabulary entry by label.

        Args:
            label: Tag label (case-insensitive)

        Returns:
            Tag object or None if not found
        """
        normalized = normalize_tag(label)

        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM vocabulary WHERE normalized = ?",
                [normalized],
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_tag(row)

    def get_tag_by_id(self, tag_id: int) -> Optional[Tag]:
        """Get vocabulary entry by ID.

        Args:
            tag_id: Tag ID

        Returns:
            Tag object or None if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM vocabulary WHERE tag_id = ?",
                [tag_id],
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_tag(row)

    def search_vocabulary(
        self,
        query: Optional[str] = None,
        min_occurrences: int = 0,
        min_confidence: float = 0.0,
        is_compound: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Tag]:
        """Search vocabulary with filters.

        Args:
            query: Text search (prefix match on normalized)
            min_occurrences: Minimum total occurrences
            min_confidence: Minimum prior confidence
            is_compound: Filter compound tags only
            limit: Maximum results
            offset: Skip first N results

        Returns:
            List of matching Tag objects
        """
        conditions = ["1=1"]
        params = []

        if query:
            conditions.append("normalized LIKE ?")
            params.append(f"{normalize_tag(query)}%")

        if min_occurrences > 0:
            conditions.append("total_occurrences >= ?")
            params.append(min_occurrences)

        if min_confidence > 0:
            conditions.append("prior_confidence >= ?")
            params.append(min_confidence)

        if is_compound is not None:
            conditions.append("is_compound = ?")
            params.append(is_compound)

        sql = f"""
            SELECT * FROM vocabulary
            WHERE {' AND '.join(conditions)}
            ORDER BY total_occurrences DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            return [self._row_to_tag(row) for row in cursor.fetchall()]

    def update_tag_statistics(
        self,
        tag_id: int,
        increment_occurrences: int = 0,
        increment_correct: int = 0,
        increment_incorrect: int = 0,
        increment_ram_plus: int = 0,
        increment_florence_2: int = 0,
        increment_siglip: int = 0,
        increment_agreement: int = 0,
    ) -> None:
        """Update tag statistics.

        Args:
            tag_id: Tag ID
            increment_*: Amount to add to each counter
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE vocabulary SET
                    total_occurrences = total_occurrences + ?,
                    confirmed_correct = confirmed_correct + ?,
                    confirmed_incorrect = confirmed_incorrect + ?,
                    ram_plus_hits = ram_plus_hits + ?,
                    florence_2_hits = florence_2_hits + ?,
                    siglip_verified = siglip_verified + ?,
                    model_agreement = model_agreement + ?,
                    last_seen = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE tag_id = ?
                """,
                [
                    increment_occurrences,
                    increment_correct,
                    increment_incorrect,
                    increment_ram_plus,
                    increment_florence_2,
                    increment_siglip,
                    increment_agreement,
                    tag_id,
                ],
            )

    def update_tag_prior(
        self,
        tag_id: int,
        prior: float,
        lower: float,
        upper: float,
    ) -> None:
        """Update tag prior confidence.

        Args:
            tag_id: Tag ID
            prior: Prior confidence value
            lower: Lower confidence bound
            upper: Upper confidence bound
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE vocabulary SET
                    prior_confidence = ?,
                    confidence_lower = ?,
                    confidence_upper = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE tag_id = ?
                """,
                [prior, lower, upper, tag_id],
            )

    # =========================================================================
    # TAG EVENT OPERATIONS
    # =========================================================================

    def record_event(
        self,
        image_id: str,
        tag_id: int,
        source: TagSource,
        unified_confidence: float,
        confidence_tier: ConfidenceTier,
        raw_confidence: Optional[float] = None,
        ram_confidence: Optional[float] = None,
        florence_found: bool = False,
        siglip_confidence: Optional[float] = None,
    ) -> int:
        """Record a tag assignment event.

        Args:
            image_id: Image identifier
            tag_id: Vocabulary entry ID
            source: Source model
            unified_confidence: Computed unified confidence
            confidence_tier: Computed tier
            raw_confidence: Raw model confidence
            ram_confidence: RAM++ confidence
            florence_found: Whether Florence-2 found tag
            siglip_confidence: SigLIP verification score

        Returns:
            event_id for the recorded event
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tag_events (
                    image_id, tag_id, source, raw_confidence,
                    unified_confidence, confidence_tier,
                    ram_confidence, florence_found, siglip_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    image_id,
                    tag_id,
                    source.value,
                    raw_confidence,
                    unified_confidence,
                    confidence_tier.value,
                    ram_confidence,
                    florence_found,
                    siglip_confidence,
                ],
            )
            return cursor.lastrowid

    def get_events_for_image(self, image_id: str) -> list[TagEvent]:
        """Get all tag events for an image.

        Args:
            image_id: Image identifier

        Returns:
            List of TagEvent objects
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM tag_events WHERE image_id = ? ORDER BY unified_confidence DESC",
                [image_id],
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_event(self, image_id: str, tag_id: int) -> Optional[TagEvent]:
        """Get specific tag event.

        Args:
            image_id: Image identifier
            tag_id: Tag ID

        Returns:
            TagEvent or None if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM tag_events WHERE image_id = ? AND tag_id = ?",
                [image_id, tag_id],
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_event(row)

    def record_feedback(
        self,
        image_id: str,
        tag_id: int,
        correct: bool,
        verified_by: Optional[str] = None,
    ) -> None:
        """Record human feedback on a tag event.

        Args:
            image_id: Image identifier
            tag_id: Tag ID
            correct: Whether tag was correct
            verified_by: User who verified
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE tag_events SET
                    human_verified = TRUE,
                    human_correct = ?,
                    verified_at = CURRENT_TIMESTAMP,
                    verified_by = ?
                WHERE image_id = ? AND tag_id = ?
                """,
                [correct, verified_by, image_id, tag_id],
            )

    def get_unverified_events(
        self,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        limit: int = 100,
    ) -> list[TagEvent]:
        """Get unverified tag events for review.

        Args:
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            limit: Maximum results

        Returns:
            List of TagEvent objects
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM tag_events
                WHERE human_verified = FALSE
                  AND unified_confidence BETWEEN ? AND ?
                ORDER BY ABS(unified_confidence - 0.5) ASC
                LIMIT ?
                """,
                [min_confidence, max_confidence, limit],
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    # =========================================================================
    # CALIBRATION OPERATIONS
    # =========================================================================

    def record_calibration_point(
        self,
        tag_id: int,
        model: str,
        raw_confidence: float,
        was_correct: bool,
    ) -> None:
        """Record a calibration data point.

        Args:
            tag_id: Tag ID
            model: Source model
            raw_confidence: Raw model confidence
            was_correct: Whether prediction was correct
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO calibration_data (tag_id, model, raw_confidence, was_correct)
                VALUES (?, ?, ?, ?)
                """,
                [tag_id, model, raw_confidence, was_correct],
            )

    def get_calibration_data(
        self,
        tag_id: Optional[int] = None,
        model: Optional[str] = None,
        min_samples: int = 0,
    ) -> list[CalibrationPoint]:
        """Get calibration data points.

        Args:
            tag_id: Filter by tag ID (None = all)
            model: Filter by model (None = all)
            min_samples: Minimum samples required

        Returns:
            List of CalibrationPoint objects
        """
        conditions = ["1=1"]
        params = []

        if tag_id is not None:
            conditions.append("tag_id = ?")
            params.append(tag_id)

        if model is not None:
            conditions.append("model = ?")
            params.append(model)

        sql = f"""
            SELECT * FROM calibration_data
            WHERE {' AND '.join(conditions)}
            ORDER BY raw_confidence
        """

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            return [self._row_to_calibration(row) for row in cursor.fetchall()]

    def save_calibration_model(
        self,
        tag_id: Optional[int],
        source_model: str,
        model_data: bytes,
        sample_count: int,
    ) -> None:
        """Save a calibration model.

        Args:
            tag_id: Tag ID (None for global model)
            source_model: Source model name
            model_data: Serialized model bytes
            sample_count: Number of samples used
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calibration_models
                (tag_id, source_model, model_data, sample_count)
                VALUES (?, ?, ?, ?)
                """,
                [tag_id, source_model, model_data, sample_count],
            )

    def get_calibration_model(
        self,
        tag_id: Optional[int],
        source_model: str,
    ) -> Optional[bytes]:
        """Get a calibration model.

        Args:
            tag_id: Tag ID (None for global model)
            source_model: Source model name

        Returns:
            Serialized model bytes or None
        """
        with self._connection() as conn:
            if tag_id is None:
                cursor = conn.execute(
                    "SELECT model_data FROM calibration_models WHERE tag_id IS NULL AND source_model = ?",
                    [source_model],
                )
            else:
                cursor = conn.execute(
                    "SELECT model_data FROM calibration_models WHERE tag_id = ? AND source_model = ?",
                    [tag_id, source_model],
                )
            row = cursor.fetchone()
            if row is None:
                return None
            return row["model_data"]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> VocabularyStats:
        """Get overall vocabulary statistics.

        Returns:
            VocabularyStats object
        """
        with self._connection() as conn:
            stats = VocabularyStats()

            # Total counts
            cursor = conn.execute("SELECT COUNT(*) FROM vocabulary")
            stats.total_vocabulary = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM tag_events")
            stats.total_events = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM tag_events WHERE human_verified = TRUE"
            )
            stats.total_feedback = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(DISTINCT tag_id) FROM calibration_models")
            stats.calibrated_tags = cursor.fetchone()[0]

            cursor = conn.execute("SELECT AVG(prior_confidence) FROM vocabulary")
            row = cursor.fetchone()
            stats.avg_prior_confidence = row[0] if row[0] else 0.5

            # Model-specific counts
            cursor = conn.execute(
                "SELECT COUNT(*) FROM vocabulary WHERE ram_plus_hits > 0"
            )
            stats.ram_plus_unique_tags = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM vocabulary WHERE florence_2_hits > 0"
            )
            stats.florence_2_unique_tags = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM vocabulary WHERE model_agreement > 0"
            )
            stats.model_agreement_tags = cursor.fetchone()[0]

            # Tier distribution
            cursor = conn.execute(
                """
                SELECT confidence_tier, COUNT(*) as count
                FROM tag_events
                GROUP BY confidence_tier
                """
            )
            stats.tier_counts = {row["confidence_tier"]: row["count"] for row in cursor.fetchall()}

            return stats

    def get_tags_needing_calibration(self, min_occurrences: int = 50) -> list[Tag]:
        """Get tags that need calibration (high volume, low feedback).

        Args:
            min_occurrences: Minimum occurrences to consider

        Returns:
            List of Tag objects needing calibration
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT v.* FROM vocabulary v
                LEFT JOIN calibration_models cm ON v.tag_id = cm.tag_id
                WHERE v.total_occurrences >= ?
                  AND cm.model_id IS NULL
                ORDER BY v.total_occurrences DESC
                LIMIT 100
                """,
                [min_occurrences],
            )
            return [self._row_to_tag(row) for row in cursor.fetchall()]

    # =========================================================================
    # EXPORT / IMPORT
    # =========================================================================

    def export_vocabulary(self) -> list[dict]:
        """Export entire vocabulary as list of dicts.

        Returns:
            List of tag dictionaries
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM vocabulary ORDER BY total_occurrences DESC")
            return [self._row_to_tag(row).to_dict() for row in cursor.fetchall()]

    def get_vocabulary_labels(
        self,
        min_occurrences: int = 1,
        min_confidence: float = 0.0,
        sources: list[str] | None = None,
    ) -> list[str]:
        """Get list of vocabulary labels for external use (e.g., SigLIP).

        Args:
            min_occurrences: Minimum total occurrences required
            min_confidence: Minimum prior confidence required
            sources: Filter to tags from specific sources (ram_plus, florence_2, etc.)

        Returns:
            List of tag labels sorted by occurrence count
        """
        with self._connection() as conn:
            if sources:
                # Filter by source - need to join with tag_events
                placeholders = ",".join("?" * len(sources))
                cursor = conn.execute(
                    f"""
                    SELECT DISTINCT v.label
                    FROM vocabulary v
                    JOIN tag_events te ON v.tag_id = te.tag_id
                    WHERE v.total_occurrences >= ?
                      AND v.prior_confidence >= ?
                      AND te.source IN ({placeholders})
                    ORDER BY v.total_occurrences DESC
                    """,
                    [min_occurrences, min_confidence] + sources,
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT label FROM vocabulary
                    WHERE total_occurrences >= ?
                      AND prior_confidence >= ?
                    ORDER BY total_occurrences DESC
                    """,
                    [min_occurrences, min_confidence],
                )
            return [row[0] for row in cursor.fetchall()]

    def import_vocabulary(self, tags: list[dict], merge: bool = True) -> int:
        """Import vocabulary from list of dicts.

        Args:
            tags: List of tag dictionaries
            merge: If True, merge with existing. If False, replace.

        Returns:
            Number of tags imported
        """
        imported = 0

        with self._connection() as conn:
            if not merge:
                conn.execute("DELETE FROM vocabulary")

            for tag_data in tags:
                normalized = normalize_tag(tag_data["label"])

                if merge:
                    # Check if exists
                    cursor = conn.execute(
                        "SELECT tag_id FROM vocabulary WHERE normalized = ?",
                        [normalized],
                    )
                    if cursor.fetchone():
                        # Update existing
                        conn.execute(
                            """
                            UPDATE vocabulary SET
                                prior_confidence = MAX(prior_confidence, ?),
                                total_occurrences = total_occurrences + ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE normalized = ?
                            """,
                            [
                                tag_data.get("prior_confidence", 0.5),
                                tag_data.get("total_occurrences", 0),
                                normalized,
                            ],
                        )
                        imported += 1
                        continue

                # Insert new
                conn.execute(
                    """
                    INSERT INTO vocabulary (
                        label, normalized, is_compound,
                        total_occurrences, confirmed_correct, confirmed_incorrect,
                        prior_confidence, confidence_lower, confidence_upper,
                        ram_plus_hits, florence_2_hits, siglip_verified, model_agreement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tag_data["label"],
                        normalized,
                        tag_data.get("is_compound", is_compound_tag(tag_data["label"])),
                        tag_data.get("total_occurrences", 0),
                        tag_data.get("confirmed_correct", 0),
                        tag_data.get("confirmed_incorrect", 0),
                        tag_data.get("prior_confidence", 0.5),
                        tag_data.get("confidence_lower", 0.0),
                        tag_data.get("confidence_upper", 1.0),
                        tag_data.get("ram_plus_hits", 0),
                        tag_data.get("florence_2_hits", 0),
                        tag_data.get("siglip_verified", 0),
                        tag_data.get("model_agreement_count", 0),
                    ],
                )
                imported += 1

        return imported

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _row_to_tag(self, row: sqlite3.Row) -> Tag:
        """Convert database row to Tag object."""
        return Tag(
            tag_id=row["tag_id"],
            label=row["label"],
            normalized=row["normalized"],
            is_compound=bool(row["is_compound"]),
            total_occurrences=row["total_occurrences"],
            confirmed_correct=row["confirmed_correct"],
            confirmed_incorrect=row["confirmed_incorrect"],
            prior_confidence=row["prior_confidence"],
            confidence_lower=row["confidence_lower"],
            confidence_upper=row["confidence_upper"],
            ram_plus_hits=row["ram_plus_hits"],
            florence_2_hits=row["florence_2_hits"],
            siglip_verified=row["siglip_verified"],
            model_agreement_count=row["model_agreement"],
            first_seen=datetime.fromisoformat(row["first_seen"]) if row["first_seen"] else None,
            last_seen=datetime.fromisoformat(row["last_seen"]) if row["last_seen"] else None,
        )

    def _row_to_event(self, row: sqlite3.Row) -> TagEvent:
        """Convert database row to TagEvent object."""
        return TagEvent(
            event_id=row["event_id"],
            image_id=row["image_id"],
            tag_id=row["tag_id"],
            source=TagSource(row["source"]),
            raw_confidence=row["raw_confidence"],
            unified_confidence=row["unified_confidence"],
            confidence_tier=ConfidenceTier(row["confidence_tier"]),
            ram_confidence=row["ram_confidence"],
            florence_found=bool(row["florence_found"]),
            siglip_confidence=row["siglip_confidence"],
            human_verified=bool(row["human_verified"]),
            human_correct=row["human_correct"],
            verified_at=datetime.fromisoformat(row["verified_at"]) if row["verified_at"] else None,
            verified_by=row["verified_by"],
            tagged_at=datetime.fromisoformat(row["tagged_at"]) if row["tagged_at"] else None,
        )

    def _row_to_calibration(self, row: sqlite3.Row) -> CalibrationPoint:
        """Convert database row to CalibrationPoint object."""
        return CalibrationPoint(
            tag_id=row["tag_id"],
            model=row["model"],
            raw_confidence=row["raw_confidence"],
            was_correct=bool(row["was_correct"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]) if row["recorded_at"] else None,
        )
