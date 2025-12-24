"""SQLite storage for OCR vocabulary.

Tracks text extracted from images via OCR, building a vocabulary
of detected text across a collection.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OCR vocabulary table (unique text fragments)
CREATE TABLE IF NOT EXISTS ocr_vocabulary (
    text_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    text                TEXT UNIQUE NOT NULL,
    normalized          TEXT NOT NULL,

    -- Classification
    text_type           TEXT DEFAULT 'unknown',  -- sign, label, handwritten, printed, logo, etc.
    language            TEXT DEFAULT 'en',

    -- Aggregate statistics
    total_occurrences   INTEGER DEFAULT 0,
    confirmed_correct   INTEGER DEFAULT 0,
    confirmed_incorrect INTEGER DEFAULT 0,

    -- Confidence from OCR
    avg_confidence      REAL DEFAULT 0.5,
    min_confidence      REAL DEFAULT 0.0,
    max_confidence      REAL DEFAULT 1.0,

    -- Metadata
    first_seen          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ocr_text ON ocr_vocabulary(text);
CREATE INDEX IF NOT EXISTS idx_ocr_normalized ON ocr_vocabulary(normalized);
CREATE INDEX IF NOT EXISTS idx_ocr_type ON ocr_vocabulary(text_type);
CREATE INDEX IF NOT EXISTS idx_ocr_occurrences ON ocr_vocabulary(total_occurrences DESC);

-- OCR detections (text found in specific images)
CREATE TABLE IF NOT EXISTS ocr_detections (
    detection_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            TEXT NOT NULL,
    text_id             INTEGER NOT NULL REFERENCES ocr_vocabulary(text_id),

    -- Detection details
    ocr_engine          TEXT NOT NULL,  -- tesseract, paddleocr, easyocr, etc.
    confidence          REAL NOT NULL,

    -- Location in image (normalized 0-1 coordinates)
    bbox_x              REAL,
    bbox_y              REAL,
    bbox_width          REAL,
    bbox_height         REAL,

    -- Context
    surrounding_tags    TEXT,  -- JSON array of nearby tags
    scene_context       TEXT,  -- scene classification if available

    -- Feedback
    human_verified      BOOLEAN DEFAULT FALSE,
    human_correct       BOOLEAN,
    verified_at         TIMESTAMP,
    verified_by         TEXT,

    -- Metadata
    detected_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ocr_det_image ON ocr_detections(image_id);
CREATE INDEX IF NOT EXISTS idx_ocr_det_text ON ocr_detections(text_id);
CREATE INDEX IF NOT EXISTS idx_ocr_det_engine ON ocr_detections(ocr_engine);
CREATE INDEX IF NOT EXISTS idx_ocr_det_confidence ON ocr_detections(confidence DESC);

-- Text groups (user-defined groupings of related text)
CREATE TABLE IF NOT EXISTS text_groups (
    group_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT UNIQUE NOT NULL,
    description         TEXT,
    pattern             TEXT,  -- regex pattern for auto-matching
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Text-to-group mapping
CREATE TABLE IF NOT EXISTS text_group_members (
    text_id             INTEGER NOT NULL REFERENCES ocr_vocabulary(text_id),
    group_id            INTEGER NOT NULL REFERENCES text_groups(group_id),
    added_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (text_id, group_id)
);
"""


@dataclass
class OCRText:
    """OCR vocabulary entry."""

    text_id: int
    text: str
    normalized: str
    text_type: str
    language: str
    total_occurrences: int
    confirmed_correct: int
    confirmed_incorrect: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    first_seen: datetime
    last_seen: datetime

    @property
    def accuracy_rate(self) -> float:
        """Computed accuracy from feedback."""
        total = self.confirmed_correct + self.confirmed_incorrect
        if total == 0:
            return 0.5
        return self.confirmed_correct / total


@dataclass
class OCRDetection:
    """Single OCR detection instance."""

    detection_id: int
    image_id: str
    text_id: int
    ocr_engine: str
    confidence: float
    bbox_x: Optional[float]
    bbox_y: Optional[float]
    bbox_width: Optional[float]
    bbox_height: Optional[float]
    human_verified: bool
    human_correct: Optional[bool]
    detected_at: datetime


class OCRStorage:
    """SQLite storage for OCR vocabulary."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            if cursor.fetchone() is None:
                cursor.executescript(SCHEMA_SQL)
                cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
                conn.commit()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return text.lower().strip()

    def get_or_create_text(self, text: str, text_type: str = "unknown") -> int:
        """Get or create OCR vocabulary entry."""
        normalized = self._normalize_text(text)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text_id FROM ocr_vocabulary WHERE normalized = ?", (normalized,))
            row = cursor.fetchone()

            if row:
                return row["text_id"]

            cursor.execute(
                """INSERT INTO ocr_vocabulary (text, normalized, text_type)
                   VALUES (?, ?, ?)""",
                (text, normalized, text_type),
            )
            conn.commit()
            return cursor.lastrowid

    def record_detection(
        self,
        image_id: str,
        text: str,
        ocr_engine: str,
        confidence: float,
        text_type: str = "unknown",
        bbox: Optional[tuple[float, float, float, float]] = None,
        surrounding_tags: Optional[list[str]] = None,
        scene_context: Optional[str] = None,
    ) -> int:
        """Record an OCR detection."""
        import json

        text_id = self.get_or_create_text(text, text_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (None, None, None, None)
            tags_json = json.dumps(surrounding_tags) if surrounding_tags else None

            cursor.execute(
                """INSERT INTO ocr_detections
                   (image_id, text_id, ocr_engine, confidence,
                    bbox_x, bbox_y, bbox_width, bbox_height,
                    surrounding_tags, scene_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id,
                    text_id,
                    ocr_engine,
                    confidence,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    tags_json,
                    scene_context,
                ),
            )

            # Update vocabulary statistics
            cursor.execute(
                """UPDATE ocr_vocabulary SET
                   total_occurrences = total_occurrences + 1,
                   last_seen = CURRENT_TIMESTAMP,
                   avg_confidence = (avg_confidence * total_occurrences + ?) / (total_occurrences + 1),
                   min_confidence = MIN(min_confidence, ?),
                   max_confidence = MAX(max_confidence, ?)
                   WHERE text_id = ?""",
                (confidence, confidence, confidence, text_id),
            )

            conn.commit()
            return cursor.lastrowid

    def get_text(self, text: str) -> Optional[OCRText]:
        """Get OCR vocabulary entry by text."""
        normalized = self._normalize_text(text)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ocr_vocabulary WHERE normalized = ?", (normalized,))
            row = cursor.fetchone()

            if not row:
                return None

            return OCRText(
                text_id=row["text_id"],
                text=row["text"],
                normalized=row["normalized"],
                text_type=row["text_type"],
                language=row["language"],
                total_occurrences=row["total_occurrences"],
                confirmed_correct=row["confirmed_correct"],
                confirmed_incorrect=row["confirmed_incorrect"],
                avg_confidence=row["avg_confidence"],
                min_confidence=row["min_confidence"],
                max_confidence=row["max_confidence"],
                first_seen=datetime.fromisoformat(row["first_seen"]),
                last_seen=datetime.fromisoformat(row["last_seen"]),
            )

    def get_detections_for_image(self, image_id: str) -> list[OCRDetection]:
        """Get all OCR detections for an image."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT d.*, v.text FROM ocr_detections d
                   JOIN ocr_vocabulary v ON d.text_id = v.text_id
                   WHERE d.image_id = ?
                   ORDER BY d.confidence DESC""",
                (image_id,),
            )

            return [
                OCRDetection(
                    detection_id=row["detection_id"],
                    image_id=row["image_id"],
                    text_id=row["text_id"],
                    ocr_engine=row["ocr_engine"],
                    confidence=row["confidence"],
                    bbox_x=row["bbox_x"],
                    bbox_y=row["bbox_y"],
                    bbox_width=row["bbox_width"],
                    bbox_height=row["bbox_height"],
                    human_verified=bool(row["human_verified"]),
                    human_correct=row["human_correct"],
                    detected_at=datetime.fromisoformat(row["detected_at"]),
                )
                for row in cursor.fetchall()
            ]

    def search_vocabulary(
        self,
        query: Optional[str] = None,
        text_type: Optional[str] = None,
        min_occurrences: int = 0,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[OCRText]:
        """Search OCR vocabulary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if query:
                conditions.append("normalized LIKE ?")
                params.append(f"%{self._normalize_text(query)}%")

            if text_type:
                conditions.append("text_type = ?")
                params.append(text_type)

            if min_occurrences > 0:
                conditions.append("total_occurrences >= ?")
                params.append(min_occurrences)

            if min_confidence > 0:
                conditions.append("avg_confidence >= ?")
                params.append(min_confidence)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)

            cursor.execute(
                f"""SELECT * FROM ocr_vocabulary
                    WHERE {where_clause}
                    ORDER BY total_occurrences DESC
                    LIMIT ?""",
                params,
            )

            return [
                OCRText(
                    text_id=row["text_id"],
                    text=row["text"],
                    normalized=row["normalized"],
                    text_type=row["text_type"],
                    language=row["language"],
                    total_occurrences=row["total_occurrences"],
                    confirmed_correct=row["confirmed_correct"],
                    confirmed_incorrect=row["confirmed_incorrect"],
                    avg_confidence=row["avg_confidence"],
                    min_confidence=row["min_confidence"],
                    max_confidence=row["max_confidence"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )
                for row in cursor.fetchall()
            ]

    def record_feedback(
        self,
        image_id: str,
        text: str,
        correct: bool,
        verified_by: Optional[str] = None,
    ) -> None:
        """Record human feedback on OCR detection."""
        text_entry = self.get_text(text)
        if not text_entry:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Update detection
            cursor.execute(
                """UPDATE ocr_detections SET
                   human_verified = TRUE,
                   human_correct = ?,
                   verified_at = CURRENT_TIMESTAMP,
                   verified_by = ?
                   WHERE image_id = ? AND text_id = ?""",
                (correct, verified_by, image_id, text_entry.text_id),
            )

            # Update vocabulary stats
            if correct:
                cursor.execute(
                    "UPDATE ocr_vocabulary SET confirmed_correct = confirmed_correct + 1 WHERE text_id = ?",
                    (text_entry.text_id,),
                )
            else:
                cursor.execute(
                    "UPDATE ocr_vocabulary SET confirmed_incorrect = confirmed_incorrect + 1 WHERE text_id = ?",
                    (text_entry.text_id,),
                )

            conn.commit()

    def get_statistics(self) -> dict:
        """Get OCR vocabulary statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM ocr_vocabulary")
            vocab_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM ocr_detections")
            detection_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(DISTINCT image_id) as count FROM ocr_detections")
            image_count = cursor.fetchone()["count"]

            cursor.execute(
                """SELECT text_type, COUNT(*) as count
                   FROM ocr_vocabulary GROUP BY text_type"""
            )
            by_type = {row["text_type"]: row["count"] for row in cursor.fetchall()}

            cursor.execute(
                """SELECT COUNT(*) as count FROM ocr_detections WHERE human_verified = TRUE"""
            )
            verified_count = cursor.fetchone()["count"]

            return {
                "vocabulary_size": vocab_count,
                "total_detections": detection_count,
                "images_with_text": image_count,
                "by_type": by_type,
                "verified_detections": verified_count,
            }
