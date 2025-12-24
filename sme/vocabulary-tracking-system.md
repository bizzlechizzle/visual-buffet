# Vocabulary Tracking & Learning System

> **Generated**: 2025-12-24
> **Sources current as of**: December 2024
> **Scope**: Comprehensive
> **Version**: 1.0
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

A vocabulary tracking system learns tag confidence from historical usage, enabling:
- **Per-tag confidence priors** that improve over time
- **Active learning** to prioritize human review of uncertain tags
- **Cross-application sharing** of learned vocabulary statistics
- **Advanced views** with confidence-weighted search and facets

**Architecture Decision:**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Standalone library** | Reusable, testable, versionable | Extra dependency | **RECOMMENDED** |
| Part of repo-depot | Centralized | Wrong abstraction level | No |
| Part of visual-buffet | Simple | Not reusable | No |

**Recommendation:** Create `vocablearn` as a standalone Python library that can be installed via pip. Document the protocol in repo-depot for other implementations.

---

## Background & Context

### The Problem

When tagging images with ML models, we get confidence scores that are:
1. **Uncalibrated** — A 0.7 from RAM++ doesn't mean 70% accuracy
2. **Model-specific** — Can't compare RAM++ 0.7 to SigLIP 0.1
3. **Static** — Don't improve with usage or feedback
4. **Isolated** — Each image tagged independently

### The Solution

A vocabulary tracking system that:
1. Records every tag assignment with metadata
2. Aggregates statistics per tag over time
3. Calibrates confidence using isotonic regression
4. Enables active learning through uncertainty sampling
5. Shares learned priors across applications

### Why Standalone Library?

| Consideration | Standalone | In repo-depot | In visual-buffet |
|---------------|------------|---------------|------------------|
| Database agnostic | Yes | N/A (docs only) | No (specific DB) |
| Reusable in other apps | Yes | Via re-implementation | No |
| Testable | Yes (unit tests) | No | Partial |
| Versionable | Yes (semver) | Via depot version | Via app version |
| Database-level access | Yes (any SQL/ORM) | N/A | Tied to app |

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VOCABULARY TRACKING SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Ingestion  │───▶│   Storage    │───▶│   Learning   │                   │
│  │   Pipeline   │    │   (SQLite/   │    │   Engine     │                   │
│  │              │    │   Postgres)  │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Tag Event   │    │  Vocabulary  │    │  Calibration │                   │
│  │  Recording   │    │  Statistics  │    │  Models      │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Query Interface                               │   │
│  │  • Get calibrated confidence for tag                                  │   │
│  │  • Get vocabulary statistics                                          │   │
│  │  • Select uncertain samples for review                                │   │
│  │  • Export learned priors                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

```python
# vocablearn/
├── __init__.py
├── models.py          # Data models (Tag, TagEvent, Vocabulary)
├── storage/
│   ├── __init__.py
│   ├── base.py        # Abstract storage interface
│   ├── sqlite.py      # SQLite implementation
│   └── postgres.py    # PostgreSQL implementation
├── learning/
│   ├── __init__.py
│   ├── calibration.py # Isotonic regression calibrators
│   ├── priors.py      # Prior confidence calculation
│   └── active.py      # Active learning sample selection
├── api.py             # High-level API
└── cli.py             # Command-line interface
```

---

## Data Model

### Core Entities

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class TagSource(Enum):
    """Source model that generated the tag."""
    RAM_PLUS = "ram_plus"
    FLORENCE_2 = "florence_2"
    SIGLIP = "siglip"
    HUMAN = "human"
    ENSEMBLE = "ensemble"

class ConfidenceTier(Enum):
    """Computed confidence tier for display."""
    GOLD = 1      # Model agreement
    HIGH = 2      # RAM++ >= 0.7
    MEDIUM = 3    # RAM++ 0.5-0.7
    CONTEXTUAL = 4  # Florence-2 compound + SigLIP verified
    LOW = 5       # Low confidence

@dataclass
class Tag:
    """A vocabulary entry with learned statistics."""
    tag_id: int
    label: str
    normalized: str  # lowercase, trimmed
    is_compound: bool  # contains underscore

    # Aggregate statistics (learned over time)
    total_occurrences: int = 0
    confirmed_correct: int = 0
    confirmed_incorrect: int = 0

    # Computed fields
    prior_confidence: float = 0.5
    confidence_interval: tuple[float, float] = (0.0, 1.0)

    # Model-specific statistics
    ram_plus_hits: int = 0
    florence_2_hits: int = 0
    siglip_verified: int = 0
    model_agreement_count: int = 0

    # Metadata
    first_seen: datetime = None
    last_seen: datetime = None

    @property
    def sample_size(self) -> int:
        """Total feedback samples for this tag."""
        return self.confirmed_correct + self.confirmed_incorrect

    @property
    def accuracy_rate(self) -> float:
        """Historical accuracy rate (if feedback exists)."""
        if self.sample_size == 0:
            return 0.5  # No data, assume 50%
        return self.confirmed_correct / self.sample_size


@dataclass
class TagEvent:
    """A single tag assignment event."""
    event_id: int
    image_id: str
    tag_id: int

    # Source and confidence
    source: TagSource
    raw_confidence: float  # Original model confidence
    unified_confidence: float  # Calibrated/combined confidence
    confidence_tier: ConfidenceTier

    # Multi-model data (if applicable)
    ram_confidence: Optional[float] = None
    florence_found: bool = False
    siglip_confidence: Optional[float] = None

    # Feedback
    human_verified: bool = False
    human_correct: Optional[bool] = None
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None

    # Metadata
    tagged_at: datetime = None


@dataclass
class CalibrationPoint:
    """A data point for calibration curve fitting."""
    tag_id: int
    model: str
    raw_confidence: float
    was_correct: bool
    recorded_at: datetime
```

### Database Schema (SQLite)

```sql
-- =============================================================================
-- VOCABULARY TRACKING SCHEMA v1.0
-- =============================================================================

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- VOCABULARY TABLE
-- Stores learned statistics for each unique tag
-- -----------------------------------------------------------------------------
CREATE TABLE vocabulary (
    tag_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    label               TEXT UNIQUE NOT NULL,
    normalized          TEXT NOT NULL,
    is_compound         BOOLEAN DEFAULT FALSE,

    -- Aggregate statistics
    total_occurrences   INTEGER DEFAULT 0,
    confirmed_correct   INTEGER DEFAULT 0,
    confirmed_incorrect INTEGER DEFAULT 0,

    -- Computed prior (updated periodically)
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
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_prior CHECK (prior_confidence BETWEEN 0 AND 1)
);

CREATE INDEX idx_vocab_label ON vocabulary(label);
CREATE INDEX idx_vocab_normalized ON vocabulary(normalized);
CREATE INDEX idx_vocab_prior ON vocabulary(prior_confidence DESC);
CREATE INDEX idx_vocab_occurrences ON vocabulary(total_occurrences DESC);

-- -----------------------------------------------------------------------------
-- TAG EVENTS TABLE
-- Records every tag assignment for learning
-- -----------------------------------------------------------------------------
CREATE TABLE tag_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            TEXT NOT NULL,
    tag_id              INTEGER NOT NULL REFERENCES vocabulary(tag_id),

    -- Source and confidence
    source              TEXT NOT NULL,  -- 'ram_plus', 'florence_2', etc.
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
    tagged_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_confidence CHECK (unified_confidence BETWEEN 0 AND 1),
    CONSTRAINT valid_tier CHECK (confidence_tier BETWEEN 1 AND 5)
);

CREATE INDEX idx_events_image ON tag_events(image_id);
CREATE INDEX idx_events_tag ON tag_events(tag_id);
CREATE INDEX idx_events_confidence ON tag_events(unified_confidence DESC);
CREATE INDEX idx_events_unverified ON tag_events(human_verified) WHERE human_verified = FALSE;
CREATE INDEX idx_events_source ON tag_events(source);

-- -----------------------------------------------------------------------------
-- CALIBRATION DATA TABLE
-- Stores data points for isotonic regression calibration
-- -----------------------------------------------------------------------------
CREATE TABLE calibration_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id              INTEGER NOT NULL REFERENCES vocabulary(tag_id),
    model               TEXT NOT NULL,
    raw_confidence      REAL NOT NULL,
    was_correct         BOOLEAN NOT NULL,
    recorded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_confidence CHECK (raw_confidence BETWEEN 0 AND 1)
);

CREATE INDEX idx_calibration_tag ON calibration_data(tag_id);
CREATE INDEX idx_calibration_model ON calibration_data(model);
CREATE INDEX idx_calibration_tag_model ON calibration_data(tag_id, model);

-- -----------------------------------------------------------------------------
-- CALIBRATION MODELS TABLE
-- Stores serialized calibration models (isotonic regression)
-- -----------------------------------------------------------------------------
CREATE TABLE calibration_models (
    model_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id              INTEGER REFERENCES vocabulary(tag_id),  -- NULL = global
    source_model        TEXT NOT NULL,  -- 'ram_plus', 'florence_2', etc.
    model_data          BLOB NOT NULL,  -- Pickled isotonic regressor
    sample_count        INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at          TIMESTAMP,  -- When to rebuild

    UNIQUE(tag_id, source_model)
);

-- -----------------------------------------------------------------------------
-- FEEDBACK QUEUE TABLE
-- Tracks images selected for human review
-- -----------------------------------------------------------------------------
CREATE TABLE feedback_queue (
    queue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            TEXT NOT NULL,
    uncertainty_score   REAL NOT NULL,
    uncertain_tags      TEXT NOT NULL,  -- JSON array of tag_ids
    priority            INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'pending',  -- pending, assigned, completed
    assigned_to         TEXT,
    assigned_at         TIMESTAMP,
    completed_at        TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_status CHECK (status IN ('pending', 'assigned', 'completed'))
);

CREATE INDEX idx_queue_status ON feedback_queue(status);
CREATE INDEX idx_queue_priority ON feedback_queue(priority DESC, uncertainty_score DESC);

-- -----------------------------------------------------------------------------
-- VIEWS
-- -----------------------------------------------------------------------------

-- High-value tags for calibration (high volume, uncertain)
CREATE VIEW v_needs_calibration AS
SELECT
    v.tag_id,
    v.label,
    v.total_occurrences,
    v.prior_confidence,
    v.confirmed_correct + v.confirmed_incorrect as feedback_count,
    ABS(v.prior_confidence - 0.5) as certainty
FROM vocabulary v
WHERE v.total_occurrences > 50
  AND (v.confirmed_correct + v.confirmed_incorrect) < 10
ORDER BY v.total_occurrences DESC;

-- Vocabulary with model agreement statistics
CREATE VIEW v_vocabulary_stats AS
SELECT
    v.tag_id,
    v.label,
    v.is_compound,
    v.total_occurrences,
    v.prior_confidence,
    v.ram_plus_hits,
    v.florence_2_hits,
    v.model_agreement,
    CASE
        WHEN v.model_agreement > 0 THEN 'gold'
        WHEN v.prior_confidence >= 0.8 THEN 'high'
        WHEN v.prior_confidence >= 0.6 THEN 'medium'
        ELSE 'low'
    END as quality_tier,
    v.first_seen,
    v.last_seen
FROM vocabulary v
ORDER BY v.total_occurrences DESC;

-- Calibration curve data for analysis
CREATE VIEW v_calibration_curve AS
SELECT
    tag_id,
    model,
    ROUND(raw_confidence, 1) as confidence_bucket,
    AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) as actual_accuracy,
    COUNT(*) as sample_count
FROM calibration_data
GROUP BY tag_id, model, ROUND(raw_confidence, 1)
ORDER BY tag_id, model, confidence_bucket;
```

---

## API Design

### High-Level Python API

```python
"""vocablearn - Vocabulary Learning Library API."""

from typing import Iterator, Optional
from pathlib import Path

class VocabLearn:
    """Main API for vocabulary tracking and learning."""

    def __init__(self, db_path: str | Path):
        """Initialize with database path.

        Args:
            db_path: Path to SQLite database (created if not exists)
        """
        ...

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
                - label: Tag text
                - confidence: Raw model confidence (optional)
                - ram_confidence: RAM++ confidence (optional)
                - florence_found: Whether Florence-2 found it (optional)
                - siglip_confidence: SigLIP verification score (optional)
            source: Source model ('ram_plus', 'florence_2', 'ensemble')

        Returns:
            List of event_ids for recorded tags
        """
        ...

    def record_feedback(
        self,
        image_id: str,
        tag_label: str,
        correct: bool,
        verified_by: Optional[str] = None,
    ) -> None:
        """Record human feedback on a tag.

        Updates:
        - Tag event record
        - Vocabulary statistics
        - Calibration data

        Args:
            image_id: Image identifier
            tag_label: Tag that was verified
            correct: Whether tag was correct
            verified_by: Optional user identifier
        """
        ...

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_tag(self, label: str) -> Optional[Tag]:
        """Get vocabulary entry for a tag.

        Args:
            label: Tag label (case-insensitive)

        Returns:
            Tag object with learned statistics, or None if not found
        """
        ...

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
        ...

    def get_tags_for_image(self, image_id: str) -> list[TagEvent]:
        """Get all tags assigned to an image.

        Args:
            image_id: Image identifier

        Returns:
            List of TagEvent objects
        """
        ...

    def search_vocabulary(
        self,
        query: Optional[str] = None,
        min_occurrences: int = 0,
        min_confidence: float = 0.0,
        is_compound: Optional[bool] = None,
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
        ...

    # =========================================================================
    # LEARNING
    # =========================================================================

    def update_priors(self, min_samples: int = 5) -> int:
        """Update prior confidence for all tags.

        Recalculates prior_confidence based on feedback data.

        Args:
            min_samples: Minimum feedback samples required

        Returns:
            Number of tags updated
        """
        ...

    def rebuild_calibrators(
        self,
        min_samples: int = 50,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Rebuild isotonic regression calibrators.

        Args:
            min_samples: Minimum calibration points required
            tags: Specific tags to rebuild (None = all eligible)

        Returns:
            Number of calibrators rebuilt
        """
        ...

    # =========================================================================
    # ACTIVE LEARNING
    # =========================================================================

    def select_for_review(
        self,
        n: int = 100,
        strategy: str = "uncertainty",
    ) -> list[dict]:
        """Select images for human review.

        Strategies:
        - 'uncertainty': Tags near decision boundary
        - 'diversity': Diverse tag coverage
        - 'high_volume': Most common uncertain tags

        Args:
            n: Number of images to select
            strategy: Selection strategy

        Returns:
            List of dicts with image_id, uncertain_tags, priority
        """
        ...

    def get_review_queue(
        self,
        status: str = "pending",
        limit: int = 50,
    ) -> list[dict]:
        """Get images in the review queue.

        Args:
            status: Queue status filter
            limit: Maximum results

        Returns:
            List of queue entries
        """
        ...

    # =========================================================================
    # EXPORT / IMPORT
    # =========================================================================

    def export_vocabulary(self, path: str | Path) -> None:
        """Export vocabulary to JSON file.

        Exports:
        - All vocabulary entries with statistics
        - Calibration models (serialized)
        """
        ...

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
        ...

    def export_calibration_curves(self, path: str | Path) -> None:
        """Export calibration curve data for analysis.

        CSV format with columns:
        tag, model, confidence_bucket, actual_accuracy, sample_count
        """
        ...

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> dict:
        """Get overall system statistics.

        Returns dict with:
        - total_vocabulary: Number of unique tags
        - total_events: Number of tag assignments
        - total_feedback: Number of human verifications
        - calibrated_tags: Tags with calibration models
        - avg_prior_confidence: Average vocabulary confidence
        - model_hit_rates: Per-model hit rates
        """
        ...
```

### CLI Interface

```bash
# Initialize database
vocablearn init --db ./vocab.db

# Record tags from JSON
vocablearn record --db ./vocab.db --image img001.jpg --tags tags.json

# Record feedback
vocablearn feedback --db ./vocab.db --image img001.jpg --tag "bar" --correct

# Get statistics
vocablearn stats --db ./vocab.db

# Select for review
vocablearn review-queue --db ./vocab.db --n 50 --strategy uncertainty

# Export vocabulary
vocablearn export --db ./vocab.db --output vocab.json

# Rebuild calibrators
vocablearn calibrate --db ./vocab.db --min-samples 50
```

---

## Learning Algorithms

### 1. Prior Confidence Calculation

```python
def calculate_prior_confidence(tag: Tag) -> tuple[float, float, float]:
    """Calculate prior confidence with confidence interval.

    Uses Beta distribution for Bayesian updating.

    Returns:
        (prior, lower_bound, upper_bound)
    """
    from scipy import stats

    # Beta prior parameters (weakly informative)
    alpha_prior = 1  # Prior successes
    beta_prior = 1   # Prior failures

    # Posterior parameters
    alpha = alpha_prior + tag.confirmed_correct
    beta = beta_prior + tag.confirmed_incorrect

    # Posterior mean
    prior = alpha / (alpha + beta)

    # 95% credible interval
    lower = stats.beta.ppf(0.025, alpha, beta)
    upper = stats.beta.ppf(0.975, alpha, beta)

    return prior, lower, upper
```

### 2. Isotonic Regression Calibration

```python
from sklearn.isotonic import IsotonicRegression
import pickle

class TagCalibrator:
    """Per-tag confidence calibrator using isotonic regression."""

    def __init__(self, tag_id: int, model: str):
        self.tag_id = tag_id
        self.model = model
        self.calibrator: Optional[IsotonicRegression] = None
        self.sample_count = 0

    def fit(self, confidences: list[float], outcomes: list[bool]) -> None:
        """Fit calibrator from historical data.

        Args:
            confidences: Raw model confidence scores
            outcomes: Whether tag was correct (True/False)
        """
        if len(confidences) < 10:
            return  # Not enough data

        self.calibrator = IsotonicRegression(
            out_of_bounds='clip',
            y_min=0.0,
            y_max=1.0,
        )
        self.calibrator.fit(confidences, outcomes)
        self.sample_count = len(confidences)

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence.

        Args:
            raw_confidence: Raw model confidence

        Returns:
            Calibrated confidence
        """
        if self.calibrator is None:
            return raw_confidence
        return float(self.calibrator.predict([raw_confidence])[0])

    def serialize(self) -> bytes:
        """Serialize calibrator for storage."""
        return pickle.dumps(self.calibrator)

    @classmethod
    def deserialize(cls, tag_id: int, model: str, data: bytes) -> "TagCalibrator":
        """Deserialize calibrator from storage."""
        cal = cls(tag_id, model)
        cal.calibrator = pickle.loads(data)
        return cal
```

### 3. Active Learning Sample Selection

```python
def select_uncertain_samples(
    db: VocabLearnDB,
    n: int = 100,
    threshold: float = 0.3,
) -> list[dict]:
    """Select images with most uncertain tags.

    Uses uncertainty sampling: prioritize images where
    unified_confidence is near the decision threshold.

    Args:
        db: Database connection
        n: Number of images to select
        threshold: Confidence threshold (0.5 = max uncertainty)

    Returns:
        List of {image_id, uncertain_tags, uncertainty_score}
    """
    # Query for images with uncertain tags
    query = """
    SELECT
        e.image_id,
        GROUP_CONCAT(v.label) as uncertain_tags,
        AVG(ABS(e.unified_confidence - ?)) as avg_uncertainty,
        COUNT(*) as uncertain_count
    FROM tag_events e
    JOIN vocabulary v ON e.tag_id = v.tag_id
    WHERE e.human_verified = FALSE
      AND ABS(e.unified_confidence - ?) < ?
    GROUP BY e.image_id
    ORDER BY uncertain_count DESC, avg_uncertainty DESC
    LIMIT ?
    """

    results = db.execute(query, [threshold, threshold, threshold, n])

    return [
        {
            "image_id": row[0],
            "uncertain_tags": row[1].split(","),
            "uncertainty_score": row[2],
        }
        for row in results
    ]
```

---

## Integration Patterns

### Pattern 1: Post-Tagging Recording

```python
# After tagging an image with visual-buffet
from vocablearn import VocabLearn

vocab = VocabLearn("./vocab.db")

# Tag with RAM++ and Florence-2
ram_result = ram_plugin.tag(image_path)
florence_result = florence_plugin.tag(image_path)

# Record RAM++ tags
vocab.record_tags(
    image_id=image_id,
    tags=[
        {
            "label": t.label,
            "confidence": t.confidence,
            "ram_confidence": t.confidence,
        }
        for t in ram_result.tags
    ],
    source="ram_plus",
)

# Record Florence-2 tags
vocab.record_tags(
    image_id=image_id,
    tags=[
        {
            "label": t.label,
            "florence_found": True,
        }
        for t in florence_result.tags
    ],
    source="florence_2",
)
```

### Pattern 2: Calibrated Confidence Lookup

```python
# When displaying tags, use calibrated confidence
vocab = VocabLearn("./vocab.db")

for tag in image_tags:
    calibrated = vocab.get_calibrated_confidence(
        label=tag.label,
        raw_confidence=tag.confidence,
        source="ram_plus",
    )

    display_tag(tag.label, calibrated)
```

### Pattern 3: Feedback Loop in UI

```python
# When user confirms/rejects a tag
vocab = VocabLearn("./vocab.db")

@app.route("/api/feedback", methods=["POST"])
def record_feedback():
    data = request.json
    vocab.record_feedback(
        image_id=data["image_id"],
        tag_label=data["tag"],
        correct=data["correct"],
        verified_by=current_user.id,
    )
    return {"status": "ok"}
```

### Pattern 4: Batch Calibration Update

```python
# Periodic calibration job (daily)
vocab = VocabLearn("./vocab.db")

# Update priors from feedback
updated = vocab.update_priors(min_samples=5)
print(f"Updated priors for {updated} tags")

# Rebuild calibrators
rebuilt = vocab.rebuild_calibrators(min_samples=50)
print(f"Rebuilt {rebuilt} calibrators")
```

---

## Advanced Views (Database-Level)

### 1. Confidence-Weighted Search

```sql
-- Search with learned confidence weighting
SELECT
    e.image_id,
    GROUP_CONCAT(v.label || ':' || ROUND(e.unified_confidence, 2)) as tags,
    SUM(e.unified_confidence) as total_confidence,
    COUNT(*) as tag_count
FROM tag_events e
JOIN vocabulary v ON e.tag_id = v.tag_id
WHERE v.normalized IN ('abandoned', 'bar', 'stool')
GROUP BY e.image_id
ORDER BY total_confidence DESC
LIMIT 100;
```

### 2. Tag Co-occurrence Analysis

```sql
-- Find tags that frequently appear together
WITH tag_pairs AS (
    SELECT
        e1.tag_id as tag1,
        e2.tag_id as tag2,
        COUNT(*) as co_occurrences
    FROM tag_events e1
    JOIN tag_events e2 ON e1.image_id = e2.image_id AND e1.tag_id < e2.tag_id
    GROUP BY e1.tag_id, e2.tag_id
    HAVING COUNT(*) > 10
)
SELECT
    v1.label as tag1,
    v2.label as tag2,
    tp.co_occurrences,
    tp.co_occurrences * 1.0 / LEAST(v1.total_occurrences, v2.total_occurrences) as jaccard
FROM tag_pairs tp
JOIN vocabulary v1 ON tp.tag1 = v1.tag_id
JOIN vocabulary v2 ON tp.tag2 = v2.tag_id
ORDER BY jaccard DESC
LIMIT 50;
```

### 3. Model Agreement Dashboard

```sql
-- Dashboard view of model agreement rates
SELECT
    CASE
        WHEN model_agreement > 0 THEN 'Both Models'
        WHEN ram_plus_hits > 0 AND florence_2_hits = 0 THEN 'RAM++ Only'
        WHEN florence_2_hits > 0 AND ram_plus_hits = 0 THEN 'Florence-2 Only'
        ELSE 'Unknown'
    END as source_category,
    COUNT(*) as tag_count,
    AVG(prior_confidence) as avg_confidence,
    SUM(total_occurrences) as total_occurrences
FROM vocabulary
GROUP BY source_category
ORDER BY total_occurrences DESC;
```

### 4. Calibration Quality Report

```sql
-- Check calibration quality by comparing predicted vs actual
SELECT
    v.label,
    cd.model,
    ROUND(cd.raw_confidence, 1) as confidence_bucket,
    AVG(CASE WHEN cd.was_correct THEN 1.0 ELSE 0.0 END) as actual_accuracy,
    COUNT(*) as sample_count,
    ABS(ROUND(cd.raw_confidence, 1) - AVG(CASE WHEN cd.was_correct THEN 1.0 ELSE 0.0 END)) as calibration_error
FROM calibration_data cd
JOIN vocabulary v ON cd.tag_id = v.tag_id
GROUP BY v.label, cd.model, ROUND(cd.raw_confidence, 1)
HAVING COUNT(*) >= 10
ORDER BY calibration_error DESC
LIMIT 50;
```

---

## Deployment Recommendations

### For repo-depot: Document Protocol

Add to `repo-depot/specs/vocabulary-tracking-protocol.md`:
- Schema specification (portable across databases)
- API contract (language-agnostic)
- Integration patterns

### For visual-buffet: Use Library

```bash
pip install vocablearn
```

```python
# In visual-buffet CLI
from vocablearn import VocabLearn

vocab = VocabLearn(Path.home() / ".visual-buffet" / "vocab.db")
```

### For Other Apps: Same Library

Any app can use the same library with its own database:
- national-treasure: Media organization
- wake-n-blake: Asset management
- etc.

### Shared Vocabulary (Optional)

Apps can export/import vocabulary to share learned priors:

```bash
# Export from visual-buffet
vocablearn export --db ~/.visual-buffet/vocab.db --output vocab-export.json

# Import into national-treasure
vocablearn import --db ~/.national-treasure/vocab.db --input vocab-export.json --merge
```

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Distributed/multi-user vocabulary merging
- Real-time streaming updates
- GPU-accelerated calibration
- Privacy considerations for vocabulary sharing

### Knowledge Gaps

- Optimal calibration curve smoothing for sparse data
- Transfer learning of priors across domains (e.g., photos → art)
- Long-term drift detection and re-calibration triggers

---

## Recommendations

1. **Create `vocablearn` as standalone Python library** with SQLite backend
2. **Document protocol in repo-depot** for cross-language implementations
3. **Integrate into visual-buffet** as primary consumer
4. **Enable vocabulary export/import** for sharing across apps
5. **Build calibration UI** for viewing calibration curves and quality metrics
6. **Implement periodic calibration** as background job (daily/weekly)

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html) | 2024 | Primary | Isotonic regression |
| 2 | [Encord Active Learning](https://encord.com/blog/active-learning-machine-learning-guide/) | 2024 | Secondary | Active learning concepts |
| 3 | [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599) | 2017 | Primary | Calibration theory |
| 4 | [Database Tagging Schemas](https://charlesleifer.com/blog/a-tour-of-tagging-schemas-many-to-many-bitmaps-and-more/) | 2024 | Secondary | Schema design |
| 5 | [GeeksforGeeks Tagging DB](https://www.geeksforgeeks.org/dbms/how-to-design-a-database-for-tagging-service/) | 2024 | Secondary | Schema patterns |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-24 | Initial comprehensive specification |

---

*Created for: visual-buffet image tagging pipeline*
*Recommended location: Standalone library (vocablearn)*
