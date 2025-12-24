"""SigLIP Verification Test Suite - Comprehensive Multi-Model Integration Testing.

Tests three configurations:
1. SigLIP + RAM++ Words: SigLIP scores RAM++ discovered vocabulary
2. SigLIP + Florence-2: SigLIP scores Florence-2 discovered vocabulary
3. SigLIP + RAM++ + Florence-2: SigLIP scores combined vocabulary

Analyzes:
- Independent verification vs combined results
- Confidence term building strategies
- Florence-2 implicit confidence tiers (from SigLIP scoring)

Run: uv run python tests/benchmark/siglip_verification_suite.py
"""

import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "plugins"))
sys.path.insert(0, str(ROOT))

# =============================================================================
# Configuration
# =============================================================================

# Test images
TEST_IMAGES_DIR = ROOT / "images"
TEST_IMAGES = [
    "testimage01.jpg",  # Abandoned diner with bar/stools
    "testimage02.jpg",  # Abandoned interior
    "testimage03.jpg",  # Abandoned church
    "testimage04.jpg",  # Abandoned interior with decay
    "testimage05.jpg",  # Abandoned structure
    "testimage06.jpg",  # Industrial/exterior
    "testimage07.jpg",  # Urban exploration
    "testimage08.jpg",  # Abandoned interior
    "testimage09.jpg",  # Parking lot with cars
    "testimage10.jpg",  # Factory by river
]

# SigLIP confidence thresholds for tier classification
# Based on SME: SigLIP sigmoid outputs are typically 0.01-0.30
CONFIDENCE_TIERS = {
    "high": 0.10,      # >= 0.10 is high confidence for SigLIP
    "medium": 0.03,    # >= 0.03 is medium confidence
    "low": 0.01,       # >= 0.01 is low confidence
    "noise": 0.0,      # < 0.01 is likely noise
}

# RAM++ settings (from SME: threshold 0.5 recommended)
RAM_PLUS_THRESHOLD = 0.5

# Florence-2 tasks to use (from SME benchmark: best efficiency)
FLORENCE_TASKS = ["<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TagScore:
    """A tag with confidence scores from multiple sources."""
    label: str
    # Source model that discovered this tag
    source: str  # "ram_plus", "florence_2", or "both"
    # Original confidence from source (None for Florence-2)
    source_confidence: float | None = None
    # SigLIP verification score
    siglip_confidence: float | None = None
    # Computed confidence tier
    confidence_tier: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {"label": self.label, "source": self.source}
        if self.source_confidence is not None:
            d["source_confidence"] = round(self.source_confidence, 4)
        if self.siglip_confidence is not None:
            d["siglip_confidence"] = round(self.siglip_confidence, 4)
        if self.confidence_tier:
            d["confidence_tier"] = self.confidence_tier
        return d


@dataclass
class ImageResult:
    """Results for a single image across all configurations."""
    image_path: str
    image_name: str

    # Raw results from each model
    ram_plus_tags: list[dict] = field(default_factory=list)
    florence_2_tags: list[dict] = field(default_factory=list)

    # SigLIP verification results for each configuration
    siglip_ram_plus: list[TagScore] = field(default_factory=list)
    siglip_florence_2: list[TagScore] = field(default_factory=list)
    siglip_combined: list[TagScore] = field(default_factory=list)

    # Timing info
    ram_plus_time_ms: float = 0.0
    florence_2_time_ms: float = 0.0
    siglip_ram_plus_time_ms: float = 0.0
    siglip_florence_2_time_ms: float = 0.0
    siglip_combined_time_ms: float = 0.0

    # Analysis metrics
    ram_plus_unique_tags: int = 0
    florence_2_unique_tags: int = 0
    overlap_tags: int = 0
    combined_unique_tags: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "image_name": self.image_name,
            "raw_results": {
                "ram_plus": {
                    "tags": self.ram_plus_tags,
                    "count": len(self.ram_plus_tags),
                    "time_ms": self.ram_plus_time_ms,
                },
                "florence_2": {
                    "tags": self.florence_2_tags,
                    "count": len(self.florence_2_tags),
                    "time_ms": self.florence_2_time_ms,
                },
            },
            "siglip_verification": {
                "ram_plus_vocabulary": {
                    "tags": [t.to_dict() for t in self.siglip_ram_plus],
                    "count": len(self.siglip_ram_plus),
                    "time_ms": self.siglip_ram_plus_time_ms,
                },
                "florence_2_vocabulary": {
                    "tags": [t.to_dict() for t in self.siglip_florence_2],
                    "count": len(self.siglip_florence_2),
                    "time_ms": self.siglip_florence_2_time_ms,
                },
                "combined_vocabulary": {
                    "tags": [t.to_dict() for t in self.siglip_combined],
                    "count": len(self.siglip_combined),
                    "time_ms": self.siglip_combined_time_ms,
                },
            },
            "vocabulary_analysis": {
                "ram_plus_unique": self.ram_plus_unique_tags,
                "florence_2_unique": self.florence_2_unique_tags,
                "overlap": self.overlap_tags,
                "combined_total": self.combined_unique_tags,
            },
        }


@dataclass
class BenchmarkResults:
    """Aggregated results across all images."""
    timestamp: str
    test_images: list[str]
    configuration: dict
    image_results: list[ImageResult] = field(default_factory=list)

    # Aggregate statistics
    aggregate_stats: dict = field(default_factory=dict)

    # Confidence tier analysis
    tier_analysis: dict = field(default_factory=dict)

    # Recommendations
    recommendations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "test_images": self.test_images,
            "configuration": self.configuration,
            "image_results": [r.to_dict() for r in self.image_results],
            "aggregate_stats": self.aggregate_stats,
            "tier_analysis": self.tier_analysis,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def classify_confidence_tier(confidence: float) -> str:
    """Classify a SigLIP confidence score into tiers."""
    if confidence >= CONFIDENCE_TIERS["high"]:
        return "high"
    elif confidence >= CONFIDENCE_TIERS["medium"]:
        return "medium"
    elif confidence >= CONFIDENCE_TIERS["low"]:
        return "low"
    else:
        return "noise"


def normalize_tag(tag: str) -> str:
    """Normalize a tag for comparison (lowercase, strip, replace spaces)."""
    return tag.lower().strip().replace(" ", "_")


def compute_vocabulary_overlap(
    ram_tags: set[str], florence_tags: set[str]
) -> tuple[set[str], set[str], set[str]]:
    """Compute vocabulary overlap between RAM++ and Florence-2.

    Returns:
        (ram_only, florence_only, overlap)
    """
    overlap = ram_tags & florence_tags
    ram_only = ram_tags - florence_tags
    florence_only = florence_tags - ram_tags
    return ram_only, florence_only, overlap


def aggregate_confidence_stats(scores: list[float]) -> dict:
    """Compute aggregate statistics for confidence scores."""
    if not scores:
        return {"count": 0}

    return {
        "count": len(scores),
        "mean": round(mean(scores), 4),
        "median": round(median(scores), 4),
        "std": round(stdev(scores), 4) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "high_confidence_count": sum(1 for s in scores if s >= CONFIDENCE_TIERS["high"]),
        "medium_confidence_count": sum(
            1 for s in scores if CONFIDENCE_TIERS["medium"] <= s < CONFIDENCE_TIERS["high"]
        ),
        "low_confidence_count": sum(
            1 for s in scores if CONFIDENCE_TIERS["low"] <= s < CONFIDENCE_TIERS["medium"]
        ),
    }


# =============================================================================
# Model Loaders
# =============================================================================

def load_ram_plus():
    """Load the RAM++ plugin."""
    from ram_plus import RamPlusPlugin
    plugin_dir = ROOT / "plugins" / "ram_plus"
    plugin = RamPlusPlugin(plugin_dir)
    if not plugin.is_available():
        raise RuntimeError("RAM++ model not available. Run: visual-buffet plugins setup ram_plus")
    return plugin


def load_florence_2():
    """Load the Florence-2 plugin."""
    from florence_2 import Florence2Plugin
    plugin_dir = ROOT / "plugins" / "florence_2"
    plugin = Florence2Plugin(plugin_dir)
    if not plugin.is_available():
        raise RuntimeError("Florence-2 not available. Install transformers>=4.40.0")
    return plugin


def load_siglip():
    """Load the SigLIP plugin."""
    from siglip import SigLIPPlugin
    plugin_dir = ROOT / "plugins" / "siglip"
    plugin = SigLIPPlugin(plugin_dir)
    if not plugin.is_available():
        raise RuntimeError("SigLIP not available. Install transformers>=4.47.0")
    return plugin


# =============================================================================
# Main Test Functions
# =============================================================================

def run_ram_plus(plugin, image_path: Path) -> tuple[list[dict], float]:
    """Run RAM++ tagging and return tags with timing."""
    start = time.perf_counter()
    result = plugin.tag(image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    tags = [
        {"label": normalize_tag(t.label), "confidence": t.confidence}
        for t in result.tags
        if t.confidence is not None and t.confidence >= RAM_PLUS_THRESHOLD
    ]
    return tags, elapsed_ms


def run_florence_2(plugin, image_path: Path, tasks: list[str]) -> tuple[list[dict], float]:
    """Run Florence-2 with multiple tasks and return combined tags with timing."""
    all_tags = []
    seen = set()
    total_time = 0.0

    for task in tasks:
        plugin.configure(task_prompt=task)
        start = time.perf_counter()
        result = plugin.tag(image_path)
        total_time += (time.perf_counter() - start) * 1000

        for tag in result.tags:
            normalized = normalize_tag(tag.label)
            if normalized not in seen:
                seen.add(normalized)
                # Florence-2 has no confidence, mark as None
                all_tags.append({"label": normalized, "confidence": None})

    return all_tags, total_time


def run_siglip_verification(
    plugin, image_path: Path, vocabulary: list[str]
) -> tuple[list[TagScore], float]:
    """Run SigLIP to score a vocabulary against an image.

    Returns scored tags sorted by confidence (highest first).
    """
    if not vocabulary:
        return [], 0.0

    # Set custom vocabulary
    plugin.set_vocabulary(vocabulary)

    start = time.perf_counter()
    result = plugin.tag(image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Reset vocabulary for next call
    plugin.reset_vocabulary()

    # Convert to TagScore objects with tier classification
    scored_tags = []
    for tag in result.tags:
        conf = tag.confidence or 0.0
        tier = classify_confidence_tier(conf)
        scored_tags.append(TagScore(
            label=tag.label,
            source="siglip",
            siglip_confidence=conf,
            confidence_tier=tier,
        ))

    # Sort by confidence (highest first)
    scored_tags.sort(key=lambda t: t.siglip_confidence or 0, reverse=True)

    return scored_tags, elapsed_ms


def process_single_image(
    image_path: Path,
    ram_plugin,
    florence_plugin,
    siglip_plugin,
) -> ImageResult:
    """Process a single image through all configurations."""
    result = ImageResult(
        image_path=str(image_path),
        image_name=image_path.name,
    )

    print(f"  Processing {image_path.name}...")

    # 1. Run RAM++
    print(f"    Running RAM++...")
    ram_tags, ram_time = run_ram_plus(ram_plugin, image_path)
    result.ram_plus_tags = ram_tags
    result.ram_plus_time_ms = ram_time
    print(f"    RAM++: {len(ram_tags)} tags in {ram_time:.0f}ms")

    # 2. Run Florence-2
    print(f"    Running Florence-2...")
    florence_tags, florence_time = run_florence_2(florence_plugin, image_path, FLORENCE_TASKS)
    result.florence_2_tags = florence_tags
    result.florence_2_time_ms = florence_time
    print(f"    Florence-2: {len(florence_tags)} tags in {florence_time:.0f}ms")

    # Extract vocabularies
    ram_vocab = {t["label"] for t in ram_tags}
    florence_vocab = {t["label"] for t in florence_tags}
    combined_vocab = ram_vocab | florence_vocab

    # Compute overlap statistics
    ram_only, florence_only, overlap = compute_vocabulary_overlap(ram_vocab, florence_vocab)
    result.ram_plus_unique_tags = len(ram_only)
    result.florence_2_unique_tags = len(florence_only)
    result.overlap_tags = len(overlap)
    result.combined_unique_tags = len(combined_vocab)

    print(f"    Vocabulary: RAM-only={len(ram_only)}, Florence-only={len(florence_only)}, "
          f"Overlap={len(overlap)}, Combined={len(combined_vocab)}")

    # 3. SigLIP + RAM++ vocabulary
    print(f"    Running SigLIP with RAM++ vocabulary ({len(ram_vocab)} terms)...")
    siglip_ram_scores, siglip_ram_time = run_siglip_verification(
        siglip_plugin, image_path, list(ram_vocab)
    )
    # Annotate source
    for score in siglip_ram_scores:
        score.source = "ram_plus"
        # Add original RAM++ confidence
        orig = next((t for t in ram_tags if t["label"] == score.label), None)
        if orig:
            score.source_confidence = orig["confidence"]
    result.siglip_ram_plus = siglip_ram_scores
    result.siglip_ram_plus_time_ms = siglip_ram_time
    high_count = sum(1 for s in siglip_ram_scores if s.confidence_tier == "high")
    print(f"    SigLIP+RAM++: {len(siglip_ram_scores)} scored, {high_count} high-confidence, {siglip_ram_time:.0f}ms")

    # 4. SigLIP + Florence-2 vocabulary
    print(f"    Running SigLIP with Florence-2 vocabulary ({len(florence_vocab)} terms)...")
    siglip_florence_scores, siglip_florence_time = run_siglip_verification(
        siglip_plugin, image_path, list(florence_vocab)
    )
    # Annotate source
    for score in siglip_florence_scores:
        score.source = "florence_2"
    result.siglip_florence_2 = siglip_florence_scores
    result.siglip_florence_2_time_ms = siglip_florence_time
    high_count = sum(1 for s in siglip_florence_scores if s.confidence_tier == "high")
    print(f"    SigLIP+Florence-2: {len(siglip_florence_scores)} scored, {high_count} high-confidence, {siglip_florence_time:.0f}ms")

    # 5. SigLIP + Combined vocabulary
    print(f"    Running SigLIP with combined vocabulary ({len(combined_vocab)} terms)...")
    siglip_combined_scores, siglip_combined_time = run_siglip_verification(
        siglip_plugin, image_path, list(combined_vocab)
    )
    # Annotate source (ram_plus, florence_2, or both)
    for score in siglip_combined_scores:
        label = score.label
        in_ram = label in ram_vocab
        in_florence = label in florence_vocab
        if in_ram and in_florence:
            score.source = "both"
        elif in_ram:
            score.source = "ram_plus"
        else:
            score.source = "florence_2"
        # Add original RAM++ confidence if available
        if in_ram:
            orig = next((t for t in ram_tags if t["label"] == label), None)
            if orig:
                score.source_confidence = orig["confidence"]
    result.siglip_combined = siglip_combined_scores
    result.siglip_combined_time_ms = siglip_combined_time
    high_count = sum(1 for s in siglip_combined_scores if s.confidence_tier == "high")
    print(f"    SigLIP+Combined: {len(siglip_combined_scores)} scored, {high_count} high-confidence, {siglip_combined_time:.0f}ms")

    return result


def analyze_results(image_results: list[ImageResult]) -> tuple[dict, dict, dict]:
    """Analyze results across all images.

    Returns:
        (aggregate_stats, tier_analysis, recommendations)
    """
    # Collect all scores by configuration
    all_ram_siglip_scores = []
    all_florence_siglip_scores = []
    all_combined_siglip_scores = []

    # Track tag sources in combined
    combined_source_counts = {"ram_plus": 0, "florence_2": 0, "both": 0}
    combined_source_high_conf = {"ram_plus": 0, "florence_2": 0, "both": 0}

    # Track Florence-2 tier distribution (since it has no native confidence)
    florence_tier_counts = {"high": 0, "medium": 0, "low": 0, "noise": 0}

    # Track RAM++ correlation (source confidence vs SigLIP confidence)
    ram_correlations = []  # List of (source_conf, siglip_conf) tuples

    for result in image_results:
        # RAM++ configuration
        for score in result.siglip_ram_plus:
            if score.siglip_confidence is not None:
                all_ram_siglip_scores.append(score.siglip_confidence)
                if score.source_confidence is not None:
                    ram_correlations.append((score.source_confidence, score.siglip_confidence))

        # Florence-2 configuration
        for score in result.siglip_florence_2:
            if score.siglip_confidence is not None:
                all_florence_siglip_scores.append(score.siglip_confidence)
                florence_tier_counts[score.confidence_tier] += 1

        # Combined configuration
        for score in result.siglip_combined:
            if score.siglip_confidence is not None:
                all_combined_siglip_scores.append(score.siglip_confidence)
                combined_source_counts[score.source] += 1
                if score.confidence_tier == "high":
                    combined_source_high_conf[score.source] += 1

    # Compute aggregate statistics
    aggregate_stats = {
        "siglip_ram_plus": aggregate_confidence_stats(all_ram_siglip_scores),
        "siglip_florence_2": aggregate_confidence_stats(all_florence_siglip_scores),
        "siglip_combined": aggregate_confidence_stats(all_combined_siglip_scores),
        "vocabulary_stats": {
            "avg_ram_plus_tags": mean([len(r.ram_plus_tags) for r in image_results]),
            "avg_florence_2_tags": mean([len(r.florence_2_tags) for r in image_results]),
            "avg_overlap": mean([r.overlap_tags for r in image_results]),
            "avg_combined_total": mean([r.combined_unique_tags for r in image_results]),
        },
        "timing_stats": {
            "avg_ram_plus_ms": mean([r.ram_plus_time_ms for r in image_results]),
            "avg_florence_2_ms": mean([r.florence_2_time_ms for r in image_results]),
            "avg_siglip_ram_plus_ms": mean([r.siglip_ram_plus_time_ms for r in image_results]),
            "avg_siglip_florence_2_ms": mean([r.siglip_florence_2_time_ms for r in image_results]),
            "avg_siglip_combined_ms": mean([r.siglip_combined_time_ms for r in image_results]),
        },
    }

    # Tier analysis
    tier_analysis = {
        "florence_2_tiers": {
            "high": florence_tier_counts["high"],
            "medium": florence_tier_counts["medium"],
            "low": florence_tier_counts["low"],
            "noise": florence_tier_counts["noise"],
            "high_percentage": round(
                florence_tier_counts["high"] / sum(florence_tier_counts.values()) * 100, 1
            ) if sum(florence_tier_counts.values()) > 0 else 0,
        },
        "combined_source_distribution": combined_source_counts,
        "combined_high_confidence_by_source": combined_source_high_conf,
        "ram_plus_siglip_correlation": _compute_correlation(ram_correlations),
    }

    # Generate recommendations
    recommendations = _generate_recommendations(aggregate_stats, tier_analysis)

    return aggregate_stats, tier_analysis, recommendations


def _compute_correlation(pairs: list[tuple[float, float]]) -> dict:
    """Compute correlation statistics between RAM++ and SigLIP scores."""
    if len(pairs) < 2:
        return {"correlation": None, "sample_size": len(pairs)}

    source_scores = [p[0] for p in pairs]
    siglip_scores = [p[1] for p in pairs]

    # Compute Pearson correlation coefficient
    n = len(pairs)
    mean_source = mean(source_scores)
    mean_siglip = mean(siglip_scores)

    numerator = sum((s - mean_source) * (g - mean_siglip) for s, g in pairs)
    denom_source = sum((s - mean_source) ** 2 for s in source_scores) ** 0.5
    denom_siglip = sum((g - mean_siglip) ** 2 for g in siglip_scores) ** 0.5

    if denom_source == 0 or denom_siglip == 0:
        correlation = 0.0
    else:
        correlation = numerator / (denom_source * denom_siglip)

    return {
        "pearson_correlation": round(correlation, 4),
        "sample_size": n,
        "interpretation": (
            "strong_positive" if correlation > 0.7 else
            "moderate_positive" if correlation > 0.4 else
            "weak_positive" if correlation > 0.1 else
            "negligible" if correlation > -0.1 else
            "weak_negative" if correlation > -0.4 else
            "moderate_negative" if correlation > -0.7 else
            "strong_negative"
        ),
    }


def _generate_recommendations(aggregate_stats: dict, tier_analysis: dict) -> dict:
    """Generate recommendations based on analysis."""
    recommendations = {}

    # 1. Best configuration for confidence
    ram_high = aggregate_stats["siglip_ram_plus"]["high_confidence_count"]
    florence_high = aggregate_stats["siglip_florence_2"]["high_confidence_count"]
    combined_high = aggregate_stats["siglip_combined"]["high_confidence_count"]

    if combined_high > ram_high and combined_high > florence_high:
        recommendations["best_high_confidence_config"] = "combined"
        recommendations["best_config_reasoning"] = (
            f"Combined vocabulary produces {combined_high} high-confidence tags vs "
            f"RAM++:{ram_high}, Florence-2:{florence_high}"
        )
    elif ram_high >= florence_high:
        recommendations["best_high_confidence_config"] = "ram_plus"
        recommendations["best_config_reasoning"] = (
            f"RAM++ vocabulary produces {ram_high} high-confidence tags, "
            f"best among all configurations"
        )
    else:
        recommendations["best_high_confidence_config"] = "florence_2"
        recommendations["best_config_reasoning"] = (
            f"Florence-2 vocabulary produces {florence_high} high-confidence tags, "
            f"best among all configurations"
        )

    # 2. Florence-2 confidence tier recommendation
    florence_high_pct = tier_analysis["florence_2_tiers"]["high_percentage"]
    if florence_high_pct > 30:
        recommendations["florence_2_quality"] = "high"
        recommendations["florence_2_verdict"] = (
            f"{florence_high_pct:.1f}% of Florence-2 tags verified as high-confidence by SigLIP. "
            "Florence-2 vocabulary is reliable."
        )
    elif florence_high_pct > 15:
        recommendations["florence_2_quality"] = "medium"
        recommendations["florence_2_verdict"] = (
            f"{florence_high_pct:.1f}% of Florence-2 tags verified as high-confidence. "
            "Use SigLIP filtering to remove low-confidence tags."
        )
    else:
        recommendations["florence_2_quality"] = "low"
        recommendations["florence_2_verdict"] = (
            f"Only {florence_high_pct:.1f}% of Florence-2 tags verified as high-confidence. "
            "Florence-2 vocabulary may contain hallucinations."
        )

    # 3. RAM++ vs SigLIP correlation interpretation
    correlation = tier_analysis["ram_plus_siglip_correlation"]["pearson_correlation"]
    if correlation and correlation > 0.4:
        recommendations["ram_plus_siglip_agreement"] = "high"
        recommendations["ram_plus_verdict"] = (
            f"Strong correlation ({correlation:.2f}) between RAM++ and SigLIP confidence. "
            "RAM++ confidence scores are reliable."
        )
    elif correlation and correlation > 0.1:
        recommendations["ram_plus_siglip_agreement"] = "moderate"
        recommendations["ram_plus_verdict"] = (
            f"Moderate correlation ({correlation:.2f}) between RAM++ and SigLIP. "
            "Use SigLIP to verify RAM++ tags for higher precision."
        )
    else:
        recommendations["ram_plus_siglip_agreement"] = "low"
        recommendations["ram_plus_verdict"] = (
            f"Low correlation ({correlation:.2f if correlation else 'N/A'}) between models. "
            "Models may be capturing different visual features."
        )

    # 4. Strategy recommendation
    recommendations["recommended_strategy"] = {
        "for_database_building": (
            "Use combined vocabulary (RAM++ + Florence-2) scored by SigLIP. "
            "Filter to high-confidence tier (>= 0.10) for reliable tags."
        ),
        "for_high_precision": (
            "Use RAM++ vocabulary only, filtered by SigLIP high-confidence tier. "
            "This provides both RAM++ calibrated confidence and SigLIP verification."
        ),
        "for_maximum_coverage": (
            "Use combined vocabulary with medium-confidence threshold (>= 0.03). "
            "Accept more tags with slightly lower precision."
        ),
        "confidence_building": {
            "high_confidence_tags": "SigLIP >= 0.10 (independent verification strong)",
            "medium_confidence_tags": "SigLIP >= 0.03 AND source in [ram_plus, both]",
            "contextual_tags": "Source=florence_2 AND SigLIP >= 0.03",
        },
    }

    return recommendations


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the full benchmark suite."""
    print("=" * 80)
    print("SigLIP Verification Test Suite")
    print("=" * 80)
    print()

    # Verify test images exist
    image_paths = []
    for img_name in TEST_IMAGES:
        path = TEST_IMAGES_DIR / img_name
        if path.exists():
            image_paths.append(path)
        else:
            print(f"Warning: Test image not found: {path}")

    if not image_paths:
        print("Error: No test images found!")
        return

    print(f"Found {len(image_paths)} test images")
    print()

    # Load models
    print("Loading models...")
    print("  Loading RAM++...")
    ram_plugin = load_ram_plus()
    print("  Loading Florence-2...")
    florence_plugin = load_florence_2()
    print("  Loading SigLIP...")
    siglip_plugin = load_siglip()
    print("Models loaded.")
    print()

    # Process all images
    print("Processing images...")
    print("-" * 80)

    image_results = []
    for image_path in image_paths:
        result = process_single_image(
            image_path, ram_plugin, florence_plugin, siglip_plugin
        )
        image_results.append(result)
        print()

    # Analyze results
    print("-" * 80)
    print("Analyzing results...")
    aggregate_stats, tier_analysis, recommendations = analyze_results(image_results)

    # Create final results
    benchmark_results = BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        test_images=[str(p) for p in image_paths],
        configuration={
            "ram_plus_threshold": RAM_PLUS_THRESHOLD,
            "florence_2_tasks": FLORENCE_TASKS,
            "confidence_tiers": CONFIDENCE_TIERS,
        },
        image_results=image_results,
        aggregate_stats=aggregate_stats,
        tier_analysis=tier_analysis,
        recommendations=recommendations,
    )

    # Save results
    output_dir = ROOT / "tests" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"siglip_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(benchmark_results.to_dict(), f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print("Aggregate Statistics:")
    print(f"  RAM++ vocabulary:")
    print(f"    Average tags/image: {aggregate_stats['vocabulary_stats']['avg_ram_plus_tags']:.1f}")
    print(f"    SigLIP high-confidence: {aggregate_stats['siglip_ram_plus']['high_confidence_count']}")
    print(f"    Mean SigLIP score: {aggregate_stats['siglip_ram_plus']['mean']:.4f}")
    print()
    print(f"  Florence-2 vocabulary:")
    print(f"    Average tags/image: {aggregate_stats['vocabulary_stats']['avg_florence_2_tags']:.1f}")
    print(f"    SigLIP high-confidence: {aggregate_stats['siglip_florence_2']['high_confidence_count']}")
    print(f"    Mean SigLIP score: {aggregate_stats['siglip_florence_2']['mean']:.4f}")
    print()
    print(f"  Combined vocabulary:")
    print(f"    Average tags/image: {aggregate_stats['vocabulary_stats']['avg_combined_total']:.1f}")
    print(f"    SigLIP high-confidence: {aggregate_stats['siglip_combined']['high_confidence_count']}")
    print(f"    Mean SigLIP score: {aggregate_stats['siglip_combined']['mean']:.4f}")
    print()

    print("Tier Analysis:")
    print(f"  Florence-2 tags by SigLIP tier:")
    for tier, count in tier_analysis["florence_2_tiers"].items():
        if tier != "high_percentage":
            print(f"    {tier}: {count}")
    print(f"    High-confidence %: {tier_analysis['florence_2_tiers']['high_percentage']:.1f}%")
    print()
    print(f"  RAM++ vs SigLIP correlation:")
    corr = tier_analysis["ram_plus_siglip_correlation"]
    print(f"    Pearson r: {corr['pearson_correlation']}")
    print(f"    Interpretation: {corr['interpretation']}")
    print()

    print("Recommendations:")
    print(f"  Best configuration: {recommendations['best_high_confidence_config']}")
    print(f"  Reasoning: {recommendations['best_config_reasoning']}")
    print()
    print(f"  Florence-2 quality: {recommendations['florence_2_quality']}")
    print(f"  {recommendations['florence_2_verdict']}")
    print()
    print(f"  RAM++ agreement: {recommendations['ram_plus_siglip_agreement']}")
    print(f"  {recommendations['ram_plus_verdict']}")
    print()

    print("Strategy Recommendations:")
    strategy = recommendations["recommended_strategy"]
    print(f"  For database building: {strategy['for_database_building']}")
    print(f"  For high precision: {strategy['for_high_precision']}")
    print(f"  For maximum coverage: {strategy['for_maximum_coverage']}")
    print()

    print("Confidence Building Rules:")
    conf_rules = strategy["confidence_building"]
    for rule_name, rule in conf_rules.items():
        print(f"  {rule_name}: {rule}")
    print()

    print("=" * 80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
