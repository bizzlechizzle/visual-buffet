"""RAM++ Threshold Quality Benchmark.

Tests tag quality at different threshold levels across 10 test images.

KEY INSIGHT: Resolution has NO impact on RAM++ quality. The model internally
resizes everything to 384×384. Only THRESHOLD matters for output quality.

This benchmark tests UNCAPPED tag output at each threshold level:
- 0.4 - Maximum discovery (capture everything)
- 0.5 - Comprehensive coverage
- 0.6 - Balanced (recommended default)
- 0.7 - High confidence only
- 0.8 - Very selective

All tests use 480px (little) for fastest processing with NO tag limits.

Collects metrics:
- Number of tags at each threshold
- Average confidence at each threshold
- Tag distribution and quality
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from visual_buffet.core.engine import TaggingEngine
from visual_buffet.plugins.schemas import ImageSize


# Threshold levels to test - ALL uncapped (limit=0) to measure true tag quality
# NOTE: All use "little" (480px) because resolution has NO impact on RAM++ quality.
THRESHOLDS = {
    "0.4": {
        "threshold": 0.4,
        "description": "Maximum discovery - capture everything",
    },
    "0.5": {
        "threshold": 0.5,
        "description": "Comprehensive coverage",
    },
    "0.6": {
        "threshold": 0.6,
        "description": "Balanced (recommended default)",
    },
    "0.7": {
        "threshold": 0.7,
        "description": "High confidence only",
    },
    "0.8": {
        "threshold": 0.8,
        "description": "Very selective - top tags only",
    },
}

# Common settings for all tests
COMMON_SETTINGS = {
    "size": "little",  # 480px - fastest, no quality difference
    "limit": 0,        # Uncapped - measure true output
}

# Test images
IMAGES_DIR = Path(__file__).parent.parent.parent / "images"
TEST_IMAGES = [
    "testimage01.jpg",
    "testimage02.jpg",
    "testimage03.jpg",
    "testimage04.jpg",
    "testimage05.jpg",
    "testimage06.jpg",
    "testimage07.jpg",
    "testimage08.jpg",
    "testimage09.jpg",
    "testimage10.jpg",
]


def run_benchmark() -> dict[str, Any]:
    """Run the full benchmark suite."""
    print("=" * 60)
    print("RAM++ Threshold Quality Benchmark")
    print("=" * 60)
    print("\nTesting UNCAPPED tag output at each threshold level")
    print(f"Common settings: size={COMMON_SETTINGS['size']}, limit={COMMON_SETTINGS['limit']} (unlimited)")

    # Initialize engine with only RAM++ plugin
    print("\nLoading RAM++ model...")
    engine = TaggingEngine()

    if "ram_plus" not in engine.plugins:
        print("ERROR: RAM++ plugin not found")
        return {}

    if not engine.plugins["ram_plus"].is_available():
        print("ERROR: RAM++ model not downloaded. Run: visual-buffet plugins setup ram_plus")
        return {}

    print("RAM++ ready.\n")

    results = {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "threshold_quality",
        "common_settings": COMMON_SETTINGS,
        "thresholds_tested": list(THRESHOLDS.keys()),
        "images_tested": len(TEST_IMAGES),
        "results": {},
    }

    # Run each threshold level
    for threshold_name, threshold_config in THRESHOLDS.items():
        threshold = threshold_config["threshold"]
        print(f"\n{'='*60}")
        print(f"Testing Threshold: {threshold}")
        print(f"  Description: {threshold_config['description']}")
        print("=" * 60)

        threshold_results = {
            "threshold": threshold,
            "description": threshold_config["description"],
            "images": {},
            "summary": {},
        }

        total_time = 0
        total_tags = 0
        total_confidence = 0
        tag_counts = []

        for image_name in TEST_IMAGES:
            image_path = IMAGES_DIR / image_name
            if not image_path.exists():
                print(f"  SKIP: {image_name} not found")
                continue

            print(f"\n  Processing: {image_name}...")

            try:
                result = engine.tag_image(
                    image_path,
                    plugin_names=["ram_plus"],
                    threshold=threshold,
                    size=COMMON_SETTINGS["size"],
                    limit=COMMON_SETTINGS["limit"],
                    save_tags=False,
                )

                plugin_result = result["results"].get("ram_plus", {})

                if "error" in plugin_result:
                    print(f"    ERROR: {plugin_result['error']}")
                    continue

                tags = plugin_result.get("tags", [])
                inference_time = plugin_result.get("inference_time_ms", 0)

                # Calculate metrics
                num_tags = len(tags)
                avg_conf = sum(t.get("confidence", 0) for t in tags) / num_tags if num_tags > 0 else 0

                # Top 10 tags for review
                top_tags = [t["label"] for t in tags[:10]]

                threshold_results["images"][image_name] = {
                    "inference_time_ms": inference_time,
                    "num_tags": num_tags,
                    "avg_confidence": round(avg_conf, 4),
                    "top_tags": top_tags,
                    "all_tags": [{"label": t["label"], "confidence": t.get("confidence")} for t in tags],
                }

                total_time += inference_time
                total_tags += num_tags
                total_confidence += avg_conf
                tag_counts.append(num_tags)

                print(f"    Tags: {num_tags} | Avg conf: {avg_conf:.3f}")
                print(f"    Top 5: {', '.join(top_tags[:5])}")

            except Exception as e:
                print(f"    EXCEPTION: {e}")
                threshold_results["images"][image_name] = {"error": str(e)}

        # Calculate summary stats
        num_images = len([i for i in threshold_results["images"].values() if "error" not in i])
        if num_images > 0:
            threshold_results["summary"] = {
                "avg_time_ms": round(total_time / num_images, 2),
                "avg_tags_per_image": round(total_tags / num_images, 2),
                "avg_confidence": round(total_confidence / num_images, 4),
                "min_tags": min(tag_counts) if tag_counts else 0,
                "max_tags": max(tag_counts) if tag_counts else 0,
                "images_processed": num_images,
            }

            print(f"\n  SUMMARY for threshold {threshold}:")
            print(f"    Avg tags: {threshold_results['summary']['avg_tags_per_image']:.1f}")
            print(f"    Tag range: {threshold_results['summary']['min_tags']}-{threshold_results['summary']['max_tags']}")
            print(f"    Avg confidence: {threshold_results['summary']['avg_confidence']:.4f}")

        results["results"][threshold_name] = threshold_results

    return results


def print_comparison(results: dict) -> None:
    """Print side-by-side comparison of thresholds."""
    print("\n" + "=" * 90)
    print("THRESHOLD QUALITY COMPARISON")
    print("=" * 90)

    # Get all threshold results
    thresholds = ["0.4", "0.5", "0.6", "0.7", "0.8"]
    headers = ["Metric"] + [f"t={t}" for t in thresholds]

    # Collect data for each threshold
    data = {}
    for t in thresholds:
        data[t] = results["results"].get(t, {}).get("summary", {})

    rows = [
        ["Avg Tags"] + [f"{data[t].get('avg_tags_per_image', 0):.0f}" for t in thresholds],
        ["Tag Range"] + [f"{data[t].get('min_tags', 0)}-{data[t].get('max_tags', 0)}" for t in thresholds],
        ["Avg Confidence"] + [f"{data[t].get('avg_confidence', 0):.3f}" for t in thresholds],
    ]

    # Print table
    col_widths = [15] + [12] * len(thresholds)
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    # Print recommendation
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)
    print("  threshold 0.4 - Maximum discovery, catch all possible tags")
    print("  threshold 0.5 - Comprehensive coverage")
    print("  threshold 0.6 - RECOMMENDED: Balanced quality and coverage")
    print("  threshold 0.7 - High confidence only, fewer tags")
    print("  threshold 0.8 - Very selective, top tags only")


def save_results(results: dict, output_path: Path) -> None:
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_benchmark()

    if results:
        print_comparison(results)

        # Save to benchmark results
        output_dir = Path(__file__).parent
        output_file = output_dir / f"ram_plus_threshold_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_file)
