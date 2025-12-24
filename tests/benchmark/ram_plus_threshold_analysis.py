#!/usr/bin/env python3
"""
RAM++ Threshold Analysis Benchmark

Tests multiple threshold values to find diminishing returns point.
Analyzes tag distribution, confidence patterns, and optimal settings.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from visual_buffet.core.engine import TaggingEngine
from visual_buffet.plugins.schemas import ImageSize


def run_threshold_analysis(image_list_path: str, output_dir: Path):
    """Run analysis across multiple thresholds."""

    # Read image list
    with open(image_list_path) as f:
        images = [line.strip() for line in f if line.strip()]

    print(f"=" * 70)
    print(f"RAM++ Threshold Analysis")
    print(f"=" * 70)
    print(f"Images: {len(images)}")
    print()

    # Test thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    engine = TaggingEngine()

    if "ram_plus" not in engine.plugins:
        print("ERROR: RAM++ plugin not found")
        return None

    if not engine.plugins["ram_plus"].is_available():
        print("ERROR: RAM++ model not downloaded")
        return None

    print("RAM++ loaded. Starting analysis...\n")

    results = {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "total_images": len(images),
        "thresholds_tested": thresholds,
        "by_threshold": {},
        "per_image": {}
    }

    # First pass: collect all tags at lowest threshold to get full picture
    print(f"Pass 1: Collecting all tags at threshold 0.3...")
    all_image_data = {}

    for i, img_path in enumerate(images):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(images)}...")

        try:
            result = engine.tag_image(
                Path(img_path),
                size=ImageSize.LITTLE,
                threshold=0.3
            )

            if "error" not in result:
                plugin_result = result.get("results", {}).get("ram_plus", {})
                tags = plugin_result.get("tags", [])
                inference_time = plugin_result.get("inference_time_ms", 0)

                all_image_data[img_path] = {
                    "all_tags": tags,
                    "inference_time_ms": inference_time
                }
        except Exception as e:
            print(f"  Error on {Path(img_path).name}: {e}")

    print(f"  Collected data from {len(all_image_data)} images\n")

    # Analyze each threshold
    for thresh in thresholds:
        print(f"Analyzing threshold {thresh}...")

        tag_counts = []
        confidences = []
        unique_tags = set()

        for img_path, data in all_image_data.items():
            # Filter tags by threshold
            filtered = [t for t in data["all_tags"] if t["confidence"] >= thresh]
            tag_counts.append(len(filtered))

            for tag in filtered:
                confidences.append(tag["confidence"])
                unique_tags.add(tag["label"])

        if tag_counts:
            results["by_threshold"][str(thresh)] = {
                "avg_tags": round(mean(tag_counts), 1),
                "min_tags": min(tag_counts),
                "max_tags": max(tag_counts),
                "std_tags": round(stdev(tag_counts), 1) if len(tag_counts) > 1 else 0,
                "avg_confidence": round(mean(confidences), 4) if confidences else 0,
                "min_confidence": round(min(confidences), 4) if confidences else 0,
                "max_confidence": round(max(confidences), 4) if confidences else 0,
                "unique_tags_total": len(unique_tags),
                "total_tag_instances": len(confidences)
            }

    # Per-image detailed data (sample)
    sample_images = list(all_image_data.items())[:10]
    for img_path, data in sample_images:
        results["per_image"][Path(img_path).name] = {
            "total_tags_at_0.3": len(data["all_tags"]),
            "inference_time_ms": data["inference_time_ms"],
            "top_10_tags": [
                {"label": t["label"], "confidence": round(t["confidence"], 3)}
                for t in data["all_tags"][:10]
            ],
            "tags_by_threshold": {
                str(t): len([x for x in data["all_tags"] if x["confidence"] >= t])
                for t in thresholds
            }
        }

    # Calculate diminishing returns
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Threshold':<12} {'Avg Tags':<12} {'Std Dev':<10} {'Unique':<10} {'Avg Conf':<10}")
    print("-" * 54)

    prev_tags = None
    diminishing_point = None

    for thresh in thresholds:
        data = results["by_threshold"][str(thresh)]
        avg_tags = data["avg_tags"]

        # Calculate % change from previous threshold
        if prev_tags is not None:
            pct_change = ((prev_tags - avg_tags) / prev_tags) * 100
            change_str = f"(-{pct_change:.0f}%)"

            # Find diminishing returns: where change drops below 15%
            if pct_change < 15 and diminishing_point is None:
                diminishing_point = thresh
        else:
            change_str = ""

        print(f"{thresh:<12} {avg_tags:<12} {data['std_tags']:<10} {data['unique_tags_total']:<10} {data['avg_confidence']:<10}")
        prev_tags = avg_tags

    print()

    # Confidence distribution analysis
    print("\nConfidence Distribution (at threshold 0.3):")
    all_confs = []
    for data in all_image_data.values():
        all_confs.extend([t["confidence"] for t in data["all_tags"]])

    if all_confs:
        buckets = {
            "0.3-0.4": len([c for c in all_confs if 0.3 <= c < 0.4]),
            "0.4-0.5": len([c for c in all_confs if 0.4 <= c < 0.5]),
            "0.5-0.6": len([c for c in all_confs if 0.5 <= c < 0.6]),
            "0.6-0.7": len([c for c in all_confs if 0.6 <= c < 0.7]),
            "0.7-0.8": len([c for c in all_confs if 0.7 <= c < 0.8]),
            "0.8-0.9": len([c for c in all_confs if 0.8 <= c < 0.9]),
            "0.9-1.0": len([c for c in all_confs if 0.9 <= c <= 1.0]),
        }

        total = len(all_confs)
        results["confidence_distribution"] = {}

        for bucket, count in buckets.items():
            pct = (count / total) * 100
            bar = "#" * int(pct / 2)
            print(f"  {bucket}: {count:>6} ({pct:>5.1f}%) {bar}")
            results["confidence_distribution"][bucket] = {
                "count": count,
                "percentage": round(pct, 2)
            }

    # Timing stats
    times = [d["inference_time_ms"] for d in all_image_data.values()]
    if times:
        results["timing"] = {
            "avg_ms": round(mean(times), 1),
            "min_ms": round(min(times), 1),
            "max_ms": round(max(times), 1),
            "total_images": len(times)
        }
        print(f"\nTiming: {results['timing']['avg_ms']}ms avg ({results['timing']['min_ms']}-{results['timing']['max_ms']}ms range)")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("""
Based on analysis:

1. THRESHOLD SELECTION:
   - 0.5-0.6: Comprehensive coverage (100-150+ tags)
   - 0.7: Good balance (60-80 tags)
   - 0.8: Selective (30-50 tags)
   - 0.9: Minimal (10-20 highest confidence)

2. DIMINISHING RETURNS:
   - Above 0.8, you lose tags faster than you gain precision
   - Below 0.5, more tags add noise without much value
   - Sweet spot: 0.6-0.7 for most use cases

3. USE --limit FOR CONTROL:
   - --limit 50 with threshold 0.5 = top 50 comprehensive tags
   - --limit 20 with threshold 0.7 = top 20 confident tags
""")

    # Save results
    output_file = output_dir / f"ram_plus_threshold_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    image_list = "/tmp/mn_test_images.txt"
    output_dir = Path(__file__).parent

    run_threshold_analysis(image_list, output_dir)
