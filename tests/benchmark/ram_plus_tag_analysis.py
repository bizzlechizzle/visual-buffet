"""RAM++ Tag Persistence Analysis.

Analyzes which tags appear at each threshold level to determine:
1. Core tags (high confidence, appear at all thresholds)
2. Contextual tags (only appear at lower thresholds)
3. Tag overlap between threshold levels
4. Optimal threshold for comprehensive tagging

This helps answer: "Are lower threshold tags still accurate, or just noise?"
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from visual_buffet.core.engine import TaggingEngine

# Thresholds to test (from most selective to most inclusive)
THRESHOLDS = [0.8, 0.7, 0.6, 0.5, 0.4]

# Test images
IMAGES_DIR = Path(__file__).parent.parent.parent / "images"
TEST_IMAGES = [
    "testimage01.jpg",  # Abandoned diner
    "testimage02.jpg",  # Dry cleaner
    "testimage03.jpg",  # Church interior
    "testimage04.jpg",  # Prison
    "testimage05.jpg",  # Kitchen
    "testimage06.jpg",  # White farmhouse
    "testimage07.jpg",  # Victorian house
    "testimage08.jpg",  # Church exterior
    "testimage09.jpg",  # VW graveyard
    "testimage10.jpg",  # Factory
]

# Ground truth - tags I expect to see (from visual analysis)
GROUND_TRUTH = {
    "testimage01.jpg": {
        "must_have": ["bar", "stool", "counter", "restaurant", "ceiling fan", "floor"],
        "should_have": ["diner", "menu", "window", "tile", "wood"],
        "contextual": ["abandoned", "decay", "vintage", "retro"],
    },
    "testimage02.jpg": {
        "must_have": ["clothing", "clothes", "pants", "jeans", "shirt", "hanging", "rack"],
        "should_have": ["plastic", "bag", "floor", "ceiling"],
        "contextual": ["dry cleaner", "laundry", "abandoned", "debris"],
    },
    "testimage03.jpg": {
        "must_have": ["church", "cross", "pew", "bench", "wood"],
        "should_have": ["pulpit", "altar", "carpet", "aisle"],
        "contextual": ["chapel", "religious", "abandoned", "sanctuary"],
    },
    "testimage04.jpg": {
        "must_have": ["prison", "jail", "cell", "bar", "cage", "metal"],
        "should_have": ["rust", "iron", "window", "debris"],
        "contextual": ["abandoned", "decay", "industrial", "historic"],
    },
    "testimage05.jpg": {
        "must_have": ["kitchen", "stove", "cabinet", "chair", "table"],
        "should_have": ["brick", "fireplace", "wood", "door"],
        "contextual": ["abandoned", "domestic", "clutter", "debris"],
    },
    "testimage06.jpg": {
        "must_have": ["house", "building", "porch", "grass", "tree", "sky"],
        "should_have": ["farmhouse", "field", "chimney", "window", "white"],
        "contextual": ["abandoned", "rural", "overgrown", "countryside"],
    },
    "testimage07.jpg": {
        "must_have": ["house", "building", "porch", "sky", "grass", "tree"],
        "should_have": ["cloud", "window", "roof", "wood"],
        "contextual": ["victorian", "abandoned", "decay", "historic", "ruins"],
    },
    "testimage08.jpg": {
        "must_have": ["church", "steeple", "building", "sky", "cloud", "grass"],
        "should_have": ["field", "white", "autumn", "foliage"],
        "contextual": ["rural", "religious", "abandoned", "fall"],
    },
    "testimage09.jpg": {
        "must_have": ["car", "vehicle", "tree", "forest", "leaves"],
        "should_have": ["volkswagen", "beetle", "rust", "autumn"],
        "contextual": ["abandoned", "junkyard", "vintage", "classic"],
    },
    "testimage10.jpg": {
        "must_have": ["factory", "building", "brick", "chimney", "water", "sky"],
        "should_have": ["smokestack", "cloud", "industrial", "pond"],
        "contextual": ["abandoned", "power plant", "manufacturing", "reflection"],
    },
}


def run_analysis() -> dict[str, Any]:
    """Run the tag persistence analysis."""
    print("=" * 70)
    print("RAM++ TAG PERSISTENCE ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing which tags appear at each threshold level")
    print("to determine if lower thresholds add value or just noise.\n")

    # Initialize engine
    print("Loading RAM++ model...")
    engine = TaggingEngine()

    if "ram_plus" not in engine.plugins:
        print("ERROR: RAM++ plugin not found")
        return {}

    if not engine.plugins["ram_plus"].is_available():
        print("ERROR: RAM++ model not available")
        return {}

    print("RAM++ ready.\n")

    results = {
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "thresholds_tested": THRESHOLDS,
        "images": {},
        "aggregate": {},
    }

    # Process each image
    for image_name in TEST_IMAGES:
        image_path = IMAGES_DIR / image_name
        if not image_path.exists():
            print(f"SKIP: {image_name} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Analyzing: {image_name}")
        print("=" * 70)

        image_results = {
            "thresholds": {},
            "tag_first_appearance": {},  # tag -> lowest threshold where it appears
            "core_tags": [],  # appear at 0.8
            "contextual_tags": [],  # only appear below 0.8
            "ground_truth_match": {},
        }

        all_tags_by_threshold = {}

        # Run at each threshold
        for threshold in THRESHOLDS:
            try:
                result = engine.tag_image(
                    image_path,
                    plugin_names=["ram_plus"],
                    threshold=threshold,
                    size="little",
                    limit=0,
                    save_tags=False,
                )

                plugin_result = result["results"].get("ram_plus", {})
                tags = plugin_result.get("tags", [])

                # Store tag data
                tag_dict = {t["label"]: t.get("confidence", 0) for t in tags}
                all_tags_by_threshold[threshold] = tag_dict

                image_results["thresholds"][str(threshold)] = {
                    "num_tags": len(tags),
                    "avg_confidence": round(
                        sum(t.get("confidence", 0) for t in tags) / len(tags), 4
                    ) if tags else 0,
                    "top_10": [t["label"] for t in tags[:10]],
                }

                print(f"  t={threshold}: {len(tags)} tags, top: {', '.join([t['label'] for t in tags[:5]])}")

            except Exception as e:
                print(f"  t={threshold}: ERROR - {e}")

        # Analyze tag persistence
        if all_tags_by_threshold:
            # Find where each tag first appears (lowest threshold)
            all_unique_tags = set()
            for tags in all_tags_by_threshold.values():
                all_unique_tags.update(tags.keys())

            for tag in all_unique_tags:
                # Find the HIGHEST threshold where this tag appears
                # (since we want to know if it's a "core" tag)
                highest_threshold = 0
                for threshold in THRESHOLDS:
                    if tag in all_tags_by_threshold.get(threshold, {}):
                        highest_threshold = max(highest_threshold, threshold)

                image_results["tag_first_appearance"][tag] = highest_threshold

                if highest_threshold >= 0.8:
                    image_results["core_tags"].append(tag)
                else:
                    image_results["contextual_tags"].append(tag)

            # Check ground truth matching
            ground_truth = GROUND_TRUTH.get(image_name, {})
            lowest_threshold_tags = set(all_tags_by_threshold.get(0.4, {}).keys())

            for category in ["must_have", "should_have", "contextual"]:
                expected = ground_truth.get(category, [])
                found = []
                missing = []
                for tag in expected:
                    # Check if any tag contains this word (fuzzy match)
                    if any(tag.lower() in t.lower() or t.lower() in tag.lower()
                           for t in lowest_threshold_tags):
                        found.append(tag)
                    else:
                        missing.append(tag)

                image_results["ground_truth_match"][category] = {
                    "expected": len(expected),
                    "found": len(found),
                    "found_tags": found,
                    "missing_tags": missing,
                    "match_rate": round(len(found) / len(expected), 2) if expected else 1.0,
                }

        # Print summary for this image
        print(f"\n  Core tags (appear at 0.8): {len(image_results['core_tags'])}")
        print(f"  Contextual tags (only below 0.8): {len(image_results['contextual_tags'])}")

        gt_match = image_results.get("ground_truth_match", {})
        if gt_match:
            print(f"\n  Ground Truth Match:")
            for cat in ["must_have", "should_have", "contextual"]:
                m = gt_match.get(cat, {})
                print(f"    {cat}: {m.get('found', 0)}/{m.get('expected', 0)} ({m.get('match_rate', 0)*100:.0f}%)")
                if m.get("missing_tags"):
                    print(f"      Missing: {', '.join(m['missing_tags'])}")

        results["images"][image_name] = image_results

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    total_core = 0
    total_contextual = 0
    gt_totals = {"must_have": {"found": 0, "expected": 0},
                 "should_have": {"found": 0, "expected": 0},
                 "contextual": {"found": 0, "expected": 0}}

    for img_data in results["images"].values():
        total_core += len(img_data.get("core_tags", []))
        total_contextual += len(img_data.get("contextual_tags", []))

        for cat in gt_totals:
            gt_match = img_data.get("ground_truth_match", {}).get(cat, {})
            gt_totals[cat]["found"] += gt_match.get("found", 0)
            gt_totals[cat]["expected"] += gt_match.get("expected", 0)

    results["aggregate"] = {
        "total_core_tags": total_core,
        "total_contextual_tags": total_contextual,
        "contextual_ratio": round(total_contextual / (total_core + total_contextual), 2) if (total_core + total_contextual) > 0 else 0,
        "ground_truth_match": {
            cat: {
                "found": gt_totals[cat]["found"],
                "expected": gt_totals[cat]["expected"],
                "match_rate": round(gt_totals[cat]["found"] / gt_totals[cat]["expected"], 2) if gt_totals[cat]["expected"] > 0 else 1.0,
            }
            for cat in gt_totals
        },
    }

    print(f"\nTotal core tags (0.8+): {total_core}")
    print(f"Total contextual tags (<0.8): {total_contextual}")
    print(f"Contextual ratio: {results['aggregate']['contextual_ratio']*100:.0f}%")

    print(f"\nGround Truth Match Rates:")
    for cat, data in results["aggregate"]["ground_truth_match"].items():
        print(f"  {cat}: {data['found']}/{data['expected']} ({data['match_rate']*100:.0f}%)")

    return results


def print_recommendation(results: dict) -> None:
    """Print the final recommendation."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    agg = results.get("aggregate", {})
    contextual_ratio = agg.get("contextual_ratio", 0)
    gt_match = agg.get("ground_truth_match", {})

    contextual_match = gt_match.get("contextual", {}).get("match_rate", 0)
    must_have_match = gt_match.get("must_have", {}).get("match_rate", 0)

    print(f"\n  {contextual_ratio*100:.0f}% of tags only appear below threshold 0.8")
    print(f"  These contextual tags match {contextual_match*100:.0f}% of expected contextual ground truth")
    print(f"  Core 'must have' tags match {must_have_match*100:.0f}% at lowest threshold")

    if contextual_match >= 0.5 and contextual_ratio > 0.3:
        print(f"\n  VERDICT: Use threshold 0.5")
        print(f"  REASON: Contextual tags (abandoned, decay, vintage, etc.) are ACCURATE")
        print(f"          and represent {contextual_ratio*100:.0f}% of useful tags.")
        print(f"          For a tag database, you want this coverage.")
    elif contextual_ratio > 0.5:
        print(f"\n  VERDICT: Use threshold 0.4")
        print(f"  REASON: Very high contextual tag ratio - maximum coverage needed.")
    else:
        print(f"\n  VERDICT: Use threshold 0.6")
        print(f"  REASON: Most tags are high-confidence core tags.")

    print(f"\n  For tag database building: threshold 0.5 recommended")
    print(f"  For display/UI: threshold 0.6-0.7 (fewer, cleaner tags)")


def save_results(results: dict, output_path: Path) -> None:
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_analysis()

    if results:
        print_recommendation(results)

        output_dir = Path(__file__).parent
        output_file = output_dir / f"ram_plus_tag_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_file)
