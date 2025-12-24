"""Florence-2 Task Analysis Benchmark - EXHAUSTIVE COMBINATORIAL TEST.

Tests ALL 31 possible task combinations (2^5 - 1) to find optimal configuration.

KEY INSIGHTS:
1. Florence-2 is TASK-BASED, not threshold-based like RAM++
2. Florence-2 does NOT provide confidence scores (all tags equal weight)
3. Slugified compound phrases (white_house, abandoned_building) are built-in
4. Different tasks produce different output types:
   - <CAPTION>: Brief description (5-10 tags)
   - <DETAILED_CAPTION>: Extended description (15-25 tags)
   - <MORE_DETAILED_CAPTION>: Comprehensive (25-50 tags)
   - <OD>: Object detection - ONLY 2-5 labels (limited!)
   - <DENSE_REGION_CAPTION>: Region-specific (15-30 tags)
   - <OCR>: Text extraction (returns text:content format, excluded from combinations)

31 PROFILES TESTED:
- 5 single-task profiles
- 10 two-task combinations
- 10 three-task combinations
- 5 four-task combinations
- 1 five-task combination (all)

This benchmark will identify which combination provides the best ground truth match
for your specific use case.
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from visual_buffet.core.engine import TaggingEngine

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

# Tasks to test
TASKS = {
    "<CAPTION>": "Brief caption - short description",
    "<DETAILED_CAPTION>": "Detailed caption - extended description",
    "<MORE_DETAILED_CAPTION>": "More detailed caption - comprehensive description",
    "<OD>": "Object detection - labels + bounding boxes",
    "<DENSE_REGION_CAPTION>": "Dense region caption - region descriptions",
    "<OCR>": "OCR - text extraction",
}

# ============================================================================
# ALL POSSIBLE TASK COMBINATIONS
# ============================================================================
# 5 tasks = 31 possible non-empty combinations (2^5 - 1)
# Testing every single one to find the optimal configuration
# ============================================================================

PROFILES = {
    # =========================================================================
    # SINGLE TASK PROFILES (5)
    # =========================================================================
    "caption_only": {
        "description": "Single: Brief caption only",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>"],
    },
    "detailed_only": {
        "description": "Single: Detailed caption only",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>"],
    },
    "more_detailed_only": {
        "description": "Single: Most detailed caption only",
        "model_variant": "large-ft",
        "tasks": ["<MORE_DETAILED_CAPTION>"],
    },
    "od_only": {
        "description": "Single: Object detection only (WARNING: limited!)",
        "model_variant": "large-ft",
        "tasks": ["<OD>"],
    },
    "dense_region_only": {
        "description": "Single: Dense region caption only",
        "model_variant": "large-ft",
        "tasks": ["<DENSE_REGION_CAPTION>"],
    },

    # =========================================================================
    # TWO-TASK COMBINATIONS (10)
    # =========================================================================
    "caption_detailed": {
        "description": "Two: Caption + Detailed",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>"],
    },
    "caption_more_detailed": {
        "description": "Two: Caption + More Detailed",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<MORE_DETAILED_CAPTION>"],
    },
    "caption_od": {
        "description": "Two: Caption + Object Detection",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<OD>"],
    },
    "caption_dense": {
        "description": "Two: Caption + Dense Region",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "detailed_more_detailed": {
        "description": "Two: Detailed + More Detailed",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
    },
    "detailed_od": {
        "description": "Two: Detailed + Object Detection",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<OD>"],
    },
    "detailed_dense": {
        "description": "Two: Detailed + Dense Region",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "more_detailed_od": {
        "description": "Two: More Detailed + Object Detection",
        "model_variant": "large-ft",
        "tasks": ["<MORE_DETAILED_CAPTION>", "<OD>"],
    },
    "more_detailed_dense": {
        "description": "Two: More Detailed + Dense Region",
        "model_variant": "large-ft",
        "tasks": ["<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "od_dense": {
        "description": "Two: Object Detection + Dense Region",
        "model_variant": "large-ft",
        "tasks": ["<OD>", "<DENSE_REGION_CAPTION>"],
    },

    # =========================================================================
    # THREE-TASK COMBINATIONS (10)
    # =========================================================================
    "all_captions": {
        "description": "Three: All caption levels (no detection)",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
    },
    "caption_detailed_od": {
        "description": "Three: Caption + Detailed + OD",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<OD>"],
    },
    "caption_detailed_dense": {
        "description": "Three: Caption + Detailed + Dense",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "caption_more_detailed_od": {
        "description": "Three: Caption + More Detailed + OD",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>"],
    },
    "caption_more_detailed_dense": {
        "description": "Three: Caption + More Detailed + Dense",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "caption_od_dense": {
        "description": "Three: Caption + OD + Dense",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
    },
    "detailed_more_detailed_od": {
        "description": "Three: Detailed + More Detailed + OD",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>"],
    },
    "detailed_more_detailed_dense": {
        "description": "Three: Detailed + More Detailed + Dense",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "detailed_od_dense": {
        "description": "Three: Detailed + OD + Dense",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
    },
    "more_detailed_od_dense": {
        "description": "Three: More Detailed + OD + Dense (ULTIMATE)",
        "model_variant": "large-ft",
        "tasks": ["<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
    },

    # =========================================================================
    # FOUR-TASK COMBINATIONS (5)
    # =========================================================================
    "no_caption": {
        "description": "Four: All except Caption",
        "model_variant": "large-ft",
        "tasks": ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
    },
    "no_detailed": {
        "description": "Four: All except Detailed",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
    },
    "no_more_detailed": {
        "description": "Four: All except More Detailed",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
    },
    "no_od": {
        "description": "Four: All except OD (captions + dense)",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
    },
    "no_dense": {
        "description": "Four: All except Dense Region",
        "model_variant": "large-ft",
        "tasks": ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>"],
    },

    # =========================================================================
    # FIVE-TASK COMBINATION (1) - THE ULTIMATE
    # =========================================================================
    "all": {
        "description": "Five: EVERY task combined - maximum extraction",
        "model_variant": "large-ft",
        "tasks": [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<OD>",
            "<DENSE_REGION_CAPTION>",
        ],
    },
}

# Ground truth - expected keywords for each image
GROUND_TRUTH = {
    "testimage01.jpg": ["bar", "stool", "diner", "restaurant", "counter", "ceiling fan", "menu"],
    "testimage02.jpg": ["clothing", "clothes", "pants", "jeans", "dry cleaner", "hanging", "rack"],
    "testimage03.jpg": ["church", "cross", "pew", "bench", "chapel", "religious", "pulpit"],
    "testimage04.jpg": ["prison", "jail", "cell", "bar", "cage", "metal", "rust"],
    "testimage05.jpg": ["kitchen", "stove", "cabinet", "chair", "table", "brick", "fireplace"],
    "testimage06.jpg": ["house", "farmhouse", "porch", "grass", "field", "tree", "white"],
    "testimage07.jpg": ["house", "victorian", "porch", "sky", "grass", "tree", "decay"],
    "testimage08.jpg": ["church", "steeple", "sky", "cloud", "grass", "field", "autumn"],
    "testimage09.jpg": ["car", "volkswagen", "beetle", "forest", "tree", "rust", "leaves"],
    "testimage10.jpg": ["factory", "building", "brick", "chimney", "water", "industrial", "smokestack"],
}


def extract_tags_from_caption(caption: str) -> list[str]:
    """Extract meaningful words from caption as tags."""
    # Common words to skip
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "and", "but", "or", "nor", "for", "yet", "so", "in", "on", "at",
        "by", "from", "to", "with", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "over",
        "this", "that", "these", "those", "it", "its", "of", "as",
        "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
        "there", "here", "some", "any", "no", "not", "very", "just", "also",
        "image", "picture", "photo", "photograph", "scene", "view", "shows",
        "appears", "seems", "looks", "see", "seen", "visible", "showing",
        "features", "includes", "contains", "displays", "depicts",
    }

    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]+\b', caption.lower())

    # Filter
    tags = []
    for word in words:
        if len(word) >= 3 and word not in stop_words:
            tags.append(word)

    # Return unique tags preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    return unique_tags


def extract_tags_from_od(od_result: dict) -> list[str]:
    """Extract labels from object detection result."""
    labels = od_result.get("labels", [])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for label in labels:
        label_lower = label.lower()
        if label_lower not in seen:
            seen.add(label_lower)
            unique.append(label_lower)
    return unique


def calculate_ground_truth_match(tags: list[str], ground_truth: list[str]) -> dict:
    """Calculate how many ground truth keywords are found in tags."""
    tags_lower = [t.lower() for t in tags]
    tags_set = set(tags_lower)

    found = []
    missing = []

    for gt in ground_truth:
        gt_lower = gt.lower()
        # Check if ground truth word appears in any tag
        matched = False
        for tag in tags_lower:
            if gt_lower in tag or tag in gt_lower:
                matched = True
                break

        if matched:
            found.append(gt)
        else:
            missing.append(gt)

    return {
        "found": found,
        "missing": missing,
        "match_rate": len(found) / len(ground_truth) if ground_truth else 0,
    }


def run_analysis() -> dict[str, Any]:
    """Run the Florence-2 task analysis."""
    print("=" * 70)
    print("FLORENCE-2 TASK ANALYSIS BENCHMARK")
    print("=" * 70)
    print("\nAnalyzing which tasks produce the best tags for different use cases.\n")

    # Initialize engine
    print("Loading Florence-2 model...")
    engine = TaggingEngine()

    if "florence_2" not in engine.plugins:
        print("ERROR: Florence-2 plugin not found")
        return {}

    if not engine.plugins["florence_2"].is_available():
        print("ERROR: Florence-2 model not available")
        return {}

    print("Florence-2 ready.\n")

    results = {
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "profiles_tested": list(PROFILES.keys()),
        "tasks_tested": list(TASKS.keys()),
        "task_results": {},
        "profile_results": {},
        "recommendations": {},
    }

    # First, test each task individually on all images
    print("\n" + "=" * 70)
    print("PHASE 1: Testing Individual Tasks")
    print("=" * 70)

    for task, description in TASKS.items():
        print(f"\n--- Testing {task}: {description} ---")

        task_data = {
            "description": description,
            "images": {},
            "summary": {},
        }

        all_tags = []
        all_match_rates = []

        for image_name in TEST_IMAGES[:3]:  # Test on first 3 images for speed
            image_path = IMAGES_DIR / image_name
            if not image_path.exists():
                continue

            print(f"  Processing: {image_name}...")

            try:
                result = engine.tag_image(
                    image_path,
                    plugin_names=["florence_2"],
                    task=task,
                    save_tags=False,
                )

                plugin_result = result["results"].get("florence_2", {})

                if "error" in plugin_result:
                    print(f"    ERROR: {plugin_result['error']}")
                    continue

                tags = plugin_result.get("tags", [])
                raw_output = plugin_result.get("raw_output", "")
                inference_time = plugin_result.get("inference_time_ms", 0)

                # Get tag labels
                tag_labels = [t.get("label", "") for t in tags]

                # Calculate ground truth match
                gt = GROUND_TRUTH.get(image_name, [])
                match = calculate_ground_truth_match(tag_labels, gt)

                task_data["images"][image_name] = {
                    "num_tags": len(tags),
                    "inference_time_ms": inference_time,
                    "tags": tag_labels[:20],  # First 20 tags
                    "raw_output_preview": raw_output[:200] if raw_output else "",
                    "ground_truth_match": match,
                }

                all_tags.append(len(tags))
                all_match_rates.append(match["match_rate"])

                print(f"    Tags: {len(tags)} | Time: {inference_time:.0f}ms | GT Match: {match['match_rate']*100:.0f}%")
                print(f"    Sample: {', '.join(tag_labels[:5])}")

            except Exception as e:
                print(f"    EXCEPTION: {e}")

        if all_tags:
            task_data["summary"] = {
                "avg_tags": round(sum(all_tags) / len(all_tags), 1),
                "min_tags": min(all_tags),
                "max_tags": max(all_tags),
                "avg_gt_match": round(sum(all_match_rates) / len(all_match_rates), 3),
            }

        results["task_results"][task] = task_data

    # Phase 2: Test ALL profiles on all images
    print("\n" + "=" * 70)
    print("PHASE 2: Testing ALL Profiles (Single → Compound → ALL)")
    print("=" * 70)

    for profile_name, profile_config in PROFILES.items():
        print(f"\n{'='*60}")
        print(f"Profile: {profile_name.upper()}")
        print(f"  Model: {profile_config['model_variant']}")
        print(f"  Tasks: {', '.join(profile_config['tasks'])}")
        print(f"  num_beams: {profile_config['num_beams']}")
        print("=" * 60)

        profile_data = {
            "config": profile_config,
            "images": {},
            "summary": {},
        }

        all_tags = []
        all_times = []
        all_match_rates = []

        for image_name in TEST_IMAGES:
            image_path = IMAGES_DIR / image_name
            if not image_path.exists():
                continue

            print(f"\n  Processing: {image_name}...")

            combined_tags = []
            total_time = 0

            try:
                for task in profile_config["tasks"]:
                    result = engine.tag_image(
                        image_path,
                        plugin_names=["florence_2"],
                        task=task,
                        save_tags=False,
                    )

                    plugin_result = result["results"].get("florence_2", {})

                    if "error" not in plugin_result:
                        tags = plugin_result.get("tags", [])
                        tag_labels = [t.get("label", "") for t in tags]
                        combined_tags.extend(tag_labels)
                        total_time += plugin_result.get("inference_time_ms", 0)

                # Deduplicate combined tags
                seen = set()
                unique_tags = []
                for tag in combined_tags:
                    tag_lower = tag.lower()
                    if tag_lower not in seen:
                        seen.add(tag_lower)
                        unique_tags.append(tag)

                # Calculate ground truth match
                gt = GROUND_TRUTH.get(image_name, [])
                match = calculate_ground_truth_match(unique_tags, gt)

                profile_data["images"][image_name] = {
                    "num_tags": len(unique_tags),
                    "inference_time_ms": total_time,
                    "tags": unique_tags[:30],
                    "ground_truth_match": match,
                }

                all_tags.append(len(unique_tags))
                all_times.append(total_time)
                all_match_rates.append(match["match_rate"])

                print(f"    Tags: {len(unique_tags)} | Time: {total_time:.0f}ms | GT Match: {match['match_rate']*100:.0f}%")
                print(f"    Sample: {', '.join(unique_tags[:8])}")

            except Exception as e:
                print(f"    EXCEPTION: {e}")

        if all_tags:
            profile_data["summary"] = {
                "avg_tags": round(sum(all_tags) / len(all_tags), 1),
                "min_tags": min(all_tags),
                "max_tags": max(all_tags),
                "avg_time_ms": round(sum(all_times) / len(all_times), 1),
                "avg_gt_match": round(sum(all_match_rates) / len(all_match_rates), 3),
            }

            print(f"\n  SUMMARY for {profile_name.upper()}:")
            print(f"    Avg tags: {profile_data['summary']['avg_tags']}")
            print(f"    Avg time: {profile_data['summary']['avg_time_ms']:.0f}ms")
            print(f"    Avg GT match: {profile_data['summary']['avg_gt_match']*100:.0f}%")

        results["profile_results"][profile_name] = profile_data

    return results


def print_recommendations(results: dict) -> None:
    """Print recommendations based on results."""
    print("\n" + "=" * 80)
    print("RESULTS - ALL 31 PROFILE COMBINATIONS RANKED BY GROUND TRUTH MATCH")
    print("=" * 80)

    # Compare profiles - SORTED by GT match rate
    profiles = results.get("profile_results", {})

    # Sort by ground truth match rate descending
    sorted_profiles = sorted(
        profiles.items(),
        key=lambda x: x[1].get("summary", {}).get("avg_gt_match", 0),
        reverse=True
    )

    print("\n" + "-" * 80)
    print(f"{'Rank':<5} {'Profile':<30} {'Tags':<8} {'Time':<10} {'GT Match':<10}")
    print("-" * 80)

    for rank, (name, data) in enumerate(sorted_profiles, 1):
        summary = data.get("summary", {})
        tags = summary.get('avg_tags', 0)
        time_ms = summary.get('avg_time_ms', 0)
        gt_match = summary.get('avg_gt_match', 0) * 100
        print(f"{rank:<5} {name:<30} {tags:<8.1f} {time_ms:<10.0f}ms {gt_match:<10.1f}%")

    # Show top 5
    print("\n" + "=" * 80)
    print("TOP 5 PROFILES BY GROUND TRUTH MATCH")
    print("=" * 80)
    for rank, (name, data) in enumerate(sorted_profiles[:5], 1):
        summary = data.get("summary", {})
        config = data.get("config", {})
        tasks = config.get("tasks", [])
        print(f"\n#{rank}: {name}")
        print(f"    Tasks: {', '.join(tasks)}")
        print(f"    Avg Tags: {summary.get('avg_tags', 0):.1f}")
        print(f"    Avg Time: {summary.get('avg_time_ms', 0):.0f}ms")
        print(f"    GT Match: {summary.get('avg_gt_match', 0)*100:.1f}%")

    # Task comparison
    tasks = results.get("task_results", {})

    print("\n" + "=" * 80)
    print("INDIVIDUAL TASK PERFORMANCE")
    print("=" * 80)
    print(f"{'Task':<30} {'Avg Tags':<12} {'GT Match':<12}")
    print("-" * 80)

    for task, data in tasks.items():
        summary = data.get("summary", {})
        if summary:
            print(f"{task:<30} {summary.get('avg_tags', 0):<12.1f} {summary.get('avg_gt_match', 0)*100:<12.1f}%")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
1. Florence-2 does NOT provide confidence scores - all tags equal weight
2. Compound phrases (white_house, abandoned_building) are built-in via slugification
3. <OD> alone is LIMITED - only 2-5 labels per image
4. More tasks = more tags, but with diminishing returns on unique coverage
5. Best profiles combine caption + region tasks

RECOMMENDATION: Use the top-ranked profile from this benchmark for your use case.
For confidence-based filtering, use RAM++ (threshold 0.5) instead.
""")


def save_results(results: dict, output_path: Path) -> None:
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_analysis()

    if results:
        print_recommendations(results)

        output_dir = Path(__file__).parent
        output_file = output_dir / f"florence_2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_file)
