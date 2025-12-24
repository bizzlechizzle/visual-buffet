"""Florence-2 Comprehensive Benchmark - All 31 Task Combinations.

Tests every possible task combination against manually verified ground truth
to find the optimal configuration for archival/database building.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from visual_buffet.core.engine import TaggingEngine

# Test images directory
IMAGES_DIR = Path(__file__).parent.parent.parent / "images"

# ============================================================================
# GROUND TRUTH - Manually verified tags for each image
# Organized by category for false positive analysis
# ============================================================================

GROUND_TRUTH = {
    "testimage01.jpg": {
        "description": "Abandoned diner interior with bar counter and stools",
        "must_have": ["bar", "counter", "stool", "diner", "restaurant", "ceiling", "fan", "menu", "floor", "window"],
        "should_have": ["tile", "wood", "chrome", "metal", "debris", "vending", "machine", "indoor", "interior"],
        "contextual": ["abandoned", "decay", "empty", "old", "vintage", "retro"],
        "false_positive_triggers": ["person", "food", "outdoor", "nature", "animal", "car"],
    },
    "testimage02.jpg": {
        "description": "Abandoned dry cleaner with hanging clothes",
        "must_have": ["clothing", "clothes", "pants", "jeans", "hanger", "rack", "room", "ceiling"],
        "should_have": ["plastic", "bag", "conveyor", "window", "floor", "laundry", "dry", "cleaner"],
        "contextual": ["abandoned", "mess", "debris", "decay", "indoor", "interior"],
        "false_positive_triggers": ["person", "outdoor", "nature", "animal", "car", "food"],
    },
    "testimage03.jpg": {
        "description": "Abandoned church interior with pews and cross",
        "must_have": ["church", "cross", "pew", "bench", "pulpit", "altar", "wood", "carpet"],
        "should_have": ["religious", "chapel", "aisle", "wall", "green", "debris"],
        "contextual": ["abandoned", "indoor", "interior", "spiritual", "christian"],
        "false_positive_triggers": ["person", "outdoor", "car", "animal", "modern"],
    },
    "testimage04.jpg": {
        "description": "Abandoned prison/jail cell block",
        "must_have": ["prison", "jail", "cell", "bar", "cage", "metal", "rust"],
        "should_have": ["walkway", "debris", "window", "brick", "concrete", "collapsed", "roof"],
        "contextual": ["abandoned", "decay", "industrial", "interior", "indoor", "dark"],
        "false_positive_triggers": ["person", "outdoor", "nature", "car", "animal", "food"],
    },
    "testimage05.jpg": {
        "description": "Abandoned kitchen with stove and brick fireplace",
        "must_have": ["kitchen", "stove", "cabinet", "chair", "table", "brick", "fireplace"],
        "should_have": ["door", "appliance", "box", "bucket", "wood", "floor", "ceiling"],
        "contextual": ["abandoned", "mess", "debris", "decay", "indoor", "interior"],
        "false_positive_triggers": ["person", "outdoor", "nature", "car", "animal", "modern"],
    },
    "testimage06.jpg": {
        "description": "Abandoned white farmhouse exterior",
        "must_have": ["house", "farmhouse", "building", "porch", "window", "grass", "field", "tree", "sky"],
        "should_have": ["chimney", "white", "wood", "rural", "overgrown"],
        "contextual": ["abandoned", "decay", "outdoor", "exterior", "countryside"],
        "false_positive_triggers": ["person", "car", "animal", "indoor", "modern", "urban"],
    },
    "testimage07.jpg": {
        "description": "Abandoned Victorian mansion exterior with collapsed section",
        "must_have": ["house", "mansion", "building", "window", "porch", "grass", "tree", "sky", "cloud"],
        "should_have": ["victorian", "wood", "gray", "collapse", "damage", "steps"],
        "contextual": ["abandoned", "decay", "outdoor", "exterior", "historic", "old"],
        "false_positive_triggers": ["person", "car", "animal", "indoor", "modern"],
    },
    "testimage08.jpg": {
        "description": "Abandoned church exterior with steeple",
        "must_have": ["church", "steeple", "building", "grass", "field", "sky", "cloud"],
        "should_have": ["tower", "bell", "white", "wood", "autumn", "fall", "foliage", "bush"],
        "contextual": ["abandoned", "rural", "outdoor", "exterior", "religious"],
        "false_positive_triggers": ["person", "car", "animal", "indoor", "modern", "urban"],
    },
    "testimage09.jpg": {
        "description": "VW Beetle car graveyard in forest",
        "must_have": ["car", "vehicle", "automobile", "tree", "forest", "leaves"],
        "should_have": ["volkswagen", "beetle", "rust", "junkyard", "graveyard", "multiple", "winter", "bare"],
        "contextual": ["abandoned", "decay", "outdoor", "vintage", "collection"],
        "false_positive_triggers": ["person", "indoor", "building", "modern", "animal"],
    },
    "testimage10.jpg": {
        "description": "Abandoned factory/power plant by water",
        "must_have": ["factory", "building", "chimney", "smokestack", "brick", "water", "pond", "sky", "cloud"],
        "should_have": ["industrial", "power", "plant", "tree", "bush", "reflection"],
        "contextual": ["abandoned", "decay", "outdoor", "exterior"],
        "false_positive_triggers": ["person", "car", "animal", "indoor", "modern", "residential"],
    },
}

# ============================================================================
# ALL 31 TASK COMBINATIONS
# ============================================================================

TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"]

def generate_all_combinations():
    """Generate all 31 non-empty task combinations."""
    from itertools import combinations

    all_combos = {}
    combo_id = 1

    # Single tasks (5)
    for task in TASKS:
        name = task.replace("<", "").replace(">", "").lower()
        all_combos[f"single_{name}"] = [task]

    # Two-task combinations (10)
    for combo in combinations(TASKS, 2):
        name = "_".join([t.replace("<", "").replace(">", "").lower()[:3] for t in combo])
        all_combos[f"two_{name}"] = list(combo)

    # Three-task combinations (10)
    for combo in combinations(TASKS, 3):
        name = "_".join([t.replace("<", "").replace(">", "").lower()[:3] for t in combo])
        all_combos[f"three_{name}"] = list(combo)

    # Four-task combinations (5)
    for combo in combinations(TASKS, 4):
        name = "_".join([t.replace("<", "").replace(">", "").lower()[:3] for t in combo])
        all_combos[f"four_{name}"] = list(combo)

    # All five tasks (1)
    all_combos["all_five"] = TASKS.copy()

    return all_combos


def analyze_results(tags: list[str], ground_truth: dict) -> dict:
    """Analyze tags against ground truth."""
    tags_lower = set(t.lower().replace("_", " ") for t in tags)
    tags_words = set()
    for t in tags:
        # Split compound tags
        tags_words.update(t.lower().replace("_", " ").split())

    # Check must_have
    must_have_found = []
    must_have_missing = []
    for gt in ground_truth["must_have"]:
        found = False
        for tag in tags_words:
            if gt.lower() in tag or tag in gt.lower():
                found = True
                break
        if found:
            must_have_found.append(gt)
        else:
            must_have_missing.append(gt)

    # Check should_have
    should_have_found = []
    for gt in ground_truth["should_have"]:
        for tag in tags_words:
            if gt.lower() in tag or tag in gt.lower():
                should_have_found.append(gt)
                break

    # Check contextual
    contextual_found = []
    for gt in ground_truth["contextual"]:
        for tag in tags_words:
            if gt.lower() in tag or tag in gt.lower():
                contextual_found.append(gt)
                break

    # Check false positives
    false_positives = []
    for fp in ground_truth["false_positive_triggers"]:
        for tag in tags_words:
            if fp.lower() in tag or tag in fp.lower():
                false_positives.append(f"{fp} (found as: {tag})")
                break

    must_have_rate = len(must_have_found) / len(ground_truth["must_have"]) if ground_truth["must_have"] else 0
    should_have_rate = len(should_have_found) / len(ground_truth["should_have"]) if ground_truth["should_have"] else 0

    return {
        "total_tags": len(tags),
        "must_have_rate": must_have_rate,
        "must_have_found": must_have_found,
        "must_have_missing": must_have_missing,
        "should_have_found": should_have_found,
        "should_have_rate": should_have_rate,
        "contextual_found": contextual_found,
        "false_positives": false_positives,
        "false_positive_count": len(false_positives),
    }


def run_comprehensive_benchmark():
    """Run benchmark on all 31 combinations."""
    print("=" * 80)
    print("FLORENCE-2 COMPREHENSIVE BENCHMARK - ALL 31 TASK COMBINATIONS")
    print("=" * 80)
    print(f"\nTesting against manually verified ground truth for {len(GROUND_TRUTH)} images")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize engine
    print("Loading Florence-2 model...")
    engine = TaggingEngine()

    if "florence_2" not in engine.plugins:
        print("ERROR: Florence-2 plugin not found")
        return None

    if not engine.plugins["florence_2"].is_available():
        print("ERROR: Florence-2 model not available")
        return None

    print("Florence-2 ready.\n")

    # Generate all combinations
    all_combos = generate_all_combinations()
    print(f"Testing {len(all_combos)} task combinations:\n")

    results = {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "num_images": len(GROUND_TRUTH),
        "num_combinations": len(all_combos),
        "combinations": {},
        "rankings": {},
    }

    # Test each combination
    for combo_name, tasks in all_combos.items():
        print(f"\n{'='*60}")
        print(f"Testing: {combo_name}")
        print(f"Tasks: {', '.join(tasks)}")
        print("=" * 60)

        combo_results = {
            "tasks": tasks,
            "images": {},
            "summary": {},
        }

        all_must_have_rates = []
        all_should_have_rates = []
        all_false_positives = []
        all_tag_counts = []
        all_times = []

        for image_name, gt in GROUND_TRUTH.items():
            image_path = IMAGES_DIR / image_name
            if not image_path.exists():
                print(f"  SKIP: {image_name} not found")
                continue

            print(f"  Processing: {image_name}...", end=" ")

            combined_tags = []
            total_time = 0

            try:
                for task in tasks:
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

                # Deduplicate
                seen = set()
                unique_tags = []
                for tag in combined_tags:
                    tag_lower = tag.lower()
                    if tag_lower not in seen:
                        seen.add(tag_lower)
                        unique_tags.append(tag)

                # Analyze
                analysis = analyze_results(unique_tags, gt)

                combo_results["images"][image_name] = {
                    "tags": unique_tags[:50],  # Store first 50
                    "total_tags": len(unique_tags),
                    "time_ms": total_time,
                    "analysis": analysis,
                }

                all_must_have_rates.append(analysis["must_have_rate"])
                all_should_have_rates.append(analysis["should_have_rate"])
                all_false_positives.append(analysis["false_positive_count"])
                all_tag_counts.append(len(unique_tags))
                all_times.append(total_time)

                print(f"Tags: {len(unique_tags)} | Must: {analysis['must_have_rate']*100:.0f}% | FP: {analysis['false_positive_count']}")

            except Exception as e:
                print(f"ERROR: {e}")

        # Calculate summary
        if all_must_have_rates:
            combo_results["summary"] = {
                "avg_must_have_rate": sum(all_must_have_rates) / len(all_must_have_rates),
                "avg_should_have_rate": sum(all_should_have_rates) / len(all_should_have_rates),
                "avg_false_positives": sum(all_false_positives) / len(all_false_positives),
                "avg_tag_count": sum(all_tag_counts) / len(all_tag_counts),
                "avg_time_ms": sum(all_times) / len(all_times),
                "total_false_positives": sum(all_false_positives),
            }

            print(f"\n  SUMMARY: Must-Have: {combo_results['summary']['avg_must_have_rate']*100:.1f}% | "
                  f"Avg Tags: {combo_results['summary']['avg_tag_count']:.0f} | "
                  f"FP Total: {combo_results['summary']['total_false_positives']}")

        results["combinations"][combo_name] = combo_results

    # Create rankings
    print("\n" + "=" * 80)
    print("FINAL RANKINGS")
    print("=" * 80)

    # Sort by must_have_rate (primary), then by false_positives (secondary, lower is better)
    ranked = sorted(
        results["combinations"].items(),
        key=lambda x: (
            -x[1].get("summary", {}).get("avg_must_have_rate", 0),
            x[1].get("summary", {}).get("avg_false_positives", 100),
        )
    )

    print("\n" + "-" * 80)
    print(f"{'Rank':<5} {'Combination':<25} {'Must%':<8} {'Should%':<8} {'FP':<5} {'Tags':<6} {'Time':<8}")
    print("-" * 80)

    for rank, (name, data) in enumerate(ranked, 1):
        s = data.get("summary", {})
        print(f"{rank:<5} {name:<25} {s.get('avg_must_have_rate', 0)*100:<8.1f} "
              f"{s.get('avg_should_have_rate', 0)*100:<8.1f} "
              f"{s.get('total_false_positives', 0):<5.0f} "
              f"{s.get('avg_tag_count', 0):<6.0f} "
              f"{s.get('avg_time_ms', 0):<8.0f}ms")

    results["rankings"] = [name for name, _ in ranked]

    return results


def print_top_recommendations(results: dict):
    """Print detailed analysis of top 3 recommendations."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("TOP 3 RECOMMENDATIONS FOR ARCHIVAL/DATABASE BUILDING")
    print("=" * 80)

    rankings = results.get("rankings", [])[:3]

    for rank, combo_name in enumerate(rankings, 1):
        combo = results["combinations"][combo_name]
        summary = combo.get("summary", {})

        print(f"\n{'#'*3} RANK #{rank}: {combo_name} {'#'*3}")
        print(f"\nTasks: {', '.join(combo['tasks'])}")
        print(f"\nMetrics:")
        print(f"  - Must-Have Coverage: {summary.get('avg_must_have_rate', 0)*100:.1f}%")
        print(f"  - Should-Have Coverage: {summary.get('avg_should_have_rate', 0)*100:.1f}%")
        print(f"  - False Positives (total): {summary.get('total_false_positives', 0):.0f}")
        print(f"  - Average Tags per Image: {summary.get('avg_tag_count', 0):.0f}")
        print(f"  - Average Time: {summary.get('avg_time_ms', 0):.0f}ms")

        # Show per-image breakdown
        print(f"\nPer-Image Analysis:")
        for img_name, img_data in combo["images"].items():
            analysis = img_data.get("analysis", {})
            print(f"  {img_name}:")
            print(f"    Tags: {img_data['total_tags']} | Must: {analysis['must_have_rate']*100:.0f}%")
            if analysis.get("must_have_missing"):
                print(f"    Missing: {', '.join(analysis['must_have_missing'][:5])}")
            if analysis.get("false_positives"):
                print(f"    FALSE POSITIVES: {', '.join(analysis['false_positives'])}")


def save_results(results: dict, output_path: Path):
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_comprehensive_benchmark()

    if results:
        print_top_recommendations(results)

        output_dir = Path(__file__).parent
        output_file = output_dir / f"florence_2_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, output_file)
