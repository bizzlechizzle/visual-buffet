#!/usr/bin/env python3
"""
Analyze tagging results across image variants.
Calculates stability scores, cross-plugin agreement, and generates master tag lists.

Usage: python scripts/analyze_results.py test-results/
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from analysis_utils import (
    analyze_format_impact,
    analyze_quality_impact,
    analyze_resolution_impact,
    calculate_tag_stability,
    extract_tags_by_plugin,
    generate_master_tag_list,
)


def load_results(results_dir: Path) -> dict:
    """Load all JSON result files from a directory."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        if json_file.name in ("manifest.json", "analysis.json", "master_tags.json"):
            continue
        with open(json_file) as f:
            data = json.load(f)
            # Handle both single result and array formats
            if isinstance(data, list):
                for item in data:
                    file_path = item.get("file", "")
                    results[file_path] = item.get("results", {})
            else:
                results[json_file.stem] = data
    return results


def parse_variant_info(file_path: str) -> dict:
    """Parse resolution, format, quality from variant filename."""
    path = Path(file_path)
    name = path.stem

    parts = name.split("-")

    info = {
        "image_name": parts[0] if parts else "",
        "resolution": "original",
        "format": path.suffix.lower().strip("."),
        "quality": "original",
    }

    for part in parts[1:]:
        if part.endswith("px"):
            try:
                info["resolution"] = int(part[:-2])
            except ValueError:
                pass
        elif part in ("jpg", "jpeg", "webp", "avif", "png"):
            info["format"] = part
        elif part.startswith("q") and part[1:].isdigit():
            info["quality"] = int(part[1:])
        elif part == "original":
            info["resolution"] = "original"

    return info


def analyze_single_image(image_name: str, variant_results: list[dict]) -> dict:
    """Perform full analysis on a single image's variants."""
    tag_stability = calculate_tag_stability(variant_results)
    resolution_impact = analyze_resolution_impact(variant_results)
    format_impact = analyze_format_impact(variant_results)
    quality_impact = analyze_quality_impact(variant_results)
    master_tags = generate_master_tag_list(tag_stability)

    # Calculate summary stats
    stable_tags = [t for t in master_tags if t["stability"] >= 0.8]
    moderate_tags = [t for t in master_tags if 0.4 <= t["stability"] < 0.8]
    unstable_tags = [t for t in master_tags if t["stability"] < 0.4]
    cross_plugin_tags = [t for t in master_tags if t["cross_plugin"]]

    return {
        "image_name": image_name,
        "variant_count": len(variant_results),
        "summary": {
            "total_unique_tags": len(tag_stability),
            "stable_tags": len(stable_tags),
            "moderate_tags": len(moderate_tags),
            "unstable_tags": len(unstable_tags),
            "cross_plugin_confirmed": len(cross_plugin_tags),
        },
        "master_tags": master_tags,
        "tag_stability": tag_stability,
        "resolution_impact": resolution_impact,
        "format_impact": format_impact,
        "quality_impact": quality_impact,
    }


def print_global_summary(analysis: dict) -> None:
    """Print global analysis summary to console."""
    print("\n" + "=" * 60)
    print("GLOBAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nAverage stable tags per image: {analysis['global_stats']['avg_stable_tags_per_image']}")

    print("\nTags found by resolution:")
    for res, avg in analysis["global_stats"]["resolution_comparison"].items():
        print(f"  {res}: {avg} avg tags")

    print("\nTags found by format:")
    for fmt, avg in analysis["global_stats"]["format_comparison"].items():
        print(f"  {fmt}: {avg} avg tags")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_dir>")
        print("Example: python analyze_results.py test-results/")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Load manifest
    manifest_path = results_dir / "variants" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"images": {}}

    # Load tagging results
    tagging_results_dir = results_dir / "tagging"
    if not tagging_results_dir.exists():
        print(f"Tagging results not found at {tagging_results_dir}")
        print("Run tagging first: visual-buffet tag test-results/variants/*/")
        sys.exit(1)

    print(f"Loading results from {tagging_results_dir}")
    raw_results = load_results(tagging_results_dir)

    # Group results by source image
    results_by_image = defaultdict(list)
    for file_path, results in raw_results.items():
        variant_info = parse_variant_info(file_path)
        image_name = variant_info["image_name"]

        results_by_image[image_name].append({
            "file_path": file_path,
            "variant_info": variant_info,
            "tags_by_plugin": extract_tags_by_plugin(results),
        })

    print(f"Found results for {len(results_by_image)} images")

    # Analyze each image
    analysis = {
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(results_dir),
        "images": {},
        "global_stats": {},
    }

    all_master_tags = {}
    all_stable_counts = []
    all_resolutions = defaultdict(list)
    all_formats = defaultdict(list)

    for image_name, variants in sorted(results_by_image.items()):
        print(f"\nAnalyzing: {image_name} ({len(variants)} variants)")
        image_analysis = analyze_single_image(image_name, variants)
        analysis["images"][image_name] = image_analysis
        all_master_tags[image_name] = image_analysis["master_tags"]

        # Print summary
        summary = image_analysis["summary"]
        print(f"  Tags: {summary['total_unique_tags']} total, "
              f"{summary['stable_tags']} stable, "
              f"{summary['cross_plugin_confirmed']} cross-plugin")

        # Collect global stats
        all_stable_counts.append(summary["stable_tags"])
        for res, data in image_analysis["resolution_impact"].items():
            all_resolutions[res].append(data["avg_tag_count"])
        for fmt, data in image_analysis["format_impact"].items():
            all_formats[fmt].append(data["avg_tag_count"])

    # Global statistics
    analysis["global_stats"] = {
        "avg_stable_tags_per_image": (
            round(sum(all_stable_counts) / len(all_stable_counts), 1)
            if all_stable_counts else 0
        ),
        "resolution_comparison": {
            res: round(sum(counts) / len(counts), 1)
            for res, counts in sorted(
                all_resolutions.items(),
                key=lambda x: (isinstance(x[0], str), x[0])
            )
        },
        "format_comparison": {
            fmt: round(sum(counts) / len(counts), 1)
            for fmt, counts in sorted(all_formats.items())
        },
    }

    # Save analysis
    analysis_path = results_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n\nFull analysis saved to {analysis_path}")

    # Save master tags separately for easy access
    master_tags_path = results_dir / "master_tags.json"
    with open(master_tags_path, "w") as f:
        json.dump(all_master_tags, f, indent=2)
    print(f"Master tag lists saved to {master_tags_path}")

    # Print global summary
    print_global_summary(analysis)


if __name__ == "__main__":
    main()
