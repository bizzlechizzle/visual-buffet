"""Utility functions for analyzing tagging results.

Provides functions for calculating tag stability, resolution impact,
format impact, and quality impact across image variants.
"""

from collections import defaultdict


def extract_tags_by_plugin(results: dict) -> dict:
    """Extract tag lists organized by plugin."""
    tags_by_plugin = defaultdict(list)

    for plugin_name, plugin_result in results.items():
        if isinstance(plugin_result, dict) and "tags" in plugin_result:
            tags = plugin_result["tags"]
            for tag in tags:
                label = tag.get("label", "").lower().strip()
                confidence = tag.get("confidence")
                if label:
                    tags_by_plugin[plugin_name].append({
                        "label": label,
                        "confidence": confidence,
                    })

    return dict(tags_by_plugin)


def calculate_tag_stability(all_variant_results: list[dict]) -> dict:
    """
    Calculate stability scores for each tag across all variants.

    Returns dict of tag -> {
        stability: float (0-1),
        variant_count: int,
        total_variants: int,
        avg_confidence: float or None,
        plugins: list of plugin names that found it,
        resolutions: list of resolutions where found,
        min_resolution: smallest resolution where found,
    }
    """
    total_variants = len(all_variant_results)
    if total_variants == 0:
        return {}

    tag_data = defaultdict(lambda: {
        "appearances": 0,
        "confidences": [],
        "plugins": set(),
        "resolutions": set(),
    })

    for variant in all_variant_results:
        variant_info = variant.get("variant_info", {})
        resolution = variant_info.get("resolution", "original")
        tags_by_plugin = variant.get("tags_by_plugin", {})

        # Track which tags appear in this variant (dedupe across plugins)
        variant_tags = set()

        for plugin_name, tags in tags_by_plugin.items():
            for tag in tags:
                label = tag["label"]
                variant_tags.add(label)
                tag_data[label]["plugins"].add(plugin_name)
                if tag.get("confidence") is not None:
                    tag_data[label]["confidences"].append(tag["confidence"])

        # Count appearances and track resolutions
        for label in variant_tags:
            tag_data[label]["appearances"] += 1
            if resolution != "original":
                tag_data[label]["resolutions"].add(resolution)

    # Calculate final scores
    result = {}
    for label, data in tag_data.items():
        resolutions = sorted([r for r in data["resolutions"] if isinstance(r, int)])

        result[label] = {
            "stability": round(data["appearances"] / total_variants, 3),
            "variant_count": data["appearances"],
            "total_variants": total_variants,
            "avg_confidence": (
                round(sum(data["confidences"]) / len(data["confidences"]), 3)
                if data["confidences"]
                else None
            ),
            "plugins": sorted(data["plugins"]),
            "resolutions": resolutions,
            "min_resolution": min(resolutions) if resolutions else "original",
        }

    return result


def analyze_resolution_impact(all_variant_results: list[dict]) -> dict:
    """Analyze how resolution affects tag count and consistency."""
    by_resolution = defaultdict(list)

    for variant in all_variant_results:
        variant_info = variant.get("variant_info", {})
        resolution = variant_info.get("resolution", "original")
        tags_by_plugin = variant.get("tags_by_plugin", {})

        # Count unique tags across all plugins
        all_tags = set()
        for tags in tags_by_plugin.values():
            all_tags.update(t["label"] for t in tags)

        by_resolution[resolution].append(len(all_tags))

    result = {}
    for resolution, counts in by_resolution.items():
        result[str(resolution)] = {
            "avg_tag_count": round(sum(counts) / len(counts), 1) if counts else 0,
            "min_tag_count": min(counts) if counts else 0,
            "max_tag_count": max(counts) if counts else 0,
            "variant_count": len(counts),
        }

    return result


def analyze_format_impact(all_variant_results: list[dict]) -> dict:
    """Analyze how format affects tag count."""
    by_format = defaultdict(list)

    for variant in all_variant_results:
        variant_info = variant.get("variant_info", {})
        fmt = variant_info.get("format", "unknown")
        tags_by_plugin = variant.get("tags_by_plugin", {})

        all_tags = set()
        for tags in tags_by_plugin.values():
            all_tags.update(t["label"] for t in tags)

        by_format[fmt].append(len(all_tags))

    result = {}
    for fmt, counts in by_format.items():
        result[fmt] = {
            "avg_tag_count": round(sum(counts) / len(counts), 1) if counts else 0,
            "variant_count": len(counts),
        }

    return result


def analyze_quality_impact(all_variant_results: list[dict]) -> dict:
    """Analyze how compression quality affects tags (at test resolution only)."""
    by_quality = defaultdict(list)

    for variant in all_variant_results:
        variant_info = variant.get("variant_info", {})
        quality = variant_info.get("quality", "original")
        tags_by_plugin = variant.get("tags_by_plugin", {})

        all_tags = set()
        for tags in tags_by_plugin.values():
            all_tags.update(t["label"] for t in tags)

        by_quality[str(quality)].append(len(all_tags))

    result = {}
    for quality, counts in by_quality.items():
        result[quality] = {
            "avg_tag_count": round(sum(counts) / len(counts), 1) if counts else 0,
            "variant_count": len(counts),
        }

    return result


def generate_master_tag_list(
    tag_stability: dict, min_stability: float = 0.3
) -> list[dict]:
    """
    Generate master tag list sorted by stability and confidence.
    Only includes tags above minimum stability threshold.
    """
    master_tags = []

    for label, data in tag_stability.items():
        if data["stability"] >= min_stability:
            master_tags.append({
                "label": label,
                "stability": data["stability"],
                "confidence": data["avg_confidence"],
                "min_resolution": data["min_resolution"],
                "plugins": data["plugins"],
                "cross_plugin": len(data["plugins"]) > 1,
            })

    # Sort by stability (desc), then confidence (desc), then label
    master_tags.sort(key=lambda t: (
        -t["stability"],
        -(t["confidence"] or 0),
        t["label"]
    ))

    return master_tags
