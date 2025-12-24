#!/usr/bin/env python3
"""Standalone Florence-2 benchmark - no visual-buffet dependencies."""

import json
import re
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Configuration
IMAGES_DIR = Path(__file__).parent.parent.parent / "images"
MODEL_ID = "multimodalart/Florence-2-large-no-flash-attn"

# All 5 tagging tasks
TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"]

# Ground truth for each image
GROUND_TRUTH = {
    "testimage01.jpg": {
        "must_have": ["bar", "counter", "stool", "diner", "restaurant", "ceiling", "fan", "menu", "floor", "window"],
        "false_positives": ["person", "food", "outdoor", "nature", "animal", "car"],
    },
    "testimage02.jpg": {
        "must_have": ["clothing", "clothes", "pants", "jeans", "hanger", "rack", "room", "ceiling"],
        "false_positives": ["person", "outdoor", "nature", "animal", "car", "food"],
    },
    "testimage03.jpg": {
        "must_have": ["church", "cross", "pew", "bench", "pulpit", "wood", "carpet"],
        "false_positives": ["person", "outdoor", "car", "animal", "modern"],
    },
    "testimage04.jpg": {
        "must_have": ["prison", "jail", "cell", "bar", "cage", "metal", "rust"],
        "false_positives": ["person", "outdoor", "nature", "car", "animal", "food"],
    },
    "testimage05.jpg": {
        "must_have": ["kitchen", "stove", "cabinet", "chair", "table", "brick", "fireplace"],
        "false_positives": ["person", "outdoor", "nature", "car", "animal", "modern"],
    },
    "testimage06.jpg": {
        "must_have": ["house", "building", "porch", "window", "grass", "field", "tree", "sky"],
        "false_positives": ["person", "car", "animal", "indoor", "modern", "urban"],
    },
    "testimage07.jpg": {
        "must_have": ["house", "building", "window", "porch", "grass", "tree", "sky", "cloud"],
        "false_positives": ["person", "car", "animal", "indoor", "modern"],
    },
    "testimage08.jpg": {
        "must_have": ["church", "steeple", "building", "grass", "field", "sky", "cloud"],
        "false_positives": ["person", "car", "animal", "indoor", "modern", "urban"],
    },
    "testimage09.jpg": {
        "must_have": ["car", "vehicle", "tree", "forest", "leaves"],
        "false_positives": ["person", "indoor", "building", "modern", "animal"],
    },
    "testimage10.jpg": {
        "must_have": ["factory", "building", "chimney", "brick", "water", "sky", "cloud"],
        "false_positives": ["person", "car", "animal", "indoor", "modern", "residential"],
    },
}

# Stop words for caption parsing
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "and", "but", "or", "for", "in", "on", "at", "by", "from", "to", "with",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "this", "that", "these", "those", "it", "its", "of", "as", "which",
    "there", "here", "some", "any", "no", "not", "very", "just", "also",
    "image", "shows", "showing", "appears", "seen", "visible", "features",
}


def extract_tags_from_caption(caption: str) -> list[str]:
    """Extract tags from caption with slugified bigrams."""
    phrases = re.split(r"[,.;:!?\-\(\)\[\]\"']", caption.lower())
    seen = set()
    compound_tags = []
    single_tags = []

    for phrase in phrases:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", phrase)
        # Bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in STOP_WORDS and w2 not in STOP_WORDS:
                compound = f"{w1}_{w2}"
                if compound not in seen:
                    seen.add(compound)
                    compound_tags.append(compound)
        # Single words
        for word in words:
            if word not in STOP_WORDS and word not in seen:
                seen.add(word)
                single_tags.append(word)

    return compound_tags + single_tags


def extract_tags_from_od(result: dict) -> list[str]:
    """Extract tags from object detection result."""
    labels = result.get("labels", [])
    seen = set()
    tags = []
    for label in labels:
        label_lower = label.lower()
        if label_lower not in seen:
            seen.add(label_lower)
            tags.append(label_lower)
    return tags


def run_task(model, processor, image, task: str, device, dtype) -> list[str]:
    """Run a single Florence-2 task."""
    inputs = processor(text=task, images=image, return_tensors="pt").to(device, dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))

    if task == "<OD>":
        return extract_tags_from_od(parsed.get(task, {}))
    elif task == "<DENSE_REGION_CAPTION>":
        key = task
        if key in parsed and "labels" in parsed[key]:
            all_text = " ".join(parsed[key]["labels"])
            return extract_tags_from_caption(all_text)
        return []
    else:
        caption = parsed.get(task, "")
        return extract_tags_from_caption(caption) if caption else []


def analyze_tags(tags: list[str], gt: dict) -> dict:
    """Analyze tags against ground truth."""
    tags_words = set()
    for t in tags:
        tags_words.update(t.lower().replace("_", " ").split())

    must_found = []
    must_missing = []
    for m in gt["must_have"]:
        found = any(m.lower() in t or t in m.lower() for t in tags_words)
        if found:
            must_found.append(m)
        else:
            must_missing.append(m)

    fp_found = []
    for fp in gt["false_positives"]:
        if any(fp.lower() in t or t in fp.lower() for t in tags_words):
            fp_found.append(fp)

    return {
        "must_have_rate": len(must_found) / len(gt["must_have"]) if gt["must_have"] else 0,
        "must_found": must_found,
        "must_missing": must_missing,
        "false_positives": fp_found,
        "fp_count": len(fp_found),
    }


def generate_combinations():
    """Generate all 31 task combinations."""
    combos = {}
    for r in range(1, 6):
        for combo in combinations(TASKS, r):
            name = "+".join([t.replace("<", "").replace(">", "")[:3] for t in combo])
            combos[name] = list(combo)
    return combos


def main():
    print("=" * 80)
    print("FLORENCE-2 COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print(f"\nLoading model: {MODEL_ID}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Using device: {device}, dtype: {dtype}")

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    # Get all combinations
    combos = generate_combinations()
    print(f"Testing {len(combos)} task combinations on {len(GROUND_TRUTH)} images\n")

    results = {}

    for combo_name, tasks in combos.items():
        print(f"\n{'='*60}")
        print(f"COMBO: {combo_name} ({len(tasks)} tasks)")
        print(f"Tasks: {', '.join(tasks)}")
        print("=" * 60)

        combo_data = {"tasks": tasks, "images": {}, "totals": {"must_rate": [], "fp": [], "tags": [], "time": []}}

        for img_name, gt in GROUND_TRUTH.items():
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            all_tags = []
            total_time = 0

            for task in tasks:
                start = time.perf_counter()
                tags = run_task(model, processor, image, task, device, dtype)
                elapsed = (time.perf_counter() - start) * 1000
                all_tags.extend(tags)
                total_time += elapsed

            # Deduplicate
            seen = set()
            unique_tags = []
            for t in all_tags:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    unique_tags.append(t)

            analysis = analyze_tags(unique_tags, gt)
            combo_data["images"][img_name] = {
                "tags": unique_tags[:30],
                "total": len(unique_tags),
                "time_ms": total_time,
                "analysis": analysis,
            }
            combo_data["totals"]["must_rate"].append(analysis["must_have_rate"])
            combo_data["totals"]["fp"].append(analysis["fp_count"])
            combo_data["totals"]["tags"].append(len(unique_tags))
            combo_data["totals"]["time"].append(total_time)

            print(f"  {img_name}: {len(unique_tags)} tags | Must: {analysis['must_have_rate']*100:.0f}% | FP: {analysis['fp_count']}")

        # Summary
        n = len(combo_data["totals"]["must_rate"])
        combo_data["summary"] = {
            "avg_must_rate": sum(combo_data["totals"]["must_rate"]) / n,
            "avg_fp": sum(combo_data["totals"]["fp"]) / n,
            "total_fp": sum(combo_data["totals"]["fp"]),
            "avg_tags": sum(combo_data["totals"]["tags"]) / n,
            "avg_time": sum(combo_data["totals"]["time"]) / n,
        }
        print(f"\n  AVG: Must={combo_data['summary']['avg_must_rate']*100:.1f}% | FP={combo_data['summary']['total_fp']} | Tags={combo_data['summary']['avg_tags']:.0f}")

        results[combo_name] = combo_data

    # Final rankings
    print("\n" + "=" * 80)
    print("FINAL RANKINGS (sorted by Must-Have %, then lowest FP)")
    print("=" * 80)

    ranked = sorted(
        results.items(),
        key=lambda x: (-x[1]["summary"]["avg_must_rate"], x[1]["summary"]["total_fp"])
    )

    print(f"\n{'Rank':<5} {'Combo':<25} {'Must%':<8} {'FP':<5} {'Tags':<6} {'Time':<8}")
    print("-" * 60)
    for i, (name, data) in enumerate(ranked, 1):
        s = data["summary"]
        print(f"{i:<5} {name:<25} {s['avg_must_rate']*100:<8.1f} {s['total_fp']:<5} {s['avg_tags']:<6.0f} {s['avg_time']:<8.0f}ms")

    # Top 3 recommendations
    print("\n" + "=" * 80)
    print("TOP 3 RECOMMENDATIONS FOR ARCHIVAL")
    print("=" * 80)

    for i, (name, data) in enumerate(ranked[:3], 1):
        print(f"\n### #{i}: {name}")
        print(f"Tasks: {', '.join(data['tasks'])}")
        print(f"Must-Have Coverage: {data['summary']['avg_must_rate']*100:.1f}%")
        print(f"False Positives (total): {data['summary']['total_fp']}")
        print(f"Avg Tags: {data['summary']['avg_tags']:.0f}")
        print(f"Avg Time: {data['summary']['avg_time']:.0f}ms")

    # Save results
    output_file = Path(__file__).parent / f"florence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({"rankings": [name for name, _ in ranked], "results": {k: {"summary": v["summary"], "tasks": v["tasks"]} for k, v in results.items()}}, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
