#!/usr/bin/env python3
"""
Run complete tag quality test pipeline.
1. Generate image variants at different resolutions/formats
2. Tag all variants with available plugins
3. Analyze results and generate master tag lists

Usage: python scripts/run_quality_test.py images/ test-results/
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_quality_test.py <source_images_dir> <output_dir>")
        print("Example: python run_quality_test.py images/ test-results/")
        sys.exit(1)

    source_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(1)

    # Create output structure
    variants_dir = output_dir / "variants"
    tagging_dir = output_dir / "tagging"
    variants_dir.mkdir(parents=True, exist_ok=True)
    tagging_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    venv_imlage = Path(__file__).parent.parent / ".venv" / "bin" / "imlage"

    print("\nTag Quality Test Pipeline")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}")

    # Step 1: Generate variants
    success = run_command(
        [str(venv_python), str(scripts_dir / "generate_variants.py"), str(source_dir), str(variants_dir)],
        "Generate image variants"
    )
    if not success:
        print("ERROR: Failed to generate variants")
        sys.exit(1)

    # Step 2: Collect all variant images
    print(f"\n{'='*60}")
    print("STEP: Collecting variant images for tagging")
    print(f"{'='*60}")

    variant_images = []
    for image_dir in variants_dir.iterdir():
        if image_dir.is_dir():
            for img_file in image_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
                    variant_images.append(str(img_file))

    print(f"Found {len(variant_images)} variant images to tag")

    if not variant_images:
        print("ERROR: No variant images found")
        sys.exit(1)

    # Step 3: Tag all variants
    # Do this in batches to avoid command line length limits
    batch_size = 20
    all_results = []

    for i in range(0, len(variant_images), batch_size):
        batch = variant_images[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(variant_images) + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"STEP: Tagging batch {batch_num}/{total_batches}")
        print(f"{'='*60}")

        output_file = tagging_dir / f"results_batch_{batch_num:03d}.json"

        cmd = [str(venv_imlage), "tag"] + batch + ["--output", str(output_file)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Batch {batch_num} had errors")
            print(result.stderr)
        else:
            print(f"Batch {batch_num} complete: {output_file.name}")

    # Step 4: Merge all results into single file
    print(f"\n{'='*60}")
    print("STEP: Merging results")
    print(f"{'='*60}")

    all_results = []
    for result_file in sorted(tagging_dir.glob("results_batch_*.json")):
        with open(result_file) as f:
            batch_results = json.load(f)
            if isinstance(batch_results, list):
                all_results.extend(batch_results)

    merged_path = tagging_dir / "all_results.json"
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Merged {len(all_results)} results into {merged_path}")

    # Step 5: Analyze results
    success = run_command(
        [str(venv_python), str(scripts_dir / "analyze_results.py"), str(output_dir)],
        "Analyze results"
    )
    if not success:
        print("ERROR: Failed to analyze results")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print("\nOutputs:")
    print(f"  Variants: {variants_dir}/")
    print(f"  Tagging:  {tagging_dir}/")
    print(f"  Analysis: {output_dir / 'analysis.json'}")
    print(f"  Master:   {output_dir / 'master_tags.json'}")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
