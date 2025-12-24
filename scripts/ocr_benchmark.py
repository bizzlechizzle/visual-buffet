#!/usr/bin/env python3
"""OCR Engine Benchmark Script.

Runs all available OCR engines (PaddleOCR, EasyOCR, docTR, Florence-2) on test images
and compares their results to identify complementary strengths and weaknesses.

Usage:
    python scripts/ocr_benchmark.py [--images-dir ./images] [--output ./ocr_benchmark_results.json]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add plugins to path
plugins_path = Path(__file__).parent.parent / "plugins"
if str(plugins_path) not in sys.path:
    sys.path.insert(0, str(plugins_path))


@dataclass
class OCRResult:
    """Result from a single OCR engine on a single image."""
    engine: str
    texts: list[str]
    confidences: list[float]
    inference_time_ms: float
    error: str | None = None


@dataclass
class ImageBenchmark:
    """Benchmark results for a single image."""
    image_path: str
    image_name: str
    results: dict[str, OCRResult] = field(default_factory=dict)
    has_text: bool = False
    unique_findings: dict[str, list[str]] = field(default_factory=dict)


def run_paddleocr(image_path: Path, threshold: float = 0.3) -> OCRResult:
    """Run PaddleOCR on an image."""
    try:
        from paddle_ocr import PaddleOCRPlugin

        plugin = PaddleOCRPlugin(plugins_path / "paddle_ocr")
        plugin.configure(threshold=threshold)

        start = time.perf_counter()
        result = plugin.tag(image_path)
        elapsed = (time.perf_counter() - start) * 1000

        texts = [t.label for t in result.tags]
        confs = [t.confidence or 0.0 for t in result.tags]

        return OCRResult(
            engine="PaddleOCR",
            texts=texts,
            confidences=confs,
            inference_time_ms=elapsed,
        )
    except Exception as e:
        return OCRResult(
            engine="PaddleOCR",
            texts=[],
            confidences=[],
            inference_time_ms=0,
            error=str(e),
        )


def run_easyocr(image_path: Path, threshold: float = 0.3) -> OCRResult:
    """Run EasyOCR on an image."""
    try:
        from easyocr import EasyOCRPlugin

        plugin = EasyOCRPlugin(plugins_path / "easyocr")
        plugin.configure(threshold=threshold)

        start = time.perf_counter()
        result = plugin.tag(image_path)
        elapsed = (time.perf_counter() - start) * 1000

        texts = [t.label for t in result.tags]
        confs = [t.confidence or 0.0 for t in result.tags]

        return OCRResult(
            engine="EasyOCR",
            texts=texts,
            confidences=confs,
            inference_time_ms=elapsed,
        )
    except Exception as e:
        return OCRResult(
            engine="EasyOCR",
            texts=[],
            confidences=[],
            inference_time_ms=0,
            error=str(e),
        )


def run_doctr(image_path: Path, threshold: float = 0.3) -> OCRResult:
    """Run docTR on an image."""
    try:
        from doctr import DocTRPlugin

        plugin = DocTRPlugin(plugins_path / "doctr")
        plugin.configure(threshold=threshold)

        start = time.perf_counter()
        result = plugin.tag(image_path)
        elapsed = (time.perf_counter() - start) * 1000

        texts = [t.label for t in result.tags]
        confs = [t.confidence or 0.0 for t in result.tags]

        return OCRResult(
            engine="docTR",
            texts=texts,
            confidences=confs,
            inference_time_ms=elapsed,
        )
    except Exception as e:
        return OCRResult(
            engine="docTR",
            texts=[],
            confidences=[],
            inference_time_ms=0,
            error=str(e),
        )


def run_florence2_ocr(image_path: Path) -> OCRResult:
    """Run Florence-2 OCR on an image."""
    try:
        from florence_2 import Florence2Plugin
        from PIL import Image

        plugin = Florence2Plugin(plugins_path / "florence_2")

        start = time.perf_counter()

        # Load model if needed
        if plugin._model is None:
            plugin._load_model()

        # Run OCR task directly
        image = Image.open(image_path).convert("RGB")
        inputs = plugin._processor(
            text="<OCR>",
            images=image,
            return_tensors="pt"
        ).to(plugin._device)

        outputs = plugin._model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

        generated_text = plugin._processor.batch_decode(outputs, skip_special_tokens=False)[0]
        result = plugin._processor.post_process_generation(
            generated_text,
            task="<OCR>",
            image_size=image.size
        )

        elapsed = (time.perf_counter() - start) * 1000

        # Florence-2 OCR returns text without confidence scores
        ocr_text = result.get("<OCR>", "")
        texts = [t.strip() for t in ocr_text.split('\n') if t.strip()] if ocr_text else []

        return OCRResult(
            engine="Florence-2",
            texts=texts,
            confidences=[1.0] * len(texts),  # Florence-2 doesn't provide confidence
            inference_time_ms=elapsed,
        )
    except Exception as e:
        return OCRResult(
            engine="Florence-2",
            texts=[],
            confidences=[],
            inference_time_ms=0,
            error=str(e),
        )


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def find_unique_texts(results: dict[str, OCRResult]) -> dict[str, list[str]]:
    """Find texts detected by only one engine."""
    all_texts: dict[str, set[str]] = {}

    for engine, result in results.items():
        if result.error:
            continue
        all_texts[engine] = {normalize_text(t) for t in result.texts if t.strip()}

    unique: dict[str, list[str]] = {}

    for engine, texts in all_texts.items():
        other_texts = set()
        for other_engine, other in all_texts.items():
            if other_engine != engine:
                other_texts.update(other)

        unique_for_engine = texts - other_texts
        if unique_for_engine:
            # Get original case versions
            original_texts = [t for t in results[engine].texts if normalize_text(t) in unique_for_engine]
            unique[engine] = original_texts

    return unique


def run_benchmark(images_dir: Path, threshold: float = 0.3) -> list[ImageBenchmark]:
    """Run OCR benchmark on all images in directory."""
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
    images = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # Sort for consistent ordering
    images.sort()

    print(f"\nFound {len(images)} images to benchmark")
    print("=" * 80)

    results: list[ImageBenchmark] = []

    # Track which engines are available
    engines = [
        ("PaddleOCR", run_paddleocr),
        ("EasyOCR", run_easyocr),
        ("docTR", run_doctr),
        ("Florence-2", run_florence2_ocr),
    ]

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {image_path.name}")

        benchmark = ImageBenchmark(
            image_path=str(image_path),
            image_name=image_path.name,
        )

        for engine_name, run_fn in engines:
            print(f"  Running {engine_name}...", end=" ", flush=True)

            if engine_name == "Florence-2":
                result = run_fn(image_path)
            else:
                result = run_fn(image_path, threshold)

            benchmark.results[engine_name] = result

            if result.error:
                print(f"ERROR: {result.error[:50]}...")
            else:
                print(f"Found {len(result.texts)} text(s) in {result.inference_time_ms:.0f}ms")

        # Check if any engine found text
        benchmark.has_text = any(
            len(r.texts) > 0 for r in benchmark.results.values() if not r.error
        )

        # Find unique findings
        benchmark.unique_findings = find_unique_texts(benchmark.results)

        results.append(benchmark)

    return results


def generate_report(results: list[ImageBenchmark]) -> str:
    """Generate a human-readable report."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("OCR BENCHMARK REPORT")
    report.append("=" * 80)

    # Summary statistics
    total_images = len(results)
    images_with_text = sum(1 for r in results if r.has_text)

    report.append(f"\nTotal images: {total_images}")
    report.append(f"Images with detected text: {images_with_text}")

    # Engine performance summary
    report.append("\n" + "-" * 80)
    report.append("ENGINE PERFORMANCE SUMMARY")
    report.append("-" * 80)

    engine_stats: dict[str, dict] = {}
    for result in results:
        for engine, ocr_result in result.results.items():
            if engine not in engine_stats:
                engine_stats[engine] = {
                    "total_texts": 0,
                    "total_time_ms": 0,
                    "images_with_text": 0,
                    "errors": 0,
                    "unique_finds": 0,
                }

            if ocr_result.error:
                engine_stats[engine]["errors"] += 1
            else:
                engine_stats[engine]["total_texts"] += len(ocr_result.texts)
                engine_stats[engine]["total_time_ms"] += ocr_result.inference_time_ms
                if ocr_result.texts:
                    engine_stats[engine]["images_with_text"] += 1

        for engine, unique in result.unique_findings.items():
            engine_stats[engine]["unique_finds"] += len(unique)

    for engine, stats in engine_stats.items():
        n = total_images - stats["errors"]
        avg_time = stats["total_time_ms"] / n if n > 0 else 0
        report.append(f"\n{engine}:")
        report.append(f"  Images with text: {stats['images_with_text']}/{n}")
        report.append(f"  Total texts found: {stats['total_texts']}")
        report.append(f"  Unique finds (only this engine): {stats['unique_finds']}")
        report.append(f"  Avg inference time: {avg_time:.0f}ms")
        report.append(f"  Errors: {stats['errors']}")

    # Images with text detected
    report.append("\n" + "-" * 80)
    report.append("IMAGES WITH DETECTED TEXT")
    report.append("-" * 80)

    for result in results:
        if not result.has_text:
            continue

        report.append(f"\n{result.image_name}:")

        for engine, ocr_result in result.results.items():
            if ocr_result.error:
                report.append(f"  {engine}: ERROR - {ocr_result.error[:50]}")
            elif ocr_result.texts:
                texts_preview = ocr_result.texts[:5]
                if len(ocr_result.texts) > 5:
                    texts_preview.append(f"... ({len(ocr_result.texts) - 5} more)")
                report.append(f"  {engine} ({len(ocr_result.texts)} texts):")
                for t in texts_preview:
                    report.append(f"    - {t}")
            else:
                report.append(f"  {engine}: (no text found)")

        if result.unique_findings:
            report.append("  UNIQUE FINDINGS:")
            for engine, unique in result.unique_findings.items():
                report.append(f"    {engine} only: {unique[:3]}")

    # Cascade recommendations
    report.append("\n" + "-" * 80)
    report.append("CASCADE STRATEGY ANALYSIS")
    report.append("-" * 80)

    # Calculate overlap and complement
    engine_list = list(engine_stats.keys())
    for i, eng1 in enumerate(engine_list):
        for eng2 in engine_list[i+1:]:
            overlap = 0
            complement = 0
            for result in results:
                if result.has_text:
                    r1 = result.results.get(eng1)
                    r2 = result.results.get(eng2)
                    if r1 and r2 and not r1.error and not r2.error:
                        t1 = {normalize_text(t) for t in r1.texts}
                        t2 = {normalize_text(t) for t in r2.texts}
                        if t1 and t2:
                            overlap += len(t1 & t2)
                            complement += len(t1 ^ t2)

            if overlap + complement > 0:
                report.append(f"\n{eng1} vs {eng2}:")
                report.append(f"  Overlapping texts: {overlap}")
                report.append(f"  Complementary texts: {complement}")
                if complement > overlap:
                    report.append(f"  -> COMPLEMENTARY: Running both catches more text")
                else:
                    report.append(f"  -> REDUNDANT: High overlap, one may suffice")

    return "\n".join(report)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="OCR Engine Benchmark")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "ocr_benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for OCR results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OCR ENGINE BENCHMARK")
    print("=" * 80)
    print(f"Images directory: {args.images_dir}")
    print(f"Confidence threshold: {args.threshold}")

    # Run benchmark
    results = run_benchmark(args.images_dir, args.threshold)

    # Generate and print report
    report = generate_report(results)
    print(report)

    # Save results to JSON
    output_data = {
        "threshold": args.threshold,
        "images_dir": str(args.images_dir),
        "results": [
            {
                "image_path": r.image_path,
                "image_name": r.image_name,
                "has_text": r.has_text,
                "results": {
                    engine: {
                        "engine": res.engine,
                        "texts": res.texts,
                        "confidences": res.confidences,
                        "inference_time_ms": res.inference_time_ms,
                        "error": res.error,
                    }
                    for engine, res in r.results.items()
                },
                "unique_findings": r.unique_findings,
            }
            for r in results
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
