#!/usr/bin/env python3
"""Direct OCR Engine Comparison Test.

Directly tests OCR engines without going through plugin abstraction.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress verbose output
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"


def run_paddleocr(image_path: str, threshold: float = 0.3) -> Tuple[List[Tuple[str, float]], float, Optional[str]]:
    """Run PaddleOCR on an image."""
    try:
        from paddleocr import PaddleOCR

        # Suppress PaddleOCR logging
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        ocr = PaddleOCR(lang='en', use_textline_orientation=True)

        start = time.perf_counter()
        result = ocr.predict(image_path)
        elapsed = (time.perf_counter() - start) * 1000

        texts = []
        if result and len(result) > 0:
            r = result[0]
            rec_texts = r.get("rec_texts") or []
            rec_scores = r.get("rec_scores") or []

            for text, score in zip(rec_texts, rec_scores):
                if score >= threshold and text.strip():
                    texts.append((text.strip(), float(score)))

        return texts, elapsed, None

    except Exception as e:
        return [], 0, str(e)


def run_easyocr(image_path: str, threshold: float = 0.3) -> Tuple[List[Tuple[str, float]], float, Optional[str]]:
    """Run EasyOCR on an image."""
    try:
        import easyocr

        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        start = time.perf_counter()
        results = reader.readtext(image_path)
        elapsed = (time.perf_counter() - start) * 1000

        texts = []
        for bbox, text, confidence in results:
            if confidence >= threshold and text.strip():
                texts.append((text.strip(), float(confidence)))

        return texts, elapsed, None

    except Exception as e:
        return [], 0, str(e)


def run_doctr(image_path: str, threshold: float = 0.3) -> Tuple[List[Tuple[str, float]], float, Optional[str]]:
    """Run docTR on an image."""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        model = ocr_predictor(pretrained=True)

        start = time.perf_counter()
        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        elapsed = (time.perf_counter() - start) * 1000

        texts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    if line.words:
                        line_conf = sum(w.confidence for w in line.words) / len(line.words)
                    else:
                        line_conf = 0

                    if line_conf >= threshold and line_text.strip():
                        texts.append((line_text.strip(), float(line_conf)))

        return texts, elapsed, None

    except Exception as e:
        return [], 0, str(e)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    # Remove non-alphanumeric, lowercase
    return re.sub(r'[^a-z0-9]', '', text.lower())


def main():
    # Get test images
    images_dir = Path(__file__).parent.parent / "images"
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    print("=" * 80)
    print("OCR ENGINE DIRECT COMPARISON")
    print("=" * 80)
    print(f"\nFound {len(images)} images to test")

    # Track results
    all_results = []
    engine_stats = {
        "PaddleOCR": {"texts": 0, "time_ms": 0, "images_with_text": 0, "unique": 0, "errors": 0},
        "EasyOCR": {"texts": 0, "time_ms": 0, "images_with_text": 0, "unique": 0, "errors": 0},
        "docTR": {"texts": 0, "time_ms": 0, "images_with_text": 0, "unique": 0, "errors": 0},
    }

    # Pre-load models once
    print("\nLoading models...")
    print("  Loading PaddleOCR...", end=" ", flush=True)
    try:
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        paddle_ocr = PaddleOCR(lang='en', use_textline_orientation=True)
        print("OK")
    except Exception as e:
        paddle_ocr = None
        print(f"FAILED: {e}")

    print("  Loading EasyOCR...", end=" ", flush=True)
    try:
        import easyocr
        easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("OK")
    except Exception as e:
        easy_reader = None
        print(f"FAILED: {e}")

    print("  Loading docTR...", end=" ", flush=True)
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        doctr_model = ocr_predictor(pretrained=True)
        print("OK")
    except Exception as e:
        doctr_model = None
        print(f"FAILED: {e}")

    threshold = 0.3

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")

        image_results = {
            "image": image_path.name,
            "engines": {}
        }

        # PaddleOCR
        if paddle_ocr:
            try:
                start = time.perf_counter()
                result = paddle_ocr.predict(str(image_path))
                elapsed = (time.perf_counter() - start) * 1000

                texts = []
                if result and len(result) > 0:
                    r = result[0]
                    rec_texts = r.get("rec_texts") or []
                    rec_scores = r.get("rec_scores") or []

                    for text, score in zip(rec_texts, rec_scores):
                        if score >= threshold and text.strip():
                            texts.append((text.strip(), float(score)))

                engine_stats["PaddleOCR"]["time_ms"] += elapsed
                engine_stats["PaddleOCR"]["texts"] += len(texts)
                if texts:
                    engine_stats["PaddleOCR"]["images_with_text"] += 1

                image_results["engines"]["PaddleOCR"] = {
                    "texts": texts,
                    "time_ms": round(elapsed, 1),
                    "error": None
                }
                print(f"  PaddleOCR: {len(texts)} texts ({elapsed:.0f}ms)")
                if texts:
                    for t, c in texts[:3]:
                        print(f"    - {t[:50]} ({c:.2%})")
                    if len(texts) > 3:
                        print(f"    ... +{len(texts) - 3} more")

            except Exception as e:
                engine_stats["PaddleOCR"]["errors"] += 1
                image_results["engines"]["PaddleOCR"] = {"texts": [], "time_ms": 0, "error": str(e)}
                print(f"  PaddleOCR: ERROR - {str(e)[:50]}")

        # EasyOCR
        if easy_reader:
            try:
                start = time.perf_counter()
                results = easy_reader.readtext(str(image_path))
                elapsed = (time.perf_counter() - start) * 1000

                texts = []
                for bbox, text, confidence in results:
                    if confidence >= threshold and text.strip():
                        texts.append((text.strip(), float(confidence)))

                engine_stats["EasyOCR"]["time_ms"] += elapsed
                engine_stats["EasyOCR"]["texts"] += len(texts)
                if texts:
                    engine_stats["EasyOCR"]["images_with_text"] += 1

                image_results["engines"]["EasyOCR"] = {
                    "texts": texts,
                    "time_ms": round(elapsed, 1),
                    "error": None
                }
                print(f"  EasyOCR: {len(texts)} texts ({elapsed:.0f}ms)")
                if texts:
                    for t, c in texts[:3]:
                        print(f"    - {t[:50]} ({c:.2%})")
                    if len(texts) > 3:
                        print(f"    ... +{len(texts) - 3} more")

            except Exception as e:
                engine_stats["EasyOCR"]["errors"] += 1
                image_results["engines"]["EasyOCR"] = {"texts": [], "time_ms": 0, "error": str(e)}
                print(f"  EasyOCR: ERROR - {str(e)[:50]}")

        # docTR
        if doctr_model:
            try:
                start = time.perf_counter()
                doc = DocumentFile.from_images(str(image_path))
                result = doctr_model(doc)
                elapsed = (time.perf_counter() - start) * 1000

                texts = []
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text = " ".join(word.value for word in line.words)
                            if line.words:
                                line_conf = sum(w.confidence for w in line.words) / len(line.words)
                            else:
                                line_conf = 0

                            if line_conf >= threshold and line_text.strip():
                                texts.append((line_text.strip(), float(line_conf)))

                engine_stats["docTR"]["time_ms"] += elapsed
                engine_stats["docTR"]["texts"] += len(texts)
                if texts:
                    engine_stats["docTR"]["images_with_text"] += 1

                image_results["engines"]["docTR"] = {
                    "texts": texts,
                    "time_ms": round(elapsed, 1),
                    "error": None
                }
                print(f"  docTR: {len(texts)} texts ({elapsed:.0f}ms)")
                if texts:
                    for t, c in texts[:3]:
                        print(f"    - {t[:50]} ({c:.2%})")
                    if len(texts) > 3:
                        print(f"    ... +{len(texts) - 3} more")

            except Exception as e:
                engine_stats["docTR"]["errors"] += 1
                image_results["engines"]["docTR"] = {"texts": [], "time_ms": 0, "error": str(e)}
                print(f"  docTR: ERROR - {str(e)[:50]}")

        # Find unique detections
        all_normalized = {}
        for engine, data in image_results["engines"].items():
            all_normalized[engine] = {normalize_text(t[0]) for t in data.get("texts", []) if t[0]}

        for engine in all_normalized:
            others = set()
            for other_engine, norms in all_normalized.items():
                if other_engine != engine:
                    others.update(norms)
            unique = all_normalized[engine] - others
            engine_stats[engine]["unique"] += len(unique)

            if unique:
                # Find original texts
                orig = [t[0] for t in image_results["engines"][engine]["texts"] if normalize_text(t[0]) in unique]
                if orig:
                    print(f"  ** {engine} UNIQUE: {orig[:2]}")

        all_results.append(image_results)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n_images = len(images)
    for engine, stats in engine_stats.items():
        n_valid = n_images - stats["errors"]
        avg_time = stats["time_ms"] / n_valid if n_valid > 0 else 0
        print(f"\n{engine}:")
        print(f"  Images with text: {stats['images_with_text']}/{n_valid}")
        print(f"  Total texts detected: {stats['texts']}")
        print(f"  Unique findings: {stats['unique']}")
        print(f"  Avg time: {avg_time:.0f}ms")
        print(f"  Errors: {stats['errors']}")

    # Analysis
    print("\n" + "=" * 80)
    print("CASCADE STRATEGY RECOMMENDATIONS")
    print("=" * 80)

    # Calculate complementarity
    for engine1 in engine_stats:
        for engine2 in engine_stats:
            if engine1 >= engine2:
                continue

            overlap = 0
            complement = 0
            for result in all_results:
                e1 = result["engines"].get(engine1, {})
                e2 = result["engines"].get(engine2, {})

                t1 = {normalize_text(t[0]) for t in e1.get("texts", []) if t[0]}
                t2 = {normalize_text(t[0]) for t in e2.get("texts", []) if t[0]}

                if t1 or t2:
                    overlap += len(t1 & t2)
                    complement += len(t1 ^ t2)

            print(f"\n{engine1} vs {engine2}:")
            print(f"  Overlapping texts: {overlap}")
            print(f"  Complementary texts: {complement}")

            if complement > overlap:
                print(f"  -> COMPLEMENTARY: Running both catches more unique text")
            elif overlap > complement:
                print(f"  -> REDUNDANT: High overlap, one may suffice for most cases")
            else:
                print(f"  -> BALANCED: Similar overlap and complementary coverage")

    # Save results
    output_file = Path(__file__).parent.parent / "ocr_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": engine_stats,
            "results": all_results
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
