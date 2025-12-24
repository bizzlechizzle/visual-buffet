#!/usr/bin/env python3
"""Test OCR Verification Architecture.

Runs PaddleOCR + docTR + SigLIP verification on test images
to demonstrate the trust-but-verify auto-tagging approach.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    bbox: list = field(default_factory=list)


@dataclass
class VerifiedText:
    text: str
    normalized: str
    tier: str  # VERIFIED, LIKELY, UNVERIFIED, REJECTED
    verification_score: float
    paddle_conf: float
    doctr_conf: Optional[float]
    siglip_score: Optional[float]
    auto_tag: bool
    searchable: bool


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def texts_match(t1: str, t2: str) -> bool:
    """Check if two texts match (fuzzy)."""
    n1, n2 = normalize_text(t1), normalize_text(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    if len(n1) > 3 and len(n2) > 3:
        ratio = SequenceMatcher(None, n1, n2).ratio()
        return ratio > 0.8
    return False


def compute_verification_score(
    paddle_conf: float,
    doctr_found: bool,
    siglip_score: float
) -> float:
    """Compute composite verification score."""
    return (
        0.5 * paddle_conf +
        0.3 * (1.0 if doctr_found else 0.0) +
        0.2 * min(siglip_score * 10, 1.0)
    )


def determine_tier(
    paddle_conf: float,
    doctr_found: bool,
    siglip_score: float,
    verification_score: float
) -> str:
    """Determine confidence tier."""
    # VERIFIED conditions
    if verification_score >= 0.7:
        return "VERIFIED"
    if paddle_conf >= 0.9 and doctr_found:
        return "VERIFIED"
    if paddle_conf >= 0.8 and siglip_score >= 0.01:
        return "VERIFIED"

    # LIKELY conditions
    if verification_score >= 0.5:
        return "LIKELY"
    if paddle_conf >= 0.7:
        return "LIKELY"
    if doctr_found:
        return "LIKELY"

    # REJECTED conditions
    if paddle_conf < 0.3 and not doctr_found and siglip_score < 0.001:
        return "REJECTED"

    return "UNVERIFIED"


def run_verification_pipeline(image_path: Path) -> list[VerifiedText]:
    """Run full verification pipeline on an image."""
    import torch
    from PIL import Image

    results = []

    # 1. Run PaddleOCR
    print("    Running PaddleOCR...", end=" ", flush=True)
    paddle_results = []
    try:
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        ocr = PaddleOCR(lang='en', use_textline_orientation=True)
        result = ocr.predict(str(image_path))

        if result and len(result) > 0:
            r = result[0]
            rec_texts = r.get("rec_texts") or []
            rec_scores = r.get("rec_scores") or []

            for text, score in zip(rec_texts, rec_scores):
                if score >= 0.3 and text.strip():
                    paddle_results.append(OCRResult(
                        text=text.strip(),
                        confidence=float(score),
                        engine="paddle_ocr"
                    ))

        print(f"{len(paddle_results)} texts")
    except Exception as e:
        print(f"ERROR: {e}")
        return []

    if not paddle_results:
        return []

    # 2. Run docTR
    print("    Running docTR...", end=" ", flush=True)
    doctr_results = []
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        model = ocr_predictor(pretrained=True)
        doc = DocumentFile.from_images(str(image_path))
        result = model(doc)

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    if line.words:
                        line_conf = sum(w.confidence for w in line.words) / len(line.words)
                    else:
                        line_conf = 0

                    if line_conf >= 0.3 and line_text.strip():
                        doctr_results.append(OCRResult(
                            text=line_text.strip(),
                            confidence=float(line_conf),
                            engine="doctr"
                        ))

        print(f"{len(doctr_results)} texts")
    except Exception as e:
        print(f"ERROR: {e}")

    # 3. Run SigLIP for validation
    print("    Running SigLIP validation...", end=" ", flush=True)
    siglip_scores = {}
    try:
        from transformers import AutoModel, AutoProcessor

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            torch_dtype=dtype
        ).to(device)
        model.eval()

        image = Image.open(image_path).convert("RGB")

        # Score each paddle result
        texts_to_score = [r.text for r in paddle_results]
        template = "A sign reading {}."
        prompts = [template.format(t.lower()) for t in texts_to_score]

        inputs = processor(
            text=prompts,
            images=image,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)

        if dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.sigmoid(logits).squeeze(0).cpu().float().numpy()

        for text, score in zip(texts_to_score, probs):
            siglip_scores[normalize_text(text)] = float(score)

        print(f"scored {len(siglip_scores)} texts")
    except Exception as e:
        print(f"ERROR: {e}")

    # 4. Compute verification for each PaddleOCR result
    for pr in paddle_results:
        normalized = normalize_text(pr.text)

        # Check if docTR found the same text
        doctr_match = None
        for dr in doctr_results:
            if texts_match(pr.text, dr.text):
                doctr_match = dr
                break

        doctr_found = doctr_match is not None
        doctr_conf = doctr_match.confidence if doctr_match else None
        siglip_score = siglip_scores.get(normalized, 0.0)

        verification_score = compute_verification_score(
            pr.confidence, doctr_found, siglip_score
        )

        tier = determine_tier(
            pr.confidence, doctr_found, siglip_score, verification_score
        )

        auto_tag = tier in ("VERIFIED", "LIKELY")
        searchable = tier != "REJECTED"

        results.append(VerifiedText(
            text=pr.text,
            normalized=normalized,
            tier=tier,
            verification_score=verification_score,
            paddle_conf=pr.confidence,
            doctr_conf=doctr_conf,
            siglip_score=siglip_score,
            auto_tag=auto_tag,
            searchable=searchable,
        ))

    return results


def main():
    """Run verification test on sample images."""
    images_dir = Path(__file__).parent.parent / "images"

    # Select images with known text
    test_images = [
        "the-coliseum-interior-sign.jpg",      # Clear sign
        "5c80a25be26b6.image.jpeg",            # Historical marker
        "_DSC1047.jpg",                         # Scoreboard (only PaddleOCR finds)
        "16195360_1469198399779947_1598506099658570170_n.jpg",  # Poster
    ]

    print("=" * 80)
    print("OCR VERIFICATION PIPELINE TEST")
    print("=" * 80)
    print("\nTrust but Verify: PaddleOCR → docTR → SigLIP")

    all_results = {}
    tier_counts = {"VERIFIED": 0, "LIKELY": 0, "UNVERIFIED": 0, "REJECTED": 0}

    for image_name in test_images:
        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"\nSkipping {image_name} - not found")
            continue

        print(f"\n{'='*80}")
        print(f"IMAGE: {image_name}")
        print(f"{'='*80}")

        results = run_verification_pipeline(image_path)
        all_results[image_name] = results

        # Display results by tier
        print(f"\n  Results ({len(results)} texts):")

        for tier in ["VERIFIED", "LIKELY", "UNVERIFIED", "REJECTED"]:
            tier_results = [r for r in results if r.tier == tier]
            if tier_results:
                print(f"\n  {tier} ({len(tier_results)}):")
                for r in tier_results[:5]:  # Show first 5
                    doctr_str = f"docTR:{r.doctr_conf:.0%}" if r.doctr_conf else "docTR:✗"
                    siglip_str = f"SigLIP:{r.siglip_score:.4f}"
                    print(f"    {'✓' if r.auto_tag else '✗'} '{r.text[:40]}' "
                          f"[Paddle:{r.paddle_conf:.0%} {doctr_str} {siglip_str}] "
                          f"score:{r.verification_score:.2f}")
                if len(tier_results) > 5:
                    print(f"    ... +{len(tier_results) - 5} more")

                tier_counts[tier] += len(tier_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = sum(tier_counts.values())
    print(f"\nTotal texts processed: {total}")
    for tier, count in tier_counts.items():
        pct = count / total * 100 if total > 0 else 0
        action = {
            "VERIFIED": "→ Auto-tag, prime search",
            "LIKELY": "→ Auto-tag, secondary search",
            "UNVERIFIED": "→ Manual review queue",
            "REJECTED": "→ Not stored (noise)",
        }[tier]
        print(f"  {tier}: {count} ({pct:.0f}%) {action}")

    auto_tagged = tier_counts["VERIFIED"] + tier_counts["LIKELY"]
    print(f"\nAuto-tagging rate: {auto_tagged}/{total} ({auto_tagged/total*100:.0f}%)")
    print(f"Noise filtered: {tier_counts['REJECTED']}/{total} ({tier_counts['REJECTED']/total*100:.0f}%)")


if __name__ == "__main__":
    main()
