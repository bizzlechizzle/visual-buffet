#!/usr/bin/env python3
"""Test: Can SigLIP validate OCR-detected text?

Tests whether SigLIP scores higher for text that actually appears in an image
vs text that doesn't. This determines if SigLIP is useful for OCR validation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

def test_siglip_ocr_validation():
    """Test SigLIP's ability to validate OCR findings."""

    # Test images with known text
    test_cases = [
        {
            "image": "the-coliseum-interior-sign.jpg",
            "real_text": ["RINK", "POLICY", "SKATE", "ROLLER", "MANAGEMENT"],
            "fake_text": ["PIZZA", "COMPUTER", "ELEPHANT", "KEYBOARD", "SUBMARINE"],
        },
        {
            "image": "5c80a25be26b6.image.jpeg",
            "real_text": ["CERES", "POWER", "HOUSE", "ENGINE", "1906"],
            "fake_text": ["APPLE", "MICROSOFT", "BANANA", "CLOUD", "WATER"],
        },
        {
            "image": "_DSC1047.jpg",  # Scoreboard - only PaddleOCR found text here
            "real_text": ["HOME", "GUEST", "STRONG"],
            "fake_text": ["PIZZA", "LAPTOP", "FLOWER", "OCEAN", "MUSIC"],
        },
    ]

    images_dir = Path(__file__).parent.parent / "images"

    print("=" * 80)
    print("SIGLIP OCR VALIDATION TEST")
    print("=" * 80)
    print("\nQuestion: Can SigLIP distinguish text that IS in an image from text that ISN'T?")
    print("\nLoading SigLIP model...")

    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        # Load SigLIP
        model_id = "google/siglip-so400m-patch14-384"

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        print(f"  Device: {device}")

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        model.eval()

        print("  Model loaded!")

    except Exception as e:
        print(f"Failed to load SigLIP: {e}")
        return

    from PIL import Image

    # Prompt templates to test
    templates = [
        "This is a photo of {}.",
        "An image containing the word {}.",
        "Text that says {}.",
        "A sign reading {}.",
        "The word {} is visible.",
    ]

    overall_results = {
        "real_text_scores": [],
        "fake_text_scores": [],
    }

    for case in test_cases:
        image_path = images_dir / case["image"]
        if not image_path.exists():
            print(f"\nSkipping {case['image']} - not found")
            continue

        print(f"\n{'='*80}")
        print(f"IMAGE: {case['image']}")
        print(f"{'='*80}")

        image = Image.open(image_path).convert("RGB")

        # Test each template
        for template in templates:
            print(f"\nTemplate: \"{template}\"")
            print("-" * 60)

            # Combine real and fake text for single batch
            all_text = case["real_text"] + case["fake_text"]
            prompts = [template.format(text.lower()) for text in all_text]

            # Process inputs
            inputs = processor(
                text=prompts,
                images=image,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(device)

            if dtype == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = torch.sigmoid(logits).squeeze(0).cpu().float().numpy()

            # Split results
            n_real = len(case["real_text"])
            real_scores = list(zip(case["real_text"], probs[:n_real]))
            fake_scores = list(zip(case["fake_text"], probs[n_real:]))

            # Sort by score
            real_scores.sort(key=lambda x: -x[1])
            fake_scores.sort(key=lambda x: -x[1])

            avg_real = sum(s for _, s in real_scores) / len(real_scores)
            avg_fake = sum(s for _, s in fake_scores) / len(fake_scores)

            print(f"  REAL TEXT (avg: {avg_real:.4f}):")
            for text, score in real_scores:
                overall_results["real_text_scores"].append(score)
                marker = "✓" if score > avg_fake else "✗"
                print(f"    {marker} '{text}': {score:.4f}")

            print(f"  FAKE TEXT (avg: {avg_fake:.4f}):")
            for text, score in fake_scores:
                overall_results["fake_text_scores"].append(score)
                print(f"      '{text}': {score:.4f}")

            # Can we distinguish?
            separation = avg_real - avg_fake
            if separation > 0.01:
                print(f"  → SEPARABLE: Real scores {separation:.4f} higher than fake")
            elif separation > 0:
                print(f"  → MARGINAL: Real scores only {separation:.4f} higher")
            else:
                print(f"  → NOT SEPARABLE: Fake scores higher by {-separation:.4f}")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    all_real = overall_results["real_text_scores"]
    all_fake = overall_results["fake_text_scores"]

    if all_real and all_fake:
        avg_real = sum(all_real) / len(all_real)
        avg_fake = sum(all_fake) / len(all_fake)
        max_real = max(all_real)
        max_fake = max(all_fake)
        min_real = min(all_real)
        min_fake = min(all_fake)

        print(f"\nReal text scores:  avg={avg_real:.4f}, min={min_real:.4f}, max={max_real:.4f}")
        print(f"Fake text scores:  avg={avg_fake:.4f}, min={min_fake:.4f}, max={max_fake:.4f}")
        print(f"Separation:        {avg_real - avg_fake:.4f}")

        # Classification accuracy
        # Can we use a threshold to separate real from fake?
        best_threshold = 0
        best_accuracy = 0

        for thresh in [i * 0.001 for i in range(1, 100)]:
            correct = sum(1 for s in all_real if s >= thresh) + sum(1 for s in all_fake if s < thresh)
            accuracy = correct / (len(all_real) + len(all_fake))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = thresh

        print(f"\nBest threshold: {best_threshold:.3f} (accuracy: {best_accuracy:.1%})")

        if avg_real > avg_fake * 1.5:
            print("\n✓ CONCLUSION: SigLIP CAN distinguish real text from fake text")
            print("  → SigLIP OCR validation is VIABLE")
        elif avg_real > avg_fake:
            print("\n⚠ CONCLUSION: SigLIP shows WEAK separation")
            print("  → May help but not definitive")
        else:
            print("\n✗ CONCLUSION: SigLIP CANNOT distinguish real text from fake")
            print("  → SigLIP OCR validation is NOT useful")


if __name__ == "__main__":
    test_siglip_ocr_validation()
