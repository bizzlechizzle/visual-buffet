#!/usr/bin/env python3
"""Compare all OCR plugins on a given image.

Usage:
    python scripts/compare_ocr.py <image_path>

Example:
    python scripts/compare_ocr.py /path/to/image.jpg
"""

import sys
import time
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_plugin(plugin_name: str):
    """Dynamically load an OCR plugin."""
    plugin_dir = Path(__file__).parent.parent / "plugins" / plugin_name

    if plugin_name == "paddle_ocr":
        from plugins.paddle_ocr import PaddleOCRPlugin
        return PaddleOCRPlugin(plugin_dir)
    elif plugin_name == "easyocr":
        from plugins.easyocr import EasyOCRPlugin
        return EasyOCRPlugin(plugin_dir)
    elif plugin_name == "surya_ocr":
        from plugins.surya_ocr import SuryaOCRPlugin
        return SuryaOCRPlugin(plugin_dir)
    elif plugin_name == "doctr":
        from plugins.doctr import DocTRPlugin
        return DocTRPlugin(plugin_dir)
    else:
        raise ValueError(f"Unknown plugin: {plugin_name}")


def test_ocr_plugin(plugin_name: str, image_path: Path) -> dict:
    """Test a single OCR plugin and return results."""
    result = {
        "name": plugin_name,
        "available": False,
        "texts": [],
        "time_ms": 0,
        "error": None,
    }

    try:
        plugin = load_plugin(plugin_name)
        result["available"] = plugin.is_available()

        if not result["available"]:
            result["error"] = "Plugin not installed"
            return result

        # Configure for lower threshold to catch more text
        plugin.configure(threshold=0.1, limit=50)

        # Run OCR
        start = time.perf_counter()
        tag_result = plugin.tag(image_path)
        result["time_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # Extract texts
        for tag in tag_result.tags:
            result["texts"].append({
                "text": tag.label,
                "confidence": tag.confidence or 0,
            })

    except Exception as e:
        result["error"] = str(e)

    return result


def compare_ocr(image_path: Path):
    """Run all OCR plugins on an image and compare results."""

    console.print(f"\n[bold cyan]OCR Comparison: {image_path.name}[/bold cyan]\n")

    # List of OCR plugins to test - scene text focused
    # PaddleOCR and EasyOCR are the main comparison (both handle scene text)
    # Surya/docTR included for reference but are document-focused
    plugins = ["paddle_ocr", "easyocr", "doctr"]
    results = {}

    for plugin_name in plugins:
        console.print(f"Testing [yellow]{plugin_name}[/yellow]...", end=" ")
        result = test_ocr_plugin(plugin_name, image_path)
        results[plugin_name] = result

        if result["error"]:
            console.print(f"[red]Error: {result['error']}[/red]")
        elif not result["available"]:
            console.print("[red]Not installed[/red]")
        else:
            console.print(f"[green]{len(result['texts'])} texts[/green] ({result['time_ms']}ms)")

    # Create comparison table
    console.print("\n")

    # Summary table
    summary_table = Table(title="OCR Comparison Summary")
    summary_table.add_column("Engine", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Texts Found", justify="right")
    summary_table.add_column("Time (ms)", justify="right")

    for name, result in results.items():
        status = "OK" if result["available"] and not result["error"] else result["error"] or "N/A"
        texts = str(len(result["texts"])) if result["available"] else "-"
        time_ms = str(result["time_ms"]) if result["available"] else "-"
        summary_table.add_row(name, status, texts, time_ms)

    console.print(summary_table)

    # Detailed results for each engine
    console.print("\n[bold]Detailed Results:[/bold]\n")

    for name, result in results.items():
        if not result["texts"]:
            continue

        # Create panel with all detected texts
        texts_formatted = []
        for i, item in enumerate(sorted(result["texts"], key=lambda x: -x["confidence"]), 1):
            conf_color = "green" if item["confidence"] > 0.7 else "yellow" if item["confidence"] > 0.4 else "red"
            texts_formatted.append(
                f"{i:2d}. [{conf_color}]{item['confidence']:.2f}[/{conf_color}] | {item['text']}"
            )

        panel_content = "\n".join(texts_formatted) if texts_formatted else "(no text detected)"
        console.print(Panel(panel_content, title=f"[bold]{name}[/bold]", border_style="blue"))

    # Text agreement analysis
    console.print("\n[bold]Text Agreement Analysis:[/bold]\n")

    # Collect all unique texts (normalized)
    all_texts = {}
    for name, result in results.items():
        for item in result["texts"]:
            text_norm = item["text"].lower().strip()
            if text_norm not in all_texts:
                all_texts[text_norm] = {"text": item["text"], "found_by": [], "confidences": {}}
            all_texts[text_norm]["found_by"].append(name)
            all_texts[text_norm]["confidences"][name] = item["confidence"]

    # Agreement table
    agreement_table = Table(title="Cross-Engine Agreement")
    agreement_table.add_column("Text", style="white", max_width=40)
    agreement_table.add_column("PaddleOCR", justify="center")
    agreement_table.add_column("EasyOCR", justify="center")
    agreement_table.add_column("docTR", justify="center")
    agreement_table.add_column("Agreement", justify="center")

    for text_norm, data in sorted(all_texts.items(), key=lambda x: len(x[1]["found_by"]), reverse=True):
        paddle_conf = data["confidences"].get("paddle_ocr", None)
        easyocr_conf = data["confidences"].get("easyocr", None)
        doctr_conf = data["confidences"].get("doctr", None)

        paddle_str = f"{paddle_conf:.2f}" if paddle_conf else "-"
        easyocr_str = f"{easyocr_conf:.2f}" if easyocr_conf else "-"
        doctr_str = f"{doctr_conf:.2f}" if doctr_conf else "-"

        agreement = f"{len(data['found_by'])}/3"
        agreement_color = "green" if len(data["found_by"]) == 3 else "yellow" if len(data["found_by"]) == 2 else "red"

        agreement_table.add_row(
            data["text"][:40],
            paddle_str,
            easyocr_str,
            doctr_str,
            f"[{agreement_color}]{agreement}[/{agreement_color}]"
        )

    console.print(agreement_table)

    # Ground truth comparison (what we know the image says)
    expected_texts = ["International", "Fla", "SPEEDWAY", "DAYTONA BEACH"]
    console.print("\n[bold]Expected Text Match (Daytona Pennant):[/bold]\n")

    for expected in expected_texts:
        matches = []
        for name, result in results.items():
            for item in result["texts"]:
                if expected.lower() in item["text"].lower():
                    matches.append(f"{name}: '{item['text']}' ({item['confidence']:.2f})")

        if matches:
            console.print(f"  [green]✓[/green] '{expected}' found by: {', '.join(matches)}")
        else:
            console.print(f"  [red]✗[/red] '{expected}' NOT found by any engine")


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python scripts/compare_ocr.py <image_path>[/red]")
        console.print("Example: python scripts/compare_ocr.py /path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        console.print(f"[red]Image not found: {image_path}[/red]")
        sys.exit(1)

    compare_ocr(image_path)


if __name__ == "__main__":
    main()
