# Visual Buffet

**Compare visual tagging results from local ML tools.**

## Overview

Visual Buffet is a CLI-first application that processes images through multiple ML tagging plugins and aggregates comparative results. Everything runs locally with no cloud dependencies.

### Features

- **9 ML Plugins** - Image tagging, object detection, OCR, and vision-language models
- **Web GUI** - Drag-and-drop interface with lightbox view and side-by-side comparison
- **Quality Levels** - Quick (480px), Standard (1080px), Max (2048px) processing
- **RAW Support** - Process Sony ARW, Canon CR2/CR3, Nikon NEF, Adobe DNG, and more
- **HEIC/HEIF Support** - Native Apple image format support
- **Automatic Thumbnails** - Generated in `visual-buffet/` folder next to images

## Installation

```bash
# Clone the repository
git clone https://github.com/bizzlechizzle/visual-buffet.git
cd visual-buffet

# Install with uv (recommended)
uv pip install -e ".[gui,all_models]"

# Or with pip
pip install -e ".[gui,all_models]"
```

### Optional Dependencies

```bash
# Individual plugin installs
uv pip install -e ".[ram_plus]"      # RAM++ tagging
uv pip install -e ".[siglip]"        # SigLIP zero-shot
uv pip install -e ".[florence_2]"    # Florence-2 captioning
uv pip install -e ".[yolo]"          # YOLO object detection
uv pip install -e ".[easyocr]"       # EasyOCR text recognition
uv pip install -e ".[doctr]"         # docTR OCR
uv pip install -e ".[qwen3_vl]"      # Qwen3-VL (requires Ollama)

# Development tools
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Tag a single image
visual-buffet tag photo.jpg

# Tag multiple images
visual-buffet tag ./photos --recursive

# Save results to file
visual-buffet tag photo.jpg -o results.json

# List available plugins
visual-buffet plugins list

# Show hardware capabilities
visual-buffet hardware

# Launch web GUI
visual-buffet gui
```

## Commands

| Command | Description |
|---------|-------------|
| `tag PATH` | Tag image(s) using configured plugins |
| `plugins list` | List available plugins |
| `plugins setup NAME` | Download and setup a plugin |
| `plugins info NAME` | Show plugin details |
| `hardware` | Show detected hardware capabilities |
| `config show` | Show current configuration |
| `config set KEY VALUE` | Set a configuration value |
| `gui` | Launch the web GUI |

## Web GUI

Visual Buffet includes a web-based GUI for visual comparison of tagging results.

```bash
# Launch GUI (opens browser automatically)
visual-buffet gui

# Custom port
visual-buffet gui --port 9000

# Without auto-opening browser
visual-buffet gui --no-browser
```

### GUI Features

- Drag-and-drop image upload
- Thumbnail grid view with batch processing
- Full-screen tag modal with side-by-side plugin comparison
- Per-plugin settings (threshold, quality, discovery mode)
- Global quality presets (Quick/Standard/Max)
- Hardware and plugin status display
- Persistent settings saved to disk

## Plugins

### Image Tagging

| Plugin | Description | Tags | Recommended Setting |
|--------|-------------|------|---------------------|
| **RAM++** | Recognize Anything Plus Plus - general purpose tagging | ~4585 | `--threshold 0.5` |
| **SigLIP** | Google's zero-shot classifier with discovery mode | Custom vocabulary | `--threshold 0.01` |
| **Florence-2** | Microsoft vision foundation model | Captioning + tags | `--task "<DETAILED_CAPTION>"` |

### RAM++ Threshold Recommendation

**Use `--threshold 0.5` for RAM++** (not 0.6 or 0.8).

Empirical testing revealed that contextual tags (debris, damage, rust, antique, historic) have confidence scores in the 0.50-0.70 range. These tags are **accurate** but would be filtered out at higher thresholds.

| Threshold | What You Get | What You Lose |
|-----------|--------------|---------------|
| 0.8 | Primary objects only | All contextual tags |
| 0.6 | Primary + some context | rust, antique, some contextual |
| **0.5** | **All accurate tags** | **Only noise** |

```bash
# Recommended for database building
visual-buffet tag photo.jpg --threshold 0.5

# For display/UI (cleaner, fewer tags)
visual-buffet tag photo.jpg --threshold 0.6
```

See [RAM++ SME Guide](docs/sme/ram_plus.sme.md) for complete analysis.

### Florence-2 Task Recommendation

Florence-2 is **task-based, not threshold-based** and provides **no confidence scores**. Tags are extracted from captions via NLP parsing with built-in slugification (e.g., `white_house`, `abandoned_building`).

> **Critical Finding**: `<OD>` (Object Detection) only achieves 14.9% coverage vs 70%+ for caption tasks. Do NOT use `<OD>` alone!

#### Top 3 Configurations (Benchmarked on 10 Images)

| Rank | Profile | Tasks | Coverage | FP | Tags | Time |
|------|---------|-------|----------|----|----- |------|
| #1 | **Maximum** | `<DETAILED_CAPTION>` + `<MORE_DETAILED_CAPTION>` + `<DENSE_REGION_CAPTION>` | 76.1% | 6 | 77 | 2915ms |
| #2 | **Balanced** ⭐ | `<MORE_DETAILED_CAPTION>` + `<DENSE_REGION_CAPTION>` | 73.4% | 3 | 65 | 2141ms |
| #3 | **Fast** | `<MORE_DETAILED_CAPTION>` only | 70.5% | 1 | 46 | 1212ms |

#### When to Use Each

**#1 Maximum Coverage** - Archives, museums, digital asset management where completeness matters more than speed.

**#2 Balanced (Recommended)** - Best for most archival work. Removes `<DETAILED_CAPTION>` (redundant with MORE_DETAILED) for 27% faster processing with minimal coverage loss.

**#3 Fast** - High-volume processing, previews, or when you need quick results. Single task = simplest pipeline.

```bash
# Balanced (recommended for database building)
visual-buffet tag photo.jpg --plugin florence_2 --profile balanced

# Maximum coverage
visual-buffet tag photo.jpg --plugin florence_2 --profile maximum

# Fast processing
visual-buffet tag photo.jpg --plugin florence_2 --profile fast
```

#### Key Benchmark Findings

- **Caption stacking works**: Combining caption tasks yields 73-76% coverage
- **`<OD>` is worthless alone**: Only 14.9% coverage, 3 tags avg, 5 false positives
- **`<DENSE_REGION_CAPTION>` adds regional context**: Boosts coverage 3-6%
- **`<CAPTION>` is redundant**: Adds minimal value when other captions present

See [Florence-2 SME Guide](docs/sme/florence_2.sme.md) for complete benchmark data and all 31 combinations tested.

### Object Detection

| Plugin | Description |
|--------|-------------|
| **YOLO** | YOLOv8 real-time detection with 80 COCO classes + bounding boxes |

### OCR (Text Recognition)

| Plugin | Description | Languages |
|--------|-------------|-----------|
| **PaddleOCR** | High-speed text detection with bounding boxes | 100+ |
| **EasyOCR** | CRAFT+CRNN scene text recognition | 80+ |
| **docTR** | Mindee deep learning OCR | Multi-language |
| **Surya OCR** | Vision transformer with layout analysis | 90+ |

### Vision-Language Models

| Plugin | Description | Requires |
|--------|-------------|----------|
| **Qwen3-VL** | Prompted image analysis with tagging, describe, or both modes | Ollama |

## Supported Image Formats

### Standard Formats
- JPEG, PNG, WebP, BMP, TIFF
- HEIC/HEIF (Apple)

### RAW Formats
- Sony: ARW
- Canon: CR2, CR3
- Nikon: NEF
- Adobe: DNG
- Olympus: ORF
- Panasonic: RW2
- Fujifilm: RAF
- Pentax: PEF
- Samsung: SRW

## Output Format

```json
[
  {
    "file": "photo.jpg",
    "results": {
      "ram_plus": {
        "tags": [
          {"label": "dog", "confidence": 0.95},
          {"label": "outdoor", "confidence": 0.89}
        ],
        "model": "RAM++",
        "version": "1.3.0",
        "quality": "standard",
        "resolutions": [1080],
        "inference_time_ms": 142
      },
      "siglip": {
        "tags": [
          {"label": "golden retriever", "confidence": 0.87}
        ],
        "model": "SigLIP",
        "version": "1.0.0",
        "quality": "standard",
        "resolutions": [1080],
        "inference_time_ms": 89
      }
    }
  }
]
```

## Configuration

Settings are stored in `~/.config/visual-buffet/` and include:
- Plugin enable/disable states
- Per-plugin thresholds and quality levels
- Discovery mode settings for SigLIP
- VLM mode settings for Qwen3-VL

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (CUDA)
- **Optimal**: RTX 3090/4090 with 24GB VRAM for all plugins simultaneously

## License

MIT
