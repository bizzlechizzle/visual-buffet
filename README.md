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

| Plugin | Description | Tags |
|--------|-------------|------|
| **RAM++** | Recognize Anything Plus Plus - general purpose tagging | ~4585 |
| **SigLIP** | Google's zero-shot classifier with discovery mode | Custom vocabulary |
| **Florence-2** | Microsoft vision foundation model | Captioning + tags |

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
