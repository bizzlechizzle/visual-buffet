# Visual Buffet

**Compare visual tagging results from local ML tools.**

Visual Buffet is a CLI-first application that processes images through multiple ML tagging plugins and aggregates comparative results. Everything runs locally with no cloud dependencies.

## Features

- **10 ML Plugins** - Image tagging, object detection, OCR, and vision-language models
- **XMP Integration** - Works with wake-n-blake/shoemaker pipeline for unified metadata
- **Web GUI** - Drag-and-drop interface with lightbox view and side-by-side comparison
- **Quality Levels** - Little (480px), Small (1080px), Large (2048px), Huge (4096px) processing
- **RAW Support** - Process Sony ARW, Canon CR2/CR3, Nikon NEF, Adobe DNG, and more
- **HEIC/HEIF Support** - Native Apple image format support
- **Discovery Mode** - SigLIP vocabulary discovery using RAM++/Florence-2

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Optional: [exiftool](https://exiftool.org/) for XMP integration

```bash
# Clone the repository
git clone https://github.com/bizzlechizzle/visual-buffet.git
cd visual-buffet

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Install exiftool for XMP Integration

```bash
# macOS
brew install exiftool

# Ubuntu/Debian
sudo apt-get install libimage-exiftool-perl

# Windows
# Download from https://exiftool.org/
```

### Optional Plugin Dependencies

```bash
# Individual plugin installs
uv pip install -e ".[ram_plus]"      # RAM++ tagging
uv pip install -e ".[siglip]"        # SigLIP zero-shot
uv pip install -e ".[florence_2]"    # Florence-2 captioning
uv pip install -e ".[yolo]"          # YOLO object detection
uv pip install -e ".[easyocr]"       # EasyOCR text recognition
uv pip install -e ".[doctr]"         # docTR OCR
uv pip install -e ".[qwen3_vl]"      # Qwen3-VL (requires Ollama)

# All plugins
uv pip install -e ".[all_models]"

# GUI dependencies
uv pip install -e ".[gui]"

# Development tools
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Tag a single image
visual-buffet tag photo.jpg

# Tag multiple images recursively
visual-buffet tag ./photos --recursive

# Save results to file
visual-buffet tag photo.jpg -o results.json

# Use specific threshold
visual-buffet tag photo.jpg --threshold 0.5

# Use smaller image size for faster processing
visual-buffet tag photo.jpg --size small

# Discovery mode (SigLIP + vocabulary from other plugins)
visual-buffet tag photo.jpg --discover

# List available plugins
visual-buffet plugins list

# Setup a plugin
visual-buffet plugins setup ram_plus

# Show hardware capabilities
visual-buffet hardware

# Launch web GUI
visual-buffet gui
```

## CLI Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `tag PATH [OPTIONS]` | Tag image(s) using configured plugins |
| `plugins list` | List available plugins |
| `plugins setup NAME` | Download and setup a plugin |
| `plugins info NAME` | Show plugin details |
| `hardware [--refresh]` | Show detected hardware capabilities |
| `config show` | Show current configuration |
| `config set KEY VALUE` | Set a configuration value |
| `config get KEY` | Get a configuration value |
| `gui [OPTIONS]` | Launch the web GUI |

### Tag Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --plugin NAME` | all | Use specific plugin(s), can repeat |
| `-o, --output FILE` | stdout | Save JSON results to file |
| `-f, --format FORMAT` | json | Output format |
| `--threshold FLOAT` | 0.5 | Minimum confidence (0.0-1.0) |
| `--recursive` | false | Search folders recursively |
| `--size SIZE` | original | Image size: little, small, large, huge, original |
| `--discover` | false | Enable SigLIP discovery mode |

### GUI Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | 127.0.0.1 | Host to bind to |
| `--port PORT` | 8420 | Port to bind to |
| `--no-browser` | false | Don't open browser automatically |

### Image Size Presets

| Size | Resolution | Use Case |
|------|------------|----------|
| `little` | 480px | Fastest, batch processing |
| `small` | 1080px | Balanced speed/quality |
| `large` | 2048px | High detail |
| `huge` | 4096px | Maximum detail |
| `original` | Native | Default, no resize |

## XMP Pipeline Integration

Visual Buffet integrates with the wake-n-blake → shoemaker → visual-buffet pipeline:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  wake-n-blake   │ ──► │    shoemaker    │ ──► │  visual-buffet  │
│   (import)      │     │  (thumbnails)   │     │   (ML tags)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

When exiftool is available, visual-buffet:
- Reads existing XMP sidecars created by wake-n-blake
- Adds ML tags to both `vbuffet:` namespace and `dc:subject`
- Preserves existing metadata from other tools
- Updates custody chain events

Without exiftool, visual-buffet falls back to JSON files (`{filename}_tags.json`).

## Plugins

### Image Tagging

| Plugin | Description | Tags | Recommended Setting |
|--------|-------------|------|---------------------|
| **RAM++** | Recognize Anything Plus Plus | ~4585 | `--threshold 0.5` |
| **SigLIP** | Google zero-shot classifier | Custom vocabulary | `--threshold 0.01` |
| **Florence-2** | Microsoft vision foundation model | Captioning + tags | `--profile balanced` |

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

### RAM++ Threshold Recommendation

**Use `--threshold 0.5` for RAM++** (not 0.6 or 0.8).

Contextual tags (debris, damage, rust, antique, historic) have confidence scores in the 0.50-0.70 range. These tags are accurate but would be filtered out at higher thresholds.

| Threshold | What You Get | What You Lose |
|-----------|--------------|---------------|
| 0.8 | Primary objects only | All contextual tags |
| 0.6 | Primary + some context | rust, antique, some contextual |
| **0.5** | **All accurate tags** | **Only noise** |

### Florence-2 Task Recommendation

Florence-2 is task-based and provides no confidence scores. Tags are extracted from captions via NLP parsing.

| Profile | Tasks | Coverage | Time |
|---------|-------|----------|------|
| **Maximum** | DETAILED + MORE_DETAILED + DENSE_REGION | 76% | 2915ms |
| **Balanced** ⭐ | MORE_DETAILED + DENSE_REGION | 73% | 2141ms |
| **Fast** | MORE_DETAILED only | 70% | 1212ms |

## Supported Image Formats

### Standard Formats
- JPEG, PNG, WebP, BMP, TIFF

### Apple Formats
- HEIC, HEIF

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
      }
    }
  }
]
```

## Configuration

Settings are stored in `~/.config/visual-buffet/config.toml`:

```toml
[general]
default_threshold = 0.5
default_format = "json"

[plugins.ram_plus]
enabled = true
threshold = 0.5
batch_size = 4

[plugins.siglip]
enabled = true
threshold = 0.01
```

### Configuration Commands

```bash
# View all settings
visual-buffet config show

# Set a value
visual-buffet config set general.default_threshold 0.6

# Get a value
visual-buffet config get general.default_threshold
```

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
- Global quality presets (Little/Small/Large/Huge/Original)
- Hardware and plugin status display
- Persistent settings saved to disk

## Hardware Requirements

| Level | CPU | RAM | GPU |
|-------|-----|-----|-----|
| **Minimum** | Any | 8GB | None (CPU-only) |
| **Recommended** | Modern | 16GB | NVIDIA 8GB+ VRAM |
| **Optimal** | Modern | 32GB | RTX 3090/4090 24GB |

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/bizzlechizzle/visual-buffet.git
cd visual-buffet
uv sync --dev

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Run type checker
uv run mypy src
```

### Project Structure

```
visual-buffet/
├── src/visual_buffet/
│   ├── cli.py              # CLI entry point
│   ├── constants.py        # Application constants
│   ├── exceptions.py       # Exception hierarchy
│   ├── core/
│   │   ├── engine.py       # Tagging orchestrator
│   │   └── hardware.py     # Hardware detection
│   ├── plugins/
│   │   ├── base.py         # Plugin base class
│   │   ├── loader.py       # Plugin discovery
│   │   └── schemas.py      # Data schemas
│   ├── services/
│   │   └── xmp_handler.py  # XMP metadata integration
│   ├── utils/
│   │   ├── config.py       # TOML config management
│   │   └── image.py        # Image loading utilities
│   └── gui/
│       ├── server.py       # FastAPI backend
│       └── static/         # Frontend assets
├── plugins/                # ML plugin implementations
├── tests/                  # Test suite
└── docs/                   # Documentation
```

### Creating a Plugin

```python
from pathlib import Path
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult

class MyPlugin(PluginBase):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
        )

    def is_available(self) -> bool:
        return self.get_model_path().exists()

    def tag(self, image_path: Path) -> TagResult:
        # Your ML inference here
        return TagResult(
            tags=[Tag(label="example", confidence=0.95)],
            model="my_model",
            version="1.0.0",
        )
```

## Troubleshooting

### "No plugins available"

Plugins need to be set up before use:

```bash
# List all plugins
visual-buffet plugins list

# Setup a specific plugin
visual-buffet plugins setup ram_plus
```

### "exiftool not found"

XMP integration requires exiftool:

```bash
# macOS
brew install exiftool

# Ubuntu/Debian
sudo apt-get install libimage-exiftool-perl
```

### Out of memory errors

Reduce batch size or use smaller image size:

```bash
# Use smaller images
visual-buffet tag photos/ --size small

# Or configure in config.toml
visual-buffet config set plugins.ram_plus.batch_size 1
```

### GPU not detected

Check CUDA installation:

```bash
# Check hardware detection
visual-buffet hardware --refresh

# Force CPU-only mode (via environment)
CUDA_VISIBLE_DEVICES="" visual-buffet tag photo.jpg
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File not found |
| 3 | Invalid input |
| 4 | Plugin error |
| 5 | No plugins available |
| 130 | Keyboard interrupt |

## License

MIT
