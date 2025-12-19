# IMLAGE

**Image Machine Learning Aggregate** - Compare visual tagging results from local ML tools.

## Overview

IMLAGE is a CLI-first application that processes images through multiple ML tagging plugins and aggregates comparative results. Everything runs locally with no cloud dependencies.

## Installation

```bash
# Clone the repository
git clone https://github.com/bizzlechizzle/imlage.git
cd imlage

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base package
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with ML dependencies for RAM++ plugin
pip install -e ".[ram_plus]"
```

## Quick Start

```bash
# Tag a single image
imlage tag photo.jpg

# Tag multiple images
imlage tag ./photos --recursive

# Save results to file
imlage tag photo.jpg -o results.json

# List available plugins
imlage plugins list

# Setup a plugin
imlage plugins setup ram_plus

# Show hardware capabilities
imlage hardware
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
| `config get KEY` | Get a configuration value |
| `gui` | Launch the web GUI |

## Web GUI

IMLAGE includes a web-based GUI for visual comparison of tagging results.

```bash
# Launch GUI (opens browser automatically)
imlage gui

# Custom port
imlage gui --port 9000

# Without auto-opening browser
imlage gui --no-browser
```

### GUI Features

- Drag-and-drop image upload
- Thumbnail grid view
- Lightbox detail view with full tagging results
- Settings panel for threshold and limit adjustments
- Hardware and plugin status display

### GUI Dependencies

```bash
pip install fastapi uvicorn python-multipart
# Or install with GUI extra
pip install -e ".[gui]"
```

## Plugins

### RAM++ (Recognize Anything Plus Plus)

General-purpose image tagging with ~6500 tags.

```bash
# Setup
imlage plugins setup ram_plus

# Requirements
pip install -e ".[ram_plus]"
```

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
        "model": "ram_plus_swin_large",
        "version": "1.0.0",
        "inference_time_ms": 142
      }
    }
  }
]
```

## License

MIT
