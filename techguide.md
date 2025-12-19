# IMLAGE Technical Guide

Implementation details, build setup, environment configuration, and troubleshooting.

## Environment Setup

### Requirements

- Python >=3.11
- uv (recommended) or pip
- Platform: macOS, Linux, Windows

### Installation

```bash
# Clone and enter
git clone https://github.com/bizzlechizzle/imlage.git
cd imlage

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Development Installation

```bash
# Install with dev dependencies
uv sync --dev

# Verify installation
uv run imlage --version
uv run pytest
```

## Project Structure

```
imlage/
├── CLAUDE.md           # Project rules (read first)
├── techguide.md        # This file
├── lilbits.md          # Script registry
├── pyproject.toml      # Project config and dependencies
├── src/
│   └── imlage/
│       ├── __init__.py
│       ├── cli.py      # CLI entry point
│       ├── core/       # Core logic
│       ├── plugins/    # Plugin interface and registry
│       └── gui/        # Web GUI (FastAPI + frontend)
├── plugins/            # Installed plugins (git-ignored)
├── tests/
│   └── ...
└── docs/
    └── sme/            # Plugin SME files
```

## Configuration

### User Config Location

- macOS: `~/.config/imlage/config.toml`
- Linux: `~/.config/imlage/config.toml`
- Windows: `%APPDATA%\imlage\config.toml`

### Hardware Cache

Hardware detection results cached at:
- `~/.imlage/hardware.json`

Delete to force re-detection.

### Plugin Config

Each plugin has its own config section in the main config:

```toml
[plugins.ram_plus]
enabled = true
model_path = "/path/to/model"
batch_size = 4
```

## Plugin Development

### Plugin Interface

Every plugin must implement:

```python
from imlage.plugins import PluginBase, TagResult, PluginInfo

class MyPlugin(PluginBase):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="my_plugin",
            version="1.0.0",
            description="My tagging plugin",
            hardware_reqs={"gpu": False, "min_ram_gb": 4}
        )

    def is_available(self) -> bool:
        # Check if model files exist, dependencies met
        return True

    def tag(self, image_path: Path) -> TagResult:
        # Process image, return tags
        return TagResult(tags=[
            {"label": "cat", "confidence": 0.95},
            {"label": "animal", "confidence": 0.99}
        ])
```

### Plugin Directory Structure

```
plugins/my_plugin/
├── __init__.py         # Exports plugin class
├── plugin.toml         # Plugin metadata
├── my_plugin.sme.md    # SME documentation
└── models/             # Model files (git-ignored)
```

### plugin.toml Format

```toml
[plugin]
name = "my_plugin"
version = "1.0.0"
entry_point = "MyPlugin"
python_requires = ">=3.11"

[plugin.dependencies]
torch = ">=2.0"
```

## Hardware Detection

On first run, IMLAGE detects:

- GPU: CUDA version, VRAM, device name
- Metal: Apple Silicon GPU capabilities
- CPU: Core count, architecture
- RAM: Total system memory

Results inform plugin batch sizes and model variant selection.

## GUI Architecture

When GUI mode is requested:

1. FastAPI server starts on `localhost:8420`
2. Serves static frontend from `src/imlage/gui/static/`
3. CLI becomes the backend via API endpoints
4. Browser opens automatically (or user navigates manually)

### Launching the GUI

```bash
# Default (opens browser automatically)
imlage gui

# Custom port
imlage gui --port 9000

# Without auto-opening browser
imlage gui --no-browser
```

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Serve main HTML page |
| GET | `/api/status` | System status (hardware, plugins) |
| GET | `/api/plugins` | List all plugins with availability |
| GET | `/api/config` | Get configuration |
| POST | `/api/upload` | Upload image for tagging |
| GET | `/api/image/{id}` | Get full image data + results |
| POST | `/api/tag/{id}` | Tag a specific image |
| POST | `/api/tag-batch` | Tag multiple images |
| DELETE | `/api/image/{id}` | Delete uploaded image |
| DELETE | `/api/images` | Clear all images |

### Frontend Design System (Braun/Ulm)

The GUI follows Dieter Rams' Functional Minimalism principles.

**Color Palette:**
- Canvas: `#FAFAF8` (background)
- Surface: `#FFFFFF` (cards)
- Text Primary: `#1C1C1A`
- Text Secondary: `#5C5C58`
- Border: `#E2E1DE`

**Functional Colors (semantic only):**
- Success/High Confidence: `#4A8C5E`
- Info/Medium Confidence: `#5A7A94`
- Warning/Low Confidence: `#C9A227`
- Error: `#B85C4A`

**Rules:**
- 8pt spacing grid
- Border-radius max 4px
- No colored accent buttons
- No decorative shadows
- No gradients
- System sans-serif font only

### Frontend State Management

```javascript
const state = {
    images: new Map(),     // id -> {filename, thumbnail, results, ...}
    selectedImage: null,   // For lightbox view
    settings: {
        threshold: 0.5,
        limit: 50,
        plugins: [],
    },
    hardware: null,
    processing: false,
};
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=imlage

# Run specific test file
uv run pytest tests/test_cli.py

# Run specific test
uv run pytest tests/test_cli.py::test_tag_single_file
```

## Linting and Formatting

```bash
# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Building

```bash
# Build distribution
uv build

# Output in dist/
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: imlage` | Not installed | Run `uv sync` or `pip install -e .` |
| Plugin not loading | Missing dependencies | Check plugin's requirements in SME file |
| GPU not detected | Driver issues | Check CUDA/Metal installation |
| Out of memory | Batch too large | Reduce batch_size in plugin config |

### Debug Mode

```bash
# Run with debug logging
IMLAGE_DEBUG=1 uv run imlage tag image.jpg

# Or via CLI flag
uv run imlage --debug tag image.jpg
```

### Log Location

Logs written to:
- `~/.imlage/logs/imlage.log`

## Performance Tuning

### Batch Processing

For large folders, IMLAGE batches images. Adjust per-plugin:

```toml
[plugins.ram_plus]
batch_size = 8  # Higher if you have more VRAM
```

### Parallel Plugins

Run multiple plugins simultaneously:

```bash
uv run imlage tag folder/ --parallel
```

Requires sufficient RAM for all active models.
