# Visual Buffet Technical Guide

Implementation details, build setup, environment configuration, and troubleshooting.

## Environment Setup

### Requirements

- Python >=3.11
- uv (recommended) or pip
- Platform: macOS, Linux, Windows

### Installation

```bash
# Clone and enter
git clone https://github.com/bizzlechizzle/visual-buffet.git
cd visual-buffet

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
uv run visual-buffet --version
uv run pytest
```

## Project Structure

```
visual-buffet/
├── CLAUDE.md           # Project rules (read first)
├── techguide.md        # This file
├── lilbits.md          # Script registry
├── pyproject.toml      # Project config and dependencies
├── src/
│   └── visual_buffet/
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

- macOS: `~/.config/visual-buffet/config.toml`
- Linux: `~/.config/visual-buffet/config.toml`
- Windows: `%APPDATA%\visual-buffet\config.toml`

### Hardware Cache

Hardware detection results cached at:
- `~/.visual-buffet/hardware.json`

Delete to force re-detection.

### Tag Storage

Tags are stored as JSON files next to each source image:

```
/photos/vacation/
├── beach.jpg
└── beach_tags.json      # Saved tags
```

This keeps tag data with the source images, making it:
- Portable (move folder, tags follow)
- Backup-friendly (backup images, tags come along)
- Easy to inspect/debug

### Plugin Config

Each plugin has its own config section in the main config:

```toml
[plugins.ram_plus]
enabled = true
threshold = 0.0
batch_size = 4
```

See plugin-specific settings references:
- [RAM++ Settings Reference](docs/reference/ram_plus_settings.md)
- [Florence-2 Settings Reference](docs/reference/florence_2_settings.md)
- [SigLIP Settings Reference](docs/reference/siglip_settings.md)
- [PaddleOCR Settings Reference](docs/reference/paddle_ocr_settings.md)

## Plugin Development

### Plugin Interface

Every plugin must implement:

```python
from visual-buffet.plugins import PluginBase, TagResult, PluginInfo

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

On first run, Visual Buffet detects:

- GPU: CUDA version, VRAM, device name
- Metal: Apple Silicon GPU capabilities
- CPU: Core count, architecture
- RAM: Total system memory

Results inform plugin batch sizes and model variant selection.

## CLI Usage

### Basic Tagging

```bash
# Tag a single image
visual-buffet tag photo.jpg

# Tag folder contents
visual-buffet tag ./photos

# Tag recursively
visual-buffet tag ./photos --recursive
```

### Image Size Option

Control the resolution used for ML inference with `--size`:

| Size | Resolution | Use Case |
|------|------------|----------|
| `little` | 480px | Fastest, good for batch processing |
| `small` | 1080px | Balanced speed/quality |
| `large` | 2048px | High detail |
| `huge` | 4096px | Maximum detail |
| `original` | Native | Default, no resize |

```bash
# Use small resolution for faster processing
visual-buffet tag photo.jpg --size small

# Use large resolution for more detail
visual-buffet tag photo.jpg --size large
```

### Discovery Mode

Use SigLIP with RAM++/Florence-2 vocabulary discovery:

```bash
visual-buffet tag photo.jpg --discover
```

## GUI Architecture

When GUI mode is requested:

1. FastAPI server starts on `localhost:8420`
2. Serves static frontend from `src/visual-buffet/gui/static/`
3. CLI becomes the backend via API endpoints
4. Browser opens automatically (or user navigates manually)

### Launching the GUI

```bash
# Default (opens browser automatically)
visual-buffet gui

# Custom port
visual-buffet gui --port 9000

# Without auto-opening browser
visual-buffet gui --no-browser
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
        size: "original",  // little, small, large, huge, original
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
uv run pytest --cov=visual-buffet

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
| `ModuleNotFoundError: visual-buffet` | Not installed | Run `uv sync` or `pip install -e .` |
| Plugin not loading | Missing dependencies | Check plugin's requirements in SME file |
| GPU not detected | Driver issues | Check CUDA/Metal installation |
| Out of memory | Batch too large | Reduce batch_size in plugin config |

### Debug Mode

```bash
# Run with debug logging
VISUAL_BUFFET_DEBUG=1 uv run visual-buffet tag image.jpg

# Or via CLI flag
uv run visual-buffet --debug tag image.jpg
```

### Log Location

Logs written to:
- `~/.visual-buffet/logs/visual-buffet.log`

## Performance Tuning

### Batch Processing

For large folders, Visual Buffet batches images. Adjust per-plugin:

```toml
[plugins.ram_plus]
batch_size = 8  # Higher if you have more VRAM
```

### Parallel Plugins

Run multiple plugins simultaneously:

```bash
uv run visual-buffet tag folder/ --parallel
```

Requires sufficient RAM for all active models.

---

## XMP Pipeline Integration

visual-buffet is the **third stage** in the media processing pipeline, running after wake-n-blake and shoemaker.

### Pipeline Order (CRITICAL)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  wake-n-blake   │ ──► │    shoemaker    │ ──► │  visual-buffet  │
│   (import)      │     │  (thumbnails)   │     │   (ML tags)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**wake-n-blake MUST run first** to create the base XMP sidecar with provenance data.

### Current Implementation: JSON Files

**visual-buffet does NOT currently write to XMP.** Tags are stored in JSON files:

```
/photos/vacation/
├── beach.jpg
├── beach.jpg.xmp        # wake-n-blake + shoemaker data
└── beach_tags.json      # visual-buffet tags (SEPARATE FILE)
```

This breaks the unified XMP pipeline.

### TODO: XMP Integration

visual-buffet needs to write ML tags to XMP sidecars instead of (or in addition to) JSON files.

**Option A: Use wake-n-blake's namespace**

Add ML tags to the `wnb:` namespace:
```xml
<wnb:MLTags>
  <rdf:Bag>
    <rdf:li rdf:parseType="Resource">
      <wnb:Plugin>ram_plus</wnb:Plugin>
      <wnb:Label>beach</wnb:Label>
      <wnb:Confidence>0.95</wnb:Confidence>
    </rdf:li>
  </rdf:Bag>
</wnb:MLTags>
```

**Option B: Use Dublin Core keywords**

Write to standard XMP fields:
```xml
<dc:subject>
  <rdf:Bag>
    <rdf:li>beach</rdf:li>
    <rdf:li>water</rdf:li>
    <rdf:li>sand</rdf:li>
  </rdf:Bag>
</dc:subject>
```

**Option C: Custom namespace**

Create `vbuffet:` namespace:
```
Namespace URI: http://visual-buffet.dev/xmp/1.0/
Prefix: vbuffet
```

### Implementation Requirements

1. **Read existing XMP** before writing (don't overwrite)
2. **Use ExifTool or pyexiftool** to preserve other namespaces
3. **Add custody event** to `wnb:CustodyChain`:
   ```xml
   <wnb:EventAction>metadata_modification</wnb:EventAction>
   <wnb:EventTool>visual-buffet/0.1.10</wnb:EventTool>
   <wnb:EventNotes>Added ML tags from ram_plus, florence_2</wnb:EventNotes>
   ```
4. **Check `wnb:IsPrimaryFile`** to avoid tagging duplicate files
5. **Update `wnb:SidecarUpdated`** timestamp

### Proposed Dependencies

```toml
# pyproject.toml additions for XMP support
[project.optional-dependencies]
xmp = [
    "pyexiftool>=0.5.6",
]
```

### Migration Path

1. Add XMP writing alongside JSON (both outputs)
2. Add `--xmp-only` flag to skip JSON
3. Eventually deprecate JSON output
