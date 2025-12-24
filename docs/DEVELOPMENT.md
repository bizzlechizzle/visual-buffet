# Visual Buffet Development Guide

Comprehensive guide for developing and extending Visual Buffet.

## Architecture Overview

Visual Buffet follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer (cli.py)                       │
│                   User interface and commands                   │
├─────────────────────────────────────────────────────────────────┤
│                     Service Layer (services/)                   │
│                  XMP handling, tag persistence                  │
├─────────────────────────────────────────────────────────────────┤
│                     Core Layer (core/)                          │
│               TaggingEngine, hardware detection                 │
├─────────────────────────────────────────────────────────────────┤
│                    Plugin Layer (plugins/)                      │
│           PluginBase, loader, individual plugins                │
├─────────────────────────────────────────────────────────────────┤
│                    Utility Layer (utils/)                       │
│              Config, image handling, helpers                    │
└─────────────────────────────────────────────────────────────────┘
```

## Module Reference

### `visual_buffet.constants`

Application-wide constants. All magic numbers and strings should be defined here.

```python
from visual_buffet.constants import (
    DEFAULT_THRESHOLD,  # 0.5
    DEFAULT_GUI_PORT,   # 8420
    ExitCode,           # Exit code enum
    ImageSize,          # Image size enum
    ALL_SUPPORTED_EXTENSIONS,
)
```

### `visual_buffet.core.engine`

The `TaggingEngine` orchestrates plugin loading and image tagging.

```python
from visual_buffet.core.engine import TaggingEngine

engine = TaggingEngine()
results = engine.tag_batch(
    image_paths,
    plugin_names=["ram_plus", "siglip"],
    threshold=0.5,
    size="small",
)
```

### `visual_buffet.services.xmp_handler`

XMP sidecar handling for metadata persistence.

```python
from visual_buffet.services.xmp_handler import XMPHandler

handler = XMPHandler()

# Write tags
handler.write_tags(
    image_path,
    tags=[{"label": "dog", "confidence": 0.95}],
    plugins_used=["ram_plus"],
)

# Read tags
data = handler.read_tags(image_path)

# Check if tags exist
has_tags = handler.has_tags(image_path)
```

### `visual_buffet.plugins.base`

Abstract base class for all plugins.

```python
from visual_buffet.plugins.base import PluginBase

class MyPlugin(PluginBase):
    def get_info(self) -> PluginInfo: ...
    def is_available(self) -> bool: ...
    def tag(self, image_path: Path) -> TagResult: ...
    def setup(self) -> bool: ...
```

## Creating a New Plugin

### 1. Directory Structure

```
plugins/my_plugin/
├── __init__.py      # Plugin class (required)
├── plugin.toml      # Metadata (required)
├── models/          # Model files (auto-created)
└── my_plugin.sme.md # SME documentation (optional)
```

### 2. plugin.toml

```toml
[plugin]
name = "my_plugin"
version = "1.0.0"
description = "My custom plugin"
entry_point = "MyPlugin"
provides_confidence = true

[plugin.dependencies]
torch = ">=2.0"
transformers = ">=4.40"

[plugin.hardware]
gpu_recommended = true
min_ram_gb = 4
min_vram_gb = 4
```

### 3. Plugin Implementation

```python
"""My plugin implementation."""

from pathlib import Path
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult


class MyPlugin(PluginBase):
    """My custom plugin."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._model = None

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
            hardware_reqs={"gpu_recommended": True, "min_ram_gb": 4},
        )

    def is_available(self) -> bool:
        """Check if model files exist."""
        model_path = self.get_model_path() / "model.pt"
        return model_path.exists()

    def setup(self) -> bool:
        """Download model files."""
        # Download from HuggingFace, etc.
        return True

    def tag(self, image_path: Path) -> TagResult:
        """Run inference on image."""
        if self._model is None:
            self._load_model()

        # Your inference code here
        predictions = self._model.predict(image_path)

        tags = [
            Tag(label=p["label"], confidence=p["score"])
            for p in predictions
        ]

        return TagResult(
            tags=tags,
            model="my_model",
            version="1.0.0",
            inference_time_ms=100.0,
        )

    def _load_model(self):
        """Lazy load the model."""
        # Load your model
        pass
```

## XMP Pipeline Integration

Visual Buffet integrates with the wake-n-blake → shoemaker → visual-buffet pipeline.

### Pipeline Order

1. **wake-n-blake** (import) - Creates XMP sidecar with provenance
2. **shoemaker** (thumbnails) - Adds thumbnail metadata
3. **visual-buffet** (ML tags) - Adds ML tagging results

### Namespace

```
Namespace URI: http://visual-buffet.dev/xmp/1.0/
Prefix: vbuffet
```

### XMP Fields Written

| Field | Type | Description |
|-------|------|-------------|
| `vbuffet:SchemaVersion` | int | Schema version (1) |
| `vbuffet:TaggedAt` | ISO datetime | When tags were generated |
| `vbuffet:Threshold` | float | Threshold used |
| `vbuffet:SizeUsed` | string | Image size preset |
| `vbuffet:InferenceTimeMs` | float | Total inference time |
| `vbuffet:PluginsUsed` | list | Plugins that ran |
| `vbuffet:Tags` | struct list | Tags with label, confidence, plugin |

### Integration Requirements

1. Use `XMPHandler` for all tag persistence
2. Preserve existing XMP namespaces (wnb:, shoemaker:)
3. Update custody events when modifying XMP

## Testing

### Running Tests

```bash
# All tests
uv run pytest

# Specific file
uv run pytest tests/test_constants.py

# With coverage
uv run pytest --cov=visual_buffet

# Verbose output
uv run pytest -v
```

### Test Fixtures

Available fixtures in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `cli` | Click CLI test runner |
| `temp_dir` | Temporary directory |
| `test_image` | Test JPEG image |
| `test_png` | Test PNG image |
| `test_images` | Multiple test images |
| `mock_plugin_dir` | Mock plugin directory |
| `mock_tagging_engine` | Mocked TaggingEngine |
| `mock_hardware_profile` | Mocked HardwareProfile |
| `large_test_image` | 1920x1080 test image |
| `corrupted_file` | Invalid image file |
| `empty_file` | Empty file |

### Writing Tests

```python
def test_my_feature(test_image, mock_tagging_engine):
    """Test description."""
    # Arrange
    engine = mock_tagging_engine

    # Act
    result = engine.tag_batch([test_image])

    # Assert
    assert len(result) == 1
    assert "results" in result[0]
```

## Code Style

### Constants

Define all magic numbers in `constants.py`:

```python
# Good
from visual_buffet.constants import DEFAULT_THRESHOLD
if confidence < DEFAULT_THRESHOLD: ...

# Bad
if confidence < 0.5: ...
```

### Error Handling

Use specific exception types:

```python
from visual_buffet.exceptions import PluginError, ImageError

try:
    result = plugin.tag(image_path)
except PluginError as e:
    logger.error(f"Plugin failed: {e}")
    raise
```

### Type Hints

All public functions must have type hints:

```python
def process_image(
    image_path: Path,
    *,
    threshold: float = 0.5,
    size: str = "original",
) -> dict[str, Any]:
    """Process an image and return results."""
    ...
```

### Exit Codes

Use `ExitCode` enum for CLI exits:

```python
from visual_buffet.constants import ExitCode

if not images:
    console.print("[red]No images found[/red]")
    sys.exit(ExitCode.FILE_NOT_FOUND)
```

## CLI Development

### Adding a New Command

```python
@main.command()
@click.argument("path")
@click.option("--flag", is_flag=True, help="Description")
def mycommand(path: str, flag: bool) -> None:
    """Command description.

    PATH is the file or directory to process.

    Examples:

        visual-buffet mycommand file.jpg

        visual-buffet mycommand ./dir --flag
    """
    try:
        # Command logic
        pass
    except VisualBuffetError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(ExitCode.GENERAL_ERROR)
```

### Input Validation

Use Click callbacks for validation:

```python
def _validate_threshold(ctx, param, value):
    if not 0.0 <= value <= 1.0:
        raise click.BadParameter("Must be between 0.0 and 1.0")
    return value

@click.option(
    "--threshold",
    callback=_validate_threshold,
    default=0.5,
)
```

## GUI Development

### Backend (FastAPI)

Located in `gui/server.py`:

```python
from fastapi import FastAPI, HTTPException
from visual_buffet.gui import app

@app.post("/api/custom")
async def custom_endpoint(data: dict):
    """Custom endpoint."""
    return {"status": "ok"}
```

### Frontend

Static files in `gui/static/`:

- `index.html` - Main page
- `app.js` - Application logic
- `styles.css` - Styling (Braun design system)

## Versioning

- Format: `MAJOR.MINOR.PATCH` (e.g., `0.1.11`)
- Bump PATCH for each commit that modifies code
- Store in `VERSION` file
- Read via `visual_buffet.__version__`

## Common Gotchas

### ExifTool Cleanup

If using exiftool directly, always clean up:

```python
# XMPHandler handles this automatically
# For direct usage, use subprocess timeout
```

### Memory Management

For large images:
- Use `--size small` for faster processing
- Reduce batch size in config
- Monitor GPU memory with `nvidia-smi`

### Plugin Loading

Plugins are discovered at runtime:
- Ensure `plugin.toml` is valid TOML
- Entry point must match class name exactly
- Import errors are logged but don't crash

## Resources

- [Click Documentation](https://click.palletsprojects.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ExifTool Documentation](https://exiftool.org/)
- [XMP Specification](https://www.adobe.com/devnet/xmp.html)
