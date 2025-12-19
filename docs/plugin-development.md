# Plugin Development Guide

This guide explains how to create plugins for IMLAGE and how the quality-based tagging system works.

## Overview

IMLAGE uses a plugin architecture where each ML tagger is an independent plugin. Plugins:

- Live in the `plugins/` directory
- Implement a standard interface (`PluginBase`)
- Return tags in a standard format
- Support per-plugin quality settings for resolution control

## Quality-Based Tagging System

### How It Works

Instead of always tagging the original image, IMLAGE uses pre-generated thumbnails at different resolutions. This provides:

1. **Faster tagging** - Smaller images process faster
2. **Consistent results** - Same resolution = reproducible tags
3. **User control** - Trade speed for completeness

### Quality Levels

Each plugin can have its own quality setting:

| Quality | Resolution(s) | Speed | Tag Coverage | Use Case |
|---------|---------------|-------|--------------|----------|
| `quick` | 480px | ~3x faster | ~66% | Bulk imports, rough sorting |
| `standard` | 1080px | Baseline | ~87% | Daily use (default) |
| `high` | 480 + 1080 + 2048px | ~3x slower | ~98% | Final archive, important images |

### Resolution Selection

```
Quality Level    Resolutions Used    Processing
─────────────    ────────────────    ──────────────────
quick            [480]               Single pass
standard         [1080]              Single pass
high             [480, 1080, 2048]   Three passes, merged
```

### Tag Merging (HIGH Mode)

When using `high` quality, tags from all resolutions are merged:

- Duplicate tags are deduplicated (case-insensitive)
- Highest confidence score is kept
- Tags found at multiple resolutions get a `sources` count
- Results sorted by: sources (desc), confidence (desc), label

## Thumbnail System

### Standard Sizes

IMLAGE generates these thumbnails for every imported image:

| Name | Size | Purpose |
|------|------|---------|
| `grid` | 480px | Grid view thumbnails |
| `preview` | 1080px | Lightbox preview |
| `zoom` | 2048px | Lightbox zoom / full preview |

### Storage Location

Thumbnails and tags are stored in an `imlage/` folder **next to each image**:

```
/photos/vacation/
├── beach.jpg
├── sunset.jpg
└── imlage/
    ├── beach_480.webp       # Grid thumbnail
    ├── beach_1080.webp      # Preview
    ├── beach_2048.webp      # Zoom
    ├── beach_tags.json      # Saved tags
    ├── sunset_480.webp
    ├── sunset_1080.webp
    ├── sunset_2048.webp
    └── sunset_tags.json
```

### Thumbnail Naming

```
{image_stem}_{resolution}.webp

Examples:
  beach_480.webp    # Grid thumbnail
  beach_1080.webp   # Preview
  beach_2048.webp   # Zoom
```

### Tags File Format

```json
{
  "file": "/photos/vacation/beach.jpg",
  "filename": "beach.jpg",
  "tagged_at": "2025-01-15T10:30:00.000000",
  "results": {
    "ram_plus": {
      "tags": [{"label": "beach"}, {"label": "ocean"}],
      "model": "ram_plus",
      "version": "1.0.0",
      "quality": "standard",
      "resolutions": [1080]
    }
  }
}
```

### Reuse for Tagging

The tagging system reuses UI thumbnails - no extra files needed:

```
Quality    Uses Thumbnail
───────    ──────────────
quick      grid (480px)
standard   preview (1080px)
high       grid + preview + zoom
```

## Plugin Structure

### Directory Layout

```
plugins/
└── my_plugin/
    ├── __init__.py      # Plugin class (required)
    ├── plugin.toml      # Metadata (required)
    ├── my_plugin.sme.md # Documentation (recommended)
    └── models/          # Model files (git-ignored)
```

### plugin.toml Format

```toml
[plugin]
name = "my_plugin"              # Unique identifier
display_name = "My Plugin"      # Human-readable name
version = "1.0.0"
entry_point = "MyPlugin"        # Class name in __init__.py
provides_confidence = true      # Does plugin provide confidence scores?

[plugin.dependencies]
torch = ">=2.0"
transformers = ">=4.40.0"

[plugin.hardware]
gpu_recommended = true
min_ram_gb = 4

[plugin.defaults]
quality = "standard"            # Default quality level
threshold = 0.5                 # Default confidence threshold
limit = 50                      # Default max tags
```

### Plugin Class

```python
# plugins/my_plugin/__init__.py

from pathlib import Path
from imlage.plugins.base import PluginBase
from imlage.plugins.schemas import PluginInfo, TagResult, Tag


class MyPlugin(PluginBase):
    """My custom tagging plugin."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._model = None

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="my_plugin",
            version="1.0.0",
            description="My custom image tagger",
            hardware_reqs={"gpu": True, "min_ram_gb": 4},
        )

    def is_available(self) -> bool:
        """Check if plugin can run (models exist, deps installed)."""
        model_path = self.get_model_path() / "model.bin"
        return model_path.exists()

    def setup(self) -> bool:
        """Download models and prepare plugin."""
        # Download model files here
        return True

    def tag(self, image_path: Path) -> TagResult:
        """Tag an image and return results.

        IMPORTANT: The image_path may be a thumbnail, not the original.
        The tagging engine handles resolution selection based on quality
        settings. Your plugin should just process whatever image it receives.

        Args:
            image_path: Path to image (may be original or thumbnail)

        Returns:
            TagResult with list of Tag objects
        """
        import time
        start = time.perf_counter()

        # Load model if needed
        if self._model is None:
            self._load_model()

        # Run inference
        predictions = self._model.predict(image_path)

        # Convert to Tag objects
        tags = [
            Tag(label=p["label"], confidence=p["score"])
            for p in predictions
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return TagResult(
            tags=tags,
            model="my_model_v1",
            version="1.0.0",
            inference_time_ms=elapsed_ms,
        )

    def _load_model(self):
        """Load the ML model."""
        # Your model loading code here
        pass
```

## Configuration

### Per-Plugin Settings

Users can configure each plugin independently:

```toml
# ~/.config/imlage/config.toml

[plugins.ram_plus]
enabled = true
quality = "quick"      # Fast tagging
threshold = 0.5
limit = 50

[plugins.florence_2]
enabled = true
quality = "high"       # Maximum tags
threshold = 0.3
limit = 100
```

### Programmatic Configuration

```python
from imlage.core.engine import TaggingEngine
from imlage.plugins.schemas import TagQuality

engine = TaggingEngine()

# Per-plugin quality settings
result = engine.tag_image(
    Path("photo.jpg"),
    plugin_configs={
        "ram_plus": {
            "quality": "quick",
            "threshold": 0.5,
            "limit": 50,
        },
        "florence_2": {
            "quality": "high",
            "threshold": 0.3,
            "limit": 100,
        },
    },
)

# Or set default for all plugins
result = engine.tag_image(
    Path("photo.jpg"),
    default_quality=TagQuality.HIGH,
)
```

## Output Format

### Standard Result

```json
{
  "file": "path/to/image.jpg",
  "results": {
    "my_plugin": {
      "tags": [
        {"label": "dog", "confidence": 0.95},
        {"label": "outdoor", "confidence": 0.87}
      ],
      "model": "my_model_v1",
      "version": "1.0.0",
      "quality": "standard",
      "resolutions": [1080],
      "inference_time_ms": 142.5
    }
  }
}
```

### HIGH Quality Result (Merged)

When using HIGH quality, the `sources` field indicates how many resolutions found the tag:

```json
{
  "tags": [
    {"label": "dog", "confidence": 0.95, "sources": 3},
    {"label": "outdoor", "confidence": 0.87, "sources": 2},
    {"label": "grass", "confidence": 0.72, "sources": 1}
  ],
  "quality": "high",
  "resolutions": [480, 1080, 2048]
}
```

## Best Practices

### 1. Handle Any Resolution

Your plugin receives thumbnails at various resolutions. Don't assume image size:

```python
def tag(self, image_path: Path) -> TagResult:
    # Load and resize if your model needs specific dimensions
    img = Image.open(image_path)
    if self.model_input_size:
        img = img.resize(self.model_input_size)
    # ...
```

### 2. Provide Confidence Scores When Possible

Plugins with confidence scores enable better filtering and sorting:

```python
# Good: Include confidence
Tag(label="dog", confidence=0.95)

# Okay: No confidence (will be ordered by relevance)
Tag(label="dog", confidence=None)
```

Set `provides_confidence = true/false` in plugin.toml.

### 3. Return Consistent Labels

Normalize your tag labels for consistency:

```python
# Good: Lowercase, stripped
Tag(label=prediction.lower().strip())

# Bad: Mixed case, extra spaces
Tag(label="  Dog  ")
```

### 4. Document Hardware Requirements

Be explicit about what your plugin needs:

```toml
[plugin.hardware]
gpu_recommended = true    # GPU helps but not required
gpu_required = false      # Can run on CPU
min_ram_gb = 4            # Minimum RAM
min_vram_gb = 6           # Minimum GPU memory (if GPU used)
```

### 5. Implement setup() for Model Downloads

Don't bundle large models in git. Download on first use:

```python
def setup(self) -> bool:
    """Download model on first run."""
    model_path = self.get_model_path() / "model.bin"
    if not model_path.exists():
        download_model(MODEL_URL, model_path)
    return True
```

## Testing Your Plugin

```python
# test_my_plugin.py
from pathlib import Path
from plugins.my_plugin import MyPlugin

def test_plugin():
    plugin = MyPlugin(Path("plugins/my_plugin"))

    assert plugin.is_available()

    result = plugin.tag(Path("test_image.jpg"))

    assert len(result.tags) > 0
    assert result.model == "my_model_v1"
    assert result.inference_time_ms > 0
```

Run with: `uv run pytest test_my_plugin.py`

## SME Documentation

Create a Subject Matter Expert file documenting your plugin:

```markdown
# My Plugin SME

## Overview
Brief description of what this plugin does.

## Model Details
- Architecture: ...
- Training data: ...
- Tag vocabulary size: ...

## Hardware Requirements
- GPU: Recommended, not required
- RAM: 4GB minimum
- VRAM: 6GB for GPU mode

## Setup
1. Run `imlage plugins setup my_plugin`
2. Model downloads automatically (~2GB)

## Configuration
Available options and their defaults.

## Known Limitations
What the model struggles with.
```

Save as `docs/sme/my_plugin.sme.md`.
