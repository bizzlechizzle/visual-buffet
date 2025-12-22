# Visual Buffet Implementation Plan (Phases 2-4)

## Overview

This plan covers the complete implementation of Visual Buffet core functionality:
- **Phase 2:** Core Engine (plugin system, hardware detection, config)
- **Phase 3:** RAM++ Plugin (working ML tagging)
- **Phase 4:** CLI Completion (all commands working)

## File Creation Order

Files are created in dependency order (dependencies first):

### 1. Core Infrastructure (no dependencies)
```
src/visual_buffet/exceptions.py         # Custom exceptions (VisualBuffetError, PluginError, etc.)
src/visual_buffet/plugins/schemas.py    # TagResult, PluginInfo, HardwareReqs dataclasses
```

### 2. Plugin Base (depends on schemas)
```
src/visual_buffet/plugins/base.py       # PluginBase ABC
```

### 3. Utilities (no dependencies)
```
src/visual_buffet/utils/config.py       # Config file loading/saving
src/visual_buffet/utils/image.py        # Image validation and loading
src/visual_buffet/core/hardware.py      # Hardware detection
```

### 4. Plugin Infrastructure (depends on base, schemas)
```
src/visual_buffet/plugins/loader.py     # Plugin discovery and loading
src/visual_buffet/plugins/registry.py   # Plugin registration
```

### 5. Core Engine (depends on plugins, utils)
```
src/visual_buffet/core/engine.py        # Main processing engine
```

### 6. RAM++ Plugin (depends on plugin base)
```
plugins/ram_plus/__init__.py     # Plugin class
plugins/ram_plus/plugin.toml     # Plugin metadata
plugins/ram_plus/tagger.py       # RAM++ inference wrapper
docs/sme/ram_plus.sme.md         # SME documentation
```

### 7. CLI Commands (depends on everything)
```
src/visual_buffet/cli.py         # Updated with all commands
```

### 8. Tests
```
tests/test_schemas.py
tests/test_hardware.py
tests/test_plugins.py
tests/test_engine.py
tests/test_cli.py
```

## Detailed Implementation

### Phase 2A: Schemas (src/visual_buffet/plugins/schemas.py)

```python
# Data classes for plugin system
@dataclass
class Tag:
    label: str
    confidence: float

@dataclass
class TagResult:
    tags: list[Tag]
    model: str
    version: str
    inference_time_ms: float

@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    hardware_reqs: dict  # {"gpu": bool, "min_ram_gb": int}

@dataclass
class HardwareProfile:
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_type: str | None  # "cuda", "mps", None
    gpu_name: str | None
    gpu_vram_gb: float | None
```

### Phase 2B: Plugin Base (src/visual_buffet/plugins/base.py)

```python
from abc import ABC, abstractmethod
from pathlib import Path

class PluginBase(ABC):
    @abstractmethod
    def get_info(self) -> PluginInfo: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def tag(self, image_path: Path) -> TagResult: ...

    def setup(self) -> bool: ...  # Download models

    def get_model_path(self) -> Path: ...  # Returns plugin's model directory
```

### Phase 2C: Hardware Detection (src/visual_buffet/core/hardware.py)

```python
# Uses psutil for CPU/RAM
# Uses torch for GPU detection (cuda/mps)
# Caches to ~/.visual-buffet/hardware.json
# Returns HardwareProfile dataclass
```

### Phase 2D: Config Management (src/visual_buffet/utils/config.py)

```python
# Loads from ~/.config/visual-buffet/config.toml
# Creates default config if missing
# Provides get/set methods for plugin configs
```

### Phase 2E: Image Utilities (src/visual_buffet/utils/image.py)

```python
# Validates image formats (jpg, png, webp, heic)
# Loads images with Pillow
# Expands globs and folders to file lists
```

### Phase 2F: Plugin Loader (src/visual_buffet/plugins/loader.py)

```python
# Discovers plugins in plugins/ directory
# Reads plugin.toml for metadata
# Dynamically imports plugin classes
# Returns list of available plugins
```

### Phase 2G: Core Engine (src/visual_buffet/core/engine.py)

```python
class TaggingEngine:
    def __init__(self, plugins: list[PluginBase]): ...

    def tag_image(self, path: Path) -> dict: ...

    def tag_batch(self, paths: list[Path]) -> list[dict]: ...
```

### Phase 3: RAM++ Plugin

**plugins/ram_plus/__init__.py:**
```python
class RamPlusPlugin(PluginBase):
    MODEL_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/..."

    def get_info(self) -> PluginInfo: ...
    def is_available(self) -> bool: ...
    def tag(self, image_path: Path) -> TagResult: ...
    def setup(self) -> bool: ...  # Downloads model
```

**plugins/ram_plus/tagger.py:**
```python
# Wraps the RAM++ inference code
# Loads model, runs inference, returns tags
# Handles device selection (cuda/mps/cpu)
```

### Phase 4: CLI Commands

**Commands to implement:**
- `visual-buffet tag PATH` - Tag images
- `visual-buffet plugins list` - List available plugins
- `visual-buffet plugins setup NAME` - Download plugin models
- `visual-buffet hardware` - Show detected hardware
- `visual-buffet config show` - Show current config
- `visual-buffet config set KEY VALUE` - Set config value

## Dependencies to Add

```toml
# pyproject.toml additions
dependencies = [
    "click>=8.0",
    "rich>=13.0",
    "pillow>=10.0",
    "tomli>=2.0;python_version<'3.11'",
    "tomli-w>=1.0",
    "psutil>=5.9",
]

[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "torchvision>=0.15",
    "timm>=0.9",
    "huggingface-hub>=0.20",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.4",
]
```

## Test Plan

### Unit Tests
- `test_schemas.py` - Dataclass creation, JSON serialization
- `test_hardware.py` - Hardware detection (mock torch)
- `test_config.py` - Config load/save
- `test_image.py` - Image validation, glob expansion

### Integration Tests
- `test_plugins.py` - Plugin loading, registration
- `test_engine.py` - End-to-end tagging (mock plugin)
- `test_cli.py` - CLI commands work

### Manual Tests
- Run `visual-buffet tag images/testimage01.jpg` with RAM++ installed
- Verify JSON output matches contract
- Test on CPU and GPU (if available)

## Success Criteria

1. ✅ `pip install -e .` works
2. ✅ `visual-buffet --version` shows version
3. ✅ `visual-buffet hardware` shows detected hardware
4. ✅ `visual-buffet plugins list` shows RAM++
5. ✅ `visual-buffet plugins setup ram_plus` downloads model
6. ✅ `visual-buffet tag images/testimage01.jpg` outputs tags
7. ✅ `pytest` passes all tests
8. ✅ `ruff check .` has no errors
