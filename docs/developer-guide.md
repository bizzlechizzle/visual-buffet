# IMLAGE Developer Implementation Guide

A step-by-step guide for implementing IMLAGE, written for developers who may be less experienced with Python plugin architectures or ML integrations.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure Overview](#project-structure-overview)
3. [Core Concepts](#core-concepts)
4. [Implementation Steps](#implementation-steps)
5. [Testing Guide](#testing-guide)
6. [Common Pitfalls](#common-pitfalls)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed (check with `python3 --version`)
- **Git** for version control
- **A code editor** (VS Code recommended)
- **Basic Python knowledge** (functions, classes, decorators)

### Setting Up Your Environment

```bash
# 1. Navigate to project
cd /Users/bryant/Documents/projects/imlage

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 4. Install in development mode
pip install -e ".[dev]"
```

**Why virtual environments?**
They isolate project dependencies. If IMLAGE needs `torch==2.0` but another project needs `torch==1.0`, virtual environments prevent conflicts.

---

## Project Structure Overview

```
imlage/
├── src/imlage/           # Main source code
│   ├── __init__.py       # Package marker + version
│   ├── cli.py            # Command-line interface (what users run)
│   ├── exceptions.py     # Custom error types
│   │
│   ├── core/             # Core business logic
│   │   ├── engine.py     # Orchestrates tagging
│   │   └── hardware.py   # Detects CPU/GPU/RAM
│   │
│   ├── plugins/          # Plugin system
│   │   ├── base.py       # Abstract base class (contract)
│   │   ├── schemas.py    # Data structures
│   │   ├── loader.py     # Finds and loads plugins
│   │   └── registry.py   # Keeps track of plugins
│   │
│   └── utils/            # Helper functions
│       ├── config.py     # Config file handling
│       └── image.py      # Image loading/validation
│
├── plugins/              # Where plugins live
│   └── ram_plus/         # Our first plugin
│
└── tests/                # Test files
```

### Understanding the Layers

```
┌─────────────────────────────────────────────┐
│                   CLI                        │  ← User interacts here
│              (cli.py)                        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│               Core Engine                    │  ← Orchestrates everything
│            (core/engine.py)                  │
└─────────────────────┬───────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   Plugin System  │    │     Utilities    │
│  (plugins/*.py)  │    │   (utils/*.py)   │
└────────┬─────────┘    └──────────────────┘
         │
         ▼
┌──────────────────┐
│  Actual Plugins  │  ← RAM++, future plugins
│   (plugins/)     │
└──────────────────┘
```

---

## Core Concepts

### 1. Dataclasses (schemas.py)

**What are they?**
Python's way of creating classes that mainly hold data. Think of them as structured dictionaries with type hints.

```python
from dataclasses import dataclass

# Without dataclass (verbose)
class Tag:
    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

# With dataclass (clean)
@dataclass
class Tag:
    label: str
    confidence: float
```

**Why use them?**
- Less boilerplate code
- Free `__repr__`, `__eq__` methods
- Type hints for IDE support
- Easy to convert to JSON

### 2. Abstract Base Classes (base.py)

**What are they?**
A "contract" that says "any plugin MUST have these methods".

```python
from abc import ABC, abstractmethod

class PluginBase(ABC):
    @abstractmethod
    def tag(self, image_path):
        """Every plugin MUST implement this."""
        pass
```

**Why?**
If someone creates a plugin but forgets the `tag()` method, Python will throw an error immediately instead of failing later.

### 3. The Plugin Pattern

**How it works:**
1. Plugins live in `plugins/` directory
2. Each plugin has a `plugin.toml` file with metadata
3. The loader scans `plugins/`, reads each `plugin.toml`
4. It imports the plugin class dynamically
5. The engine uses plugins to tag images

```
plugins/ram_plus/
├── __init__.py      # Contains RamPlusPlugin class
├── plugin.toml      # Metadata: name, version, entry_point
├── tagger.py        # Actual ML inference code
└── models/          # Downloaded model files
```

---

## Implementation Steps

### Step 1: Create Custom Exceptions

**File: `src/imlage/exceptions.py`**

**Why?**
Custom exceptions make error handling clearer. Instead of `raise Exception("Plugin failed")`, you write `raise PluginError("RAM++ failed to load model")`.

```python
"""Custom exceptions for IMLAGE.

Each exception type represents a category of error.
Catch specific exceptions to handle errors appropriately.
"""


class ImlageError(Exception):
    """Base exception for all IMLAGE errors."""
    pass


class PluginError(ImlageError):
    """Raised when a plugin fails to load or execute."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin doesn't exist."""
    pass


class ModelNotFoundError(PluginError):
    """Raised when a plugin's model file is missing."""
    pass


class ConfigError(ImlageError):
    """Raised when configuration is invalid or missing."""
    pass


class HardwareDetectionError(ImlageError):
    """Raised when hardware detection fails."""
    pass


class ImageError(ImlageError):
    """Raised when an image cannot be loaded or is invalid."""
    pass
```

---

### Step 2: Create Data Schemas

**File: `src/imlage/plugins/schemas.py`**

**What goes here?**
All the data structures that get passed around. Think of these as the "shape" of your data.

```python
"""Data schemas for the plugin system.

These dataclasses define the structure of data exchanged between
the core engine and plugins. All plugins must use these schemas.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class Tag:
    """A single tag with confidence score.

    Attributes:
        label: The tag text (e.g., "dog", "outdoor")
        confidence: How confident the model is (0.0 to 1.0)

    Example:
        >>> tag = Tag(label="cat", confidence=0.95)
        >>> tag.label
        'cat'
    """
    label: str
    confidence: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {"label": self.label, "confidence": self.confidence}


@dataclass
class TagResult:
    """Result from a plugin's tag() method.

    Attributes:
        tags: List of Tag objects
        model: Model name/identifier used
        version: Plugin version
        inference_time_ms: How long inference took in milliseconds

    Example:
        >>> result = TagResult(
        ...     tags=[Tag("dog", 0.9)],
        ...     model="ram_plus_swin_large",
        ...     version="1.0.0",
        ...     inference_time_ms=142.5
        ... )
    """
    tags: list[Tag]
    model: str
    version: str
    inference_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary matching output contract."""
        return {
            "tags": [t.to_dict() for t in self.tags],
            "model": self.model,
            "version": self.version,
            "inference_time_ms": self.inference_time_ms
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PluginInfo:
    """Metadata about a plugin.

    Attributes:
        name: Unique plugin identifier (e.g., "ram_plus")
        version: Semantic version (e.g., "1.0.0")
        description: Human-readable description
        hardware_reqs: Dict of requirements {"gpu": bool, "min_ram_gb": int}
    """
    name: str
    version: str
    description: str
    hardware_reqs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HardwareProfile:
    """Detected hardware capabilities.

    Cached to ~/.imlage/hardware.json after first detection.

    Attributes:
        cpu_model: CPU name (e.g., "Apple M2 Pro")
        cpu_cores: Number of CPU cores
        ram_total_gb: Total RAM in gigabytes
        ram_available_gb: Currently available RAM
        gpu_type: "cuda", "mps", or None
        gpu_name: GPU name if available
        gpu_vram_gb: GPU memory in GB if available
    """
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_type: str | None = None
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        """Create HardwareProfile from dictionary."""
        return cls(**data)
```

---

### Step 3: Create Plugin Base Class

**File: `src/imlage/plugins/base.py`**

**Why an abstract class?**
It forces all plugins to implement the same interface. The engine can call `plugin.tag()` on ANY plugin without knowing which one it is.

```python
"""Abstract base class for all IMLAGE plugins.

Every plugin MUST inherit from PluginBase and implement all abstract methods.
This ensures consistent behavior across all plugins.

Example:
    class MyPlugin(PluginBase):
        def get_info(self):
            return PluginInfo(name="my_plugin", ...)

        def is_available(self):
            return self.get_model_path().exists()

        def tag(self, image_path):
            # Do ML inference
            return TagResult(...)
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import PluginInfo, TagResult


class PluginBase(ABC):
    """Base class that all plugins must inherit from.

    Plugins are discovered automatically from the plugins/ directory.
    Each plugin must have a plugin.toml file with metadata.

    Lifecycle:
        1. Plugin is discovered by loader
        2. Plugin class is instantiated
        3. is_available() is checked
        4. If not available, setup() can be called
        5. tag() is called for each image
    """

    def __init__(self, plugin_dir: Path):
        """Initialize plugin with its directory path.

        Args:
            plugin_dir: Path to the plugin's directory (e.g., plugins/ram_plus/)
        """
        self._plugin_dir = plugin_dir

    @property
    def plugin_dir(self) -> Path:
        """Get the plugin's directory path."""
        return self._plugin_dir

    def get_model_path(self) -> Path:
        """Get path to plugin's models directory.

        Returns:
            Path to plugins/<name>/models/
        """
        return self._plugin_dir / "models"

    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Return plugin metadata.

        Returns:
            PluginInfo with name, version, description, requirements
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if plugin is ready to use.

        Should verify:
            - Model files exist
            - Dependencies are installed
            - Hardware requirements are met

        Returns:
            True if plugin can run, False otherwise
        """
        pass

    @abstractmethod
    def tag(self, image_path: Path) -> TagResult:
        """Tag an image and return results.

        This is the main method called by the engine.

        Args:
            image_path: Path to image file

        Returns:
            TagResult with list of tags and metadata

        Raises:
            ImageError: If image cannot be loaded
            PluginError: If inference fails
        """
        pass

    def setup(self) -> bool:
        """Download models and prepare plugin for use.

        Called when is_available() returns False.
        Should download model files, create directories, etc.

        Returns:
            True if setup succeeded, False otherwise
        """
        # Default implementation does nothing
        return True
```

---

### Step 4: Create Hardware Detection

**File: `src/imlage/core/hardware.py`**

**What it does:**
1. Detects CPU, RAM, and GPU
2. Caches results to avoid re-detecting every time
3. Used to scale batch sizes and select model variants

```python
"""Hardware detection for performance scaling.

Detects CPU, RAM, and GPU capabilities. Results are cached to
~/.imlage/hardware.json to avoid re-detection on every run.

Usage:
    profile = detect_hardware()
    if profile.gpu_type == "cuda":
        print("NVIDIA GPU available!")
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import psutil

from ..exceptions import HardwareDetectionError
from ..plugins.schemas import HardwareProfile


# Cache location
CACHE_DIR = Path.home() / ".imlage"
CACHE_FILE = CACHE_DIR / "hardware.json"


def detect_hardware(force_refresh: bool = False) -> HardwareProfile:
    """Detect system hardware capabilities.

    Args:
        force_refresh: If True, ignore cache and re-detect

    Returns:
        HardwareProfile with detected capabilities

    Raises:
        HardwareDetectionError: If detection fails
    """
    # Try to load from cache first
    if not force_refresh and CACHE_FILE.exists():
        try:
            return _load_cached()
        except Exception:
            pass  # Fall through to re-detect

    # Detect hardware
    profile = _detect_all()

    # Cache results
    _save_cache(profile)

    return profile


def _load_cached() -> HardwareProfile:
    """Load hardware profile from cache file."""
    with open(CACHE_FILE) as f:
        data = json.load(f)
    return HardwareProfile.from_dict(data["hardware"])


def _save_cache(profile: HardwareProfile) -> None:
    """Save hardware profile to cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "detected_at": datetime.now().isoformat(),
        "hardware": profile.to_dict()
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _detect_all() -> HardwareProfile:
    """Perform full hardware detection."""
    try:
        # CPU detection
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count(logical=False) or 1

        # RAM detection
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024**3), 1)
        ram_available_gb = round(mem.available / (1024**3), 1)

        # GPU detection
        gpu_type, gpu_name, gpu_vram_gb = _detect_gpu()

        return HardwareProfile(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            gpu_type=gpu_type,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
        )
    except Exception as e:
        raise HardwareDetectionError(f"Failed to detect hardware: {e}")


def _detect_gpu() -> tuple[str | None, str | None, float | None]:
    """Detect GPU type and capabilities.

    Returns:
        Tuple of (gpu_type, gpu_name, vram_gb)
        gpu_type is "cuda", "mps", or None
    """
    # Try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Get VRAM in GB
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 1)
            return "cuda", gpu_name, vram_gb
    except ImportError:
        pass  # torch not installed
    except Exception:
        pass  # CUDA detection failed

    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose VRAM info
            return "mps", "Apple Silicon", None
    except ImportError:
        pass
    except Exception:
        pass

    # No GPU
    return None, None, None


def get_recommended_batch_size(profile: HardwareProfile) -> int:
    """Get recommended batch size based on hardware.

    Args:
        profile: Detected hardware profile

    Returns:
        Recommended batch size (1, 2, 4, or 8)
    """
    if profile.gpu_type == "cuda":
        if profile.gpu_vram_gb and profile.gpu_vram_gb >= 8:
            return 8
        return 4
    elif profile.gpu_type == "mps":
        return 4
    else:
        # CPU only
        if profile.ram_available_gb >= 16:
            return 2
        return 1
```

---

### Step 5: Create Config Management

**File: `src/imlage/utils/config.py`**

**What it does:**
- Loads config from `~/.config/imlage/config.toml`
- Creates default config if missing
- Provides get/set methods

```python
"""Configuration file management.

Config is stored in TOML format at:
- macOS/Linux: ~/.config/imlage/config.toml
- Windows: %APPDATA%\\imlage\\config.toml

Usage:
    config = load_config()
    threshold = config.get("plugins.ram_plus.threshold", 0.5)
"""

import platform
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback

import tomli_w

from ..exceptions import ConfigError


def get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if platform.system() == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    else:
        base = Path.home() / ".config"
    return base / "imlage"


def get_config_path() -> Path:
    """Get path to config file."""
    return get_config_dir() / "config.toml"


DEFAULT_CONFIG = {
    "general": {
        "default_format": "json",
        "default_threshold": 0.5,
        "default_limit": 50,
    },
    "plugins": {
        "enabled": ["ram_plus"],
    },
}


def load_config() -> dict:
    """Load configuration from file.

    Creates default config if file doesn't exist.

    Returns:
        Dict with configuration values

    Raises:
        ConfigError: If config file is malformed
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise ConfigError(f"Failed to load config: {e}")


def save_config(config: dict) -> None:
    """Save configuration to file.

    Args:
        config: Dict with configuration values
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def get_value(config: dict, key: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Config dict
        key: Dot-separated key (e.g., "plugins.ram_plus.threshold")
        default: Default value if key not found

    Returns:
        Config value or default

    Example:
        >>> config = {"plugins": {"ram_plus": {"threshold": 0.7}}}
        >>> get_value(config, "plugins.ram_plus.threshold")
        0.7
    """
    parts = key.split(".")
    current = config

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def set_value(config: dict, key: str, value: Any) -> None:
    """Set a nested config value using dot notation.

    Args:
        config: Config dict (modified in place)
        key: Dot-separated key
        value: Value to set

    Example:
        >>> config = {}
        >>> set_value(config, "plugins.ram_plus.threshold", 0.7)
        >>> config
        {'plugins': {'ram_plus': {'threshold': 0.7}}}
    """
    parts = key.split(".")
    current = config

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value
```

---

### Step 6: Create Image Utilities

**File: `src/imlage/utils/image.py`**

```python
"""Image loading and validation utilities.

Handles:
- Validating image formats
- Loading images with Pillow
- Expanding globs/folders to file lists
"""

from pathlib import Path

from PIL import Image

from ..exceptions import ImageError


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def is_supported_image(path: Path) -> bool:
    """Check if file is a supported image format.

    Args:
        path: Path to check

    Returns:
        True if supported image format
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def validate_image(path: Path) -> None:
    """Validate that an image file exists and is readable.

    Args:
        path: Path to image file

    Raises:
        ImageError: If file doesn't exist, isn't an image, or can't be read
    """
    if not path.exists():
        raise ImageError(f"Image not found: {path}")

    if not path.is_file():
        raise ImageError(f"Not a file: {path}")

    if not is_supported_image(path):
        raise ImageError(
            f"Unsupported format: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Try to open to verify it's a valid image
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ImageError(f"Cannot read image {path}: {e}")


def load_image(path: Path) -> Image.Image:
    """Load an image file.

    Args:
        path: Path to image file

    Returns:
        PIL Image object

    Raises:
        ImageError: If image cannot be loaded
    """
    validate_image(path)

    try:
        img = Image.open(path)
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise ImageError(f"Failed to load image {path}: {e}")


def expand_paths(paths: list[str], recursive: bool = False) -> list[Path]:
    """Expand paths to list of image files.

    Handles:
    - Single files
    - Directories (lists contents)
    - Glob patterns (*.jpg)

    Args:
        paths: List of path strings
        recursive: If True, search directories recursively

    Returns:
        List of Path objects to image files
    """
    result = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_file():
            if is_supported_image(path):
                result.append(path)
        elif path.is_dir():
            if recursive:
                for ext in SUPPORTED_EXTENSIONS:
                    result.extend(path.rglob(f"*{ext}"))
            else:
                for ext in SUPPORTED_EXTENSIONS:
                    result.extend(path.glob(f"*{ext}"))
        elif "*" in path_str or "?" in path_str:
            # Glob pattern
            parent = path.parent
            pattern = path.name
            result.extend(p for p in parent.glob(pattern) if is_supported_image(p))

    # Remove duplicates and sort
    return sorted(set(result))
```

---

### Step 7: Create Plugin Loader

**File: `src/imlage/plugins/loader.py`**

```python
"""Plugin discovery and loading.

Scans the plugins/ directory for valid plugins and loads them.
Each plugin must have a plugin.toml file with metadata.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Type

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from ..exceptions import PluginError, PluginNotFoundError
from .base import PluginBase
from .schemas import PluginInfo


def get_plugins_dir() -> Path:
    """Get the plugins directory path."""
    # plugins/ is at project root, not in src/
    # Go up from src/imlage/plugins/loader.py to project root
    return Path(__file__).parent.parent.parent.parent.parent / "plugins"


def discover_plugins() -> list[dict]:
    """Discover all plugins in the plugins directory.

    Returns:
        List of dicts with plugin metadata from plugin.toml
    """
    plugins_dir = get_plugins_dir()

    if not plugins_dir.exists():
        return []

    plugins = []

    for plugin_dir in plugins_dir.iterdir():
        if not plugin_dir.is_dir():
            continue

        toml_path = plugin_dir / "plugin.toml"
        if not toml_path.exists():
            continue

        try:
            with open(toml_path, "rb") as f:
                metadata = tomllib.load(f)

            metadata["_path"] = plugin_dir
            plugins.append(metadata)
        except Exception:
            # Skip invalid plugins
            continue

    return plugins


def load_plugin(plugin_dir: Path) -> PluginBase:
    """Load a plugin class from its directory.

    Args:
        plugin_dir: Path to plugin directory

    Returns:
        Instantiated plugin object

    Raises:
        PluginError: If plugin cannot be loaded
    """
    toml_path = plugin_dir / "plugin.toml"

    if not toml_path.exists():
        raise PluginNotFoundError(f"No plugin.toml in {plugin_dir}")

    # Read metadata
    try:
        with open(toml_path, "rb") as f:
            metadata = tomllib.load(f)
    except Exception as e:
        raise PluginError(f"Invalid plugin.toml: {e}")

    plugin_section = metadata.get("plugin", {})
    entry_point = plugin_section.get("entry_point")

    if not entry_point:
        raise PluginError("plugin.toml missing 'entry_point'")

    # Import the plugin module
    init_path = plugin_dir / "__init__.py"
    if not init_path.exists():
        raise PluginError(f"No __init__.py in {plugin_dir}")

    try:
        # Dynamic import
        module_name = f"imlage_plugin_{plugin_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, init_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the plugin class
        plugin_class = getattr(module, entry_point)

        if not issubclass(plugin_class, PluginBase):
            raise PluginError(f"{entry_point} does not inherit from PluginBase")

        # Instantiate and return
        return plugin_class(plugin_dir)

    except AttributeError:
        raise PluginError(f"Plugin class '{entry_point}' not found in {init_path}")
    except Exception as e:
        raise PluginError(f"Failed to load plugin: {e}")


def load_all_plugins() -> list[PluginBase]:
    """Load all discovered plugins.

    Returns:
        List of loaded plugin instances
    """
    plugins = []

    for metadata in discover_plugins():
        try:
            plugin = load_plugin(metadata["_path"])
            plugins.append(plugin)
        except PluginError:
            # Skip plugins that fail to load
            continue

    return plugins
```

---

### Step 8: Create Core Engine

**File: `src/imlage/core/engine.py`**

```python
"""Core tagging engine.

Orchestrates the tagging process:
1. Load plugins
2. Validate images
3. Run plugins on images
4. Aggregate results
"""

import time
from pathlib import Path
from typing import Any

from ..exceptions import ImageError, PluginError
from ..plugins.base import PluginBase
from ..plugins.loader import load_all_plugins
from ..plugins.schemas import TagResult
from ..utils.image import validate_image


class TaggingEngine:
    """Main engine for processing images through plugins."""

    def __init__(self, plugins: list[PluginBase] | None = None):
        """Initialize engine with plugins.

        Args:
            plugins: List of plugins to use. If None, loads all available.
        """
        if plugins is None:
            plugins = load_all_plugins()

        self._plugins = {p.get_info().name: p for p in plugins}

    @property
    def plugins(self) -> dict[str, PluginBase]:
        """Get dict of loaded plugins by name."""
        return self._plugins

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def tag_image(
        self,
        image_path: Path,
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Tag a single image.

        Args:
            image_path: Path to image file
            plugin_names: Plugins to use (None = all)
            threshold: Minimum confidence to include
            limit: Maximum tags per plugin

        Returns:
            Dict matching output contract:
            {
                "file": "path/to/image.jpg",
                "results": {
                    "plugin_name": {
                        "tags": [...],
                        "model": "...",
                        ...
                    }
                }
            }
        """
        # Validate image
        validate_image(image_path)

        # Select plugins
        if plugin_names:
            plugins = {
                name: p for name, p in self._plugins.items()
                if name in plugin_names and p.is_available()
            }
        else:
            plugins = {
                name: p for name, p in self._plugins.items()
                if p.is_available()
            }

        # Run each plugin
        results = {}
        for name, plugin in plugins.items():
            try:
                result = plugin.tag(image_path)

                # Apply threshold and limit
                filtered_tags = [
                    t for t in result.tags
                    if t.confidence >= threshold
                ]

                if limit:
                    filtered_tags = filtered_tags[:limit]

                result.tags = filtered_tags
                results[name] = result.to_dict()

            except Exception as e:
                results[name] = {"error": str(e)}

        return {
            "file": str(image_path),
            "results": results
        }

    def tag_batch(
        self,
        image_paths: list[Path],
        plugin_names: list[str] | None = None,
        threshold: float = 0.0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Tag multiple images.

        Args:
            image_paths: List of image paths
            plugin_names: Plugins to use
            threshold: Minimum confidence
            limit: Maximum tags per plugin

        Returns:
            List of result dicts, one per image
        """
        results = []

        for path in image_paths:
            try:
                result = self.tag_image(path, plugin_names, threshold, limit)
                results.append(result)
            except ImageError as e:
                results.append({"file": str(path), "error": str(e)})

        return results
```

---

### Step 9: Create RAM++ Plugin

This is the actual ML plugin. It's more complex because it handles model downloading and inference.

**File: `plugins/ram_plus/__init__.py`**

```python
"""RAM++ (Recognize Anything Plus Plus) plugin.

Uses the RAM++ model from https://github.com/xinyu1205/recognize-anything
for general-purpose image tagging with ~6500 possible tags.
"""

import time
from pathlib import Path

from imlage.plugins.base import PluginBase
from imlage.plugins.schemas import PluginInfo, TagResult, Tag
from imlage.exceptions import ModelNotFoundError, PluginError

# Version of this plugin
PLUGIN_VERSION = "1.0.0"

# Model configuration
MODEL_NAME = "ram_plus_swin_large_14m.pth"
MODEL_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
TAG_LIST_URL = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_tag_list.txt"


class RamPlusPlugin(PluginBase):
    """RAM++ image tagging plugin."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._model = None
        self._tag_list = None

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="ram_plus",
            version=PLUGIN_VERSION,
            description="Recognize Anything Plus Plus - General purpose image tagging with ~6500 tags",
            hardware_reqs={
                "gpu": False,  # Works on CPU, but GPU recommended
                "min_ram_gb": 4,
            }
        )

    def is_available(self) -> bool:
        """Check if model files are downloaded."""
        model_path = self.get_model_path() / MODEL_NAME
        tag_list_path = self.get_model_path() / "ram_tag_list.txt"
        return model_path.exists() and tag_list_path.exists()

    def setup(self) -> bool:
        """Download model files."""
        from .downloader import download_model

        try:
            download_model(
                self.get_model_path(),
                MODEL_URL,
                MODEL_NAME,
                TAG_LIST_URL,
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def tag(self, image_path: Path) -> TagResult:
        """Tag an image using RAM++."""
        if not self.is_available():
            raise ModelNotFoundError(
                "RAM++ model not found. Run 'imlage plugins setup ram_plus'"
            )

        # Lazy load model
        if self._model is None:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        return TagResult(
            tags=tags,
            model="ram_plus_swin_large",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
        )

    def _load_model(self):
        """Load the RAM++ model."""
        try:
            from .tagger import load_ram_plus_model

            model_path = self.get_model_path() / MODEL_NAME
            tag_list_path = self.get_model_path() / "ram_tag_list.txt"

            self._model, self._tag_list = load_ram_plus_model(
                str(model_path),
                str(tag_list_path),
            )
        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with: pip install torch torchvision timm\n{e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load model: {e}")

    def _run_inference(self, image_path: Path) -> list[Tag]:
        """Run inference on an image."""
        from .tagger import run_inference

        try:
            raw_tags = run_inference(self._model, self._tag_list, str(image_path))

            # Convert to Tag objects
            return [
                Tag(label=label, confidence=round(conf, 4))
                for label, conf in raw_tags
            ]
        except Exception as e:
            raise PluginError(f"Inference failed: {e}")
```

**File: `plugins/ram_plus/downloader.py`**

```python
"""Model download utilities for RAM++."""

from pathlib import Path
import urllib.request
import sys


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download
        dest: Destination path
        desc: Description for progress bar
    """
    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        sys.stdout.write(f"\r{desc}: {percent}%")
        sys.stdout.flush()

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()  # Newline after progress


def download_model(
    model_dir: Path,
    model_url: str,
    model_name: str,
    tag_list_url: str,
) -> None:
    """Download RAM++ model and tag list.

    Args:
        model_dir: Directory to save files
        model_url: URL for model weights
        model_name: Filename for model
        tag_list_url: URL for tag list
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_name
    tag_list_path = model_dir / "ram_tag_list.txt"

    if not model_path.exists():
        print(f"Downloading RAM++ model (~1.5GB)...")
        download_file(model_url, model_path, "Model")
    else:
        print(f"Model already exists: {model_path}")

    if not tag_list_path.exists():
        print("Downloading tag list...")
        download_file(tag_list_url, tag_list_path, "Tag list")
    else:
        print(f"Tag list already exists: {tag_list_path}")

    print("Setup complete!")
```

**File: `plugins/ram_plus/tagger.py`**

```python
"""RAM++ model wrapper.

This module handles loading and running inference with the RAM++ model.
"""

from pathlib import Path
from typing import Tuple, List

import torch
from PIL import Image
from torchvision import transforms


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ram_plus_model(
    model_path: str,
    tag_list_path: str,
) -> Tuple[torch.nn.Module, List[str]]:
    """Load RAM++ model and tag list.

    Args:
        model_path: Path to model weights
        tag_list_path: Path to tag list file

    Returns:
        Tuple of (model, tag_list)
    """
    device = get_device()

    # Load tag list
    with open(tag_list_path, "r") as f:
        tag_list = [line.strip() for line in f if line.strip()]

    # Build model architecture
    # RAM++ uses a Swin Transformer backbone with custom head
    # We'll use the recognize-anything library's model
    try:
        from ram import get_transform, inference_ram
        from ram.models import ram_plus

        # Load model
        model = ram_plus(
            pretrained=model_path,
            image_size=384,
            vit="swin_l",
        )
        model.eval()
        model = model.to(device)

        return model, tag_list

    except ImportError:
        # Fallback: try to load from checkpoint directly
        raise ImportError(
            "Please install the recognize-anything package:\n"
            "pip install git+https://github.com/xinyu1205/recognize-anything.git"
        )


def run_inference(
    model: torch.nn.Module,
    tag_list: List[str],
    image_path: str,
) -> List[Tuple[str, float]]:
    """Run inference on an image.

    Args:
        model: Loaded RAM++ model
        tag_list: List of possible tags
        image_path: Path to image file

    Returns:
        List of (tag, confidence) tuples, sorted by confidence
    """
    device = get_device()

    try:
        from ram import get_transform, inference_ram

        # Load and transform image
        transform = get_transform(image_size=384)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            tags, _ = inference_ram(image_tensor, model)

        # Parse results - RAM++ returns comma-separated tags
        # We need to match with tag list and get confidences
        detected_tags = [t.strip() for t in tags[0].split("|")]

        # Create result with confidence (RAM++ doesn't provide individual confidences,
        # so we'll assign decreasing confidence based on order)
        results = []
        for i, tag in enumerate(detected_tags):
            confidence = max(0.5, 1.0 - (i * 0.05))  # Decreasing from 1.0
            results.append((tag, confidence))

        return results

    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
```

**File: `plugins/ram_plus/plugin.toml`**

```toml
[plugin]
name = "ram_plus"
version = "1.0.0"
description = "Recognize Anything Plus Plus - General purpose image tagging"
entry_point = "RamPlusPlugin"
python_requires = ">=3.11"

[plugin.dependencies]
torch = ">=2.0"
torchvision = ">=0.15"
timm = ">=0.9"

[plugin.hardware]
gpu_recommended = true
min_ram_gb = 4
min_vram_gb = 4
```

---

### Step 10: Update CLI

**File: `src/imlage/cli.py`**

```python
"""IMLAGE CLI entry point.

Commands:
    tag       Tag image(s) using configured plugins
    plugins   Manage plugins (list, setup)
    hardware  Show detected hardware
    config    View/edit configuration
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from imlage import __version__
from imlage.core.engine import TaggingEngine
from imlage.core.hardware import detect_hardware, get_recommended_batch_size
from imlage.plugins.loader import discover_plugins, load_plugin, get_plugins_dir
from imlage.utils.config import load_config, save_config, get_value, set_value
from imlage.utils.image import expand_paths
from imlage.exceptions import ImlageError

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="imlage")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """IMLAGE - Compare visual tagging results from local ML tools."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# ============================================================================
# TAG COMMAND
# ============================================================================

@main.command()
@click.argument("path", nargs=-1, required=True)
@click.option("-p", "--plugin", "plugins", multiple=True, help="Plugins to use")
@click.option("-o", "--output", type=click.Path(), help="Output file")
@click.option("-f", "--format", "fmt", default="json", type=click.Choice(["json"]))
@click.option("--threshold", default=0.5, type=float, help="Minimum confidence")
@click.option("--limit", default=50, type=int, help="Max tags per plugin")
@click.option("--recursive", is_flag=True, help="Search folders recursively")
@click.pass_context
def tag(ctx, path, plugins, output, fmt, threshold, limit, recursive):
    """Tag image(s) using configured plugins."""
    try:
        # Expand paths to file list
        image_paths = expand_paths(list(path), recursive=recursive)

        if not image_paths:
            console.print("[red]No images found[/red]")
            sys.exit(1)

        console.print(f"[dim]Found {len(image_paths)} image(s)[/dim]")

        # Initialize engine
        engine = TaggingEngine()

        if not engine.plugins:
            console.print("[red]No plugins available[/red]")
            console.print("Run 'imlage plugins list' to see available plugins")
            sys.exit(1)

        # Filter to requested plugins
        plugin_names = list(plugins) if plugins else None

        # Process images
        results = engine.tag_batch(
            image_paths,
            plugin_names=plugin_names,
            threshold=threshold,
            limit=limit,
        )

        # Output results
        output_json = json.dumps(results, indent=2)

        if output:
            Path(output).write_text(output_json)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            # Pretty print to console
            for result in results:
                _print_result(result)

    except ImlageError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _print_result(result: dict) -> None:
    """Pretty print a tagging result."""
    file_path = result.get("file", "unknown")
    console.print(f"\n[bold]{file_path}[/bold]")

    if "error" in result:
        console.print(f"  [red]Error: {result['error']}[/red]")
        return

    for plugin_name, plugin_result in result.get("results", {}).items():
        if "error" in plugin_result:
            console.print(f"  [dim]{plugin_name}:[/dim] [red]{plugin_result['error']}[/red]")
            continue

        tags = plugin_result.get("tags", [])
        if not tags:
            console.print(f"  [dim]{plugin_name}:[/dim] No tags")
            continue

        tag_str = " • ".join(
            f"{t['label']} ({t['confidence']:.2f})"
            for t in tags[:10]  # Show first 10
        )
        console.print(f"  [dim]{plugin_name}:[/dim] {tag_str}")


# ============================================================================
# PLUGINS COMMAND
# ============================================================================

@main.group()
def plugins():
    """Manage plugins."""
    pass


@plugins.command("list")
def plugins_list():
    """List available plugins."""
    discovered = discover_plugins()

    if not discovered:
        console.print("[yellow]No plugins found in plugins/ directory[/yellow]")
        return

    table = Table(title="Available Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Description")

    for meta in discovered:
        plugin_section = meta.get("plugin", {})
        name = plugin_section.get("name", "unknown")
        version = plugin_section.get("version", "?")
        description = plugin_section.get("description", "")[:50]

        # Check if available
        try:
            plugin = load_plugin(meta["_path"])
            status = "[green]Ready[/green]" if plugin.is_available() else "[yellow]Setup needed[/yellow]"
        except Exception:
            status = "[red]Error[/red]"

        table.add_row(name, version, status, description)

    console.print(table)


@plugins.command("setup")
@click.argument("name")
def plugins_setup(name):
    """Download and setup a plugin's model files."""
    plugins_dir = get_plugins_dir()
    plugin_dir = plugins_dir / name

    if not plugin_dir.exists():
        console.print(f"[red]Plugin '{name}' not found[/red]")
        sys.exit(1)

    try:
        plugin = load_plugin(plugin_dir)

        if plugin.is_available():
            console.print(f"[green]{name} is already set up[/green]")
            return

        console.print(f"Setting up {name}...")
        success = plugin.setup()

        if success:
            console.print(f"[green]{name} setup complete![/green]")
        else:
            console.print(f"[red]{name} setup failed[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Setup error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# HARDWARE COMMAND
# ============================================================================

@main.command()
@click.option("--refresh", is_flag=True, help="Force re-detection")
def hardware(refresh):
    """Show detected hardware capabilities."""
    try:
        profile = detect_hardware(force_refresh=refresh)

        table = Table(title="Hardware Profile")
        table.add_column("Component", style="cyan")
        table.add_column("Details")

        table.add_row("CPU", f"{profile.cpu_model} ({profile.cpu_cores} cores)")
        table.add_row("RAM", f"{profile.ram_total_gb} GB total, {profile.ram_available_gb} GB available")

        if profile.gpu_type:
            gpu_info = f"{profile.gpu_name}"
            if profile.gpu_vram_gb:
                gpu_info += f" ({profile.gpu_vram_gb} GB VRAM)"
            table.add_row(f"GPU ({profile.gpu_type.upper()})", gpu_info)
        else:
            table.add_row("GPU", "None detected")

        console.print(table)

        batch_size = get_recommended_batch_size(profile)
        console.print(f"\n[dim]Recommended batch size: {batch_size}[/dim]")

    except Exception as e:
        console.print(f"[red]Hardware detection failed: {e}[/red]")
        sys.exit(1)


# ============================================================================
# CONFIG COMMAND
# ============================================================================

@main.group()
def config():
    """View and edit configuration."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    cfg = load_config()
    console.print_json(json.dumps(cfg, indent=2))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value."""
    cfg = load_config()

    # Try to parse as JSON for complex values
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    set_value(cfg, key, parsed_value)
    save_config(cfg)
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


if __name__ == "__main__":
    main()
```

---

## Testing Guide

### Running Tests

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cli.py

# Run with coverage
pytest --cov=imlage
```

### What Each Test File Checks

- `test_schemas.py` - Data structures work correctly
- `test_hardware.py` - Hardware detection (mocked)
- `test_config.py` - Config loading/saving
- `test_image.py` - Image validation
- `test_plugins.py` - Plugin loading
- `test_engine.py` - End-to-end tagging
- `test_cli.py` - CLI commands work

---

## Common Pitfalls

### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'imlage'`

**Solution:** Install the package in development mode:
```bash
pip install -e .
```

### 2. Plugin Not Found

**Problem:** Plugin exists but `imlage plugins list` shows nothing

**Solution:** Check that:
- Plugin has `__init__.py`
- Plugin has `plugin.toml`
- `plugin.toml` has correct format

### 3. Model Download Fails

**Problem:** `ConnectionError` during model download

**Solution:**
- Check internet connection
- Try downloading manually and placing in `plugins/ram_plus/models/`

### 4. CUDA/MPS Not Detected

**Problem:** GPU available but not detected

**Solution:**
- For CUDA: Install `torch` with CUDA support
- For MPS: Ensure macOS 12.3+ and PyTorch 1.12+

---

## Dependency Licenses

| Package | License | Notes |
|---------|---------|-------|
| click | BSD-3-Clause | CLI framework |
| rich | MIT | Terminal formatting |
| pillow | HPND | Image loading |
| tomli | MIT | TOML parsing |
| tomli-w | MIT | TOML writing |
| psutil | BSD-3-Clause | Hardware detection |
| torch | BSD-3-Clause | ML runtime |
| torchvision | BSD-3-Clause | Image transforms |
| timm | Apache-2.0 | Vision models |
| pytest | MIT | Testing |
| ruff | MIT | Linting |
