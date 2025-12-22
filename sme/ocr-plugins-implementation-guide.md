# OCR Plugins Implementation Guide

> **Audience**: Developers new to Visual Buffet plugin development
> **Prerequisites**: Basic Python knowledge, familiarity with pip/virtual environments
> **Time to Complete**: 2-3 hours

---

## Table of Contents

1. [Understanding the Plugin System](#understanding-the-plugin-system)
2. [Development Environment Setup](#development-environment-setup)
3. [Step-by-Step: Surya OCR Plugin](#step-by-step-surya-ocr-plugin)
4. [Step-by-Step: docTR Plugin](#step-by-step-doctr-plugin)
5. [Testing Your Plugins](#testing-your-plugins)
6. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Understanding the Plugin System

### What is a Plugin?

A Visual Buffet plugin is a Python module that:
1. **Detects something** in an image (text, objects, tags)
2. **Returns results** in a standardized format
3. **Lives in its own folder** under `plugins/`

### Plugin File Structure

Every plugin needs exactly these files:

```
plugins/
└── your_plugin/
    ├── __init__.py      # Your Python code (REQUIRED)
    ├── plugin.toml      # Configuration file (REQUIRED)
    └── models/          # Folder for model files (OPTIONAL)
```

### The Plugin Contract

Your plugin MUST:
1. **Inherit from `PluginBase`** - This is the base class all plugins extend
2. **Implement 3 methods**:
   - `get_info()` - Returns metadata about your plugin
   - `is_available()` - Returns True if plugin can run
   - `tag()` - Actually processes an image and returns results

### Understanding TagResult

When your plugin processes an image, it returns a `TagResult` object:

```python
TagResult(
    tags=[                          # List of detected items
        Tag(label="Hello", confidence=0.95),
        Tag(label="World", confidence=0.88),
    ],
    model="your_plugin_name",       # What model was used
    version="1.0.0",                # Your plugin version
    inference_time_ms=123.45,       # How long it took
    metadata={                      # Extra data (bounding boxes, etc.)
        "total_lines": 2,
        "boxes": [...]
    }
)
```

---

## Development Environment Setup

### Step 1: Navigate to Project

```bash
cd /mnt/nas-projects/visual-buffet
```

### Step 2: Activate the Virtual Environment

```bash
# If using uv (recommended)
source .venv/bin/activate

# Or if using standard venv
source venv/bin/activate
```

### Step 3: Verify You're in the Right Environment

```bash
which python
# Should show: /mnt/nas-projects/visual-buffet/.venv/bin/python
```

### Step 4: Install Development Dependencies

```bash
# Install the project in editable mode
pip install -e .

# Install Surya OCR
pip install surya-ocr

# Install docTR with PyTorch backend
pip install "python-doctr[torch]"
```

---

## Step-by-Step: Surya OCR Plugin

### Step 1: Create the Plugin Directory

```bash
mkdir -p plugins/surya_ocr/models
```

The `models/` folder stays empty - Surya downloads models automatically to `~/.cache/huggingface/`.

### Step 2: Create plugin.toml

Create `plugins/surya_ocr/plugin.toml`:

```toml
[plugin]
name = "surya_ocr"
display_name = "Surya OCR"
version = "1.0.0"
description = "Vision transformer OCR with layout analysis - 90+ languages"
entry_point = "SuryaOCRPlugin"
python_requires = ">=3.10"
provides_confidence = true

[plugin.dependencies]
surya-ocr = ">=0.17.0"
torch = ">=2.0"

[plugin.hardware]
gpu_recommended = true
gpu_required = false
min_ram_gb = 8
min_vram_gb = 6

[plugin.defaults]
quality = "standard"
threshold = 0.5
limit = 100
batch_size = 1

[plugin.model]
include_layout = false
det_batch_size = "auto"
rec_batch_size = "auto"

[plugin.output]
sort_by = "confidence"
include_boxes = true
```

**What each section means:**
- `[plugin]` - Basic info about your plugin
- `[plugin.dependencies]` - Python packages needed
- `[plugin.hardware]` - GPU/RAM requirements
- `[plugin.defaults]` - Default settings users can override
- `[plugin.model]` - Model-specific settings
- `[plugin.output]` - How to format results

### Step 3: Create __init__.py

Create `plugins/surya_ocr/__init__.py`. Let's build it piece by piece:

#### Part A: Imports and Setup

```python
"""Surya OCR text detection and recognition plugin.

Uses Surya's vision transformer architecture for document OCR.
Supports 90+ languages and optional layout analysis.
"""

import sys
import time
from pathlib import Path
from typing import Any

# This adds the src folder to Python's path so we can import Visual Buffet code
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the Visual Buffet plugin framework
from visual_buffet.exceptions import PluginError
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult
```

**Why this path manipulation?**
Plugins live outside the main `src/` folder. This trick lets us import Visual Buffet's code.

#### Part B: Configuration

```python
PLUGIN_VERSION = "1.0.0"

# Default settings - users can override these
DEFAULT_CONFIG = {
    "threshold": 0.5,       # Minimum confidence to include (0.0-1.0)
    "limit": 100,           # Maximum text lines to return
    "include_layout": False, # Run layout analysis (slower)
    "include_boxes": True,   # Include bounding box coordinates
    "det_batch_size": None,  # None = auto-detect based on VRAM
    "rec_batch_size": None,  # None = auto-detect based on VRAM
    "sort_by": "confidence", # How to sort results
}
```

#### Part C: The Plugin Class

```python
class SuryaOCRPlugin(PluginBase):
    """Surya OCR plugin for document text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        """Initialize the plugin.

        Args:
            plugin_dir: Path to this plugin's folder (e.g., plugins/surya_ocr/)
        """
        super().__init__(plugin_dir)

        # These will hold our loaded models (we load them lazily)
        self._det_predictor = None       # Detection model
        self._rec_predictor = None       # Recognition model
        self._foundation_predictor = None # Foundation model (required by recognition)
        self._layout_predictor = None    # Layout model (optional)

        # Copy default config so each instance has its own settings
        self._config = DEFAULT_CONFIG.copy()
```

**What is "lazy loading"?**
We don't load the models when the plugin is created. We wait until `tag()` is called. This saves memory if the plugin is never used.

#### Part D: get_info() Method

```python
    def get_info(self) -> PluginInfo:
        """Return metadata about this plugin.

        This is used by Visual Buffet to display plugin info to users.
        """
        return PluginInfo(
            name="surya_ocr",
            version=PLUGIN_VERSION,
            description="Surya OCR - Vision transformer document OCR with 90+ languages",
            hardware_reqs={
                "gpu": False,      # Not required, but recommended
                "min_ram_gb": 8,   # Minimum RAM needed
            },
            provides_confidence=True,      # Our tags have confidence scores
            recommended_threshold=0.5,     # Suggested confidence cutoff
        )
```

#### Part E: is_available() Method

```python
    def is_available(self) -> bool:
        """Check if Surya OCR is installed and ready to use.

        Returns:
            True if we can import Surya, False otherwise
        """
        try:
            # Try to import the main class we need
            from surya.detection import DetectionPredictor
            return True
        except ImportError:
            return False
```

**Why check imports?**
Users might not have Surya installed. This lets Visual Buffet know the plugin can't be used yet.

#### Part F: setup() Method

```python
    def setup(self) -> bool:
        """Install Surya if it's not already installed.

        This is called when is_available() returns False.

        Returns:
            True if installation succeeded, False otherwise
        """
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "surya-ocr"],
                check=True,
                capture_output=True
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False
```

#### Part G: configure() Method

```python
    def configure(self, **kwargs: Any) -> None:
        """Update plugin settings at runtime.

        Example:
            plugin.configure(threshold=0.7, limit=50)

        Args:
            **kwargs: Settings to update (threshold, limit, etc.)
        """
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key] = value
```

#### Part H: tag() Method (The Main Method!)

```python
    def tag(self, image_path: Path) -> TagResult:
        """Process an image and extract text.

        This is the main method - it does the actual OCR work.

        Args:
            image_path: Path to the image file

        Returns:
            TagResult with detected text lines

        Raises:
            PluginError: If OCR fails
        """
        # First, check if Surya is installed
        if not self.is_available():
            raise PluginError("Surya OCR not installed. Run: pip install surya-ocr")

        # Load models if not already loaded (lazy loading)
        if self._det_predictor is None:
            self._load_models()

        # Track how long inference takes
        start_time = time.perf_counter()

        # Run the actual OCR
        tags, metadata = self._run_inference(image_path)

        # Calculate inference time in milliseconds
        inference_time = (time.perf_counter() - start_time) * 1000

        # Return results in the standard format
        return TagResult(
            tags=tags,
            model="surya_ocr",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )
```

#### Part I: _load_models() Helper

```python
    def _load_models(self) -> None:
        """Load all required Surya models.

        This is called once, the first time tag() is used.
        Models are cached for subsequent calls.
        """
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.foundation import FoundationPredictor

        print("Loading Surya OCR models...")

        # Load models in the correct order
        # Foundation predictor is required by Recognition predictor
        self._foundation_predictor = FoundationPredictor()
        self._det_predictor = DetectionPredictor()
        self._rec_predictor = RecognitionPredictor(self._foundation_predictor)

        # Optionally load layout predictor
        if self._config["include_layout"]:
            from surya.layout import LayoutPredictor
            from surya.settings import settings
            self._layout_predictor = LayoutPredictor(
                FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
            )
```

#### Part J: _run_inference() Helper

```python
    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict]:
        """Run OCR on an image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (list of Tag objects, metadata dict)
        """
        from PIL import Image

        # Load and prepare the image
        image = Image.open(image_path).convert("RGB")

        # Surya works best on images ≤2048px wide
        # If larger, resize while maintaining aspect ratio
        if image.width > 2048:
            ratio = 2048 / image.width
            new_height = int(image.height * ratio)
            image = image.resize((2048, new_height), Image.LANCZOS)

        # Run OCR: detection finds text regions, recognition reads them
        predictions = self._rec_predictor([image], det_predictor=self._det_predictor)

        # Convert results to our Tag format
        tags = []
        boxes_data = []

        if predictions and len(predictions) > 0:
            page = predictions[0]  # We process one image at a time

            for line in page.text_lines:
                # Get confidence score (default to 0.9 if not provided)
                conf = getattr(line, 'confidence', 0.9)

                # Skip low-confidence results
                if conf < self._config["threshold"]:
                    continue

                # Get the text content
                text = line.text.strip() if hasattr(line, 'text') else str(line)
                if not text:
                    continue

                # Create a Tag for this text line
                tags.append(Tag(label=text, confidence=round(float(conf), 4)))

                # Store bounding box info
                if self._config["include_boxes"]:
                    bbox = line.bbox if hasattr(line, 'bbox') else [0, 0, 0, 0]
                    boxes_data.append({
                        "text": text,
                        "confidence": round(float(conf), 4),
                        "bbox": bbox,
                    })

        # Sort results
        sort_by = self._config["sort_by"]
        if sort_by == "confidence":
            tags.sort(key=lambda t: -(t.confidence or 0))
            boxes_data.sort(key=lambda b: -b["confidence"])
        elif sort_by == "alphabetical":
            tags.sort(key=lambda t: t.label.lower())
            boxes_data.sort(key=lambda b: b["text"].lower())

        # Apply limit
        limit = self._config["limit"]
        if limit > 0:
            tags = tags[:limit]
            boxes_data = boxes_data[:limit]

        # Build metadata
        metadata = {"total_lines": len(tags)}
        if self._config["include_boxes"]:
            metadata["boxes"] = boxes_data

        # Add layout info if requested
        if self._config["include_layout"] and self._layout_predictor:
            layout_results = self._layout_predictor([image])
            if layout_results:
                layout_data = []
                for elem in layout_results[0].bboxes:
                    layout_data.append({
                        "label": elem.label,
                        "bbox": elem.bbox,
                        "position": elem.position,
                    })
                metadata["layout"] = layout_data

        return tags, metadata
```

---

## Step-by-Step: docTR Plugin

### Step 1: Create the Plugin Directory

```bash
mkdir -p plugins/doctr/models
```

### Step 2: Create plugin.toml

Create `plugins/doctr/plugin.toml`:

```toml
[plugin]
name = "doctr"
display_name = "docTR"
version = "1.0.0"
description = "Mindee docTR - Deep learning OCR with flexible model selection"
entry_point = "DocTRPlugin"
python_requires = ">=3.10"
provides_confidence = true

[plugin.dependencies]
python-doctr = ">=0.11.0"
torch = ">=2.0"

[plugin.hardware]
gpu_recommended = true
gpu_required = false
min_ram_gb = 4
min_vram_gb = 2

[plugin.defaults]
quality = "standard"
threshold = 0.5
limit = 100
batch_size = 1

[plugin.model]
det_arch = "db_resnet50"
reco_arch = "crnn_vgg16_bn"
assume_straight_pages = true
preserve_aspect_ratio = true

[plugin.output]
sort_by = "confidence"
include_boxes = true
```

### Step 3: Create __init__.py

Create `plugins/doctr/__init__.py`:

```python
"""docTR text detection and recognition plugin.

Uses Mindee's docTR library with flexible model architecture selection.
Supports TensorFlow and PyTorch backends.
"""

import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from visual_buffet.exceptions import PluginError
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult

PLUGIN_VERSION = "1.0.0"

# Available model architectures - for documentation
DET_ARCHITECTURES = [
    "db_resnet50", "db_mobilenet_v3_large",
    "linknet_resnet18", "linknet_resnet34", "linknet_resnet50",
    "fast_tiny", "fast_small", "fast_base"
]

RECO_ARCHITECTURES = [
    "crnn_vgg16_bn", "crnn_mobilenet_v3_small", "crnn_mobilenet_v3_large",
    "sar_resnet31", "master", "vitstr_small", "vitstr_base",
    "parseq", "viptr_tiny"
]

DEFAULT_CONFIG = {
    "det_arch": "db_resnet50",        # Detection model
    "reco_arch": "crnn_vgg16_bn",     # Recognition model
    "threshold": 0.5,                  # Minimum confidence
    "limit": 100,                      # Max results
    "include_boxes": True,             # Include bounding boxes
    "assume_straight_pages": True,     # Optimize for non-rotated pages
    "preserve_aspect_ratio": True,     # Keep aspect ratio when resizing
    "sort_by": "confidence",           # Sort order
}


class DocTRPlugin(PluginBase):
    """docTR OCR plugin for document text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._model = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="doctr",
            version=PLUGIN_VERSION,
            description="docTR - Deep learning OCR with flexible model selection",
            hardware_reqs={
                "gpu": False,
                "min_ram_gb": 4,
            },
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        try:
            from doctr.models import ocr_predictor
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "python-doctr[torch]"],
                check=True, capture_output=True
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def configure(self, **kwargs: Any) -> None:
        """Configure plugin settings.

        If model architecture changes, we need to reload the model.
        """
        needs_reload = False
        reload_triggers = {"det_arch", "reco_arch", "assume_straight_pages"}

        for key, value in kwargs.items():
            if key in self._config:
                if key in reload_triggers and self._config[key] != value:
                    needs_reload = True
                self._config[key] = value

        # Unload model if architecture changed
        if needs_reload and self._model is not None:
            self._model = None

    def tag(self, image_path: Path) -> TagResult:
        if not self.is_available():
            raise PluginError(
                "docTR not installed. Run: pip install 'python-doctr[torch]'"
            )

        # Lazy load model
        if self._model is None:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags, metadata = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Include model info in name
        model_name = f"doctr_{self._config['det_arch']}_{self._config['reco_arch']}"

        return TagResult(
            tags=tags,
            model=model_name,
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_model(self) -> None:
        """Load the docTR OCR model with configured architectures."""
        from doctr.models import ocr_predictor

        det_arch = self._config["det_arch"]
        reco_arch = self._config["reco_arch"]

        print(f"Loading docTR model (det={det_arch}, reco={reco_arch})...")

        self._model = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True,
            assume_straight_pages=self._config["assume_straight_pages"],
            preserve_aspect_ratio=self._config["preserve_aspect_ratio"],
        )

    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict]:
        """Run OCR inference on an image."""
        from doctr.io import DocumentFile

        # Load image using docTR's document loader
        doc = DocumentFile.from_images(str(image_path))

        # Run OCR
        result = self._model(doc)

        tags = []
        boxes_data = []

        # docTR returns a hierarchical structure:
        # Document -> Pages -> Blocks -> Lines -> Words
        for page in result.pages:
            # Get page dimensions for coordinate conversion
            page_h, page_w = page.dimensions

            for block in page.blocks:
                for line in block.lines:
                    # Combine all words in the line into one string
                    line_text = " ".join(word.value for word in line.words)

                    # Average the confidence of all words
                    if line.words:
                        line_conf = sum(w.confidence for w in line.words) / len(line.words)
                    else:
                        line_conf = 0

                    # Skip low-confidence or empty results
                    if line_conf < self._config["threshold"]:
                        continue
                    if not line_text.strip():
                        continue

                    tags.append(Tag(
                        label=line_text.strip(),
                        confidence=round(float(line_conf), 4),
                    ))

                    if self._config["include_boxes"]:
                        # docTR coordinates are normalized (0-1)
                        # Convert to pixel coordinates
                        geom = line.geometry
                        bbox = [
                            [int(geom[0][0] * page_w), int(geom[0][1] * page_h)],
                            [int(geom[1][0] * page_w), int(geom[1][1] * page_h)],
                        ]
                        boxes_data.append({
                            "text": line_text.strip(),
                            "confidence": round(float(line_conf), 4),
                            "bbox": bbox,
                        })

        # Sort results
        sort_by = self._config["sort_by"]
        if sort_by == "confidence":
            tags.sort(key=lambda t: -(t.confidence or 0))
            boxes_data.sort(key=lambda b: -b["confidence"])
        elif sort_by == "alphabetical":
            tags.sort(key=lambda t: t.label.lower())
            boxes_data.sort(key=lambda b: b["text"].lower())

        # Apply limit
        limit = self._config["limit"]
        if limit > 0:
            tags = tags[:limit]
            boxes_data = boxes_data[:limit]

        metadata = {
            "total_lines": len(tags),
            "det_arch": self._config["det_arch"],
            "reco_arch": self._config["reco_arch"],
        }

        if self._config["include_boxes"]:
            metadata["boxes"] = boxes_data

        return tags, metadata
```

---

## Testing Your Plugins

### Quick Test: Check if Plugin Loads

```bash
# From the project root
python -c "
from pathlib import Path
from plugins.surya_ocr import SuryaOCRPlugin
from plugins.doctr import DocTRPlugin

surya = SuryaOCRPlugin(Path('plugins/surya_ocr'))
doctr = DocTRPlugin(Path('plugins/doctr'))

print(f'Surya available: {surya.is_available()}')
print(f'docTR available: {doctr.is_available()}')
print(f'Surya info: {surya.get_info()}')
print(f'docTR info: {doctr.get_info()}')
"
```

### Test on a Real Image

```bash
python -c "
from pathlib import Path
from plugins.surya_ocr import SuryaOCRPlugin

plugin = SuryaOCRPlugin(Path('plugins/surya_ocr'))
if plugin.is_available():
    result = plugin.tag(Path('images/visual-buffet/testimage04.jpg'))
    print(f'Found {len(result.tags)} text lines')
    for tag in result.tags[:5]:
        print(f'  - {tag.label} ({tag.confidence})')
else:
    print('Surya not installed')
"
```

### Using Visual Buffet CLI

```bash
# List all plugins
visual-buffet plugins list

# Tag an image with a specific plugin
visual-buffet tag images/test.jpg --plugin surya_ocr
visual-buffet tag images/test.jpg --plugin doctr
```

---

## Troubleshooting Common Issues

### "ModuleNotFoundError: No module named 'surya'"

**Cause**: Surya is not installed

**Fix**:
```bash
pip install surya-ocr
```

### "ModuleNotFoundError: No module named 'doctr'"

**Cause**: docTR is not installed

**Fix**:
```bash
pip install "python-doctr[torch]"
```

### "CUDA out of memory"

**Cause**: GPU doesn't have enough VRAM

**Fix**: Reduce batch size or use CPU:
```python
plugin.configure(det_batch_size=4, rec_batch_size=32)
```

Or set environment variable:
```bash
export TORCH_DEVICE=cpu
```

### "PIL.UnidentifiedImageError"

**Cause**: Image file is corrupted or unsupported format

**Fix**: Convert to JPEG or PNG:
```python
from PIL import Image
img = Image.open("file.webp").convert("RGB")
img.save("file.jpg")
```

### Plugin Not Showing in Visual Buffet

**Cause**: Missing or invalid plugin.toml

**Fix**: Check these things:
1. File is named exactly `plugin.toml` (not `plugin.toml.txt`)
2. TOML syntax is valid (no missing quotes or brackets)
3. `entry_point` matches your class name exactly

### Results Are Empty

**Cause**: Threshold too high or no text in image

**Fix**:
```python
plugin.configure(threshold=0.0)  # Accept all results
result = plugin.tag(image_path)
```

---

## Summary

You've learned:

1. **Plugin structure** - `__init__.py` + `plugin.toml`
2. **Required methods** - `get_info()`, `is_available()`, `tag()`
3. **Lazy loading** - Load models only when needed
4. **TagResult format** - How to return results
5. **Testing** - How to verify your plugin works

Now you're ready to implement the plugins! Follow the code templates above, and refer back to this guide when you get stuck.
