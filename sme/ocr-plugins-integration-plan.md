# OCR Plugins Integration Plan: Surya & docTR

> **Generated**: 2024-12-22
> **Version**: 1.0
> **Status**: Ready for Implementation

---

## Overview

This plan details the implementation of two new OCR plugins for Visual Buffet:

1. **Surya OCR** - Vision transformer-based document OCR with layout analysis
2. **docTR** - CNN+CRNN-based document OCR with flexible model selection

Both plugins will follow the established Visual Buffet plugin architecture, matching the pattern set by `paddle_ocr`, `florence_2`, `siglip`, and other existing plugins.

---

## Plugin Architecture Summary

### Existing Pattern Analysis

Based on analysis of existing plugins (`paddle_ocr/__init__.py`, `florence_2/__init__.py`, `siglip/__init__.py`):

| Component | Pattern |
|-----------|---------|
| **Base Class** | Inherit from `PluginBase` |
| **Required Methods** | `get_info()`, `is_available()`, `tag()` |
| **Optional Methods** | `setup()`, `configure()` |
| **Model Loading** | Lazy loading (load on first `tag()` call) |
| **Output Format** | Return `TagResult` with `List[Tag]` |
| **Metadata** | Include bounding boxes in `metadata` dict |
| **Configuration** | `DEFAULT_CONFIG` dict + `configure()` method |
| **Device Detection** | Auto-detect CUDA > MPS > CPU |

### File Structure

```
plugins/
├── surya_ocr/
│   ├── __init__.py          # Plugin implementation
│   ├── plugin.toml          # Plugin metadata and defaults
│   └── models/              # (empty - models download to HuggingFace cache)
│
└── doctr/
    ├── __init__.py          # Plugin implementation
    ├── plugin.toml          # Plugin metadata and defaults
    └── models/              # (empty - models download to doctr cache)
```

---

## Surya OCR Plugin Specification

### plugin.toml

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
# Include layout analysis (tables, headers, etc.)
include_layout = false

# Detection batch size (auto = based on VRAM)
det_batch_size = "auto"

# Recognition batch size (auto = based on VRAM)
rec_batch_size = "auto"

[plugin.detection]
# Blank threshold for line spacing (0.0-1.0)
blank_threshold = "default"

# Text threshold for text joining (0.0-1.0, must > blank_threshold)
text_threshold = "default"

[plugin.output]
sort_by = "confidence"
include_boxes = true
```

### __init__.py Structure

```python
"""Surya OCR text detection and recognition plugin.

Uses Surya's vision transformer architecture for document OCR.
Supports 90+ languages and optional layout analysis.
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

DEFAULT_CONFIG = {
    "threshold": 0.5,
    "limit": 100,
    "include_layout": False,
    "include_boxes": True,
    "det_batch_size": None,  # Auto
    "rec_batch_size": None,  # Auto
    "sort_by": "confidence",
}


class SuryaOCRPlugin(PluginBase):
    """Surya OCR plugin for document text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._det_predictor = None
        self._rec_predictor = None
        self._foundation_predictor = None
        self._layout_predictor = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="surya_ocr",
            version=PLUGIN_VERSION,
            description="Surya OCR - Vision transformer document OCR with 90+ languages",
            hardware_reqs={
                "gpu": False,
                "min_ram_gb": 8,
            },
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        try:
            from surya.detection import DetectionPredictor
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        # Models auto-download on first use
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "surya-ocr"],
                check=True, capture_output=True
            )
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def configure(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key] = value

    def tag(self, image_path: Path) -> TagResult:
        if not self.is_available():
            raise PluginError("Surya OCR not installed. Run: pip install surya-ocr")

        # Lazy load models
        if self._det_predictor is None:
            self._load_models()

        # Run inference
        start_time = time.perf_counter()
        tags, metadata = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        return TagResult(
            tags=tags,
            model="surya_ocr",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_models(self) -> None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.foundation import FoundationPredictor

        print("Loading Surya OCR models...")
        self._foundation_predictor = FoundationPredictor()
        self._det_predictor = DetectionPredictor()
        self._rec_predictor = RecognitionPredictor(self._foundation_predictor)

        if self._config["include_layout"]:
            from surya.layout import LayoutPredictor
            from surya.settings import settings
            self._layout_predictor = LayoutPredictor(
                FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
            )

    def _run_inference(self, image_path: Path) -> tuple[list[Tag], dict]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Resize if too large (Surya optimal at ≤2048px)
        if image.width > 2048:
            ratio = 2048 / image.width
            new_height = int(image.height * ratio)
            image = image.resize((2048, new_height), Image.LANCZOS)

        # Run OCR
        predictions = self._rec_predictor([image], det_predictor=self._det_predictor)

        tags = []
        boxes_data = []

        if predictions and len(predictions) > 0:
            page = predictions[0]
            for line in page.text_lines:
                conf = getattr(line, 'confidence', 0.9)  # Default if not provided

                if conf < self._config["threshold"]:
                    continue

                text = line.text.strip() if hasattr(line, 'text') else str(line)
                if not text:
                    continue

                tags.append(Tag(label=text, confidence=round(float(conf), 4)))

                if self._config["include_boxes"]:
                    bbox = line.bbox if hasattr(line, 'bbox') else [0, 0, 0, 0]
                    boxes_data.append({
                        "text": text,
                        "confidence": round(float(conf), 4),
                        "bbox": bbox,
                    })

        # Apply limit and sorting
        limit = self._config["limit"]
        sort_by = self._config["sort_by"]

        if sort_by == "confidence":
            tags.sort(key=lambda t: -(t.confidence or 0))
            boxes_data.sort(key=lambda b: -b["confidence"])
        elif sort_by == "alphabetical":
            tags.sort(key=lambda t: t.label.lower())
            boxes_data.sort(key=lambda b: b["text"].lower())

        if limit > 0:
            tags = tags[:limit]
            boxes_data = boxes_data[:limit]

        metadata = {"total_lines": len(tags)}
        if self._config["include_boxes"]:
            metadata["boxes"] = boxes_data

        # Add layout if enabled
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

## docTR Plugin Specification

### plugin.toml

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
python-doctr = ">=0.11.0"  # Updated per audit 2024-12-22
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
# Detection architecture
# Options: db_resnet50, db_mobilenet_v3_large, linknet_resnet18,
#          linknet_resnet34, linknet_resnet50, fast_tiny, fast_small, fast_base
det_arch = "db_resnet50"

# Recognition architecture
# Options: crnn_vgg16_bn, crnn_mobilenet_v3_small, crnn_mobilenet_v3_large,
#          sar_resnet31, master, vitstr_small, vitstr_base, parseq, viptr_tiny
reco_arch = "crnn_vgg16_bn"

# Assume pages are not rotated (faster)
assume_straight_pages = true

# Preserve aspect ratio during resize
preserve_aspect_ratio = true

[plugin.output]
sort_by = "confidence"
include_boxes = true
```

### __init__.py Structure

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

# Available model architectures
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
    "det_arch": "db_resnet50",
    "reco_arch": "crnn_vgg16_bn",
    "threshold": 0.5,
    "limit": 100,
    "include_boxes": True,
    "assume_straight_pages": True,
    "preserve_aspect_ratio": True,
    "sort_by": "confidence",
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
        needs_reload = False
        reload_triggers = {"det_arch", "reco_arch", "assume_straight_pages"}

        for key, value in kwargs.items():
            if key in self._config:
                if key in reload_triggers and self._config[key] != value:
                    needs_reload = True
                self._config[key] = value

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

        model_name = f"doctr_{self._config['det_arch']}_{self._config['reco_arch']}"
        return TagResult(
            tags=tags,
            model=model_name,
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _load_model(self) -> None:
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
        from doctr.io import DocumentFile

        # Load image
        doc = DocumentFile.from_images(str(image_path))

        # Run OCR
        result = self._model(doc)

        tags = []
        boxes_data = []

        # Extract words from hierarchical structure
        for page in result.pages:
            page_h, page_w = page.dimensions

            for block in page.blocks:
                for line in block.lines:
                    # Combine words into line text
                    line_text = " ".join(word.value for word in line.words)
                    line_conf = sum(w.confidence for w in line.words) / len(line.words) if line.words else 0

                    if line_conf < self._config["threshold"]:
                        continue

                    if not line_text.strip():
                        continue

                    tags.append(Tag(
                        label=line_text.strip(),
                        confidence=round(float(line_conf), 4),
                    ))

                    if self._config["include_boxes"]:
                        # Convert normalized coords to pixels
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

        # Apply sorting
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

## Implementation Checklist

### Phase 1: Setup (Both Plugins)

- [ ] Create `plugins/surya_ocr/` directory
- [ ] Create `plugins/doctr/` directory
- [ ] Create `plugins/surya_ocr/models/` (empty, for convention)
- [ ] Create `plugins/doctr/models/` (empty, for convention)

### Phase 2: Surya OCR Plugin

- [ ] Write `plugins/surya_ocr/plugin.toml`
- [ ] Write `plugins/surya_ocr/__init__.py`
- [ ] Test `is_available()` with surya-ocr not installed
- [ ] Test `setup()` installs surya-ocr
- [ ] Test `tag()` on sample image
- [ ] Test `configure()` with different settings
- [ ] Test with `include_layout=True`
- [ ] Verify output format matches PaddleOCR

### Phase 3: docTR Plugin

- [ ] Write `plugins/doctr/plugin.toml`
- [ ] Write `plugins/doctr/__init__.py`
- [ ] Test `is_available()` with doctr not installed
- [ ] Test `setup()` installs python-doctr
- [ ] Test `tag()` on sample image
- [ ] Test `configure()` with different model architectures
- [ ] Test with `assume_straight_pages=False`
- [ ] Verify output format matches PaddleOCR

### Phase 4: Integration Testing

- [ ] Test both plugins load via Visual Buffet engine
- [ ] Test plugins appear in `visual-buffet plugins list`
- [ ] Test tagging via CLI: `visual-buffet tag image.jpg --plugin surya_ocr`
- [ ] Test tagging via CLI: `visual-buffet tag image.jpg --plugin doctr`
- [ ] Test batch tagging on multiple images
- [ ] Test GUI displays results correctly

### Phase 5: Documentation

- [ ] Update README with new plugins
- [ ] Add settings reference docs for both plugins
- [ ] Update techguide.md with new plugin commands

---

## Error Handling

### Common Scenarios

| Error | Handling |
|-------|----------|
| Model not installed | Return clear install instructions in error |
| VRAM exhausted | Catch OOM, suggest reducing batch size |
| Unsupported image format | Convert to RGB via PIL |
| Empty image | Return empty TagResult with warning in metadata |
| Corrupted image | Catch PIL error, raise PluginError |

### Example Error Handling

```python
def tag(self, image_path: Path) -> TagResult:
    try:
        # ... inference code ...
    except torch.cuda.OutOfMemoryError:
        raise PluginError(
            "GPU out of memory. Try reducing batch size or using CPU:\n"
            "Set TORCH_DEVICE=cpu or reduce batch_size in config"
        )
    except Exception as e:
        raise PluginError(f"OCR inference failed: {e}")
```

---

## Testing Plan

### Unit Tests

```python
# tests/plugins/test_surya_ocr.py
def test_surya_is_available():
    plugin = SuryaOCRPlugin(Path("plugins/surya_ocr"))
    # Should return False if not installed, True if installed
    assert isinstance(plugin.is_available(), bool)

def test_surya_get_info():
    plugin = SuryaOCRPlugin(Path("plugins/surya_ocr"))
    info = plugin.get_info()
    assert info.name == "surya_ocr"
    assert info.provides_confidence == True

def test_surya_configure():
    plugin = SuryaOCRPlugin(Path("plugins/surya_ocr"))
    plugin.configure(threshold=0.7, limit=50)
    assert plugin._config["threshold"] == 0.7
    assert plugin._config["limit"] == 50
```

### Integration Tests

```python
# tests/integration/test_ocr_plugins.py
def test_surya_tag_real_image(sample_image):
    plugin = SuryaOCRPlugin(Path("plugins/surya_ocr"))
    if not plugin.is_available():
        pytest.skip("Surya not installed")

    result = plugin.tag(sample_image)
    assert isinstance(result, TagResult)
    assert result.model == "surya_ocr"
    assert result.inference_time_ms > 0

def test_doctr_tag_real_image(sample_image):
    plugin = DocTRPlugin(Path("plugins/doctr"))
    if not plugin.is_available():
        pytest.skip("docTR not installed")

    result = plugin.tag(sample_image)
    assert isinstance(result, TagResult)
    assert "doctr_" in result.model
```

---

## Dependencies to Add

### pyproject.toml Updates

```toml
[project.optional-dependencies]
surya = ["surya-ocr>=0.17.0"]
doctr = ["python-doctr[torch]>=0.9.0"]
ocr = [
    "paddlepaddle>=2.5.0",
    "paddleocr>=2.7.0",
    "surya-ocr>=0.17.0",
    "python-doctr[torch]>=0.9.0",
]
```

---

## Success Criteria

1. Both plugins load without errors
2. Both plugins return valid `TagResult` objects
3. Output format matches `paddle_ocr` plugin (tags + boxes in metadata)
4. Confidence scores are in 0.0-1.0 range
5. Threshold filtering works correctly
6. Limit setting works correctly
7. Sort options (confidence, position, alphabetical) work
8. GPU acceleration works when available
9. CPU fallback works when GPU unavailable
10. Error messages are clear and actionable
