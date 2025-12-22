# docTR (Document Text Recognition) - Subject Matter Expert Document

> **Generated**: 2024-12-22
> **Sources current as of**: December 2024
> **Scope**: Comprehensive
> **Version**: 1.0
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

docTR is an open-source OCR library by Mindee that provides end-to-end document text recognition using a two-stage pipeline: text detection followed by text recognition [1][HIGH]. Built on TensorFlow 2 and PyTorch, it offers state-of-the-art performance on public document datasets comparable to Google Vision and AWS Textract [1][HIGH].

**Key Strengths:**
- Two-stage pipeline (8 detection + 9 recognition architectures to mix and match) [1]
- Apache 2.0 license - fully open for commercial use [1]
- Works with both PyTorch and TensorFlow backends [1]
- ONNX export support for production deployment [1]
- PDF and image input support out of the box [1]
- Hierarchical output (Pages → Blocks → Lines → Words) [1]

**Key Limitations:**
- Requires Python 3.10+ [1]
- No built-in layout analysis (tables, headers) unlike Surya
- Limited scene text performance (optimized for documents)
- Larger models (MASTER) have slower inference

**Visual Buffet Integration:**
docTR complements PaddleOCR well due to:
- Different architecture (CNN+CRNN vs PaddlePaddle's optimized models)
- Different detection approaches (DBNet vs EAST/SAST)
- Apache 2.0 license (more permissive than Surya's commercial restrictions)

---

## Background & Context

### Why docTR Exists

Mindee created docTR to provide a production-ready OCR solution that bridges the gap between academic research models and real-world deployment. Unlike research-focused libraries, docTR emphasizes ease of use, deployment flexibility, and consistent API across frameworks.

### Architecture Overview

docTR uses a modular two-stage approach:

```
Input Image → Text Detection → Crop Regions → Text Recognition → Structured Output
                   ↓                                    ↓
             DBNet/LinkNet/FAST                   CRNN/ViTSTR/PARSeq
```

| Stage | Purpose | Models Available |
|-------|---------|------------------|
| Detection | Find text regions | 8 architectures |
| Recognition | Convert regions to text | 9 architectures |

### Comparison with Other OCR Tools

| Feature | docTR | PaddleOCR | Surya |
|---------|-------|-----------|-------|
| License | Apache 2.0 | Apache 2.0 | GPL + Commercial |
| Architecture | CNN + CRNN/Transformer | CNN + LSTM | Vision Transformer |
| Framework | TensorFlow/PyTorch | PaddlePaddle | PyTorch |
| Layout Analysis | No | PP-Structure | Built-in |
| Table Recognition | No | Separate module | Built-in |
| Scene Text | Moderate | Good | Poor |
| Document Text | Good | Good | Excellent |
| ONNX Export | Yes | Yes | No |

**Cross-Validation Value**: docTR's DBNet + CRNN pipeline differs from both PaddleOCR (different detection) and Surya (different overall approach). When all three agree, confidence is maximized.

---

## Installation & Setup

### Requirements

- Python 3.10+ [1][HIGH]
- TensorFlow 2.15+ OR PyTorch 2.0+ [1]
- ~2-4GB disk space for models
- GPU optional but recommended for speed

### Installation Options

```bash
# PyTorch backend (recommended for Visual Buffet compatibility)
pip install "python-doctr[torch]"

# TensorFlow backend
pip install "python-doctr[tf]"

# With visualization support
pip install "python-doctr[torch,viz]"

# All optional dependencies
pip install "python-doctr[torch,viz,html,contrib]"
```

### ONNX Runtime (Lightweight)

For production without full framework:

```bash
pip install onnxtr[cpu]  # CPU only
pip install onnxtr[gpu]  # CUDA support
```

### Docker Deployment

```bash
# GPU-enabled container
docker run -it --gpus all ghcr.io/mindee/doctr:torch-py3.9.18-2024-10 bash

# Build custom image
docker build -t doctr --build-arg FRAMEWORK=torch \
  --build-arg PYTHON_VERSION=3.10 .
```

---

## Python API

### Basic OCR

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load model (downloads on first use)
model = ocr_predictor(pretrained=True)

# Load document
doc = DocumentFile.from_images("document.jpg")
# Or: DocumentFile.from_pdf("document.pdf")

# Run OCR
result = model(doc)

# Access results
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                print(f"Text: {word.value}")
                print(f"Confidence: {word.confidence}")
                print(f"Bbox: {word.geometry}")  # ((x1,y1), (x2,y2))
```

### Custom Model Configuration

```python
# Specify detection and recognition architectures
model = ocr_predictor(
    det_arch='db_resnet50',        # Detection model
    reco_arch='crnn_vgg16_bn',     # Recognition model
    pretrained=True,
    assume_straight_pages=True,    # Faster if pages not rotated
    preserve_aspect_ratio=True,    # Maintain aspect during resize
)
```

### Available Detection Architectures

| Model | Parameters | Speed (sec/iter) | Best For |
|-------|------------|------------------|----------|
| `db_mobilenet_v3_large` | 4.2M | 0.5 | Speed priority |
| `db_resnet50` | 25.4M | 0.8 | Balanced |
| `linknet_resnet18` | 11.5M | 0.6 | Speed/accuracy balance |
| `linknet_resnet34` | 21.7M | 0.7 | Accuracy priority |
| `linknet_resnet50` | 28.8M | 0.8 | High accuracy |
| `fast_tiny` | 2.8M | 0.4 | Fastest |
| `fast_small` | 7.5M | 0.5 | Fast |
| `fast_base` | 15.8M | 0.8 | Highest accuracy |

### Available Recognition Architectures

| Model | Parameters | Speed (sec/iter) | Accuracy (FUNSD) |
|-------|------------|------------------|------------------|
| `crnn_mobilenet_v3_small` | 2.1M | 0.03 | ~80% |
| `crnn_mobilenet_v3_large` | 4.5M | 0.05 | ~83% |
| `crnn_vgg16_bn` | 15.8M | 0.08 | ~85% |
| `sar_resnet31` | 54.2M | 0.3 | ~86% |
| `master` | 67.7M | 17.6 | **88.57%** |
| `vitstr_small` | 21.5M | 0.1 | ~86% |
| `vitstr_base` | 85.8M | 0.2 | ~87% |
| `parseq` | 23.8M | 0.1 | ~87% |
| `viptr_tiny` | 3.2M | **0.08** | ~82% |

### Input Formats

```python
from doctr.io import DocumentFile

# Single image
doc = DocumentFile.from_images("page.jpg")

# Multiple images
doc = DocumentFile.from_images(["page1.jpg", "page2.jpg"])

# PDF file
doc = DocumentFile.from_pdf("document.pdf")

# From URL
doc = DocumentFile.from_url("https://example.com/doc.pdf")

# From numpy array
import numpy as np
doc = DocumentFile.from_images([np_array])
```

### Output Processing

```python
# Export to JSON
json_output = result.export()

# JSON structure
{
    "pages": [
        {
            "page_idx": 0,
            "dimensions": [height, width],
            "orientation": {"value": 0, "confidence": 1.0},
            "language": {"value": "en", "confidence": 0.95},
            "blocks": [
                {
                    "geometry": [[x1, y1], [x2, y2]],
                    "lines": [
                        {
                            "geometry": [[x1, y1], [x2, y2]],
                            "words": [
                                {
                                    "value": "Hello",
                                    "confidence": 0.98,
                                    "geometry": [[x1, y1], [x2, y2]]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}

# Visualize results
result.show()  # Opens matplotlib window

# Synthesize document (render text on blank page)
synthetic_pages = result.synthesize()
```

### Detection Only

```python
from doctr.models import detection_predictor

det_model = detection_predictor(arch='db_resnet50', pretrained=True)
det_result = det_model(doc)

# Access detection boxes
for page in det_result.pages:
    for box in page.boxes:
        print(f"Box: {box}")  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

### Recognition Only

```python
from doctr.models import recognition_predictor

rec_model = recognition_predictor(arch='crnn_vgg16_bn', pretrained=True)

# Recognition expects cropped word images
from PIL import Image
import numpy as np

crop = np.array(Image.open("word_crop.jpg"))
rec_result = rec_model([crop])

print(f"Text: {rec_result[0][0]}")
print(f"Confidence: {rec_result[0][1]}")
```

---

## Output Formats

### Document Hierarchy

```
Document
├── pages: List[Page]
│   ├── page_idx: int
│   ├── dimensions: (height, width)
│   ├── orientation: Dict  # {"value": degrees, "confidence": float}
│   ├── language: Dict     # {"value": "en", "confidence": float}
│   └── blocks: List[Block]
│       ├── geometry: [[x1,y1], [x2,y2]]
│       └── lines: List[Line]
│           ├── geometry: [[x1,y1], [x2,y2]]
│           └── words: List[Word]
│               ├── value: str
│               ├── confidence: float
│               └── geometry: [[x1,y1], [x2,y2]]
```

### Geometry Format

All geometry is normalized to `[0, 1]` relative to page dimensions:
- `[[x1, y1], [x2, y2]]` for axis-aligned boxes (straight pages)
- `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` for rotated boxes

To convert to pixel coordinates:
```python
def to_pixels(geometry, page_dims):
    """Convert normalized coords to pixels."""
    h, w = page_dims
    return [
        [int(x * w), int(y * h)]
        for x, y in geometry
    ]
```

---

## CLI Commands

### Analyze Script

```bash
# Basic usage
python -m doctr.scripts.analyze path/to/document.pdf

# With specific models
python -m doctr.scripts.analyze document.jpg \
    --det-arch db_resnet50 \
    --reco-arch crnn_vgg16_bn

# Output visualization
python -m doctr.scripts.analyze document.pdf --export-as-images
```

### Training Scripts

```bash
# Train detection model
python doctr/references/detection/train_pytorch.py \
    --arch db_resnet50 \
    --dataset FUNSD

# Train recognition model
python doctr/references/recognition/train_pytorch.py \
    --arch crnn_vgg16_bn \
    --dataset FUNSD
```

---

## Visual Buffet Integration

### Plugin Design Pattern

Following existing plugin structure (similar to `paddle_ocr`):

```python
class DocTRPlugin(PluginBase):
    """docTR OCR plugin for text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._model = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="doctr",
            version="1.0.0",
            description="docTR - Document Text Recognition with deep learning",
            hardware_reqs={"gpu": False, "min_ram_gb": 4},
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        try:
            from doctr.models import ocr_predictor
            return True
        except ImportError:
            return False

    def tag(self, image_path: Path) -> TagResult:
        # Lazy load model
        # Run detection + recognition
        # Convert to Tag objects with bboxes in metadata
        pass
```

### Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `det_arch` | str | "db_resnet50" | Detection architecture |
| `reco_arch` | str | "crnn_vgg16_bn" | Recognition architecture |
| `threshold` | float | 0.5 | Min confidence to return |
| `limit` | int | 100 | Max text lines to return |
| `include_boxes` | bool | True | Include bboxes in metadata |
| `assume_straight_pages` | bool | True | Optimize for non-rotated pages |
| `sort_by` | str | "confidence" | Sort: confidence, position, alphabetical |

### Tag Output Format

Match PaddleOCR plugin format:

```python
TagResult(
    tags=[
        Tag(label="Detected text line 1", confidence=0.95),
        Tag(label="Detected text line 2", confidence=0.88),
    ],
    model="doctr_db_resnet50_crnn_vgg16_bn",
    version="1.0.0",
    inference_time_ms=156.3,
    metadata={
        "total_lines": 12,
        "det_arch": "db_resnet50",
        "reco_arch": "crnn_vgg16_bn",
        "boxes": [
            {"text": "...", "confidence": 0.95, "bbox": [[x1,y1], [x2,y2]]},
        ]
    }
)
```

---

## Performance Optimization

### Model Selection Guidelines

| Priority | Detection | Recognition | Use Case |
|----------|-----------|-------------|----------|
| **Speed** | `fast_tiny` | `viptr_tiny` | Real-time, mobile |
| **Balanced** | `db_resnet50` | `crnn_vgg16_bn` | General use |
| **Accuracy** | `fast_base` | `master` | Quality-critical |

### GPU Memory Estimation

Based on model parameters and typical batch sizes:

| Configuration | Approx VRAM |
|--------------|-------------|
| fast_tiny + viptr_tiny | ~1GB |
| db_resnet50 + crnn_vgg16_bn | ~2-3GB |
| fast_base + master | ~4-6GB |

### Batch Processing

```python
# Process multiple images efficiently
images = [Image.open(p) for p in image_paths]
doc = DocumentFile.from_images(images)
result = model(doc)  # Batched inference
```

### Straight Page Optimization

If documents are known to be properly oriented:

```python
model = ocr_predictor(
    assume_straight_pages=True,  # Skip rotation detection
    pretrained=True
)
```

This provides ~15-20% speedup.

### ONNX Deployment

For production with minimal dependencies:

```python
# Export to ONNX
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True, export_as_straight_boxes=True)

# Use OnnxTR for inference
from onnxtr.models import ocr_predictor as onnx_predictor
onnx_model = onnx_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
```

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Fine-tuning docTR on custom datasets
- Detailed KIE (Key Information Extraction) predictor usage
- Multi-GPU training configuration
- Comparison benchmarks across all languages

### Known Limitations [1][HIGH]

1. **No layout analysis**: Unlike Surya, doesn't detect tables, headers, or document structure
2. **Scene text**: Optimized for documents; photos/signs may have lower accuracy
3. **Rotated text**: While supported, `assume_straight_pages=True` is faster
4. **Large MASTER model**: High accuracy but 17.6 sec/iter makes it impractical for batch

### Unverified Claims

| Claim | Confidence | Note |
|-------|------------|------|
| "Comparable to Google Vision/AWS Textract" | MEDIUM | Marketing claim, limited benchmark data |
| "State-of-the-art on public datasets" | MEDIUM | Benchmarks provided but conditions vary |

### License Advantages

Apache 2.0 license has no commercial restrictions, unlike:
- Surya (GPL code + commercial weight restrictions)
- Some PaddleOCR models (check individual model licenses)

---

## Recommendations

### For Visual Buffet Integration

1. **Use PyTorch backend** - Consistent with other plugins (RAM++, SigLIP, Florence-2)
2. **Default to balanced config** - `db_resnet50` + `crnn_vgg16_bn`
3. **Enable straight_pages** - Most images won't need rotation detection
4. **Match PaddleOCR output** - Consistent metadata structure

### Configuration Defaults

```python
DEFAULT_CONFIG = {
    "det_arch": "db_resnet50",
    "reco_arch": "crnn_vgg16_bn",
    "threshold": 0.5,
    "limit": 100,
    "include_boxes": True,
    "assume_straight_pages": True,
    "sort_by": "confidence",
}
```

### Recommended Use Cases

| Use Case | Recommendation |
|----------|----------------|
| Fast processing | docTR with fast_tiny models |
| Cross-validation with PaddleOCR | Default docTR config |
| Document with known layout | PaddleOCR or Surya |
| Commercial deployment | docTR (Apache 2.0) |
| Table extraction | Surya (built-in) |

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [GitHub - mindee/doctr](https://github.com/mindee/doctr) | 2024-12 | Primary | API, features, models |
| 2 | [docTR Documentation](https://mindee.github.io/doctr/) | 2024-12 | Primary | Installation, usage |
| 3 | [PyPI - python-doctr](https://pypi.org/project/python-doctr/) | 2024-12 | Primary | Versioning, deps |
| 4 | [docTR Model Guide](https://mindee.github.io/doctr/latest/using_doctr/using_models.html) | 2024-12 | Primary | Model benchmarks |
| 5 | [PyTorch Blog - docTR](https://pytorch.org/blog/doctr-joins-pytorch-ecosystem/) | 2024 | Secondary | Framework integration |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-22 | Initial version |

---

## Claims Appendix

```yaml
claims:
  - id: C001
    text: "docTR supports 8 detection architectures"
    type: quantitative
    citations: [1]
    confidence: HIGH
    source_quote: "linknet_resnet18, linknet_resnet34, linknet_resnet50, db_resnet50, db_mobilenet_v3_large, fast_tiny, fast_small, fast_base"

  - id: C002
    text: "docTR supports 9 recognition architectures"
    type: quantitative
    citations: [1]
    confidence: HIGH
    source_quote: "crnn_vgg16_bn, crnn_mobilenet_v3_small, crnn_mobilenet_v3_large, sar_resnet31, master, vitstr_small, vitstr_base, parseq, viptr_tiny"

  - id: C003
    text: "MASTER achieves 88.57% accuracy on FUNSD"
    type: quantitative
    citations: [4]
    confidence: HIGH
    source_quote: "master achieves 88.57 exact-match accuracy on FUNSD"

  - id: C004
    text: "fast_base achieves 94.39% recall on CORD"
    type: quantitative
    citations: [4]
    confidence: HIGH
    source_quote: "fast_base achieves 94.39% recall and 85.36% precision"

  - id: C005
    text: "Apache 2.0 license"
    type: factual
    citations: [1]
    confidence: HIGH
    source_quote: "Apache License 2.0"

  - id: C006
    text: "Requires Python 3.10+"
    type: factual
    citations: [1]
    confidence: HIGH
    source_quote: "python >= 3.10"

  - id: C007
    text: "Supports both TensorFlow and PyTorch"
    type: factual
    citations: [1]
    confidence: HIGH
    source_quote: "Powered by TensorFlow 2 & PyTorch"
```
