# Surya OCR - Subject Matter Expert Document

> **Generated**: 2024-12-22
> **Sources current as of**: December 2024
> **Scope**: Comprehensive
> **Version**: 1.0
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

Surya is a modern, Python-based OCR toolkit that provides text detection, recognition, layout analysis, reading order detection, and table recognition in 90+ languages [1][HIGH]. Unlike traditional OCR engines, Surya uses vision transformer architecture (EfficientViT for detection, Donut-based for recognition) optimized for document understanding [1][HIGH].

**Key Strengths:**
- Outperforms Tesseract in accuracy on document text [1]
- Layout analysis detects tables, headers, figures, captions, and 14+ element types [1]
- Table recognition with row/column detection [1]
- GPU-accelerated with configurable batch sizes for VRAM management [1]
- Works on CPU but GPU recommended for production [1]

**Key Limitations:**
- Specialized for documents - will NOT work on photographs or natural scene text [1]
- Designed for printed text, limited handwriting support [1]
- Images should be ≤2048px width for optimal performance [1]
- Requires Python 3.10+ and PyTorch [1]
- Model license restricts commercial use (< $2M revenue threshold) [1]

**Visual Buffet Integration:**
Surya is ideal as a secondary OCR engine to cross-validate PaddleOCR results. Its different architecture (vision transformer vs CNN) provides true independent verification. Best used for:
- Document-heavy images with structured layouts
- Images with tables, headers, or complex formatting
- Cross-validation of PaddleOCR text extraction

---

## Background & Context

### Why Surya Exists

Traditional OCR engines like Tesseract use LSTM-based architectures designed primarily for single-line text recognition. Modern documents contain complex layouts with tables, multi-column text, headers, footers, and embedded figures. Surya was built to handle these modern document understanding challenges using transformer-based vision models.

### Architecture Overview

| Component | Model Base | Purpose |
|-----------|------------|---------|
| Text Detection | EfficientViT (semantic segmentation) | Find text line bounding boxes |
| Text Recognition | Donut-based transformer | Convert detected regions to text |
| Layout Analysis | Foundation model | Identify document structure elements |
| Table Recognition | Specialized model | Detect rows, columns, cells in tables |
| Reading Order | Foundation model | Determine logical reading sequence |

### Comparison with PaddleOCR

| Feature | Surya | PaddleOCR |
|---------|-------|-----------|
| Architecture | Vision Transformer | CNN + LSTM |
| Layout Analysis | Built-in (14+ labels) | PP-Structure addon |
| Table Recognition | Built-in | Separate module |
| Languages | 90+ | 100+ |
| Speed | Moderate | Fast |
| Scene Text | Poor | Good |
| Document Text | Excellent | Good |
| VRAM (default) | ~9GB detection | ~2GB |

**Cross-Validation Value**: Different architectures mean independent failure modes. When both agree, confidence is high. Disagreements warrant manual review.

---

## Installation & Setup

### Requirements

- Python 3.10+ [1][HIGH]
- PyTorch (CUDA optional but recommended) [1]
- ~4GB disk space for models (downloaded on first run) [1]
- GPU: 8-12GB VRAM for default batch sizes, adjustable down to 4GB [3]

### Installation

```bash
# Basic installation
pip install surya-ocr

# Models download automatically on first use
# Cached in ~/.cache/huggingface/hub/
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_DEVICE` | auto | Force device: `cuda`, `cpu`, `mps` |
| `RECOGNITION_BATCH_SIZE` | 512 (GPU), 32 (CPU) | ~40MB VRAM per item |
| `DETECTOR_BATCH_SIZE` | 36 (GPU), 6 (CPU) | ~440MB VRAM per item |
| `LAYOUT_BATCH_SIZE` | 32 (GPU), 4 (CPU) | ~220MB VRAM per item |
| `TABLE_REC_BATCH_SIZE` | 64 (GPU), 8 (CPU) | ~150MB VRAM per item |
| `DETECTOR_BLANK_THRESHOLD` | model default | Space detection (0-1) |
| `DETECTOR_TEXT_THRESHOLD` | model default | Text joining (0-1, > BLANK) |
| `COMPILE_ALL` | false | Enable PyTorch compilation |

### VRAM Management

For RTX 3090 (24GB), default settings work well. For lower VRAM:

| GPU VRAM | Recommended Settings |
|----------|---------------------|
| 24GB | Defaults |
| 16GB | `DETECTOR_BATCH_SIZE=24`, `RECOGNITION_BATCH_SIZE=256` |
| 12GB | `DETECTOR_BATCH_SIZE=16`, `RECOGNITION_BATCH_SIZE=128` |
| 8GB | `DETECTOR_BATCH_SIZE=8`, `RECOGNITION_BATCH_SIZE=64` |
| 6GB | `DETECTOR_BATCH_SIZE=4`, `RECOGNITION_BATCH_SIZE=32` |

---

## Python API

### Basic OCR (Detection + Recognition)

```python
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

# Load image
image = Image.open("document.jpg")

# Initialize predictors (loads models on first call)
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor)

# Run OCR
predictions = rec_predictor([image], det_predictor=det_predictor)

# Access results
for page in predictions:
    for line in page.text_lines:
        print(f"Text: {line.text}")
        print(f"Confidence: {line.confidence}")
        print(f"Bbox: {line.bbox}")  # [x1, y1, x2, y2]
        print(f"Polygon: {line.polygon}")  # 4 corner points
```

### Text Detection Only

```python
from PIL import Image
from surya.detection import DetectionPredictor

image = Image.open("document.jpg")
det_predictor = DetectionPredictor()

# Detect text regions
predictions = det_predictor([image])

for page in predictions:
    for bbox in page.bboxes:
        print(f"Bbox: {bbox.bbox}")  # [x1, y1, x2, y2]
        print(f"Confidence: {bbox.confidence}")
```

### Layout Analysis

```python
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

image = Image.open("document.jpg")

# Layout predictor uses foundation model
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)

predictions = layout_predictor([image])

for page in predictions:
    for element in page.bboxes:
        print(f"Label: {element.label}")  # Caption, Table, Text, etc.
        print(f"Position: {element.position}")  # Reading order
        print(f"Bbox: {element.bbox}")
        print(f"Top-k: {element.top_k}")  # Alternative labels
```

### Table Recognition

```python
from PIL import Image
from surya.table_rec import TableRecPredictor

image = Image.open("document_with_table.jpg")
predictor = TableRecPredictor()

predictions = predictor([image])

for table in predictions:
    for row in table.rows:
        print(f"Row bbox: {row.bbox}")
        print(f"Is header: {row.is_header}")

    for col in table.cols:
        print(f"Column bbox: {col.bbox}")
        print(f"Is header: {col.is_header}")

    for cell in table.cells:
        print(f"Cell: row={cell.row_id}, col={cell.col_id}")
        print(f"Spans: row={cell.rowspan}, col={cell.colspan}")
        print(f"Text: {cell.text}")
```

### Reading Order Detection

Reading order is included in layout analysis via the `position` attribute.

```python
# Sort layout elements by reading order
elements_sorted = sorted(page.bboxes, key=lambda e: e.position)
```

---

## Output Formats

### OCR Result Structure

```python
# Each text line contains:
{
    "text": str,           # Recognized text content
    "confidence": float,   # Model confidence (0-1)
    "polygon": [           # Clockwise from top-left
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ],
    "bbox": [x1, y1, x2, y2],  # Axis-aligned bounding box
    "chars": [             # Per-character data
        {"char": "H", "bbox": [...], "confidence": 0.99}
    ],
    "words": [             # Per-word data (computed from chars)
        {"text": "Hello", "bbox": [...], "confidence": 0.98}
    ]
}
```

### Layout Labels

Surya can detect 14+ element types [1][HIGH]:

| Label | Description |
|-------|-------------|
| `Caption` | Image/figure captions |
| `Footnote` | Page footnotes |
| `Formula` | Mathematical formulas |
| `List-item` | Bulleted/numbered list items |
| `Page-footer` | Page footer content |
| `Page-header` | Page header content |
| `Picture` | Embedded images |
| `Figure` | Diagrams, charts |
| `Section-header` | Section titles |
| `Table` | Data tables |
| `Form` | Form elements |
| `Table-of-contents` | TOC entries |
| `Handwriting` | Handwritten text regions |
| `Text` | Regular body text |
| `Text-inline-math` | Inline equations |

---

## CLI Commands

### OCR

```bash
# Basic OCR
surya_ocr image.jpg

# With output directory
surya_ocr document.pdf --output_dir ./results

# Save debug images
surya_ocr image.jpg --images

# Page range for PDFs
surya_ocr document.pdf --page_range 1-5

# Task options: ocr_with_boxes (default), ocr_without_boxes, block_without_boxes
surya_ocr image.jpg --task_name ocr_without_boxes
```

### Detection

```bash
surya_detect image.jpg --output_dir ./results --images
```

### Layout Analysis

```bash
surya_layout document.pdf --output_dir ./results --images
```

### Table Recognition

```bash
# With existing layout boxes
surya_table document.pdf --output_dir ./results

# Detect table boxes first
surya_table document.pdf --detect_boxes
```

### LaTeX OCR (Math Formulas)

```bash
surya_latex_ocr equation.png --output_dir ./results
```

---

## Visual Buffet Integration

### Plugin Design Pattern

Following the existing plugin pattern (see `plugins/paddle_ocr/__init__.py`):

```python
class SuryaOCRPlugin(PluginBase):
    """Surya OCR plugin for text detection and recognition."""

    def __init__(self, plugin_dir: Path):
        super().__init__(plugin_dir)
        self._det_predictor = None
        self._rec_predictor = None
        self._foundation_predictor = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="surya_ocr",
            version="1.0.0",
            description="Surya OCR - Document text detection and recognition",
            hardware_reqs={"gpu": False, "min_ram_gb": 8},
            provides_confidence=True,
            recommended_threshold=0.5,
        )

    def is_available(self) -> bool:
        try:
            import surya
            return True
        except ImportError:
            return False

    def tag(self, image_path: Path) -> TagResult:
        # Lazy load models
        # Run detection + recognition
        # Convert to Tag objects with bboxes in metadata
        pass
```

### Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `threshold` | float | 0.5 | Min confidence to return |
| `limit` | int | 100 | Max text lines to return |
| `include_boxes` | bool | True | Include bboxes in metadata |
| `include_layout` | bool | False | Run layout analysis |
| `det_batch_size` | int | auto | Detection batch size |
| `rec_batch_size` | int | auto | Recognition batch size |
| `sort_by` | str | "confidence" | Sort: confidence, position, alphabetical |

### Tag Output Format

Match PaddleOCR plugin format for consistency:

```python
TagResult(
    tags=[
        Tag(label="Detected text line 1", confidence=0.95),
        Tag(label="Detected text line 2", confidence=0.88),
    ],
    model="surya_ocr",
    version="1.0.0",
    inference_time_ms=234.5,
    metadata={
        "total_lines": 15,
        "boxes": [
            {"text": "...", "confidence": 0.95, "bbox": [[x1,y1], ...]},
        ],
        "layout": [  # If include_layout=True
            {"label": "Table", "bbox": [...], "position": 0},
        ]
    }
)
```

---

## Performance Optimization

### Batch Processing

Process multiple images efficiently:

```python
images = [Image.open(p) for p in image_paths]
predictions = rec_predictor(images, det_predictor=det_predictor)
```

### Compilation (PyTorch 2.0+)

Enable for 3-11% speedup after warmup [1]:

```bash
export COMPILE_ALL=true
# Or individual:
export COMPILE_DETECTOR=true
export COMPILE_LAYOUT=true
export COMPILE_TABLE_REC=true
```

### Image Preprocessing

Surya works best with images ≤2048px width. For large images:

```python
from PIL import Image

def preprocess_for_surya(image_path: Path, max_width: int = 2048) -> Image.Image:
    """Resize image if too large."""
    image = Image.open(image_path).convert("RGB")

    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.LANCZOS)

    return image
```

### Memory Management

For processing many images:

```python
import gc
import torch

def process_batch(images, det_predictor, rec_predictor, batch_size=10):
    """Process images in batches to manage memory."""
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        predictions = rec_predictor(batch, det_predictor=det_predictor)
        results.extend(predictions)

        # Clear CUDA cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results
```

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Surya's training methodology and dataset details
- Fine-tuning Surya on custom datasets
- Detailed benchmark comparisons across all OCR engines
- Non-English language-specific performance characteristics

### Known Limitations [1][HIGH]

1. **Document-only**: Will NOT work on photographs, signs, or natural scene text
2. **Printed text focus**: Limited handwriting recognition capability
3. **Image size**: Optimal at ≤2048px width; larger images may need resizing
4. **Advertisements**: May ignore advertising content in documents [1]
5. **Very noisy images**: May struggle with heavily degraded documents

### License Restrictions [1][HIGH]

- **Code**: GPL licensed (copyleft)
- **Model weights**: Modified AI Pubs Open Rail-M license
  - Free for research and personal use
  - Free for startups under $2M funding/revenue
  - Commercial use above threshold requires license

### Unverified Claims

| Claim | Confidence | Note |
|-------|------------|------|
| "Outperforms Tesseract on all metrics" | MEDIUM | Benchmarks show improvement but test conditions vary |
| "90+ language support" | HIGH | Listed in source code, not all independently verified |

---

## Recommendations

### For Visual Buffet Integration

1. **Implement as secondary OCR validator** - Use alongside PaddleOCR for cross-validation
2. **Configure for document detection** - Set appropriate batch sizes for your GPU
3. **Include layout analysis option** - Unique capability not in PaddleOCR
4. **Match PaddleOCR output format** - Ensures consistent tag handling in engine

### Configuration Defaults

```python
DEFAULT_CONFIG = {
    "threshold": 0.5,
    "limit": 100,
    "include_boxes": True,
    "include_layout": False,  # Optional, adds latency
    "det_batch_size": None,   # Auto-detect based on VRAM
    "rec_batch_size": None,   # Auto-detect based on VRAM
    "sort_by": "confidence",
}
```

### Recommended Use Cases

| Use Case | Recommendation |
|----------|----------------|
| Simple text extraction | PaddleOCR (faster) |
| Document with tables | Surya (better layout) |
| Cross-validation | Both engines |
| Natural scene text | PaddleOCR only |
| Complex layouts | Surya with layout=True |

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [GitHub - VikParuchuri/surya](https://github.com/VikParuchuri/surya) | 2024-12 | Primary | API, features, limitations |
| 2 | [PyPI - surya-ocr](https://pypi.org/project/surya-ocr/) | 2024-12 | Primary | Installation, versioning |
| 3 | [GitHub Issue #183](https://github.com/VikParuchuri/surya/issues/183) | 2024 | Secondary | VRAM requirements |
| 4 | [Surya-OCR-Hardware-Benchmarking](https://github.com/Jl16ExA/Surya-OCR-Hardware-Benchmarking) | 2024 | Secondary | Performance optimization |
| 5 | [Datalab.to/surya](https://www.datalab.to/surya) | 2024-12 | Primary | Official documentation |

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
    text: "Surya supports 90+ languages for OCR"
    type: quantitative
    citations: [1]
    confidence: HIGH
    source_quote: "OCR in 90+ languages"

  - id: C002
    text: "Recognition batch size uses ~40MB VRAM per item"
    type: quantitative
    citations: [1]
    confidence: HIGH
    source_quote: "Each batch item will use ~40MB of VRAM"

  - id: C003
    text: "Detection batch size uses ~440MB VRAM per item"
    type: quantitative
    citations: [1]
    confidence: HIGH
    source_quote: "Each batch item will use 440MB of VRAM"

  - id: C004
    text: "Specialized for document OCR, not photos"
    type: factual
    citations: [1]
    confidence: HIGH
    source_quote: "This is specialized for document OCR. It will likely not work on photos"

  - id: C005
    text: "Layout analysis detects 14+ element types"
    type: quantitative
    citations: [1]
    confidence: HIGH
    source_quote: "Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Figure, Section-header, Table, Form, Table-of-contents, Handwriting, Text, Text-inline-math"

  - id: C006
    text: "Detection model based on EfficientViT"
    type: factual
    citations: [1]
    confidence: HIGH
    source_quote: "modified EfficientViT architecture for semantic segmentation"

  - id: C007
    text: "Recognition model based on Donut"
    type: factual
    citations: [1]
    confidence: HIGH
    source_quote: "the recognition model is based on Donut"

  - id: C008
    text: "Compilation speedup 3-11%"
    type: quantitative
    citations: [1]
    confidence: MEDIUM
    source_quote: "Detection +3.3%, Table Recognition +11.5%"
```
