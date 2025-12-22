# PaddleOCR Plugin Settings Reference

Complete reference for PaddleOCR configuration options in Visual Buffet.

---

## Requirements

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| GPU VRAM | - (CPU works) | 2 GB |
| CPU | x86_64 or ARM64 | 4+ cores |
| Disk | 500 MB | 1 GB |

### Dependencies

```bash
# CPU version (default)
pip install paddlepaddle>=2.5.0 paddleocr>=2.7.0

# GPU version (CUDA 11.x)
pip install paddlepaddle-gpu>=2.5.0 paddleocr>=2.7.0

# GPU version (CUDA 12.x)
pip install paddlepaddle-gpu==2.6.0.post120 paddleocr>=2.7.0
```

### Performance Benchmarks

| Device | Inference Time | Throughput |
|--------|----------------|------------|
| NVIDIA RTX 3090 | ~20ms | ~50 img/s |
| NVIDIA RTX 3080 | ~25ms | ~40 img/s |
| Apple M1 Pro | ~80ms | ~12 img/s |
| Intel i7 (CPU) | ~500ms | ~2 img/s |

---

## Quick Reference

| Setting | CLI Flag | Config Key | Default | Range |
|---------|----------|------------|---------|-------|
| Language | `--language` | `plugins.paddle_ocr.language` | `en` | en/ch/fr/etc |
| Threshold | `--threshold` | `plugins.paddle_ocr.threshold` | `0.5` | 0.0-1.0 |
| Limit | `--limit` | `plugins.paddle_ocr.limit` | `100` | 0-unlimited |
| Line Orientation | - | `plugins.paddle_ocr.use_textline_orientation` | `true` | true/false |

**Note:** PaddleOCR v3 handles GPU/CPU selection automatically based on available hardware.

---

## Settings Detail

### language

OCR language model to use. PaddleOCR supports 100+ languages.

| Value | Language |
|-------|----------|
| `en` | English |
| `ch` | Chinese (Simplified + Traditional) |
| `french` | French |
| `german` | German |
| `korean` | Korean |
| `japan` | Japanese |
| `arabic` | Arabic |
| `cyrillic` | Russian/Cyrillic |
| `devanagari` | Hindi/Devanagari |

See full list: [PaddleOCR Multi-Language Support](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md)

**CLI:**
```bash
visual-buffet tag image.jpg --plugin paddle_ocr --language ch
```

**Config:**
```toml
[plugins.paddle_ocr]
language = "ch"
```

**Note:** Changing language requires model reload. Different languages use different model files.

---

### threshold

Minimum recognition confidence for returned text. Lines below this score are filtered out.

| Value | Effect |
|-------|--------|
| `0.0` | Return all detected text (may include noise) |
| `0.5` | Balanced - filters uncertain results |
| `0.8` | Only high-confidence text |
| `0.95` | Only very high-confidence text |

**CLI:**
```bash
visual-buffet tag image.jpg --plugin paddle_ocr --threshold 0.8
```

**Config:**
```toml
[plugins.paddle_ocr]
threshold = 0.8
```

---

### limit

Maximum number of text lines to return per image.

| Value | Effect |
|-------|--------|
| `0` | Unlimited (return all detected text) |
| `50` | First 50 lines (after sorting) |
| `100` | First 100 lines (default) |

**Config:**
```toml
[plugins.paddle_ocr]
limit = 50
```

---

### use_textline_orientation

Enable text line orientation detection for rotated text.

| Value | Effect |
|-------|--------|
| `true` | Detect and correct rotated text (180 degree rotation) |
| `false` | Assume text is upright (faster) |

**Config:**
```toml
[plugins.paddle_ocr]
use_textline_orientation = true
```

**When to disable:** If all your images have upright text, disabling orientation detection provides a small speed boost.

---

### GPU Acceleration

PaddleOCR v3 automatically detects and uses GPU if available. No configuration needed.

**To use GPU acceleration:**
```bash
# Uninstall CPU version
pip uninstall paddlepaddle

# Install GPU version (CUDA 11.x)
pip install paddlepaddle-gpu

# Or for CUDA 12.x
pip install paddlepaddle-gpu==2.6.0.post120
```

---

## Detection Settings

These settings control how text regions are detected in images.

### text_det_thresh

Text region detection threshold. Lower values detect more potential text regions.

| Value | Effect |
|-------|--------|
| `0.1` | Very sensitive - may detect noise as text |
| `0.3` | Balanced (default) |
| `0.5` | Conservative - only clear text regions |

**Config:**
```toml
[plugins.paddle_ocr]
text_det_thresh = 0.3
```

---

### text_det_box_thresh

Box confidence threshold. Minimum score for keeping a detected text box.

| Value | Effect |
|-------|--------|
| `0.3` | Keep more uncertain boxes |
| `0.5` | Balanced (default) |
| `0.7` | Only high-confidence boxes |

**Config:**
```toml
[plugins.paddle_ocr]
text_det_box_thresh = 0.5
```

---

### text_det_unclip_ratio

Text region expansion ratio. Controls how much to expand detected text regions.

| Value | Effect |
|-------|--------|
| `1.0` | No expansion |
| `1.5` | Moderate expansion (default) |
| `2.0` | Large expansion - helps with touching characters |

**Config:**
```toml
[plugins.paddle_ocr]
text_det_unclip_ratio = 1.5
```

---

## Recognition Settings

These settings control how detected text is recognized.

### text_rec_score_thresh

Recognition confidence threshold. Completely discard results below this threshold (before any processing).

| Value | Effect |
|-------|--------|
| `0.3` | Keep most results |
| `0.5` | Drop low-quality results (default) |
| `0.7` | Only keep confident results |

**Config:**
```toml
[plugins.paddle_ocr]
text_rec_score_thresh = 0.5
```

---

## Output Settings

### sort_by

How to sort detected text lines in output.

| Value | Effect |
|-------|--------|
| `confidence` | Highest confidence first (default) |
| `position` | Reading order (top-to-bottom, left-to-right) |
| `alphabetical` | Alphabetical by text content |

**Config:**
```toml
[plugins.paddle_ocr]
sort_by = "position"
```

---

### include_boxes

Whether to include bounding box coordinates in metadata.

| Value | Effect |
|-------|--------|
| `true` | Include 4-point polygon coordinates (default) |
| `false` | Text and confidence only |

**Config:**
```toml
[plugins.paddle_ocr]
include_boxes = true
```

**Metadata format when enabled:**
```json
{
  "language": "en",
  "total_lines": 3,
  "boxes": [
    {
      "text": "Hello World",
      "confidence": 0.95,
      "bbox": [[10, 20], [100, 20], [100, 40], [10, 40]]
    }
  ]
}
```

---

## Example Configurations

### High-Speed Processing

```toml
[plugins.paddle_ocr]
language = "en"
use_textline_orientation = false
threshold = 0.5
limit = 50
include_boxes = false
```

### Maximum Accuracy

```toml
[plugins.paddle_ocr]
language = "en"
use_textline_orientation = true
threshold = 0.3
text_det_thresh = 0.2
text_det_box_thresh = 0.4
limit = 0
```

### Document Processing

```toml
[plugins.paddle_ocr]
language = "en"
use_textline_orientation = true
sort_by = "position"
include_boxes = true
threshold = 0.4
limit = 0
```

### Chinese Text Recognition

```toml
[plugins.paddle_ocr]
language = "ch"
use_textline_orientation = true
threshold = 0.5
```

---

## Troubleshooting

### "PaddleOCR not found"

```bash
pip install paddlepaddle paddleocr
```

### Slow performance on GPU system

Ensure you have the GPU version:
```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

### Poor accuracy on rotated text

Enable text line orientation detection:
```toml
[plugins.paddle_ocr]
use_textline_orientation = true
```

### Missing text in images

Lower detection thresholds:
```toml
[plugins.paddle_ocr]
text_det_thresh = 0.2
text_det_box_thresh = 0.3
threshold = 0.3
```

### Too much noise/false detections

Increase thresholds:
```toml
[plugins.paddle_ocr]
text_det_thresh = 0.5
threshold = 0.7
```

### Unsupported image format

PaddleOCR v3 only supports: jpg, jpeg, png, bmp, pdf

For webp/tiff images, the plugin automatically converts them to jpg before processing.
