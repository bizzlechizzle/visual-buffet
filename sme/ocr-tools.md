# Dedicated OCR Tools: A Comprehensive Guide

> **Generated**: 2025-12-22
> **Sources current as of**: December 2025
> **Scope**: Comprehensive
> **Version**: 1.0
> **Audit-Ready**: Yes

---

## Executive Summary / TLDR

Optical Character Recognition (OCR) tools have evolved significantly, with modern deep learning solutions outperforming traditional approaches. **For most use cases, PaddleOCR offers the best balance of accuracy, speed, and language support** [1][HIGH], particularly when GPU acceleration is available. For CPU-only environments, **Tesseract remains viable** but requires careful preprocessing [2][HIGH]. For **handwritten text, TrOCR or LLM-based solutions (GPT-4V, Gemini)** achieve significantly higher accuracy than traditional OCR [3][HIGH].

**Quick Recommendations:**

| Use Case | Recommended Tool | Why |
|----------|------------------|-----|
| High-volume production (GPU) | PaddleOCR | Fastest, 100+ languages, excellent Asian text |
| CPU-only deployment | Tesseract 5 | Mature, 100+ languages, no GPU required |
| Handwriting recognition | TrOCR / GPT-4V | 90%+ accuracy vs 64% traditional [3] |
| Document processing | docTR | PyTorch native, state-of-the-art on documents |
| Easy integration | EasyOCR | Simple API, 80+ languages, good GPU support |
| Multilingual scenes | Surya | 90+ languages, layout analysis included |

---

## Background & Context

OCR converts images of text into machine-readable text. Modern OCR systems use deep learning architectures (CNNs, Transformers) that significantly outperform traditional rule-based approaches [4]. The choice of OCR tool depends on:

- **Text type**: Printed vs. handwritten vs. scene text
- **Languages**: Latin scripts vs. Asian vs. Arabic/RTL
- **Hardware**: GPU available vs. CPU-only
- **Volume**: Single images vs. batch processing
- **Accuracy requirements**: Good enough vs. near-perfect

Key terminology:
- **CER** (Character Error Rate): % of characters incorrectly recognized (lower = better)
- **WER** (Word Error Rate): % of words incorrectly recognized (lower = better)
- **DPI**: Dots per inch - resolution metric (300+ recommended)

---

## Tool Comparison Matrix

### Overall Comparison

| Tool | Languages | GPU Accel | Speed (GPU) | Printed Accuracy | Handwriting | Ease of Use |
|------|-----------|-----------|-------------|------------------|-------------|-------------|
| **PaddleOCR** | 100+ | Yes | Very Fast | Excellent | Poor | Medium |
| **EasyOCR** | 80+ | Yes | Fast | Good | Poor | Easy |
| **Tesseract 5** | 100+ | No | Slow | Good | Poor | Medium |
| **TrOCR** | English | Yes | Medium | Excellent | Excellent | Medium |
| **docTR** | Multi | Yes | Fast | Excellent | Poor | Easy |
| **Surya** | 90+ | Yes | Fast | Good | Poor | Medium |

### Accuracy Benchmarks

| Tool | Printed Text | Scene Text | Handwriting | Notes |
|------|--------------|------------|-------------|-------|
| PaddleOCR v5 | 92-98% [5] | 88-94% | <70% | Best for Asian languages |
| EasyOCR | 85-95% | 80-90% | <65% | Lower WER issues [1] |
| Tesseract 5 | 95-99% [2] | 70-85% | <60% | Needs preprocessing |
| TrOCR-Large | 96.6% F1 [6] | 88-94% | 97% (2.89 CER) | Best handwriting |
| docTR | 90-97% | 85-92% | <65% | Document-focused |
| Surya | 88-95% | 85-90% | <65% | Best layout analysis |

[HIGH] confidence based on multiple benchmark sources agreeing.

---

## PaddleOCR

### Overview

PaddleOCR is a comprehensive OCR toolkit developed by Baidu, supporting 100+ languages with excellent performance on Asian text [7]. It uses a three-stage pipeline: text detection, direction classification, and text recognition.

### Key Strengths

- **Speed**: Several times faster than Tesseract on GPU [1]
- **Asian languages**: Superior Chinese, Japanese, Korean support
- **PP-OCRv5**: Latest version with 13% accuracy improvement [8]
- **Flexible architecture**: Modular components can be replaced
- **Active development**: Regular updates, large community

### Installation

```bash
pip install paddlepaddle paddleocr

# With GPU (CUDA)
pip install paddlepaddle-gpu paddleocr
```

### Basic Usage

```python
from paddleocr import PaddleOCR

# Initialize (downloads models automatically)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Process image
result = ocr.ocr('image.jpg', cls=True)

for line in result[0]:
    bbox, (text, confidence) = line
    print(f"{text} ({confidence:.2%})")
```

### Performance

| Hardware | Throughput | VRAM |
|----------|------------|------|
| RTX 3090 | ~50 img/s | 2-4 GB |
| RTX 4090 | ~80 img/s | 2-4 GB |
| CPU (8 core) | ~2 img/s | N/A |

### Limitations

- Complex installation with PaddlePaddle framework
- Less accurate on cursive handwriting [1]
- Struggles with very small text (<8pt)

---

## EasyOCR

### Overview

EasyOCR is a Python-based OCR library built on PyTorch, designed for ease of use with 80+ language support [9]. It provides a simple API that works out of the box.

### Key Strengths

- **Simplicity**: 3 lines of code to get started
- **Multi-language**: Read multiple compatible languages simultaneously
- **GPU flexibility**: Easy GPU/CPU switching
- **Active maintenance**: Regular updates

### Installation

```bash
pip install easyocr

# Models download automatically on first use
```

### Basic Usage

```python
import easyocr

# Initialize reader (downloads models first time)
reader = easyocr.Reader(['en'])  # Add more languages: ['en', 'fr', 'de']

# Process image
result = reader.readtext('image.jpg')

for bbox, text, confidence in result:
    print(f"{text} ({confidence:.2%})")

# GPU options
reader = easyocr.Reader(['en'], gpu=True)      # Use GPU (default)
reader = easyocr.Reader(['en'], gpu=False)     # CPU only
reader = easyocr.Reader(['en'], gpu='cuda:1')  # Specific GPU
```

### Language Compatibility

Languages must share script to be combined:
- **Latin**: English + French + German + Spanish (compatible)
- **Chinese**: Simplified + Traditional (compatible)
- **Mixed**: English is compatible with ALL languages

### Performance

| Hardware | Throughput | VRAM |
|----------|------------|------|
| RTX 3090 | ~30 img/s | 2-3 GB |
| CPU | ~1 img/s | N/A |

### Limitations

- Higher WER than CER suggests word segmentation issues [1]
- Slower than PaddleOCR on GPU
- Memory-intensive for multiple language models

---

## Tesseract 5

### Overview

Tesseract is the most established open-source OCR engine, originally developed by HP and now maintained by Google [10]. Version 5 added LSTM-based recognition and new binarization methods.

### Key Strengths

- **Maturity**: Decades of development, extensive documentation
- **Language support**: 100+ languages with trainable models
- **CPU-optimized**: No GPU required
- **Customizable**: PSM modes, character whitelists, custom training

### Installation

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows: Download from GitHub releases

# Python wrapper
pip install pytesseract
```

### Basic Usage

```python
import pytesseract
from PIL import Image

# Simple extraction
text = pytesseract.image_to_string(Image.open('image.jpg'))

# With configuration
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(Image.open('image.jpg'), config=custom_config)

# Get detailed data
data = pytesseract.image_to_data(Image.open('image.jpg'), output_type=pytesseract.Output.DICT)
```

### Page Segmentation Modes (PSM)

| Mode | Description | Use Case |
|------|-------------|----------|
| 3 | Auto page segmentation | Default, full pages |
| 6 | Single uniform block | Column of text |
| 7 | Single text line | One line only |
| 8 | Single word | Isolated words |
| 10 | Single character | Individual chars |
| 11 | Sparse text | Scattered text |

### OCR Engine Modes (OEM)

| Mode | Description |
|------|-------------|
| 0 | Legacy engine only |
| 1 | Neural nets LSTM only |
| 2 | Legacy + LSTM |
| 3 | Default (auto) |

### Critical: Preprocessing Required

Tesseract accuracy depends heavily on image quality [2][11]:

```python
import cv2
import pytesseract

def preprocess_for_tesseract(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)

    return pytesseract.image_to_string(denoised)
```

### Resolution Requirements

- **Minimum**: 300 DPI for standard text [11]
- **Small text** (<8pt): 400-600 DPI recommended
- **Above 600 DPI**: No accuracy improvement, just larger files

### Limitations

- No GPU acceleration
- Significantly slower than deep learning alternatives
- Poor on complex layouts, tables, handwriting [2]
- Requires careful preprocessing

---

## TrOCR

### Overview

TrOCR (Transformer-based OCR) is Microsoft's state-of-the-art model using Vision Transformer (ViT) encoder and text Transformer decoder [6]. It excels at both printed and handwritten text recognition.

### Key Strengths

- **Handwriting**: Best-in-class accuracy (2.89% CER on IAM) [6]
- **Printed text**: 96.6% F1 on SROIE benchmark
- **Pre-trained models**: Multiple sizes available
- **Hugging Face integration**: Easy to use via transformers library

### Model Variants

| Model | Parameters | Printed F1 | Handwriting CER |
|-------|------------|------------|-----------------|
| TrOCR-Small | 62M | 95.86% | 4.22% |
| TrOCR-Base | 334M | 96.34% | 3.42% |
| TrOCR-Large | 558M | 96.60% | 2.89% |

### Installation

```bash
pip install transformers torch pillow
```

### Basic Usage

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# For handwriting, use: 'microsoft/trocr-base-handwritten'

def recognize_text(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

text = recognize_text('image.jpg')
```

### Performance

| Model | Inference Time (GPU) | VRAM |
|-------|---------------------|------|
| TrOCR-Small | ~50ms | 1 GB |
| TrOCR-Base | ~100ms | 2 GB |
| TrOCR-Large | ~200ms | 4 GB |

### Limitations

- **Line-level only**: Requires text line cropping (no full-page OCR)
- **English-focused**: Limited multilingual support
- **No detection**: Only recognition, needs separate detection step

### Recommended Pipeline

```python
# Use with text detection (e.g., CRAFT, EAST, or PaddleOCR detection)
from paddleocr import PaddleOCR

# Step 1: Detect text regions
detector = PaddleOCR(use_angle_cls=True, lang='en', rec=False)
boxes = detector.ocr('image.jpg', rec=False)

# Step 2: Crop and recognize each region with TrOCR
for box in boxes[0]:
    cropped = crop_image(image, box)
    text = recognize_text(cropped)
```

---

## docTR

### Overview

docTR (Document Text Recognition) is a seamless, high-performing library by Mindee, now part of the PyTorch ecosystem [12]. It's optimized for document processing with end-to-end pipelines.

### Key Strengths

- **Document-focused**: Optimized for forms, invoices, receipts
- **PyTorch/TensorFlow**: Dual backend support
- **End-to-end**: Detection + recognition in one pipeline
- **Production-ready**: Used in commercial document processing

### Installation

```bash
# PyTorch backend (recommended)
pip install "python-doctr[torch]"

# TensorFlow backend
pip install "python-doctr[tf]"
```

### Basic Usage

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load model (downloads automatically)
model = ocr_predictor(pretrained=True)

# Process document
doc = DocumentFile.from_images("image.jpg")
result = model(doc)

# Extract text
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            text = " ".join([word.value for word in line.words])
            print(text)

# Export to JSON
result.export()
```

### Detection Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| db_resnet50 | Medium | High | Best accuracy |
| fast_base | Fast | Good | Default, balanced |
| linknet_resnet18 | Very Fast | Medium | Real-time |

### Recognition Models

| Model | Speed | Accuracy |
|-------|-------|----------|
| crnn_vgg16_bn | Fast | Good |
| master | Medium | High |
| vitstr_small | Fast | Good |

### Performance

| Hardware | Throughput |
|----------|------------|
| GPU (T4) | ~0.27 pages/s [12] |
| CPU | ~0.1 pages/s |

### Limitations

- Slower than PaddleOCR/EasyOCR
- Fewer language models available
- Document-centric (less suited for scene text)

---

## Surya

### Overview

Surya is a modern document OCR toolkit supporting 90+ languages with built-in layout analysis and reading order detection [13]. It's optimized for document understanding.

### Key Strengths

- **Layout analysis**: Detects tables, figures, headers automatically
- **Reading order**: Correct text flow for multi-column documents
- **90+ languages**: Including low-resource languages
- **Open weights**: Fully open-source models

### Installation

```bash
pip install surya-ocr
```

### Basic Usage

```python
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from PIL import Image

# Load models
det_model = segformer.load_model()
det_processor = segformer.load_processor()
rec_model = load_model()
rec_processor = load_processor()

# Process image
image = Image.open("image.jpg")
results = run_ocr(
    [image],
    [["en"]],
    det_model,
    det_processor,
    rec_model,
    rec_processor
)

for result in results:
    for line in result.text_lines:
        print(line.text)
```

### Performance

| Hardware | Layout Analysis | Full OCR |
|----------|-----------------|----------|
| A10 GPU | 0.4s/page | ~1s/page |
| CPU | 2-3s/page | ~5s/page |

### Benchmark Results

- **Layout detection**: 88% mean accuracy on Publaynet [13]
- **Sinhala**: 2.61% WER (best among tested tools) [13]
- **Multilingual**: Comparable to Google Cloud Vision

### Limitations

- Optimized for printed text, struggles with handwriting [13]
- Complex backgrounds reduce accuracy
- Newer project with smaller community

---

## Preprocessing Best Practices

Image preprocessing significantly improves OCR accuracy across all tools [11][14].

### Recommended Pipeline

```python
import cv2
import numpy as np

def preprocess_for_ocr(image_path):
    """Standard preprocessing pipeline for OCR."""
    img = cv2.imread(image_path)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Resize if needed (target 300 DPI equivalent)
    # Assuming 72 DPI input, scale by 300/72 ≈ 4.17
    height, width = gray.shape
    if width < 1000:  # Heuristic for low-res images
        scale = 2.0
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Binarization (adaptive thresholding)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 5. Deskew (optional, for scanned documents)
    # coords = np.column_stack(np.where(binary > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # ... rotate if needed

    return binary
```

### Resolution Guidelines [11]

| Text Size | Recommended DPI |
|-----------|-----------------|
| Normal (>8pt) | 300 DPI |
| Small (<8pt) | 400-600 DPI |
| Very small (<6pt) | 600 DPI |

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Low contrast | Histogram equalization, CLAHE |
| Skewed text | Deskew using Hough transform |
| Noise/artifacts | Gaussian blur + thresholding |
| Uneven lighting | Adaptive thresholding |
| Small text | Upscale to 300+ DPI equivalent |

---

## Handwriting Recognition

Traditional OCR tools achieve only ~64% accuracy on handwriting [3]. For handwriting, use specialized solutions:

### Recommended Approaches

| Approach | Accuracy | Notes |
|----------|----------|-------|
| TrOCR-Large | ~97% | Best open-source |
| GPT-4V/Gemini | ~90% | Excellent but API-based |
| AWS Textract | ~85% | Good for forms |

### TrOCR for Handwriting

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Use handwriting-specific model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
```

### LLM-Based OCR (Best Accuracy)

```python
import anthropic
import base64

def ocr_with_claude(image_path):
    """Use Claude for handwriting recognition."""
    client = anthropic.Anthropic()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                {"type": "text", "text": "Transcribe all text in this image exactly as written."}
            ]
        }]
    )
    return response.content[0].text
```

---

## Deployment Options

### Local Deployment (Self-Hosted)

| Tool | Docker | ONNX Export | Edge Support |
|------|--------|-------------|--------------|
| PaddleOCR | Yes | Yes (Paddle2ONNX) | PaddleLite |
| EasyOCR | Yes | Limited | No |
| Tesseract | Yes | N/A | Native |
| TrOCR | Yes | Yes (Optimum) | Limited |
| docTR | Yes | Yes (OnnxTR) | Yes |

### GPU Memory Requirements

| Tool | Minimum VRAM | Recommended |
|------|--------------|-------------|
| PaddleOCR | 2 GB | 4 GB |
| EasyOCR | 2 GB | 4 GB |
| TrOCR-Base | 2 GB | 4 GB |
| docTR | 2 GB | 4 GB |
| Surya | 4 GB | 8 GB |

### Batch Processing

```python
# PaddleOCR batch processing
from paddleocr import PaddleOCR
import os

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def batch_ocr(image_folder):
    results = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_folder, filename)
            result = ocr.ocr(path, cls=True)
            results[filename] = result
    return results
```

---

## Integration with Visual-Buffet

For the visual-buffet project, OCR can be integrated as a plugin or combined with existing Florence-2:

### Option 1: Florence-2 OCR (Already Available)

```python
# Use existing Florence-2 plugin with OCR task
from transformers import AutoProcessor, AutoModelForCausalLM

def florence_ocr(image):
    inputs = processor(text="<OCR>", images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    return processor.post_process_generation(text, task="<OCR>", image_size=image.size)
```

### Option 2: Dedicated OCR Plugin

For production OCR needs, a dedicated PaddleOCR or docTR plugin would provide:
- Faster inference than Florence-2 for OCR-only tasks
- Better multilingual support
- Optimized text detection

---

## Analysis & Implications

### Key Findings

1. **No single best tool**: Choice depends on hardware, languages, and text type
2. **GPU acceleration critical**: 10-50x speedup over CPU for deep learning tools
3. **Preprocessing matters**: Can improve Tesseract accuracy by 10-20%
4. **Handwriting requires specialized tools**: TrOCR or LLMs, not traditional OCR
5. **PaddleOCR leads**: Best overall for production with GPU [HIGH]

### Decision Framework

```
Need OCR?
│
├─► Handwritten text?
│   ├─► Yes → TrOCR or GPT-4V/Claude
│   └─► No → Continue
│
├─► GPU available?
│   ├─► Yes → PaddleOCR (best) or EasyOCR (easiest)
│   └─► No → Tesseract 5 with preprocessing
│
├─► Asian languages primary?
│   ├─► Yes → PaddleOCR (specialized)
│   └─► No → Any major tool works
│
├─► Document processing (forms, invoices)?
│   ├─► Yes → docTR or PaddleOCR PP-Structure
│   └─► No → General-purpose tool
│
└─► Need layout analysis?
    ├─► Yes → Surya or PaddleOCR PP-Structure
    └─► No → Recognition-only tool
```

---

## Limitations & Uncertainties

### What This Document Does NOT Cover

- Commercial/cloud OCR services (AWS Textract, Google Vision, Azure)
- Fine-tuning and custom training procedures
- Specific accuracy numbers for all 100+ languages
- Real-time video OCR

### Unverified Claims

- Exact throughput numbers vary significantly by hardware and image complexity
- Some benchmark comparisons use different datasets, limiting direct comparison

### Source Conflicts

- PaddleOCR vs Tesseract accuracy: Some sources favor each; resolved by noting Tesseract better on clean documents, PaddleOCR better with GPU and Asian text [1][5]

### Knowledge Gaps

- Long-term maintenance commitment for newer tools (Surya)
- Detailed multilingual benchmarks for all language pairs

### Recency Limitations

- PaddleOCR v5 and PP-OCRv5 are recent; some benchmarks may reference older versions
- LLM-based OCR rapidly improving; accuracy figures may be outdated quickly

---

## Recommendations

1. **For visual-buffet project**: Use Florence-2's built-in OCR for integrated tagging+OCR, or add PaddleOCR plugin for dedicated high-speed OCR

2. **For new projects with GPU**: Start with PaddleOCR; easiest path to high accuracy

3. **For CPU-only/embedded**: Tesseract 5 with proper preprocessing pipeline

4. **For handwriting**: TrOCR-large or LLM API (Claude/GPT-4V)

5. **Always preprocess**: Regardless of tool, proper image preprocessing improves results

---

## Source Appendix

| # | Source | Date | Type | Used For |
|---|--------|------|------|----------|
| 1 | [OCR Comparison: Tesseract vs EasyOCR vs PaddleOCR](https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66) | 2023 | Secondary | Tool comparison, accuracy |
| 2 | [Tesseract Documentation - Improve Quality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html) | 2024 | Primary | Tesseract best practices |
| 3 | [Handwriting Recognition Benchmark: LLMs vs OCRs](https://research.aimultiple.com/handwriting-recognition/) | 2024 | Secondary | Handwriting accuracy |
| 4 | [8 Top Open-Source OCR Models Compared](https://modal.com/blog/8-top-open-source-ocr-models-compared) | 2024 | Secondary | Model overview |
| 5 | [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) | 2024 | Primary | PaddleOCR features, languages |
| 6 | [TrOCR - Microsoft Research](https://www.microsoft.com/en-us/research/publication/trocr-transformer-based-optical-character-recognition-with-pre-trained-models/) | 2023 | Primary | TrOCR architecture, benchmarks |
| 7 | [PaddleOCR Documentation](http://www.paddleocr.ai/v2.9/en/ppocr/blog/multi_languages.html) | 2024 | Primary | Language support |
| 8 | [PaddleOCR 3.0 Technical Report](https://arxiv.org/html/2507.05595v1) | 2024 | Primary | PP-OCRv5 improvements |
| 9 | [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) | 2024 | Primary | EasyOCR features, usage |
| 10 | [Pytesseract Guide](https://nanonets.com/blog/ocr-with-tesseract/) | 2024 | Secondary | Tesseract usage |
| 11 | [Image Preprocessing for OCR](https://medium.com/technovators/survey-on-image-preprocessing-techniques-to-improve-ocr-accuracy-616ddb931b76) | 2024 | Secondary | Preprocessing techniques |
| 12 | [docTR GitHub](https://github.com/mindee/doctr) | 2024 | Primary | docTR features, benchmarks |
| 13 | [Surya OCR GitHub](https://github.com/datalab-to/surya) | 2024 | Primary | Surya features, benchmarks |
| 14 | [Chandra OCR Benchmarks](https://skywork.ai/blog/sheets/chandra-ocr-benchmark/) | 2024 | Secondary | Speed benchmarks |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-22 | Initial version |
