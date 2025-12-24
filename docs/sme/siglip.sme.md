# SigLIP Plugin — Subject Matter Expert Document

## Overview

**SigLIP** (Sigmoid Loss for Language Image Pre-Training) is Google's vision-language model that improves upon CLIP by using a pairwise sigmoid loss instead of softmax. This eliminates the need for global batch normalization, enabling better performance at smaller batch sizes and more efficient memory usage.

**SigLIP 2** (released February 2025) extends the original with decoder-based pretraining, self-distillation, and masked prediction for improved semantic understanding and dense feature extraction.

---

## 1. Default vs Configurable Settings

SigLIP has **both defaults and configurable settings**. The model ships with sensible defaults but offers significant configurability for optimization.

### Critical Defaults (DO NOT CHANGE)
| Setting | Default | Why Fixed |
|---------|---------|-----------|
| `padding` | `"max_length"` | Model was trained this way — **required** |
| `max_length` | `64` (SigLIP 2) | Training configuration |
| Prompt template | `"This is a photo of {label}."` | Matches training distribution |
| Text preprocessing | **lowercase** | Model trained on lowercased text |

---

## 2. Configurable Settings for Visual Buffet Plugin

### A. Model Selection (Practical: YES)

| Model | Parameters | Resolution | Use Case |
|-------|------------|------------|----------|
| `google/siglip-base-patch16-224` | 86M | 224x224 | Fast, low VRAM |
| `google/siglip-so400m-patch14-384` | 400M | 384x384 | **Recommended default** |
| `google/siglip2-base-patch16-224` | 86M | 224x224 | Multilingual, improved (experimental) |
| `google/siglip2-so400m-patch14-384` | 400M | 384x384 | V2 experimental |
| `google/siglip2-base-patch16-naflex` | 86M | Dynamic | Variable aspect ratio |
| `google/siglip2-large-patch16-512` | 303M | 512x512 | High resolution needs |
| `google/siglip2-giant-opt-patch16-384` | 1B | 384x384 | Maximum accuracy |

**Note**: SigLIP v1 models are more stable and recommended for production use.

**Practical for settings**: YES — Let users choose model variant based on their hardware.

### B. Attention Implementation (Practical: YES)

```python
attn_implementation = "sdpa"  # Options: "eager", "sdpa", "flash_attention_2"
```

| Option | Requirement | Memory Savings | Speed |
|--------|-------------|----------------|-------|
| `"eager"` | None | Baseline | Baseline |
| `"sdpa"` | PyTorch 2.0+ | ~20-30% | Faster |
| `"flash_attention_2"` | flash-attn package | ~40-50% | Fastest |

**Practical for settings**: YES — Auto-detect capability, let user override.

### C. Quantization (Practical: YES for low VRAM)

```python
# 4-bit quantization example
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

| Mode | VRAM Reduction | Accuracy Loss |
|------|----------------|---------------|
| `load_in_8bit` | ~50% | Minimal |
| `load_in_4bit` | ~75% | Small |

**Practical for settings**: YES — Enable for users with <8GB VRAM.

### D. NaFlex Settings (SigLIP 2 only, Practical: CONDITIONAL)

```python
max_num_patches = 256  # Default, range: 64-512+
```

- Lower values (64-128): Faster, less detail
- Default (256): Balanced
- Higher values (384-512): Better for documents/detailed images

**Practical for settings**: YES for NaFlex variants — helps with non-square images.

### E. Batch Size (Practical: YES)

Not a model config, but critical for your plugin:

| VRAM | Recommended Batch |
|------|-------------------|
| 4GB | 1-2 |
| 8GB | 4-8 |
| 16GB | 8-16 |
| 24GB+ | 16-32 |

**Practical for settings**: YES — Auto-detect based on hardware cache.

### F. Device Placement (Practical: AUTO)

```python
device_map = "auto"  # or "cuda", "mps", "cpu"
```

**Practical for settings**: Auto-detect, allow override.

### G. Confidence Threshold (Practical: YES)

SigLIP outputs probabilities via `torch.sigmoid()`. Unlike softmax, these are independent per-label.

**IMPORTANT**: SigLIP's sigmoid outputs are typically much lower than softmax-based models. A good match might only have 1-10% confidence.

```python
threshold = 0.01  # Recommended range: 0.005-0.05
# Note: 0.5 is TOO HIGH for SigLIP - you'll get no results
```

**Practical for settings**: YES — Core filtering parameter. Use much lower values than other models.

### H. Data Type (Practical: YES)

```python
dtype = torch.float16  # Options: float32, float16, bfloat16
```

| Type | VRAM | Compatibility |
|------|------|---------------|
| `float32` | 2x | All hardware |
| `float16` | 1x | Most GPUs |
| `bfloat16` | 1x | Ampere+ (RTX 30xx+), Apple Silicon |

**Practical for settings**: Auto-detect, allow override.

---

## 3. Recommended Plugin Configuration Schema

```toml
[plugins.siglip]
enabled = true

# Model selection
model = "google/siglip2-so400m-patch14-384"  # Default choice
# Alternatives:
#   "google/siglip2-base-patch16-224" (fast/low VRAM)
#   "google/siglip2-base-patch16-naflex" (variable aspect)
#   "google/siglip2-giant-opt-patch16-384" (max accuracy)

# Performance settings
batch_size = 4                    # Auto-adjusted by hardware detection
attention = "auto"                # "auto", "sdpa", "flash_attention_2", "eager"
dtype = "auto"                    # "auto", "float16", "bfloat16", "float32"
quantization = "none"             # "none", "8bit", "4bit"

# Output settings
confidence_threshold = 0.01      # 0.005-0.05 recommended (sigmoid outputs are low)
max_tags = 50                     # Limit returned tags

# NaFlex-specific (only for -naflex models)
max_num_patches = 256             # 64-512, higher = more detail
```

---

## 4. Implementation Best Practices

### Text Preprocessing
```python
# CRITICAL: SigLIP was trained on lowercase text
labels = [label.lower() for label in candidate_labels]
texts = [f"This is a photo of {label}." for label in labels]
```

### Processing Pipeline
```python
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained(
    "google/siglip2-so400m-patch14-384",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

# Process
inputs = processor(
    text=texts,
    images=image,
    padding="max_length",  # REQUIRED
    max_length=64,         # For SigLIP 2
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

# Get probabilities (sigmoid, not softmax!)
probs = torch.sigmoid(outputs.logits_per_image)
```

### Multi-Label Output (vs CLIP)
Unlike CLIP's softmax (probabilities sum to 1), SigLIP's sigmoid outputs **independent probabilities** per label. This is ideal for multi-label tagging where an image can have many valid tags.

---

## 5. Prompt Engineering for Better Accuracy

### The Default Isn't Always Best

The default prompt `"This is a photo of {label}."` works for general use, but **OpenAI's CLIP research shows that prompt ensembling can improve accuracy by 3.5-5%**.

### OpenAI's 80-Template Ensemble (CLIP)

OpenAI found that using 80 different prompt templates and averaging their embeddings significantly improves ImageNet classification. Key templates include:

| Category | Example Templates |
|----------|-------------------|
| **Quality variations** | "a bad photo of a {}", "a good photo of a {}", "a blurry photo of a {}" |
| **Size variations** | "a photo of a big {}", "a photo of a small {}" |
| **Artistic renderings** | "a painting of a {}", "a sketch of a {}", "a sculpture of a {}" |
| **Context variations** | "a {} in a video game", "the origami {}", "the toy {}" |
| **Technical views** | "a close-up photo of a {}", "a cropped photo of a {}" |
| **Lighting conditions** | "a bright photo of a {}", "a dark photo of a {}" |

### Applying to SigLIP

SigLIP shares CLIP's architecture and benefits from similar prompt strategies:

```python
# Prompt ensembling for SigLIP
PROMPT_TEMPLATES = [
    "This is a photo of {}.",           # Default
    "a photo of a {}.",                 # Simple
    "a photo of the {}.",               # Definite article
    "a good photo of a {}.",            # Quality
    "a bad photo of a {}.",             # Quality (catches poor images)
    "a close-up photo of a {}.",        # View
    "a photo of a big {}.",             # Size
    "a photo of a small {}.",           # Size
]

def get_ensemble_confidence(model, processor, image, label):
    """Get averaged confidence across multiple prompts."""
    total_conf = 0.0
    label_lower = label.lower()

    for template in PROMPT_TEMPLATES:
        text = template.format(label_lower)
        inputs = processor(text=[text], images=image,
                          padding="max_length", return_tensors="pt")
        outputs = model(**inputs)
        prob = torch.sigmoid(outputs.logits_per_image).item()
        total_conf += prob

    return total_conf / len(PROMPT_TEMPLATES)
```

### Trade-offs

| Approach | Speed | Accuracy | When to Use |
|----------|-------|----------|-------------|
| Single prompt | 1x | Baseline | Real-time tagging |
| 3-prompt ensemble | 3x | +1-2% | Balanced |
| 7-prompt ensemble | 7x | +2-3% | Quality-focused |
| 80-prompt ensemble | 80x | +3-5% | Benchmarking only |

**Recommendation**: For archive applications where accuracy matters more than speed, use a 7-prompt ensemble. OpenAI's sequential forward selection found that 7 templates captured most of the gain:

```python
SELECTED_7_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",             # Handles poor quality
    "a photo of a small {}.",           # Scale variation
    "a photo of a large {}.",           # Scale variation
    "a origami {}.",                    # Abstract rendering
    "a {} in a video game.",            # Abstract rendering
    "a close-up photo of a {}.",        # View variation
]
```

### Domain-Specific Templates

For specific image types, specialized templates outperform generic ones:

| Image Domain | Recommended Templates |
|--------------|----------------------|
| **Satellite/Aerial** | "satellite imagery of {}", "aerial photo of a {}" |
| **Documents** | "a scan of a {}", "a photo of a document showing {}" |
| **Food** | "a photo of a plate of {}", "a photo of {} food" |
| **Art** | "a painting of a {}", "artwork depicting a {}" |
| **Indoor/Scenes** | "a photo of a room with {}", "interior with {}" |

### Sources

- [OpenAI CLIP Prompts](https://github.com/openai/CLIP/blob/main/data/prompts.md)
- [ImageNet Prompt Engineering Notebook](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
- [Meta-Prompting for Zero-shot Recognition (arXiv:2403.11755)](https://arxiv.org/abs/2403.11755)

---

## 6. Hardware Requirements

| Model Variant | Min VRAM (fp16) | Recommended VRAM |
|---------------|-----------------|------------------|
| Base (86M) | 2GB | 4GB |
| Large (303M) | 4GB | 8GB |
| So400m (400M) | 6GB | 8GB |
| Giant (1B) | 12GB | 16GB |

With 4-bit quantization, reduce requirements by ~75%.

---

## 6. Quirks & Gotchas

1. **Padding is mandatory**: Always use `padding="max_length"` — model fails silently otherwise
2. **Lowercase text**: Model trained on lowercase; mixed case degrades accuracy
3. **Prompt template matters**: "This is a photo of X" performs best
4. **SigLIP 2 requires max_length=64**: Different from original SigLIP
5. **NaFlex models need Siglip2Model class**: Regular SiglipModel won't work
6. **Sigmoid outputs**: Don't normalize — values are independent probabilities

---

## 7. Output Contract

Per Visual Buffet specification, this plugin returns:

```json
{
  "tags": [
    {"label": "string", "confidence": 0.0}
  ]
}
```

---

## 8. Dependencies

```toml
[dependencies]
transformers = ">=4.47.0"
torch = ">=2.0.0"
pillow = ">=10.0.0"

[optional-dependencies]
flash-attn = ">=2.0.0"      # For flash_attention_2
bitsandbytes = ">=0.41.0"   # For quantization
accelerate = ">=0.25.0"     # For device_map="auto"
```

---

## Sources

- [HuggingFace SigLIP Documentation](https://huggingface.co/docs/transformers/en/model_doc/siglip)
- [HuggingFace SigLIP2 Documentation](https://huggingface.co/docs/transformers/en/model_doc/siglip2)
- [Original Paper: Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343)
- [SigLIP 2 Paper (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [SigLIP 2 Blog Post](https://huggingface.co/blog/siglip2)
- [Roboflow SigLIP Guide](https://roboflow.com/model/siglip)
- [OpenVINO Zero-Shot Classification Guide](https://docs.openvino.ai/2024/notebooks/siglip-zero-shot-image-classification-with-output.html)
