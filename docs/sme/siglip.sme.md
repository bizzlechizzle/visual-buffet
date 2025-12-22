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

## 5. Hardware Requirements

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
