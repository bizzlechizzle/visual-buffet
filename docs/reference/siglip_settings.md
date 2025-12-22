# SigLIP Plugin Settings Reference

Complete reference for SigLIP configuration options in Visual Buffet.

---

## Critical Quirks (READ FIRST)

### 1. Padding is MANDATORY

SigLIP **requires** `padding="max_length"` during text preprocessing. Without it, the model fails silently with incorrect results. The plugin handles this automatically—do not modify.

### 2. SigLIP 2 Requires max_length=64

SigLIP 2 models require `max_length=64` (different from original SigLIP). This is handled automatically but should not be modified.

### 3. Lowercase Text Only

SigLIP was trained on **lowercase text only**. Mixed case degrades accuracy significantly. The plugin automatically lowercases all tag candidates.

### 4. Prompt Template Matters

The template `"This is a photo of {label}."` performs best. The plugin uses this automatically.

### 5. NaFlex Requires Siglip2Model Class

NaFlex variants (`v2-naflex`) require the `Siglip2Model` class from transformers. The regular `SiglipModel` class will not work.

### 6. Sigmoid ≠ Softmax

SigLIP outputs are **independent probabilities** (sigmoid), not a distribution (softmax). This is why thresholds are 0.01-0.10, not 0.5-0.99.

---

## Dependencies

### Required
```bash
pip install transformers>=4.47.0 torch>=2.0.0 pillow>=10.0.0 accelerate>=0.25.0
```

### Optional (for advanced features)
```bash
pip install flash-attn>=2.0.0    # Flash Attention 2 support
pip install bitsandbytes>=0.41.0 # 4-bit/8-bit quantization
```

---

## Quick Reference

| Setting | CLI Flag | Config Key | Default | Range/Options |
|---------|----------|------------|---------|---------------|
| Quality | `--quality` | `plugins.siglip.quality` | `standard` | quick/standard/max |
| Threshold | `--threshold` | `plugins.siglip.threshold` | `0.01` | 0.001-0.10 |
| Limit | `--limit` | `plugins.siglip.max_tags` | `50` | 0-unlimited |
| Variant | `--variant` | `plugins.siglip.variant` | `so400m` | base/large/so400m/giant |
| Batch Size | `--batch-size` | `plugins.siglip.batch_size` | `4` | 1-32 |
| Quantization | `--quantization` | `plugins.siglip.quantization` | `none` | none/8bit/4bit |
| Attention | `--attention` | `plugins.siglip.attention` | `auto` | auto/sdpa/flash_attention_2/eager |
| dtype | `--dtype` | `plugins.siglip.dtype` | `auto` | auto/float16/bfloat16/float32 |

### Batch Size Guidelines

| Available VRAM | Recommended Batch Size |
|----------------|------------------------|
| 4 GB | 1-2 |
| 8 GB | 4-8 |
| 16 GB | 8-16 |
| 24+ GB | 16-32 |

The plugin auto-detects optimal batch size. Override only if needed.

---

## Critical: SigLIP Threshold Values

**SigLIP uses MUCH LOWER threshold values than other models.**

Unlike softmax-based models where probabilities sum to 1.0, SigLIP uses **sigmoid activation** producing independent probabilities per label. A strong match typically scores **0.01-0.10**, not 0.8-0.99.

| Threshold | Effect |
|-----------|--------|
| `0.001` | Very permissive, many tags |
| `0.01` | Balanced **(RECOMMENDED)** |
| `0.05` | More selective |
| `0.10` | Very selective, fewer tags |
| `0.50` | **TOO HIGH** - will return no results! |

---

## Model Variants

| Variant | Parameters | Resolution | VRAM (fp16) | Use Case |
|---------|------------|------------|-------------|----------|
| `base` | 86M | 224×224 | 2 GB | Fast, low VRAM |
| `large` | 303M | 512×512 | 4 GB | High resolution |
| `so400m` | 400M | 384×384 | 6 GB | **RECOMMENDED** |
| `giant` | 1B | 384×384 | 12 GB | Maximum accuracy |

### SigLIP 2 Variants (Experimental)

| Variant | Parameters | Resolution | Notes |
|---------|------------|------------|-------|
| `v2-base` | 86M | 224×224 | Improved semantics, multilingual |
| `v2-so400m` | 400M | 384×384 | V2 balanced |
| `v2-naflex` | 86M | Dynamic | Variable aspect ratio |
| `v2-giant` | 1B | 384×384 | Maximum accuracy |

**CLI:**
```bash
visual-buffet tag image.jpg --plugin siglip --variant so400m
```

**Config:**
```toml
[plugins.siglip]
variant = "so400m"
use_v2 = false  # Set true for SigLIP 2
```

---

## Settings Detail

### quality

Controls how many resolution passes are used for tagging.

| Value | Resolutions | Passes | Speed | Coverage |
|-------|-------------|--------|-------|----------|
| `quick` | 1080px | 1 | Fastest | ~87% |
| `standard` | 480 + 2048px | 2 | Fast | ~92% |
| `high` | 480 + 1080 + 2048px | 3 | Slow | ~96% |
| `max` | 480 + 1080 + 2048 + 4096 + original | 5 | Slowest | 100% |

---

### threshold

Minimum confidence score for returned tags.

**IMPORTANT:** SigLIP confidence scores are typically much lower than other models!

| Value | Effect |
|-------|--------|
| `0.001` | Very permissive |
| `0.01` | Balanced (default) |
| `0.05` | Selective |
| `0.10` | Very selective |

**CLI:**
```bash
visual-buffet tag image.jpg --plugin siglip --threshold 0.02
```

---

### attention

Attention implementation affects speed and memory usage.

| Value | Requirements | Memory Savings | Speed |
|-------|--------------|----------------|-------|
| `auto` | None | Auto-detect | Auto |
| `eager` | None | Baseline | Baseline |
| `sdpa` | PyTorch 2.0+ | ~20-30% | Faster |
| `flash_attention_2` | flash-attn package | ~40-50% | Fastest |

**Config:**
```toml
[plugins.siglip]
attention = "sdpa"
```

---

### dtype

Data type for model weights and inference.

| Value | VRAM Usage | Compatibility |
|-------|------------|---------------|
| `auto` | Optimal | Auto-detect hardware |
| `float32` | 2× | All hardware |
| `float16` | 1× | Most GPUs |
| `bfloat16` | 1× | Ampere+ (RTX 30xx+), Apple Silicon |

---

### quantization

Reduces VRAM usage at slight accuracy cost.

| Value | VRAM Reduction | Accuracy Loss | Requirements |
|-------|----------------|---------------|--------------|
| `none` | 0% | None | - |
| `8bit` | ~50% | Minimal | bitsandbytes |
| `4bit` | ~75% | Small | bitsandbytes |

**Config:**
```toml
[plugins.siglip]
quantization = "4bit"
```

---

### NaFlex Settings (SigLIP 2 only)

For variable aspect ratio support with `v2-naflex` variant.

| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| `max_num_patches` | 256 | 64-512 | Higher = more detail for non-square images |

**Config:**
```toml
[plugins.siglip]
variant = "v2-naflex"
use_v2 = true

[plugins.siglip.naflex]
max_num_patches = 384
```

---

### Custom Vocabulary

SigLIP can use a custom tag vocabulary instead of the built-in one.

| Setting | Description |
|---------|-------------|
| `custom_vocabulary` | Path to text file with one tag per line |
| `max_candidates` | Maximum labels to evaluate (default: 1000) |

**Config:**
```toml
[plugins.siglip]
custom_vocabulary = "/path/to/my_tags.txt"
max_candidates = 500
```

**Tags file format:**
```
dog
cat
outdoor
portrait
landscape
...
```

---

## Hardware Requirements

| Variant | Min VRAM (fp16) | Recommended | With 4-bit |
|---------|-----------------|-------------|------------|
| base | 2 GB | 4 GB | 1 GB |
| large | 4 GB | 8 GB | 2 GB |
| so400m | 6 GB | 8 GB | 2 GB |
| giant | 12 GB | 16 GB | 4 GB |

---

## Configuration Examples

### Minimal Config

```toml
[plugins.siglip]
enabled = true
```

### Maximum Quality

```toml
[plugins.siglip]
enabled = true
quality = "max"
variant = "giant"
threshold = 0.005
limit = 0
attention = "flash_attention_2"
```

### Low VRAM (~4GB)

```toml
[plugins.siglip]
enabled = true
variant = "base"
quantization = "4bit"
attention = "sdpa"
threshold = 0.01
limit = 30
```

### Custom Vocabulary

```toml
[plugins.siglip]
enabled = true
custom_vocabulary = "~/.config/visual-buffet/my_tags.txt"
max_candidates = 500
threshold = 0.02
```

### SigLIP 2 with Variable Aspect Ratio

```toml
[plugins.siglip]
enabled = true
variant = "v2-naflex"
use_v2 = true
threshold = 0.01

[plugins.siglip.naflex]
max_num_patches = 384
```

---

## CLI Examples

```bash
# Basic tagging
visual-buffet tag photo.jpg --plugin siglip

# Maximum quality
visual-buffet tag photo.jpg --plugin siglip --variant giant --quality max

# Low VRAM mode
visual-buffet tag photo.jpg --plugin siglip --variant base --quantization 4bit

# Custom threshold
visual-buffet tag photo.jpg --plugin siglip --threshold 0.02

# With custom vocabulary
visual-buffet tag photo.jpg --plugin siglip --vocabulary ~/my_tags.txt

# Batch processing
visual-buffet tag photos/ --plugin siglip --batch-size 8
```

---

## GUI Settings Location

In the Visual Buffet GUI:

1. **Settings Panel** (gear icon) → **Plugins** → **SigLIP**
2. Available controls:
   - Model variant dropdown
   - Quality dropdown
   - Threshold slider (0.001-0.10 range)
   - Limit input
   - Quantization dropdown
3. Advanced settings:
   - Attention implementation
   - Data type
   - Custom vocabulary path

---

## Technical Notes

### Why Low Thresholds?

SigLIP uses **sigmoid activation** instead of softmax:

```python
# Softmax (CLIP): probabilities sum to 1.0
probs = torch.softmax(logits, dim=-1)
# Result: [0.7, 0.2, 0.05, 0.05] - one dominant class

# Sigmoid (SigLIP): independent probabilities
probs = torch.sigmoid(logits)
# Result: [0.08, 0.06, 0.03, 0.02] - all classes independent
```

This is **ideal for multi-label tagging** where an image can have many valid tags, but requires lower thresholds.

### Text Preprocessing

SigLIP was trained on lowercase text. The plugin automatically:
1. Lowercases all tag candidates
2. Uses the prompt template: "This is a photo of {label}."

Do not modify these behaviors.

---

## Troubleshooting

### No tags returned

- **Lower threshold** to 0.005 or 0.001
- Verify vocabulary contains relevant tags
- Check if model loaded correctly

### All confidence scores are very low

This is expected! SigLIP sigmoid outputs are typically 0.01-0.10 for good matches. This is not a bug.

### Slow inference

- Use `attention = "sdpa"` or `"flash_attention_2"`
- Use `dtype = "float16"`
- Reduce `max_candidates` if using custom vocabulary
- Use smaller variant (base instead of so400m)

### Out of memory

- Use `quantization = "4bit"`
- Use smaller variant
- Reduce `batch_size`

### "padding" warnings

Ensure the plugin uses `padding="max_length"`. This is handled automatically.

---

## Comparison with Other Plugins

| Aspect | SigLIP | RAM++ | Florence-2 |
|--------|--------|-------|------------|
| Confidence | Real (sigmoid) | Real (extracted) | Synthetic |
| Threshold range | 0.001-0.10 | 0.5-0.99 | 0.0-1.0 |
| Custom vocabulary | Yes | No | No |
| Multi-label | Native | Native | Parsed |
| Speed | Slow | Fast | Medium |
| Zero-shot | Yes | No | Limited |
