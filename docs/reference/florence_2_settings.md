# Florence-2 Plugin Settings Reference

Complete reference for Florence-2 configuration options in Visual Buffet.

---

## Critical Warnings

### Apple Silicon (Mac) Users

Florence-2 with `float16` on MPS produces **empty or garbled output** and runs 40x slower. The plugin auto-detects this and forces `float32`, but if you experience issues:

```toml
[plugins.florence_2]
variant = "large-no-flash"  # Uses community-patched version without flash_attn
torch_dtype = "float32"      # Required for MPS
```

### 4K Token Limit

Florence-2 has a **4,096 token context window** that limits:
- High object count scenes (100+ objects)
- Very detailed captions
- Complex OCR documents

**Symptoms**: Truncated output, missing objects, incomplete text

**Workaround**: For complex images, use `<DETAILED_CAPTION>` instead of `<MORE_DETAILED_CAPTION>`, or tile large images.

### Transformers Version

**CRITICAL**: Use transformers `4.40.0 - 4.49.x` only. Version 4.50+ breaks Florence-2.

```bash
pip install "transformers>=4.40.0,<4.50.0"
```

---

## Quick Reference

| Setting | CLI Flag | Config Key | Default | Range/Options |
|---------|----------|------------|---------|---------------|
| Quality | `--quality` | `plugins.florence_2.quality` | `standard` | quick/standard/high/max |
| Threshold | `--threshold` | `plugins.florence_2.threshold` | `0.0` | 0.0-1.0 |
| Limit | `--limit` | `plugins.florence_2.limit` | `50` | 0-unlimited |
| Variant | `--variant` | `plugins.florence_2.variant` | `large-ft` | base/large/base-ft/large-ft/large-no-flash |
| Task | `--task` | `plugins.florence_2.task_prompt` | `<MORE_DETAILED_CAPTION>` | See task table |
| Beams | `--beams` | `plugins.florence_2.num_beams` | `3` | 1-5 |
| Device | `--device` | `plugins.florence_2.device` | `auto` | auto/cuda/mps/cpu |
| dtype | `--dtype` | `plugins.florence_2.torch_dtype` | `auto` | auto/float16/float32 |

### Hardware Requirements

| Variant | Min RAM | Min VRAM | Download Size |
|---------|---------|----------|---------------|
| base / base-ft | 4 GB | 2 GB | ~500 MB |
| large / large-ft | 8 GB | 4 GB | ~1.5 GB |

### Performance Benchmarks

| Device | Variant | Inference Time |
|--------|---------|----------------|
| NVIDIA RTX 3080 | large-ft | ~100ms |
| NVIDIA RTX 3060 | large-ft | ~150ms |
| Apple M1 Pro | large-no-flash | ~300ms |
| Intel i7 (CPU) | large-ft | ~3000ms |
| Intel i7 (CPU) | base-ft | ~1500ms |

---

## Model Variants

Florence-2 offers multiple model sizes. Choose based on your hardware and quality needs.

| Variant | Parameters | Size | Speed | Quality | Use Case |
|---------|------------|------|-------|---------|----------|
| `base` | 0.23B | ~0.5 GB | Fastest | Good | Low VRAM, quick tagging |
| `large` | 0.77B | ~1.5 GB | Slower | Better | Standard use |
| `base-ft` | 0.23B | ~0.5 GB | Fastest | Better | Fine-tuned, general tasks |
| `large-ft` | 0.77B | ~1.5 GB | Slower | Best | **RECOMMENDED** |
| `large-no-flash` | 0.77B | ~1.5 GB | Slower | Best | Mac/MPS compatible |

**CLI:**
```bash
visual-buffet tag image.jpg --plugin florence_2 --variant large-ft
```

**Config:**
```toml
[plugins.florence_2]
variant = "large-ft"
```

---

## Task Prompts

The task prompt controls what type of output Florence-2 generates. This is the most important setting for tagging quality.

| Task | Prompt | Tags | Best For |
|------|--------|------|----------|
| **More Detailed Caption** | `<MORE_DETAILED_CAPTION>` | 25-50 | Maximum tags **(DEFAULT)** |
| Detailed Caption | `<DETAILED_CAPTION>` | 15-25 | Balanced output |
| Dense Region Caption | `<DENSE_REGION_CAPTION>` | 15-30 | Spatial/region tags |
| Caption | `<CAPTION>` | 5-10 | Brief description |
| Object Detection | `<OD>` | 2-5 | Object labels only **(LIMITED!)** |
| OCR | `<OCR>` | varies | Text extraction |

**WARNING:** The `<OD>` task has a very limited vocabulary and typically returns only 2-5 object labels. For comprehensive tagging, use `<MORE_DETAILED_CAPTION>`.

**CLI:**
```bash
visual-buffet tag image.jpg --plugin florence_2 --task "<DETAILED_CAPTION>"
```

**Config:**
```toml
[plugins.florence_2]
task_prompt = "<DETAILED_CAPTION>"
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

**CLI:**
```bash
visual-buffet tag image.jpg --plugin florence_2 --quality high
```

---

### threshold

Minimum confidence score for returned tags.

Florence-2 uses synthetic confidence scores derived from:
- Token generation probabilities
- Tag position in parsed output

| Value | Effect |
|-------|--------|
| `0.0` | Return all parsed tags (default) |
| `0.5` | Only medium-high confidence tags |
| `0.8` | Only high confidence tags |

**CLI:**
```bash
visual-buffet tag image.jpg --plugin florence_2 --threshold 0.5
```

---

### limit

Maximum number of tags to return per image.

| Value | Effect |
|-------|--------|
| `0` | Unlimited |
| `50` | Top 50 tags (default) |

---

### num_beams

Beam search width for generation. Higher = better quality but slower.

| Value | Speed | Quality |
|-------|-------|---------|
| `1` | Fastest (greedy) | Good |
| `3` | Balanced | Better (default) |
| `5` | Slowest | Best |

**CLI:**
```bash
visual-buffet tag image.jpg --plugin florence_2 --beams 5
```

---

### max_new_tokens

Maximum tokens to generate. Affects output length and detail.

| Value | Use Case |
|-------|----------|
| `256` | Quick, brief output |
| `1024` | Standard (default) |
| `2048` | Very detailed descriptions |
| `4096` | Maximum detail (may hit token limit) |

**Note:** Florence-2 has a 4K token limit. Very complex images may be truncated.

---

### Generation Settings (Advanced)

These settings control the text generation behavior.

| Setting | Default | Description |
|---------|---------|-------------|
| `do_sample` | `false` | Enable random sampling |
| `temperature` | `1.0` | Sampling randomness (only if do_sample=true) |
| `top_p` | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | `1.0` | Penalize repeated tokens |

**Config:**
```toml
[plugins.florence_2.generation]
do_sample = true
temperature = 0.7
top_p = 0.9
```

---

### Output Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `sort_by` | `confidence` | Sort order: confidence/alphabetical/position |
| `include_bboxes` | `false` | Include bounding boxes (OD/DENSE tasks) |
| `include_raw_caption` | `false` | Include original caption text |

---

## Hardware Requirements

| Variant | Min VRAM | Recommended VRAM | CPU Inference |
|---------|----------|------------------|---------------|
| base / base-ft | 2 GB | 4 GB | ~1500ms |
| large / large-ft | 4 GB | 8 GB | ~3000ms |

### dtype Selection

| Hardware | Recommended dtype |
|----------|-------------------|
| CUDA GPU | `float16` (fastest) |
| Apple Silicon (MPS) | `float32` (required) |
| CPU | `float32` |

**IMPORTANT:** Using float16 on Apple Silicon causes empty output and 40x slowdown. The plugin auto-detects this.

---

## Configuration Examples

### Minimal Config

```toml
[plugins.florence_2]
enabled = true
```

### Maximum Quality

```toml
[plugins.florence_2]
enabled = true
quality = "max"
variant = "large-ft"
task_prompt = "<MORE_DETAILED_CAPTION>"
num_beams = 5
max_new_tokens = 2048
limit = 0
```

### Fast/Low VRAM

```toml
[plugins.florence_2]
enabled = true
quality = "quick"
variant = "base-ft"
task_prompt = "<DETAILED_CAPTION>"
num_beams = 1
max_new_tokens = 512
limit = 30
```

### Region-Specific Tagging

```toml
[plugins.florence_2]
enabled = true
task_prompt = "<DENSE_REGION_CAPTION>"
include_bboxes = true
```

### Text Extraction (OCR)

```toml
[plugins.florence_2]
enabled = true
task_prompt = "<OCR>"
```

---

## CLI Examples

```bash
# Basic tagging
visual-buffet tag photo.jpg --plugin florence_2

# Maximum quality
visual-buffet tag photo.jpg --plugin florence_2 --quality max --beams 5

# Fast tagging
visual-buffet tag photo.jpg --plugin florence_2 --quality quick --variant base-ft

# Specific task
visual-buffet tag photo.jpg --plugin florence_2 --task "<DENSE_REGION_CAPTION>"

# OCR mode
visual-buffet tag document.jpg --plugin florence_2 --task "<OCR>"

# Batch processing
visual-buffet tag photos/ --plugin florence_2 --batch-size 4
```

---

## GUI Settings Location

In the Visual Buffet GUI:

1. **Settings Panel** (gear icon) → **Plugins** → **Florence-2**
2. Available controls:
   - Model variant dropdown
   - Task prompt dropdown
   - Quality dropdown
   - Threshold slider
   - Limit input
   - Beams slider
3. Advanced settings expandable section

---

## Troubleshooting

### Empty or few tags

- Use `<MORE_DETAILED_CAPTION>` instead of `<OD>`
- Increase `max_new_tokens`
- Lower `threshold` to 0.0

### Slow inference

- Use `variant = "base-ft"`
- Reduce `num_beams` to 1
- Use `quality = "quick"`

### Out of memory

- Use `variant = "base"` or `base-ft`
- Reduce `batch_size`
- Ensure not running other GPU processes

### Empty output on Mac

- Use `large-no-flash` variant
- Plugin should auto-detect and use float32

### "trust_remote_code" error

Ensure transformers version is 4.40.0-4.49.x (not 4.50+)

---

## Comparison with Other Plugins

| Aspect | Florence-2 | RAM++ | SigLIP |
|--------|------------|-------|--------|
| Confidence | Synthetic | Real (extracted) | Real (sigmoid) |
| Tag source | Caption parsing | Fixed vocabulary | Custom vocabulary |
| OCR | Yes | No | No |
| Bounding boxes | Yes | No | No |
| Typical tags | 25-50 | 10-30 | 20-50 |
| Speed | Medium | Fast | Slow |
