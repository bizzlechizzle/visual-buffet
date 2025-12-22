# RAM++ Plugin Settings Reference

Complete reference for RAM++ configuration options in Visual Buffet.

---

## Requirements

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| GPU VRAM | - (CPU works) | 4 GB |
| CPU | x86_64 or ARM64 | 4+ cores |
| Disk | 3 GB | 3 GB |

### Dependencies

```bash
pip install torch>=2.0 torchvision>=0.15 timm>=0.9
pip install scipy>=1.10.0 fairscale>=0.4.0 einops>=0.7.0
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

**WARNING:** The RAM package does not properly declare all dependencies. You must manually install `scipy`, `fairscale`, and `einops` or inference will fail silently.

### Performance Benchmarks

| Device | Inference Time | Cost Efficiency |
|--------|----------------|-----------------|
| NVIDIA RTX 3080 | ~100ms | Excellent |
| NVIDIA RTX 2080 | ~150ms | Best $/image |
| Apple M1 Pro | ~150ms | Good |
| Intel i7 (CPU) | ~2000ms | Slow |

---

## Quick Reference

| Setting | CLI Flag | Config Key | Default | Range |
|---------|----------|------------|---------|-------|
| Quality | `--quality` | `plugins.ram_plus.quality` | `standard` | quick/standard/high/max |
| Threshold | `--threshold` | `plugins.ram_plus.threshold` | `0.0` | 0.0-1.0 |
| Limit | `--limit` | `plugins.ram_plus.limit` | `50` | 0-unlimited |
| Batch Size | `--batch-size` | `plugins.ram_plus.batch_size` | `4` | 1-32 |
| Chinese | `--chinese` | `plugins.ram_plus.include_chinese` | `false` | true/false |

---

## Settings Detail

### quality

Controls how many resolution passes are used for tagging. More passes = more complete tags but slower.

| Value | Resolutions | Passes | Speed | Coverage |
|-------|-------------|--------|-------|----------|
| `quick` | 1080px | 1 | Fastest | ~87% |
| `standard` | 480 + 2048px | 2 | Fast | ~92% |
| `high` | 480 + 1080 + 2048px | 3 | Slow | ~96% |
| `max` | 480 + 1080 + 2048 + 4096 + original | 5 | Slowest | 100% |

**CLI:**
```bash
visual-buffet tag image.jpg --quality high
visual-buffet tag image.jpg -q quick
```

**Config:**
```toml
[plugins.ram_plus]
quality = "high"
```

**GUI:** Quality dropdown in plugin settings panel.

---

### threshold

Minimum confidence score for returned tags. Tags below this score are filtered out.

**Important:** RAM++ already applies per-class calibrated thresholds internally (averaging ~0.68). This setting is an *additional* filter. Since returned tags already passed internal thresholding, confidence scores typically range 0.68-0.99.

| Value | Effect |
|-------|--------|
| `0.0` | Return all tags that passed model's internal threshold |
| `0.7` | Only high-confidence tags |
| `0.9` | Only very high-confidence tags |

**CLI:**
```bash
visual-buffet tag image.jpg --threshold 0.8
visual-buffet tag image.jpg -t 0.8
```

**Config:**
```toml
[plugins.ram_plus]
threshold = 0.8
```

**GUI:** Threshold slider (0.0-1.0) in plugin settings.

---

### limit

Maximum number of tags to return per image.

| Value | Effect |
|-------|--------|
| `0` | Unlimited (return all matching tags) |
| `10` | Top 10 tags only |
| `50` | Top 50 tags (default) |
| `100` | Top 100 tags |

**CLI:**
```bash
visual-buffet tag image.jpg --limit 20
visual-buffet tag image.jpg -l 20
```

**Config:**
```toml
[plugins.ram_plus]
limit = 20
```

**GUI:** Limit input field in plugin settings.

---

### batch_size

Number of images to process simultaneously. Higher values use more VRAM but process faster.

| VRAM | Recommended |
|------|-------------|
| 4 GB | 1-2 |
| 8 GB | 4-8 |
| 16 GB | 8-16 |
| 24+ GB | 16-32 |

**Note:** Visual Buffet auto-detects hardware and adjusts batch size. Override only if needed.

**CLI:**
```bash
visual-buffet tag folder/ --batch-size 8
```

**Config:**
```toml
[plugins.ram_plus]
batch_size = 8
```

**GUI:** Batch size input in advanced settings.

---

### include_chinese

Include Chinese translations of tags in output.

**CLI:**
```bash
visual-buffet tag image.jpg --chinese
```

**Config:**
```toml
[plugins.ram_plus]
include_chinese = true
```

**Output format when enabled:**
```json
{
  "tags": [
    {"label": "dog", "label_zh": "狗", "confidence": 0.95},
    {"label": "outdoor", "label_zh": "户外", "confidence": 0.87}
  ]
}
```

---

### sort_by

How to sort returned tags.

| Value | Description |
|-------|-------------|
| `confidence` | Highest confidence first (default) |
| `alphabetical` | A-Z by label |
| `model_order` | Order returned by model (internal relevance) |

**Config:**
```toml
[plugins.ram_plus]
sort_by = "confidence"
```

---

### include_thresholds

Include the per-class threshold used for each tag (debugging).

**Config:**
```toml
[plugins.ram_plus]
include_thresholds = true
```

**Output when enabled:**
```json
{
  "tags": [
    {"label": "dog", "confidence": 0.95, "threshold": 0.68},
    {"label": "husky", "confidence": 0.72, "threshold": 0.71}
  ]
}
```

---

## Model Settings (Read-Only)

These settings are fixed by the model architecture. Documented for reference.

| Setting | Value | Notes |
|---------|-------|-------|
| `variant` | `swin_l` | Swin Transformer Large backbone |
| `image_size` | `384` | Input resolution (fixed) |
| `parameters` | ~440M | Model parameters |
| `base_threshold` | `0.68` | Average per-class threshold |
| `num_tags` | `4,585` | After synonym reduction (original: ~6,500) |
| `model_size` | `~2.9 GB` | Download size (`ram_plus_swin_large_14m.pth`) |

### Tag Vocabulary

RAM++ recognizes **4,585 semantic categories** after synonym reduction from an original ~6,500 tags. Categories include:

- **Objects**: camera, furniture, vehicle, animal, food...
- **Scenes**: beach, forest, city, indoor, outdoor...
- **Actions**: running, swimming, cooking, reading...
- **Attributes**: colorful, vintage, 3D, abstract...
- **Concepts**: celebration, accident, adaptation...

### Per-Class Thresholds

RAM++ uses individually calibrated thresholds per tag (loaded from `ram_tag_list_threshold.txt`). These are optimized based on class frequency and difficulty. The `0.68` base threshold is an average—individual tags may have thresholds from 0.5 to 0.85.

---

## Configuration File Examples

### Minimal Config

```toml
# ~/.config/visual-buffet/config.toml

[plugins.ram_plus]
enabled = true
```

### Speed-Optimized

```toml
[plugins.ram_plus]
enabled = true
quality = "quick"
threshold = 0.8
limit = 20
batch_size = 8
```

### Quality-Optimized

```toml
[plugins.ram_plus]
enabled = true
quality = "max"
threshold = 0.0
limit = 0
include_chinese = true
```

### Multi-Plugin Setup

```toml
[plugins.ram_plus]
enabled = true
quality = "standard"
threshold = 0.0
limit = 50

[plugins.siglip]
enabled = true
quality = "high"
threshold = 0.01
limit = 30
```

---

## CLI Examples

```bash
# Basic tagging
visual-buffet tag photo.jpg

# High quality, all tags
visual-buffet tag photo.jpg --quality max --limit 0

# Quick tagging with filter
visual-buffet tag photo.jpg --quality quick --threshold 0.85 --limit 10

# Batch processing
visual-buffet tag photos/ --quality standard --batch-size 8

# With Chinese output
visual-buffet tag photo.jpg --chinese

# JSON output to file
visual-buffet tag photo.jpg --output results.json

# Specific plugin only
visual-buffet tag photo.jpg --plugin ram_plus
```

---

## GUI Settings Location

In the Visual Buffet GUI:

1. **Settings Panel** (gear icon) → **Plugins** → **RAM++**
2. Available controls:
   - Quality dropdown
   - Threshold slider
   - Limit input
   - Chinese toggle
3. Changes apply to next tagging operation
4. Use "Save as Default" to persist to config file

---

## Troubleshooting

### Tags not appearing

- Lower `threshold` (try 0.0)
- Increase `limit` (try 0 for unlimited)
- Try `quality = "max"` for maximum coverage

### Processing too slow

- Use `quality = "quick"`
- Reduce `batch_size` if running out of VRAM
- Ensure GPU is being used (check hardware detection)

### Out of memory

- Reduce `batch_size`
- Use `quality = "quick"` (fewer resolution passes)
- Process smaller batches of images

### Confidence scores all high (>0.9)

This is expected. RAM++ only returns tags that exceed per-class thresholds, so returned tags are already high-confidence. Use `include_thresholds = true` to see the internal thresholds.
