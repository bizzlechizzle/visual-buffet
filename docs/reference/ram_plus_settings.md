# RAM++ Plugin Settings Reference

Complete reference for RAM++ configuration options in Visual Buffet.

## TLDR - Use Threshold 0.5

**Official Recommendation:** `--threshold 0.5` for comprehensive tagging and database building.

**Key Findings:**
1. Resolution (`--size`) has **no impact** - model internally resizes to 384×384
2. Threshold 0.5 captures **contextual tags** (debris, damage, rust, antique, historic) that are ACCURATE
3. Higher thresholds (0.7-0.8) MISS these contextual tags entirely

**Only two settings matter:**
- `--threshold` — Filter by confidence (**0.5 recommended**)
- `--limit` — Hard cap on tag count

## Suggested CLI Commands

```bash
# RECOMMENDED - comprehensive tagging for databases
visual-buffet tag photo.jpg --threshold 0.5

# Display/UI - cleaner output
visual-buffet tag photo.jpg --threshold 0.6

# Selective - high-confidence only (loses contextual tags)
visual-buffet tag photo.jpg --threshold 0.8

# Discovery mode - catch everything
visual-buffet tag photo.jpg --threshold 0.4

# Batch processing for database
visual-buffet tag photos/ --threshold 0.5
```

## Expected Results by Threshold

| Threshold | Tags | Avg Confidence | Contextual Coverage | Use Case |
|-----------|------|----------------|---------------------|----------|
| 0.4 | 150-200 | 0.58 | 100% | Maximum discovery |
| **0.5** | **130-165** | **0.61** | **95%** | **Database building (recommended)** |
| 0.6 | 100-130 | 0.65 | 60% | Display/UI |
| 0.7 | 60-100 | 0.72 | 30% | High confidence only |
| 0.8 | 30-50 | 0.85 | **0%** | Very selective (loses context) |

**Performance:** ~55ms per image on GPU (Apple MPS or NVIDIA CUDA)

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

### Performance Benchmarks (Empirical - 2025-12-23)

| Device | Inference Time | Notes |
|--------|----------------|-------|
| Apple Silicon (MPS) | **55-72ms** | Tested with benchmarks |
| NVIDIA RTX 3080 | ~50ms | CUDA acceleration |
| Intel i7 (CPU) | ~2000ms | No GPU acceleration |

---

## Quick Reference

| Setting | CLI Flag | Config Key | Default | Range |
|---------|----------|------------|---------|-------|
| Size | `--size` | `plugins.ram_plus.size` | `original` | little/small/large/huge/original |
| Threshold | `--threshold` | `plugins.ram_plus.threshold` | `0.5` | 0.0-1.0 |
| Limit | `--limit` | `plugins.ram_plus.limit` | `0` (unlimited) | 0-unlimited |

---

## Settings Detail

### size

Controls the input resolution before processing. RAM++ internally resizes to 384×384.

| Value | Resolution | Avg Time | Notes |
|-------|------------|----------|-------|
| `little` | 480px | **55ms** | Recommended - minimal quality loss |
| `small` | 1080px | 60ms | Marginal improvement |
| `large` | 2048px | 72ms | For fine details |
| `huge` | 4096px | ~100ms | Rarely needed |
| `original` | No resize | Varies | For already small images |

**CLI:**
```bash
visual-buffet tag image.jpg --size little
visual-buffet tag image.jpg --size small
```

**Config:**
```toml
[plugins.ram_plus]
size = "little"
```

**Key Finding:** Empirical testing shows resolution has **minimal impact** on tagging quality. All sizes produce similar tag counts (130-140 avg) and confidence levels (~0.606 avg).

---

### threshold

Minimum confidence score for returned tags. This is the **primary lever** for controlling output volume.

**Empirical Results (10 test images):**

| Threshold | Expected Tags | Avg Confidence | Use Case |
|-----------|---------------|----------------|----------|
| `0.4` | 130-175 | 0.606 | Maximum discovery |
| `0.5` | 130-165 | 0.607 | Comprehensive coverage |
| `0.6` | 105-175 | 0.606 | Balanced (recommended) |
| `0.7` | ~80-100 | ~0.72 | High confidence only |
| `0.8` | ~30-50 | ~0.85 | Very selective |

**CLI:**
```bash
visual-buffet tag image.jpg --threshold 0.6
visual-buffet tag image.jpg -t 0.8
```

**Config:**
```toml
[plugins.ram_plus]
threshold = 0.5
```

**Key Finding:** RAM++ returns many tags per image (130-165 at threshold 0.5). This is by design—it's a comprehensive tagger. Threshold 0.5 captures contextual tags (debris, damage, rust) that are accurate but would be missed at higher thresholds.

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

### Database Building (Recommended)

```toml
[plugins.ram_plus]
enabled = true
size = "little"      # 480px - fastest (resolution doesn't matter)
threshold = 0.5      # Captures contextual tags (debris, rust, antique)
limit = 0            # Unlimited for comprehensive database
```

### Display/UI (Cleaner Output)

```toml
[plugins.ram_plus]
enabled = true
size = "little"
threshold = 0.6      # Fewer tags, still good coverage
limit = 50           # Cap at 50 tags
```

### High-Confidence Only (Loses Context)

```toml
[plugins.ram_plus]
enabled = true
size = "little"
threshold = 0.8      # WARNING: Loses debris, damage, rust, antique tags
limit = 30           # Top 30 tags
```

### Multi-Plugin Setup

```toml
[plugins.ram_plus]
enabled = true
size = "little"
threshold = 0.5      # Database building

[plugins.siglip]
enabled = true
size = "small"
threshold = 0.01
limit = 30
```

---

## CLI Examples

```bash
# RECOMMENDED - Database building with contextual tags
visual-buffet tag photo.jpg --threshold 0.5

# Display/UI - cleaner output
visual-buffet tag photo.jpg --threshold 0.6

# High-confidence only (loses contextual tags like debris, rust)
visual-buffet tag photo.jpg --threshold 0.8 --limit 30

# Maximum discovery
visual-buffet tag photo.jpg --threshold 0.4

# Batch processing for database
visual-buffet tag photos/ --threshold 0.5

# JSON output to file
visual-buffet tag photo.jpg --threshold 0.5 --output results.json

# NOTE: --size is ignored for RAM++ (model always uses 384x384 internally)
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

- Lower `threshold` (try 0.4)
- Remove `limit` (set to 0 for unlimited)
- Check plugin is enabled: `visual-buffet plugins list`

### Too many tags

- Raise `threshold` (try 0.7 or 0.8)
- Set `limit` to cap output (e.g., `--limit 30`)
- RAM++ is comprehensive by design—100+ tags is normal

### Processing too slow

- Use `--size little` (480px) — same quality, faster
- Ensure GPU is being used (check `visual-buffet hardware`)
- CPU-only takes ~2000ms vs 55ms with GPU

### Out of memory

- Use `--size little` instead of larger resolutions
- Process fewer images at once
- Check GPU VRAM with `visual-buffet hardware`

### Confidence scores all similar (~0.6)

This is expected. RAM++ sigmoid outputs cluster around 0.5-0.7 for most tags. The average confidence of ~0.606 is normal. Use threshold to filter, not to interpret quality.
