# Florence-2 Subject Matter Expert Guide

## TLDR - Key Differences from RAM++

**Critical Differences:**

| Feature | Florence-2 | RAM++ |
|---------|------------|-------|
| **Configuration** | TASK-BASED | THRESHOLD-BASED |
| **Confidence Scores** | **NO** - tags have no confidence values | YES - per-tag confidence |
| **Output Format** | Parsed from captions/descriptions | Direct tag list |
| **Compound Phrases** | Built-in slugification (`white_house`, `abandoned_building`) | Single-word tags only |

**Florence-2 does NOT provide confidence scores.** All tags are equal weight. Use RAM++ (threshold 0.5) if you need confidence-based filtering.

## Empirical Benchmark Results (2024-12-23)

All 31 possible task combinations were tested on 10 diverse images (abandoned interiors, exteriors, vehicles, industrial). Results ranked by ground truth coverage.

### Top 3 Recommendations for Archival/Database Building

#### 🥇 #1 BEST VALUE: `<DETAILED_CAPTION>` + `<MORE_DETAILED_CAPTION>` + `<DENSE_REGION_CAPTION>`

| Metric | Value |
|--------|-------|
| **Must-Have Coverage** | **76.1%** |
| **False Positives** | 6 total (across 10 images) |
| **Avg Tags/Image** | 77 |
| **Avg Time** | 2915ms |

**Why:** Highest coverage. Skips `<CAPTION>` (redundant) and `<OD>` (useless). Combines the three most productive tasks.

```bash
# Run all three tasks and combine results
visual-buffet tag photo.jpg --plugin florence_2 --task "<DETAILED_CAPTION>"
visual-buffet tag photo.jpg --plugin florence_2 --task "<MORE_DETAILED_CAPTION>"
visual-buffet tag photo.jpg --plugin florence_2 --task "<DENSE_REGION_CAPTION>"
```

---

#### 🥈 #2 BEST EFFICIENCY: `<MORE_DETAILED_CAPTION>` + `<DENSE_REGION_CAPTION>`

| Metric | Value |
|--------|-------|
| **Must-Have Coverage** | **73.4%** |
| **False Positives** | **3 total** (lowest!) |
| **Avg Tags/Image** | 65 |
| **Avg Time** | **2141ms** |

**Why:** Only 2.7% less coverage than #1, but **HALF the false positives** and 27% faster. Best balance of accuracy and speed.

```bash
# RECOMMENDED for most use cases
visual-buffet tag photo.jpg --plugin florence_2 --task "<MORE_DETAILED_CAPTION>"
visual-buffet tag photo.jpg --plugin florence_2 --task "<DENSE_REGION_CAPTION>"
```

---

#### 🥉 #3 SINGLE TASK CHAMPION: `<MORE_DETAILED_CAPTION>` only

| Metric | Value |
|--------|-------|
| **Must-Have Coverage** | **70.5%** |
| **False Positives** | **1 total** (best!) |
| **Avg Tags/Image** | 46 |
| **Avg Time** | **1212ms** (fastest) |

**Why:** If speed matters or you want minimal false positives, this single task achieves 70% coverage with virtually no hallucinations.

```bash
# Fast and accurate
visual-buffet tag photo.jpg --plugin florence_2 --task "<MORE_DETAILED_CAPTION>"
```

---

### Key Findings from Benchmark

1. **`<OD>` is WORTHLESS for tagging** - Only 14.9% coverage, 3 tags avg. Adding it to any combination provides 0% improvement.

2. **More tasks ≠ better results** - Diminishing returns after 3 tasks. The 5-task "all" profile has same 76.1% coverage as 3-task #1.

3. **False positive pattern** - Florence-2 occasionally hallucinates "person" in empty abandoned scenes. Simpler profiles have fewer hallucinations.

4. **`<CAPTION>` is redundant** - Its content is subsumed by `<DETAILED_CAPTION>`. Skip it.

### Full Benchmark Rankings

| Rank | Combination | Must% | FP | Tags | Time |
|------|-------------|-------|-----|------|------|
| 1 | DET+MOR+DEN | 76.1% | 6 | 77 | 2915ms |
| 2 | CAP+DET+MOR+DEN | 76.1% | 6 | 79 | 3405ms |
| 3 | DET+MOR+OD+DEN | 76.1% | 6 | 78 | 3655ms |
| 4 | ALL (5 tasks) | 76.1% | 6 | 80 | 4133ms |
| 5 | **MOR+DEN** | **73.4%** | **3** | 65 | 2141ms |
| 6 | CAP+MOR+DEN | 73.4% | 3 | 68 | 2583ms |
| 7 | MOR+OD+DEN | 73.4% | 3 | 66 | 2838ms |
| 13 | **MOR** | **70.5%** | **1** | 46 | 1212ms |
| 23 | DET | 56.6% | 5 | 22 | 772ms |
| 30 | CAP | 28.0% | 0 | 7 | 443ms |
| 31 | OD | 14.9% | 0 | 3 | 620ms |

*DET=DETAILED_CAPTION, MOR=MORE_DETAILED_CAPTION, DEN=DENSE_REGION_CAPTION, CAP=CAPTION, OD=Object Detection*

## Built-in Slugification

Florence-2 automatically extracts compound phrases as slugified tags:

```
Caption: "A white house with an abandoned building in the background"
↓
Tags: ["white_house", "abandoned_building", "white", "house", "abandoned", "building", "background"]
```

Compound phrases (bigrams of meaningful words) are joined with underscores and appear BEFORE individual words. This preserves semantic relationships like "swimming_pool" vs just "swimming" and "pool".

## Suggested CLI Commands

```bash
# 🥇 BEST - Maximum coverage (76.1%), 3 tasks
visual-buffet tag photo.jpg --plugin florence_2 --task "<DETAILED_CAPTION>"
visual-buffet tag photo.jpg --plugin florence_2 --task "<MORE_DETAILED_CAPTION>"
visual-buffet tag photo.jpg --plugin florence_2 --task "<DENSE_REGION_CAPTION>"

# 🥈 RECOMMENDED - Best efficiency (73.4%), lowest false positives
visual-buffet tag photo.jpg --plugin florence_2 --task "<MORE_DETAILED_CAPTION>"
visual-buffet tag photo.jpg --plugin florence_2 --task "<DENSE_REGION_CAPTION>"

# 🥉 FAST - Single task champion (70.5%), minimal hallucinations
visual-buffet tag photo.jpg --plugin florence_2 --task "<MORE_DETAILED_CAPTION>"

# Quick preview only
visual-buffet tag photo.jpg --plugin florence_2 --task "<CAPTION>"
```

---

## Overview

Florence-2 is a vision foundation model developed by Microsoft Research, released in June 2024. It uses a unified, prompt-based approach to handle multiple vision and vision-language tasks including captioning, object detection, OCR, and segmentation. Despite its compact size (0.23B-0.77B parameters), it achieves state-of-the-art performance on many benchmarks.

## Model Details

| Attribute | Value |
|-----------|-------|
| Developer | Microsoft Research |
| Release | June 2024 |
| Architecture | DaViT encoder + BERT + Transformer decoder |
| Training Data | FLD-5B (5.4B annotations, 126M images) |
| License | MIT |
| Context Length | 4,096 tokens |

### Model Variants

| Variant | Parameters | Size | Best For |
|---------|------------|------|----------|
| `florence-2-base` | 0.23B | ~0.5 GB | Fast inference, resource-constrained |
| `florence-2-large` | 0.77B | ~1.5 GB | Better accuracy, more detail |
| `florence-2-base-ft` | 0.23B | ~0.5 GB | General downstream tasks |
| `florence-2-large-ft` | 0.77B | ~1.5 GB | Best quality, recommended default |

The `-ft` (fine-tuned) variants are trained on additional downstream tasks and generally perform better for practical use.

**Note on Mac/MPS Compatibility**: The plugin uses `multimodalart/Florence-2-large-no-flash-attn` for the `large` and `large-ft` variants. This community-patched version removes the flash_attn dependency, enabling inference on Apple Silicon (MPS) and CPU devices. Original Microsoft variants are available as `large-original` and `large-ft-original` for CUDA users.

## Capabilities

### Supported Tasks

| Task | Prompt | Output Type | Tagging Use |
|------|--------|-------------|-------------|
| Caption | `<CAPTION>` | Text | Extract nouns as tags |
| Detailed Caption | `<DETAILED_CAPTION>` | Text | More tags from description |
| More Detailed Caption | `<MORE_DETAILED_CAPTION>` | Text | Maximum tag extraction |
| Object Detection | `<OD>` | Labels + bboxes | Direct labels as tags |
| Dense Region Caption | `<DENSE_REGION_CAPTION>` | Labels + bboxes | Region-specific tags |
| Region Proposal | `<REGION_PROPOSAL>` | Bboxes only | Not useful for tagging |
| OCR | `<OCR>` | Text | Text content in image |
| OCR with Region | `<OCR_WITH_REGION>` | Text + quads | Located text content |

### Tasks Requiring Input Text (Not for Auto-Tagging)

| Task | Prompt | Purpose |
|------|--------|---------|
| Phrase Grounding | `<CAPTION_TO_PHRASE_GROUNDING>` | Locate described objects |
| Referring Expression Segmentation | `<REFERRING_EXPRESSION_SEGMENTATION>` | Segment described objects |
| Open Vocabulary Detection | `<OPEN_VOCABULARY_DETECTION>` | Find specific objects |
| Region to Description | `<REGION_TO_DESCRIPTION>` | Describe bounding box |

### Strengths

- **Multi-task**: Single model handles captioning, detection, OCR, segmentation
- **Compact**: High performance despite small size (0.23B-0.77B)
- **Flexible Output**: Different detail levels available
- **MIT License**: Permissive for commercial use
- **Fast Inference**: Efficient on consumer hardware

### Categories Covered

- Objects (general detection vocabulary)
- Scenes and environments
- Activities and actions
- Text and typography (OCR)
- Spatial relationships

## Limitations

### Known Weaknesses

1. **NO CONFIDENCE SCORES**: Florence-2 does not provide per-tag confidence. Tags are returned without confidence values. This is fundamentally different from RAM++ which provides calibrated confidence scores per tag.
2. **No Predefined Tag List**: Unlike RAM++, outputs free-form text requiring parsing
3. **Variable Output Format**: Caption tasks return sentences, not tags
4. **Detection Vocabulary**: OD task has limited label set compared to specialized taggers (only 2-5 labels!)
5. **Requires Parsing**: Raw output needs post-processing to extract tags

### When to Use RAM++ Instead

Use RAM++ (threshold 0.5) when you need:
- Confidence-based filtering
- Consistent tag vocabulary
- Per-tag confidence scores for ranking
- Threshold-based control over output volume

### Edge Cases

- Very small objects may be missed by detection
- Unusual or domain-specific objects
- Non-English text (OCR optimized for English)
- Abstract or artistic imagery

## Plugin Settings

### Recommended Settings to Expose

| Setting | Type | Default | Options/Range | Purpose |
|---------|------|---------|---------------|---------|
| `model_variant` | enum | `florence-2-large-ft` | base, large, base-ft, large-ft | Quality vs speed tradeoff |
| `task_prompt` | enum | `<OD>` | See supported tasks | Output type |
| `max_new_tokens` | int | 1024 | 256-4096 | Limit output length |
| `num_beams` | int | 3 | 1-5 | Quality vs speed (beam search) |
| `device` | enum | auto | auto, cuda, cpu | Hardware selection |

### Advanced Settings (Optional)

| Setting | Type | Default | Purpose |
|---------|------|---------|---------|
| `do_sample` | bool | False | Enable sampling (non-deterministic) |
| `temperature` | float | 1.0 | Sampling randomness (only if do_sample=True) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `repetition_penalty` | float | 1.0 | Penalize repeated tokens |

### Fixed Settings (Do Not Expose)

| Setting | Value | Reason |
|---------|-------|--------|
| `trust_remote_code` | True | Required for model loading |
| `torch_dtype` | auto | Determined by device |

## Hardware Requirements

| Configuration | Minimum | Recommended |
|---------------|---------|-------------|
| RAM | 4 GB | 8 GB |
| GPU VRAM | 2 GB (base) | 4 GB (large) |
| CPU | Any x86_64/ARM64 | 4+ cores |

### Performance by Hardware

| Device | Model | Inference Time |
|--------|-------|----------------|
| NVIDIA RTX 3080 | large-ft | ~100ms |
| NVIDIA RTX 3060 | large-ft | ~150ms |
| Apple M1 Pro | large-ft | ~300ms |
| Intel i7 (CPU) | large-ft | ~3000ms |
| Intel i7 (CPU) | base-ft | ~1500ms |

### dtype Selection

| Hardware | torch_dtype | Notes |
|----------|-------------|-------|
| CUDA GPU | float16 | Recommended, fastest |
| Apple Silicon | float32 | Required (float16 causes empty output + 40x slowdown) |
| CPU | float32 | Required for CPU |

## Installation

### Dependencies

```bash
pip install torch torchvision
pip install "transformers>=4.40.0,<4.50.0"
pip install einops
```

**Important**: Transformers versions 4.50+ have compatibility issues with Florence-2's custom model code. Use version 4.44.x for best results.

### Model Download

Models are downloaded automatically from HuggingFace on first use:

```bash
visual-buffet plugins setup florence_2
```

Or specify variant:

```bash
visual-buffet plugins setup florence_2 --variant large-ft
```

Models are cached in HuggingFace's default cache (`~/.cache/huggingface/`).

### Manual Download

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
)
```

## Output Normalization

Florence-2 outputs vary by task and require normalization to Visual Buffet format.

**IMPORTANT:** Florence-2 does NOT provide confidence scores. Tags are returned without confidence values.

### Object Detection (`<OD>`)

```python
# Florence-2 raw output
{
    '<OD>': {
        'bboxes': [[x1, y1, x2, y2], ...],
        'labels': ['cat', 'dog', 'person']
    }
}

# Normalized to Visual Buffet format (NO confidence)
{
    "tags": [
        {"label": "cat"},
        {"label": "dog"},
        {"label": "person"}
    ]
}
```

### Caption Tasks (with Slugification)

```python
# Florence-2 raw output
"A black cat sitting on a wooden table next to a window"

# Normalized to Visual Buffet format with compound phrases
{
    "tags": [
        {"label": "black_cat"},      # Compound phrase first
        {"label": "wooden_table"},   # Compound phrase
        {"label": "black"},          # Then individual words
        {"label": "cat"},
        {"label": "sitting"},
        {"label": "wooden"},
        {"label": "table"},
        {"label": "window"}
    ]
}
```

### No Confidence Scores

Florence-2 cannot provide meaningful confidence scores:

- **Caption tasks**: Free-form text generation has no per-word confidence
- **Detection tasks**: Labels are extracted from generated text, not detection logits
- **Current implementation**: Tags have no confidence value

**If you need confidence-based filtering**, use RAM++ with threshold 0.5 instead.

## Usage Tips

### Best Practices

1. Use `florence-2-large-ft` for best quality
2. Use `<MORE_DETAILED_CAPTION>` for maximum tags (default, 25-50 tags)
3. Use `<DENSE_REGION_CAPTION>` for region-specific descriptions
4. Avoid `<OD>` for tagging - it only returns 2-5 object labels
5. Set `max_new_tokens=1024` unless expecting very long outputs

### Task Selection Guide

| Goal | Recommended Task | Expected Tags | Notes |
|------|------------------|---------------|-------|
| **Database building** | `<DETAILED_CAPTION>` | 15-25 tags | **RECOMMENDED default** |
| Maximum coverage | `<MORE_DETAILED_CAPTION>` | 25-50 tags | Comprehensive |
| Quick preview | `<CAPTION>` | 5-10 tags | Fast |
| Region-specific | `<DENSE_REGION_CAPTION>` | 15-30 tags | Good for complex scenes |
| Object labels | `<OD>` | 2-5 tags | **NOT recommended** - very limited! |
| Text extraction | `<OCR>` | Text content | Use for documents/signs |

### WARNING: Do Not Use `<OD>` for Tagging

The `<OD>` (Object Detection) task has a **very limited vocabulary** and typically returns only **2-5 object labels**. This is a common mistake!

**Wrong approach:**
```bash
visual-buffet tag photo.jpg --task "<OD>"  # Only returns 2-5 labels!
```

**Correct approach:**
```bash
visual-buffet tag photo.jpg --task "<DETAILED_CAPTION>"  # Returns 15-25 rich tags
```

For comprehensive tagging, use `<MORE_DETAILED_CAPTION>` which generates rich descriptions that are parsed into 25-50 meaningful tags.

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "trust_remote_code" error | Ensure `trust_remote_code=True` in both model and processor |
| CUDA out of memory | Use base variant or reduce batch size |
| Slow inference | Install CUDA-enabled PyTorch, use GPU |
| Empty output | Check task prompt format (include angle brackets) |
| Garbled output | Ensure correct post-processing for task type |

### Debug Commands

```bash
# Check model files cached
ls -la ~/.cache/huggingface/hub/models--microsoft--Florence-2-large-ft/

# Verify PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Test basic inference
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)
print('Model loaded successfully')
"

# Test with Visual Buffet
visual-buffet tag test.jpg --plugin florence_2 --debug
```

## References

- [Paper: Florence-2](https://arxiv.org/abs/2311.06242)
- [HuggingFace: Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- [HuggingFace: Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)
- [Microsoft Research Blog](https://www.microsoft.com/en-us/research/publication/florence-2-advancing-a-unified-representation-for-a-variety-of-vision-tasks/)
- [HuggingFace Fine-tuning Guide](https://huggingface.co/blog/finetune-florence2)

## License

Florence-2 is released under the **MIT License**, permitting commercial use, modification, and distribution.
