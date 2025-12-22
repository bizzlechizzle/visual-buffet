# Florence-2 Subject Matter Expert Guide

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

1. **No Predefined Tag List**: Unlike RAM++, outputs free-form text requiring parsing
2. **Variable Output Format**: Caption tasks return sentences, not tags
3. **Detection Vocabulary**: OD task has limited label set compared to specialized taggers
4. **Confidence Scores**: Not natively provided; must be derived from generation scores
5. **Requires Parsing**: Raw output needs post-processing to extract tags

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

### Object Detection (`<OD>`)

```python
# Florence-2 raw output
{
    '<OD>': {
        'bboxes': [[x1, y1, x2, y2], ...],
        'labels': ['cat', 'dog', 'person']
    }
}

# Normalized to Visual Buffet format
{
    "tags": [
        {"label": "cat", "confidence": 0.95},
        {"label": "dog", "confidence": 0.90},
        {"label": "person", "confidence": 0.85}
    ]
}
```

Note: Confidence scores derived from generation beam scores or assigned positionally.

### Caption Tasks

```python
# Florence-2 raw output
"A black cat sitting on a wooden table next to a window"

# Normalized to Visual Buffet format (extract nouns/phrases)
{
    "tags": [
        {"label": "cat", "confidence": 0.90},
        {"label": "table", "confidence": 0.85},
        {"label": "window", "confidence": 0.80},
        {"label": "black", "confidence": 0.75},
        {"label": "wooden", "confidence": 0.70}
    ]
}
```

### Confidence Score Strategy

Since Florence-2 doesn't provide native confidence scores:

1. **For OD/Detection**: Use generation beam scores if available, else assign by position
2. **For Captions**: Parse nouns/adjectives, assign decreasing confidence by order
3. **Alternative**: Use `output_scores=True` in generation to get token probabilities

## Usage Tips

### Best Practices

1. Use `florence-2-large-ft` for best quality
2. Use `<MORE_DETAILED_CAPTION>` for maximum tags (default, 25-50 tags)
3. Use `<DENSE_REGION_CAPTION>` for region-specific descriptions
4. Avoid `<OD>` for tagging - it only returns 2-5 object labels
5. Set `max_new_tokens=1024` unless expecting very long outputs

### Task Selection Guide

| Goal | Recommended Task | Expected Tags |
|------|------------------|---------------|
| **Maximum tags (DEFAULT)** | `<MORE_DETAILED_CAPTION>` | 25-50 tags |
| Rich descriptive tags | `<DETAILED_CAPTION>` | 15-25 tags |
| Region-specific tags | `<DENSE_REGION_CAPTION>` | 15-30 tags |
| Object detection only | `<OD>` | 2-5 tags (limited!) |
| Brief description | `<CAPTION>` | 5-10 tags |
| Text in images | `<OCR>` | Text content |

**Important**: The `<OD>` (Object Detection) task has a very limited vocabulary and typically returns only 2-5 object labels. For comprehensive tagging, use `<MORE_DETAILED_CAPTION>` which generates rich descriptions that are parsed into 25-50 meaningful tags.

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
