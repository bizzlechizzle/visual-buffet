# RAM++ Subject Matter Expert Guide

## Overview

RAM++ (Recognize Anything Plus Plus) is an advanced image tagging model developed by researchers at DAMO Academy, Alibaba Group. It can recognize and tag images with approximately 6,500 different labels covering a wide range of concepts.

## Model Details

| Attribute | Value |
|-----------|-------|
| Model Name | ram_plus_swin_large_14m |
| Architecture | Swin Transformer Large |
| Input Size | 384 x 384 pixels |
| Parameters | ~440M |
| File Size | ~1.5 GB |
| Training Data | 14 million image-tag pairs |

## Capabilities

### Strengths
- **Broad Coverage**: Recognizes ~6,500 different concepts
- **General Purpose**: Works well on diverse image types
- **Zero-shot**: Can identify objects without fine-tuning
- **Multi-label**: Outputs multiple relevant tags per image

### Categories Covered
- Objects (animals, vehicles, furniture, etc.)
- Scenes (indoor, outdoor, landscapes)
- Activities (sports, cooking, reading)
- Attributes (colors, textures, materials)
- Abstract concepts (emotion, style)

## Limitations

### Known Weaknesses
1. **~~No Confidence Scores~~**: RESOLVED - Visual Buffet now extracts sigmoid probabilities from model internals
2. **Western Bias**: Better at recognizing Western objects/concepts
3. **Artistic Images**: May struggle with abstract or stylized art
4. **Text in Images**: Does not OCR or read text
5. **Fine-grained**: May not distinguish between similar species/models

### Edge Cases
- Very dark or overexposed images
- Heavily cropped subjects
- Unusual camera angles
- Synthetic/rendered images

## Hardware Requirements

| Configuration | Minimum | Recommended |
|---------------|---------|-------------|
| RAM | 4 GB | 8 GB |
| GPU VRAM | - | 4 GB |
| CPU | Any x86_64/ARM64 | 4+ cores |

### Performance by Hardware

| Device | Inference Time |
|--------|----------------|
| NVIDIA RTX 3080 | ~50ms |
| Apple M1 Pro | ~150ms |
| Intel i7 (CPU) | ~2000ms |

## Installation

### Dependencies

```bash
pip install torch torchvision timm
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

### Model Download

Models are downloaded automatically on first use, or manually via:

```bash
visual-buffet plugins setup ram_plus
```

Files are stored in `plugins/ram_plus/models/`:
- `ram_plus_swin_large_14m.pth` (~1.5 GB)
- `ram_tag_list.txt` (~100 KB)

## Usage Tips

### Best Practices
1. Use images at least 384x384 pixels for best results
2. Ensure good lighting and focus
3. Center the main subject when possible
4. Use threshold of 0.5-0.7 for balanced precision/recall

### Confidence Interpretation
Visual Buffet extracts **real sigmoid probabilities** from RAM++'s internal logits:
- Scores represent `sigmoid(logits)` - actual model confidence
- Tags only appear if they exceed their per-class calibrated threshold
- Typical range: 0.68-0.99 (since they already passed thresholding)
- Higher scores = stronger model confidence for that tag

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Model not found" | Run `visual-buffet plugins setup ram_plus` |
| "CUDA out of memory" | Use CPU or reduce batch size |
| "Missing ram package" | Install recognize-anything package |
| Slow inference | Install CUDA-enabled PyTorch |

### Debug Commands

```bash
# Check model files
ls -la plugins/ram_plus/models/

# Verify PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Test inference
visual-buffet tag test.jpg --debug
```

## References

- [Paper: Recognize Anything](https://arxiv.org/abs/2306.03514)
- [GitHub: recognize-anything](https://github.com/xinyu1205/recognize-anything)
- [HuggingFace Model](https://huggingface.co/xinyu1205/recognize-anything-plus-model)

## License

The RAM++ model is released under the Apache 2.0 License.
