# SME Audit Report

> **Audit Date**: 2025-12-21
> **Audit Target**: `plugins/siglip/__init__.py`
> **SME Reference**: `/mnt/nas-projects/repo-depot/skills/vlm/references/siglip.md`
> **Auditor**: Claude (audit skill v0.1.0)
> **Strictness**: Standard

---

## Executive Summary

**Overall Grade: B** (82%)

| Dimension | Score | Grade |
|-----------|-------|-------|
| Citation Integrity | N/A | N/A (code, not docs) |
| Accuracy | 90% | A |
| Coverage | 75% | C |
| Currency | 85% | B |

### Key Finding

**The hardcoded vocabulary is NOT a bug — it's architecturally required.**

SigLIP is a **zero-shot classifier**, not a tagger. Unlike RAM++ which has 6,500 built-in tag categories, SigLIP requires you to provide candidate labels upfront. The SME confirms this:

> "texts = ["a photo of a cat", "a photo of a dog"]" — you must supply the labels.

The plugin author correctly implemented SigLIP but had to choose a vocabulary to make it work as a tagging plugin. They chose ~200 general-purpose labels.

### Critical Issues

1. **Vocabulary mismatch**: 200 labels vs RAM++'s 6,500 makes comparison unfair
2. **Missing: Load vocabulary from RAM++ tag list** — trivial fix for fair comparison

---

## Detailed Findings

### Accuracy Analysis (90%)

| Implementation Claim | SME Reference | Result | Notes |
|---------------------|---------------|--------|-------|
| Uses sigmoid, NOT softmax | "Apply sigmoid (NOT softmax!)" | **VERIFIED** | Line 504: `torch.sigmoid(logits)` |
| `padding="max_length"` required | "IMPORTANT: Use padding='max_length'" | **VERIFIED** | Line 471: `padding="max_length"` |
| Prompt template "This is a photo of {}" | "Use 'This is a photo of {label}.' for best results" | **VERIFIED** | Line 44: `PROMPT_TEMPLATE` |
| Lowercase text required | "CRITICAL: SigLIP was trained on lowercase text" | **VERIFIED** | Line 463: `labels_lower = [label.lower()...]` |
| Max sequence length 64 | "Max sequence length: 64 tokens" | **VERIFIED** | Line 474: `max_length=64` |
| SO400M as default | "Recommended: siglip-so400m-patch14-384" | **VERIFIED** | Line 40: `DEFAULT_MODEL = "so400m"` |
| Supports quantization | "4-bit quantization with minimal quality loss" | **VERIFIED** | Lines 309-325: BitsAndBytesConfig |
| SDPA attention | "attn_implementation='sdpa'" | **VERIFIED** | Lines 263-285: auto-detects SDPA |

**Contradictions Found**: None

**Minor Deviations**:
| Claim | SME Says | Severity | Notes |
|-------|----------|----------|-------|
| `recommended_threshold=0.01` | Not specified in SME | Minor | Reasonable choice given sigmoid output range |

### Coverage Analysis (75%)

**Implemented from SME:**
- [x] Zero-shot classification pipeline
- [x] Multiple model variants (v1 and v2)
- [x] Device auto-detection (CUDA, MPS, CPU)
- [x] dtype auto-selection (bfloat16/float16/float32)
- [x] Quantization support (4-bit, 8-bit)
- [x] Attention implementation selection
- [x] NaFlex variant support

**Gaps Identified:**

| Gap | SME Section | Severity | Recommendation |
|-----|-------------|----------|----------------|
| Fixed 200-label vocabulary | N/A (design choice) | **SIGNIFICANT** | Load from RAM++ tag list |
| No embedding extraction API | "Extracting Embeddings" | Minor | Add `get_embedding()` method |
| No batch processing | Common pattern | Minor | Process multiple images at once |

### Currency Analysis (85%)

| Item | Implementation | SME Current | Status |
|------|---------------|-------------|--------|
| Model IDs | v1 + v2 variants | v1 + v2 documented | **CURRENT** |
| Transformers version | `>=4.47.0` | `>=4.37.0` minimum | **CURRENT** |
| SigLIP 2 NaFlex | Included | Documented | **CURRENT** |

**Currency Issue:**
- Line 106: Checks for transformers `>=4.47.0`, but SME says `>=4.37.0` is minimum. The plugin is more conservative — acceptable.

---

## Root Cause: Why Hardcoded Labels?

### SigLIP Architecture Requires Candidate Labels

From the SME:
```python
# You MUST provide texts (labels) to score against
texts = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=texts, images=image, ...)
```

**SigLIP computes similarity between an image and each provided text prompt.** It cannot generate tags on its own — it only scores candidates.

### Comparison with RAM++

| Model | Architecture | Labels |
|-------|--------------|--------|
| **RAM++** | Classification head with 6,500 fixed categories | Built-in |
| **SigLIP** | Vision-language contrastive (like CLIP) | User-provided |
| **Florence-2** | Generative (caption → parse) | Generated |

**SigLIP is fundamentally different** — it's a similarity model, not a classifier with fixed categories.

### The 200-Label Vocabulary

The plugin author created `_get_tag_vocabulary()` (lines 356-452) with ~200 curated labels covering:
- Animals, people, vehicles
- Buildings, nature, plants
- Food, objects, actions
- Attributes, scenes, emotions
- Art styles, photography terms

This is a **reasonable default** but makes comparison with RAM++ (6,500 tags) unfair.

---

## Recommendations

### Must Fix (Critical)

1. **Load vocabulary from external source**
   ```python
   def _load_labels(self) -> None:
       # Option 1: Use RAM++ tag list for fair comparison
       ram_tags_path = self.plugin_dir.parent / "ram_plus" / "ram_tag_list.txt"
       if ram_tags_path.exists():
           self._labels = ram_tags_path.read_text().splitlines()
       else:
           self._labels = self._get_tag_vocabulary()  # fallback
   ```

### Should Fix (Important)

2. **Make vocabulary configurable**
   ```python
   def configure(self, **kwargs):
       if "vocabulary" in kwargs:
           self._labels = kwargs["vocabulary"]
           # or load from file path
   ```

3. **Document the architectural difference** in plugin docstring

### Consider (Minor)

4. Add batch processing for multiple images
5. Expose `get_image_embedding()` for vector search use cases

---

## Answer to Your Question

**Why does SigLIP have hardcoded labels?**

Because SigLIP is architecturally a **zero-shot classifier** (like CLIP), not a tagger. It requires candidate labels to score against — it cannot generate tags on its own.

The plugin correctly implements SigLIP per the SME. The 200-label vocabulary was a necessary design choice to make it function as a tagging plugin. However, it should be expanded to RAM++'s ~6,500 tags for fair comparison.

---

## Audit Metadata

### Methodology
- Extracted implementation patterns from plugin source
- Cross-referenced against SME reference document
- Verified code follows documented best practices

### Confidence: HIGH
- Clear technical claims in both documents
- Unambiguous matching criteria
- No subjective interpretation required

---

*Generated with Claude Code Audit Skill*
