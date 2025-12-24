# OCR Verification Architecture for Auto-Tagging

> **Philosophy**: Trust but verify. Capture everything, verify before auto-tagging.

## The Problem

Auto-tagging without verification means noise goes directly into search:
- PaddleOCR: 22% low-confidence results (potential noise)
- No human filter = polluted search results
- Need automated verification pipeline

## Multi-Engine Verification Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMAGE INPUT                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  PaddleOCR   │ │    docTR     │ │   SigLIP     │
            │  (Primary)   │ │  (Verifier)  │ │  (Validator) │
            │  threshold   │ │  threshold   │ │  threshold   │
            │    0.3       │ │    0.3       │ │    0.001     │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    │  All texts    │  All texts    │  Scores for
                    │  + confidence │  + confidence │  each text
                    ▼               ▼               ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │                    VERIFICATION ENGINE                           │
            │                                                                  │
            │  For each text found by PaddleOCR:                              │
            │    1. Did docTR also find it? (normalized match)                │
            │    2. What's the SigLIP score? (concept validation)             │
            │    3. What's the OCR confidence? (detection confidence)         │
            │                                                                  │
            │  Compute verification score:                                     │
            │    - paddle_conf: 0-1 (OCR confidence)                          │
            │    - doctr_found: 0 or 1 (binary verification)                  │
            │    - siglip_score: 0-1 (concept validation)                     │
            │                                                                  │
            │  verification_score = (                                          │
            │      0.5 * paddle_conf +                                        │
            │      0.3 * doctr_found +                                        │
            │      0.2 * min(siglip_score * 10, 1)  # Scale SigLIP to 0-1     │
            │  )                                                               │
            └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │                    CONFIDENCE TIERS                              │
            │                                                                  │
            │  VERIFIED (auto-tag, auto-search):                              │
            │    verification_score >= 0.7                                     │
            │    OR (paddle_conf >= 0.9 AND doctr_found)                      │
            │    OR (paddle_conf >= 0.8 AND siglip_score >= 0.01)             │
            │                                                                  │
            │  LIKELY (auto-tag, auto-search with lower rank):                │
            │    verification_score >= 0.5                                     │
            │    OR paddle_conf >= 0.7                                        │
            │                                                                  │
            │  UNVERIFIED (store but don't auto-tag, manual review queue):    │
            │    verification_score < 0.5                                      │
            │    AND paddle_conf < 0.7                                        │
            │                                                                  │
            │  REJECTED (don't store):                                         │
            │    paddle_conf < 0.3 AND NOT doctr_found AND siglip < 0.001    │
            └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │                      OUTPUT                                      │
            │                                                                  │
            │  {                                                               │
            │    "text": "RINK",                                              │
            │    "tier": "VERIFIED",                                          │
            │    "verification_score": 0.92,                                  │
            │    "sources": {                                                 │
            │      "paddle_ocr": {"confidence": 0.99, "bbox": [...]},        │
            │      "doctr": {"confidence": 0.98, "bbox": [...]},             │
            │      "siglip": {"score": 0.99}                                 │
            │    },                                                           │
            │    "auto_tag": true,                                            │
            │    "searchable": true                                           │
            │  }                                                               │
            └─────────────────────────────────────────────────────────────────┘
```

## Why Three Engines?

| Engine | What It Validates | Catches |
|--------|-------------------|---------|
| **PaddleOCR** | "There is text here that looks like X" | Maximum recall (95%) |
| **docTR** | "I also see text that looks like X" | Cross-OCR verification |
| **SigLIP** | "This image conceptually contains X" | Semantic validation |

### Verification Matrix

| PaddleOCR | docTR | SigLIP | Tier | Action |
|-----------|-------|--------|------|--------|
| High (≥0.8) | Found | High (≥0.01) | VERIFIED | Auto-tag |
| High (≥0.8) | Found | Low | VERIFIED | Auto-tag (prominent text) |
| High (≥0.8) | Not found | High | LIKELY | Auto-tag (SigLIP confirms) |
| High (≥0.8) | Not found | Low | LIKELY | Auto-tag (OCR confident) |
| Medium (0.5-0.8) | Found | Any | LIKELY | Auto-tag |
| Medium (0.5-0.8) | Not found | High | LIKELY | Auto-tag (SigLIP confirms) |
| Medium (0.5-0.8) | Not found | Low | UNVERIFIED | Manual review |
| Low (<0.5) | Found | Any | LIKELY | Auto-tag (docTR confirms) |
| Low (<0.5) | Not found | High | UNVERIFIED | Manual review |
| Low (<0.5) | Not found | Low | REJECTED | Don't store |

## Edge Cases

### 1. Small/Distant Text (Scoreboard Example)
- PaddleOCR: "HOME" @ 99%
- docTR: Not found
- SigLIP: 0.0007 (can't validate small text)
- **Decision**: LIKELY (high OCR confidence, auto-tag despite no verification)

### 2. Prominent Sign
- PaddleOCR: "RINK" @ 99%
- docTR: "RINK" @ 98%
- SigLIP: 0.99
- **Decision**: VERIFIED (all three agree)

### 3. OCR Noise/Hallucination
- PaddleOCR: "XYZABC" @ 45%
- docTR: Not found
- SigLIP: 0.0000
- **Decision**: REJECTED (low confidence, no verification)

### 4. Real Text, Low OCR Confidence
- PaddleOCR: "WARNING" @ 52%
- docTR: "WARNING" @ 48%
- SigLIP: 0.03
- **Decision**: LIKELY (both OCR engines agree)

## Implementation Notes

### Text Normalization for Matching
```python
def normalize_for_match(text: str) -> str:
    """Normalize text for cross-engine comparison."""
    import re
    # Lowercase, remove punctuation, collapse whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def texts_match(t1: str, t2: str) -> bool:
    """Check if two OCR results refer to the same text."""
    n1, n2 = normalize_for_match(t1), normalize_for_match(t2)
    # Exact match or high similarity
    if n1 == n2:
        return True
    # One contains the other (handles line breaks, partial matches)
    if n1 in n2 or n2 in n1:
        return True
    # Levenshtein distance for typos
    if len(n1) > 3 and len(n2) > 3:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, n1, n2).ratio()
        return ratio > 0.8
    return False
```

### SigLIP Prompt for OCR Validation
```python
# Best performing template from testing
SIGLIP_OCR_TEMPLATE = "A sign reading {}."

# For prominent text: "A sign reading RINK." → 0.99
# For small text: "A sign reading HOME." → 0.0007 (won't help)
```

### Confidence Tier Thresholds
```python
TIER_THRESHOLDS = {
    "VERIFIED": {
        "verification_score": 0.7,
        "alt_rules": [
            {"paddle_conf": 0.9, "doctr_found": True},
            {"paddle_conf": 0.8, "siglip_score": 0.01},
        ]
    },
    "LIKELY": {
        "verification_score": 0.5,
        "alt_rules": [
            {"paddle_conf": 0.7},
            {"doctr_found": True},
        ]
    },
    "UNVERIFIED": {
        # Everything else that wasn't rejected
    },
    "REJECTED": {
        "paddle_conf_max": 0.3,
        "doctr_found": False,
        "siglip_max": 0.001,
    }
}
```

## Performance Considerations

| Engine | Time (GPU) | Time (CPU) | Run When |
|--------|------------|------------|----------|
| PaddleOCR | ~3s | ~63s | Always (primary) |
| docTR | ~1s | ~1.4s | Always (fast verifier) |
| SigLIP | ~0.5s | ~2s | Only for texts needing validation |

**Total pipeline time**: ~4.5s per image (GPU) - acceptable for batch processing.

### Optimization: Lazy SigLIP
```python
# Only run SigLIP when needed
def needs_siglip_validation(paddle_conf: float, doctr_found: bool) -> bool:
    """Determine if SigLIP validation would change the outcome."""
    if paddle_conf >= 0.9 and doctr_found:
        return False  # Already VERIFIED
    if paddle_conf < 0.3 and not doctr_found:
        return False  # Will be REJECTED anyway
    return True  # SigLIP might help
```

## Storage Schema

```json
{
  "image_id": "abc123",
  "ocr_results": [
    {
      "text": "RINK POLICY",
      "normalized": "rink policy",
      "tier": "VERIFIED",
      "verification_score": 0.92,
      "auto_tagged": true,
      "searchable": true,
      "sources": {
        "paddle_ocr": {
          "confidence": 0.99,
          "bbox": [[10, 20], [100, 20], [100, 50], [10, 50]],
          "inference_time_ms": 145
        },
        "doctr": {
          "confidence": 0.97,
          "bbox": [[12, 22], [98, 22], [98, 48], [12, 48]],
          "inference_time_ms": 89
        },
        "siglip": {
          "score": 0.85,
          "template": "A sign reading {}.",
          "inference_time_ms": 42
        }
      }
    }
  ],
  "processing": {
    "total_time_ms": 4521,
    "verified_count": 8,
    "likely_count": 3,
    "unverified_count": 1,
    "rejected_count": 2
  }
}
```

## Summary

For auto-tagging: **Trust but verify**

1. **PaddleOCR** (trust) - Maximum recall, catches everything
2. **docTR** (verify) - Cross-OCR confirmation
3. **SigLIP** (validate) - Semantic confirmation for prominent text

Confidence tiers enable:
- VERIFIED → Auto-tag, prime search results
- LIKELY → Auto-tag, secondary search results
- UNVERIFIED → Store, manual review queue
- REJECTED → Don't pollute the archive
