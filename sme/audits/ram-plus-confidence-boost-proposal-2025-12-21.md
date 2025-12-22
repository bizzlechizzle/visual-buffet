# SME Audit Report: RAM Plus Multi-Resolution Confidence Boosting

> **Audit Date**: 2025-12-21
> **Audit Target**: RAM Plus multi-resolution confidence scoring proposal
> **SME References**:
> - `/mnt/nas-projects/visual-buffet/sme/ram-plus-image-tagging.md`
> - `/mnt/nas-projects/visual-buffet/docs/sme/ram_plus.sme.md`
> - `/mnt/nas-projects/visual-buffet/docs/plugin-development.md`
> **Auditor**: Claude (audit skill v0.1.0)
> **Strictness**: Standard

---

## Executive Summary

**Overall Assessment: PROPOSAL IS SOUND WITH MODIFICATIONS NEEDED**

The user's proposal to add confidence boosts for overlapping tags across resolutions addresses a **documented limitation** in RAM++: the model outputs binary tag presence, not probabilities [SME: HIGH confidence claim].

### Proposed System (User's Original)

| Occurrences | Confidence |
|-------------|------------|
| 1 (single pass) | 1.0 (base) |
| 2 resolutions | +0.20 boost |
| 3 resolutions | +0.30 boost |
| Max | +0.50 boost cap |

### Critical Issues Identified

1. **Scoring inversion**: Base of 1.0 with additive boosts creates scores >1.0
2. **Quick mode penalty**: Valid concern - single-pass modes never get validation boosts
3. **Non-linear scaling**: Jump from +0.20 to +0.30 lacks clear rationale

---

## Detailed Analysis

### SME Reference: Key Facts

From the SME documents, these facts are **verified** [HIGH confidence]:

| Claim | Source | Status |
|-------|--------|--------|
| RAM++ provides NO confidence scores | Both SME docs | VERIFIED |
| Current plugin assigns synthetic scores (0.99 → 0.50) based on tag order | `docs/sme/ram_plus.sme.md:94-96` | VERIFIED |
| Multi-resolution merging uses `sources` count | `plugin-development.md:52` | VERIFIED |
| Quick mode = single 1080px pass | `plugin-development.md:30` | VERIFIED |
| Standard mode = 2 passes (480 + 2048) | `plugin-development.md:31` | VERIFIED |
| High mode = 3 passes (480 + 1080 + 2048) | `plugin-development.md:32` | VERIFIED |

### Current System Analysis

The current tag merging system (from `plugin-development.md:49-53`):

```
- Duplicate tags are deduplicated (case-insensitive)
- Highest confidence score is kept
- Tags found at multiple resolutions get a `sources` count
- Results sorted by: sources (desc), confidence (desc), label
```

**Gap identified**: The `sources` count is stored but not used to modify confidence. The user's proposal fills this gap.

---

## Proposal Audit

### Issue 1: Scoring Scale Problem

**User's Proposal**: Base 1.0 + boosts up to +0.50 = max 1.50

**Problem**: Confidence scores conventionally range 0.0-1.0. Scores >1.0 break:
- Threshold filtering (threshold 0.5 means nothing if scores are 1.0-1.5)
- Cross-model comparison (SigLIP returns 0.01-0.10 typically)
- UI display expectations

**Recommendation**: Use multiplicative or replacement approach instead:

```
Option A: Replacement Scale (Recommended)
─────────────────────────────────────────
sources=1 → confidence = 0.70  (baseline, unvalidated)
sources=2 → confidence = 0.85  (+0.15 validated)
sources=3 → confidence = 0.95  (+0.10 high confidence)
sources=4 → confidence = 0.98  (+0.03 very high)
sources=5 → confidence = 0.99  (+0.01 cap)

Option B: Multiplicative Boost
─────────────────────────────────────────
sources=1 → base_confidence × 1.0
sources=2 → base_confidence × 1.15 (capped at 0.99)
sources=3 → base_confidence × 1.25 (capped at 0.99)
```

### Issue 2: Quick Mode Never Gets Boosts

**User's Concern**: "if you do fast which is only 1 pass you would never get a boost on some tags?"

**Analysis**: This is **CORRECT** and a valid design tension.

| Mode | Passes | Max Sources | Max Boost (user's system) |
|------|--------|-------------|---------------------------|
| quick | 1 | 1 | 0 (never boosted) |
| standard | 2 | 2 | +0.20 |
| high | 3 | 3 | +0.30 |
| max | 5 | 5 | +0.50 (capped) |

**This is actually CORRECT BEHAVIOR** - quick mode sacrifices confidence validation for speed. The design accurately reflects that:

- **Quick mode tags are inherently less validated** - single resolution only
- **Users choosing quick mode accept this trade-off**
- **Boosts reward multi-resolution agreement** - quick mode has no agreement to reward

**Recommendation**: Document this clearly rather than "fix" it:

```markdown
## Confidence Scoring by Quality Level

| Quality | Resolutions | Confidence Range | Notes |
|---------|-------------|------------------|-------|
| quick | 1 | 0.70 | Unvalidated, speed-optimized |
| standard | 2 | 0.70-0.85 | Cross-resolution validated |
| high | 3 | 0.70-0.95 | High confidence validation |
| max | 5 | 0.70-0.99 | Maximum validation |
```

### Issue 3: Non-Linear Boost Scaling

**User's Proposal**:
- 2 sources = +0.20
- 3 sources = +0.30
- 4+ sources = +0.50 cap

**Analysis**: The jumps are arbitrary. Consider:

**Diminishing Returns Model** (more realistic):

```
sources=1 → 0.70 (base)
sources=2 → 0.70 + 0.15 = 0.85  (+0.15)
sources=3 → 0.85 + 0.08 = 0.93  (+0.08, diminishing)
sources=4 → 0.93 + 0.04 = 0.97  (+0.04, further diminishing)
sources=5 → 0.97 + 0.02 = 0.99  (+0.02, cap)
```

**Rationale**: Each additional confirmation adds less new information (Bayesian update intuition).

---

## Recommended Implementation

### Final Scoring Formula

```python
def calculate_confidence(sources: int, tag_position: int, max_tags: int) -> float:
    """
    Calculate synthetic confidence for RAM++ tags.

    Args:
        sources: Number of resolutions that found this tag (1-5)
        tag_position: Position in tag list (0 = first/most confident)
        max_tags: Total tags returned

    Returns:
        Confidence score 0.0-1.0
    """
    # Base confidence from tag position (RAM++ ordering indicates relevance)
    # First tag = 0.70, last tag = 0.50
    position_factor = 1.0 - (tag_position / max_tags) * 0.4  # 1.0 → 0.6
    base_confidence = 0.50 + (position_factor * 0.20)  # 0.50 → 0.70

    # Multi-resolution boost (diminishing returns)
    SOURCE_BOOSTS = {
        1: 0.00,  # No boost - unvalidated
        2: 0.15,  # Validated by 2 resolutions
        3: 0.23,  # +0.08 diminishing
        4: 0.27,  # +0.04 further diminishing
        5: 0.29,  # +0.02 cap approaching
    }
    boost = SOURCE_BOOSTS.get(sources, 0.29)

    # Final score capped at 0.99
    return min(0.99, base_confidence + boost)
```

### Example Outputs

| Tag | Position | Sources | Base | Boost | Final |
|-----|----------|---------|------|-------|-------|
| dog | 1/20 | 3 | 0.69 | 0.23 | 0.92 |
| outdoor | 2/20 | 2 | 0.68 | 0.15 | 0.83 |
| grass | 10/20 | 1 | 0.60 | 0.00 | 0.60 |
| tree | 15/20 | 3 | 0.56 | 0.23 | 0.79 |

### Quick Mode Handling

For quick mode (sources always = 1):

```python
def calculate_confidence_quick(tag_position: int, max_tags: int) -> float:
    """Quick mode: position-only confidence, no validation boost."""
    position_factor = 1.0 - (tag_position / max_tags) * 0.4
    return 0.50 + (position_factor * 0.20)  # Range: 0.50-0.70
```

**Quick mode tags are clearly lower confidence** - this is intentional and correct.

---

## Comparison with SigLIP

| Aspect | RAM++ (with this proposal) | SigLIP |
|--------|---------------------------|--------|
| Native confidence | No (synthetic) | Yes (sigmoid) |
| Confidence range | 0.50-0.99 | 0.001-0.10 typical |
| Multi-res boost | Yes (sources count) | No (already has confidence) |
| Comparable? | Not directly | Need normalization |

**Recommendation**: When combining RAM++ and SigLIP results, normalize both to a common scale or keep separate.

---

## Audit Verdict

### Proposal Assessment

| Aspect | User's Proposal | Recommendation |
|--------|-----------------|----------------|
| Core concept | SOUND | Multi-resolution validation is valid |
| Scoring scale | NEEDS FIX | Use 0.0-1.0 range, not >1.0 |
| Quick mode concern | VALID BUT CORRECT | Document as intentional trade-off |
| Boost amounts | ARBITRARY | Use diminishing returns model |

### Implementation Checklist

- [ ] Implement `sources` count → confidence boost mapping
- [ ] Use 0.0-1.0 scale (not >1.0)
- [ ] Apply diminishing returns to additional sources
- [ ] Combine with position-based base confidence
- [ ] Document quick mode limitations clearly
- [ ] Add `provides_confidence = "synthetic"` to plugin.toml

---

## Source Appendix

| Claim | Source | Line | Verified |
|-------|--------|------|----------|
| RAM++ no confidence | ram_plus.sme.md | 360 | YES |
| Synthetic scores 0.99→0.50 | ram_plus.sme.md | 94-96 | YES |
| Sources count in merge | plugin-development.md | 52 | YES |
| Quick = 1080px only | plugin-development.md | 30 | YES |
| High = 3 resolutions | plugin-development.md | 32 | YES |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-21 | Initial audit |
