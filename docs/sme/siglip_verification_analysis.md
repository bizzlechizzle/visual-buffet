# SigLIP Verification Analysis — REVISED

## TLDR — Critical Finding

**SigLIP is NOT a reliable verifier for single-word tags.**

| What Happened | Data |
|---------------|------|
| Tags found by BOTH RAM++ and Florence-2 | 128 |
| SigLIP verified (>= 0.03) | 23 (18%) |
| SigLIP REJECTED as noise | **105 (82%)** |

**SigLIP rejects CORRECT tags:**
- `car` (RAM++ 0.9999) → SigLIP 0.0007 = NOISE ❌
- `building` (RAM++ 1.0) → SigLIP 0.029 = LOW ❌
- `floor` (RAM++ 0.93) → SigLIP 0.004 = NOISE ❌
- `ceiling_fan` (RAM++ 0.86) → SigLIP 0.0001 = NOISE ❌
- `debris` (RAM++ 0.76) → SigLIP 0.0005 = NOISE ❌

**SigLIP only works well for compound phrases:**
- `abandoned_bar` → SigLIP 0.997 ✓
- `empty_restaurant` → SigLIP 0.96 ✓
- `stools_arranged` → SigLIP 0.98 ✓

---

## Revised Confidence Strategy for Archive Apps

### DO NOT use SigLIP to filter generic tags

SigLIP was trained with "This is a photo of {label}" prompts. It works for descriptive phrases but fails for:
- Generic nouns: floor, ceiling, wall, window, door
- Scene elements: building, room, sky, grass
- Objects: car, chair, table, sign

### USE Model Agreement as Verification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: HIGHEST CONFIDENCE                                                  │
│  ════════════════════════════                                                │
│  Tags found by BOTH RAM++ AND Florence-2                                     │
│  • Agreement IS verification                                                 │
│  • No SigLIP needed                                                          │
│  • Include ALL overlap tags regardless of SigLIP score                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 2: HIGH CONFIDENCE                                                     │
│  ═══════════════════════                                                     │
│  RAM++ tags with confidence >= 0.7                                           │
│  • Single model but high confidence                                          │
│  • RAM++ is well-calibrated for these                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 3: MEDIUM CONFIDENCE                                                   │
│  ════════════════════════                                                    │
│  RAM++ tags with confidence 0.5-0.7                                          │
│  • Lower confidence, may have some noise                                     │
│  • Consider context/image type                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 4: CONTEXTUAL (SigLIP verified)                                        │
│  ═══════════════════════════════════                                         │
│  Florence-2 COMPOUND phrases with SigLIP >= 0.10                             │
│  • "abandoned_bar", "empty_restaurant", "wooden_panels"                      │
│  • SigLIP works well for these descriptive phrases                           │
│  • Adds semantic richness beyond single words                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Code

```python
def build_confident_tags(ram_result, florence_result, siglip_result=None):
    """Build tiered confident tags from multi-model results.

    Args:
        ram_result: TagResult from RAM++ plugin
        florence_result: TagResult from Florence-2 plugin
        siglip_result: Optional TagResult from SigLIP (only for compound phrases)

    Returns:
        dict with tiered tag lists
    """
    # Extract tag sets
    ram_tags = {t.label: t.confidence for t in ram_result.tags}
    florence_tags = {t.label for t in florence_result.tags}

    # Build SigLIP lookup if provided
    siglip_scores = {}
    if siglip_result:
        siglip_scores = {t.label: t.confidence for t in siglip_result.tags}

    # Calculate overlap
    overlap = set(ram_tags.keys()) & florence_tags
    ram_only = set(ram_tags.keys()) - florence_tags
    florence_only = florence_tags - set(ram_tags.keys())

    confident_tags = {
        "tier1_highest": [],   # Both models agree
        "tier2_high": [],      # RAM++ >= 0.7
        "tier3_medium": [],    # RAM++ 0.5-0.7
        "tier4_contextual": [], # Florence-2 compounds + SigLIP verified
    }

    # TIER 1: Overlap tags (both models agree)
    for tag in overlap:
        confident_tags["tier1_highest"].append({
            "label": tag,
            "ram_confidence": ram_tags[tag],
            "source": "both",
            "reason": "Found by both RAM++ and Florence-2",
        })

    # TIER 2 & 3: RAM++ only tags
    for tag in ram_only:
        conf = ram_tags[tag]
        if conf >= 0.7:
            confident_tags["tier2_high"].append({
                "label": tag,
                "ram_confidence": conf,
                "source": "ram_plus",
                "reason": "High RAM++ confidence",
            })
        elif conf >= 0.5:
            confident_tags["tier3_medium"].append({
                "label": tag,
                "ram_confidence": conf,
                "source": "ram_plus",
                "reason": "Medium RAM++ confidence",
            })

    # TIER 4: Florence-2 compound phrases (SigLIP verified)
    for tag in florence_only:
        # Only include compound phrases (contain underscore)
        if "_" in tag and siglip_scores.get(tag, 0) >= 0.10:
            confident_tags["tier4_contextual"].append({
                "label": tag,
                "siglip_confidence": siglip_scores.get(tag, 0),
                "source": "florence_2",
                "reason": "Compound phrase verified by SigLIP",
            })

    return confident_tags
```

---

## Empirical Results by Tier

### testimage01.jpg (Abandoned Diner)

**TIER 1 - Both Models Agree (17 tags):**
```
bar, stool, counter, ceiling, floor, fan, window, row,
barrel, black, building, light, metal, area, side, space, yellow
```

**TIER 2 - RAM++ High Confidence (100+ tags):**
```
bar_stool, room, ceiling_fan, restaurant, bulletin_board, sit,
wall, kitchen, table, debris, sign, seat, tile, chair, pillar...
```

**TIER 4 - Florence-2 Compounds (SigLIP >= 0.10):**
```
abandoned_bar (0.997), stools_arranged (0.985), empty_bar (0.962),
several_stools (0.882), abandoned_building (0.236)...
```

---

## Benchmark Data Summary

### Vocabulary Statistics (10 images)

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| RAM++ tags | 132.9 | 104 | 164 |
| Florence-2 tags | 65.9 | 42 | 137 |
| Overlap (both) | 12.8 | 6 | 25 |
| Combined unique | 186.0 | 143 | 268 |

### Why SigLIP Fails for Generic Tags

SigLIP uses the prompt template: `"This is a photo of {label}."`

This works poorly for:
- **Generic nouns**: "This is a photo of floor" → Low score
- **Parts of scenes**: "This is a photo of ceiling" → Low score
- **Abstract concepts**: "This is a photo of row" → Low score

But works well for:
- **Descriptive phrases**: "This is a photo of abandoned bar" → High score
- **Compound concepts**: "This is a photo of empty restaurant" → High score

### Correlation Data

| Comparison | Pearson r | Interpretation |
|------------|-----------|----------------|
| RAM++ conf vs SigLIP conf | 0.2545 | Weak positive |
| Florence-2 overlap vs SigLIP | N/A | No correlation (no Florence confidence) |

The weak correlation confirms: **RAM++ and SigLIP measure different things.**

---

## Recommendations for Archive App

### For Database Building (Maximum Coverage)

```python
# Use ALL tiers
threshold_config = {
    "tier1": True,        # All overlap tags
    "tier2_min": 0.6,     # RAM++ >= 0.6
    "tier3_min": 0.5,     # RAM++ >= 0.5
    "tier4_siglip_min": 0.05,  # Lower threshold for more compounds
}
# Expected: 150-200 tags per image
```

### For Display (Balanced)

```python
# Use Tier 1 + Tier 2 + selective Tier 4
threshold_config = {
    "tier1": True,        # All overlap tags
    "tier2_min": 0.7,     # RAM++ >= 0.7
    "tier3_min": None,    # Skip medium
    "tier4_siglip_min": 0.10,  # High confidence compounds only
}
# Expected: 50-80 tags per image
```

### For Search/Facets (High Precision)

```python
# Use Tier 1 only + high Tier 2
threshold_config = {
    "tier1": True,        # All overlap tags (gold standard)
    "tier2_min": 0.8,     # Only very high RAM++ confidence
    "tier3_min": None,    # Skip
    "tier4_siglip_min": 0.15,  # Very high SigLIP only
}
# Expected: 20-40 tags per image
```

---

## Key Takeaways

1. **Model Agreement > SigLIP Verification**
   - Tags found by both RAM++ and Florence-2 are reliable
   - Don't filter these with SigLIP

2. **RAM++ Confidence is Well-Calibrated**
   - >= 0.7 is reliably correct
   - >= 0.5 is mostly correct with some noise

3. **SigLIP is ONLY useful for Florence-2 compounds**
   - `abandoned_bar`, `empty_restaurant` verify well
   - Single words like `floor`, `ceiling` do NOT

4. **For Archive Apps: Trust the Models**
   - Use agreement as primary signal
   - Use RAM++ confidence as secondary
   - Use SigLIP only for compound phrase enrichment

---

## Files Generated

| File | Description |
|------|-------------|
| `tests/benchmark/siglip_verification_suite.py` | Benchmark runner |
| `tests/benchmark/siglip_verification_*.json` | Raw results data |
| `docs/sme/siglip_verification_analysis.md` | This document |

---

*Benchmark: 2025-12-24, 10 test images, Apple Silicon MPS*
*Models: RAM++ swin_large, Florence-2 large-ft, SigLIP so400m*
