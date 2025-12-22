# Grid + Slide Panel UI Design Plan

> **Status**: DRAFT - Awaiting approval
> **Created**: 2025-12-22
> **Updated**: 2025-12-22
> **Design System**: Braun/Ulm (Functional Minimalism)

---

## Executive Summary

Redesign the image interface to use a **Grid + Slide Panel** pattern:

| Layer | Purpose | Content |
|-------|---------|---------|
| **Grid** | Browse | Simplified cards: thumbnail + filename + status |
| **Panel** | Inspect | Full tag breakdown with all details |
| **Lightbox** | View | Full-size image only |

This follows Braun's master-detail pattern: **browse → inspect → view**.

---

## Part 1: Backend — Boosted Confidence

### 1.1 Add boost_confidence() Function

**File**: `src/visual_buffet/plugins/schemas.py`

```python
import math

def boost_confidence(
    raw_confidence: float,
    sources: int,
    boost_per_source: float = 0.15
) -> float:
    """
    Boost confidence in log-odds space for multi-resolution agreement.

    Args:
        raw_confidence: Original sigmoid probability (0-1)
        sources: Number of resolutions that detected this tag
        boost_per_source: Log-odds boost per additional source (default 0.15)

    Returns:
        Boosted confidence (0-1), never exceeds 1.0

    Example:
        >>> boost_confidence(0.70, sources=5)
        0.8175  # ~82%
    """
    if sources <= 1 or raw_confidence <= 0 or raw_confidence >= 1:
        return raw_confidence

    # Convert to log-odds (logit)
    log_odds = math.log(raw_confidence / (1 - raw_confidence))

    # Add boost per additional source
    boosted_log_odds = log_odds + (sources - 1) * boost_per_source

    # Convert back to probability (sigmoid)
    return 1 / (1 + math.exp(-boosted_log_odds))
```

### 1.2 Update MergedTag Dataclass

**File**: `src/visual_buffet/plugins/schemas.py`

```python
@dataclass
class MergedTag:
    """A tag with metadata from multi-resolution merging."""

    label: str
    raw_confidence: float | None = None       # Original max confidence
    boosted_confidence: float | None = None   # After multi-res boost
    sources: int = 1                          # Resolutions that found this tag
    max_sources: int = 1                      # Total resolutions used
    min_resolution: int | None = None         # Smallest res where found

    @property
    def confidence(self) -> float | None:
        """Return boosted confidence if available, else raw."""
        return self.boosted_confidence or self.raw_confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"label": self.label}
        if self.boosted_confidence is not None:
            result["confidence"] = round(self.boosted_confidence, 4)
            result["raw_confidence"] = round(self.raw_confidence, 4)
        elif self.raw_confidence is not None:
            result["confidence"] = round(self.raw_confidence, 4)
        if self.sources > 1:
            result["sources"] = self.sources
            result["max_sources"] = self.max_sources
        return result
```

### 1.3 Update merge_tags() Function

```python
def merge_tags(
    all_tags: list[Tag],
    resolutions_used: list[int] | None = None,
    boost_per_source: float = 0.15
) -> list[MergedTag]:
    """Merge tags from multiple resolutions with boosted confidence."""
    merged: dict[str, MergedTag] = {}
    max_sources = len(resolutions_used) if resolutions_used else 1

    for tag in all_tags:
        label = tag.label.lower().strip()
        if not label:
            continue

        if label not in merged:
            merged[label] = MergedTag(
                label=label,
                raw_confidence=tag.confidence,
                sources=1,
                max_sources=max_sources,
            )
        else:
            merged[label].sources += 1
            # Keep highest raw confidence
            if tag.confidence is not None:
                if merged[label].raw_confidence is None:
                    merged[label].raw_confidence = tag.confidence
                else:
                    merged[label].raw_confidence = max(
                        merged[label].raw_confidence, tag.confidence
                    )

    # Calculate boosted confidence for all tags
    for tag in merged.values():
        if tag.raw_confidence is not None:
            tag.boosted_confidence = boost_confidence(
                tag.raw_confidence,
                tag.sources,
                boost_per_source
            )

    # Sort by boosted confidence (desc), then sources (desc), then label
    result = sorted(
        merged.values(),
        key=lambda t: (-(t.boosted_confidence or 0), -t.sources, t.label),
    )

    return result
```

### 1.4 Expected Boost Values

| Raw | Sources | Boosted | Change |
|-----|---------|---------|--------|
| 0.50 | 1/5 | 0.50 | — |
| 0.50 | 3/5 | 0.57 | +7% |
| 0.50 | 5/5 | 0.65 | +15% |
| 0.70 | 1/5 | 0.70 | — |
| 0.70 | 3/5 | 0.76 | +6% |
| 0.70 | 5/5 | 0.82 | +12% |
| 0.85 | 1/5 | 0.85 | — |
| 0.85 | 3/5 | 0.89 | +4% |
| 0.85 | 5/5 | 0.92 | +7% |
| 0.95 | 5/5 | 0.97 | +2% |

---

## Part 2: API Response Shape

### 2.1 Current Response (Reference)

```json
{
  "results": {
    "ram_plus": {
      "tags": [
        {"label": "sunset", "confidence": 0.92}
      ],
      "model": "ram_plus_swin_large",
      "inference_time_ms": 142
    }
  }
}
```

### 2.2 New Response Shape

```json
{
  "results": {
    "ram_plus": {
      "tags": [
        {
          "label": "sunset",
          "confidence": 0.92,
          "raw_confidence": 0.87,
          "sources": 5,
          "max_sources": 5
        },
        {
          "label": "ocean",
          "confidence": 0.89,
          "raw_confidence": 0.85,
          "sources": 4,
          "max_sources": 5
        },
        {
          "label": "person",
          "confidence": 0.71,
          "sources": 1,
          "max_sources": 5
        }
      ],
      "model": "ram_plus_swin_large",
      "version": "1.3.0",
      "inference_time_ms": 142,
      "quality": "max",
      "resolutions_used": [480, 1080, 2048, 4096, 0]
    }
  }
}
```

### 2.3 API Endpoints Updates

| Endpoint | Change |
|----------|--------|
| `GET /api/image/{id}/meta` | Include full tag results with boosted scores |
| `POST /api/tag/{id}` | Return boosted scores in response |
| `GET /api/images` | Include summary: `tagged: true`, `tag_count: 23` |

---

## Part 3: Frontend State

### 3.1 State Shape

```javascript
const state = {
    images: new Map(),          // id -> ImageData
    selectedImageId: null,      // Currently selected card
    panelOpen: false,           // Is detail panel visible
    panelLoading: false,        // Is panel content loading
    settings: { ... },
    hardware: null,
    processing: false,
};

// ImageData shape
{
    id: "abc123",
    filename: "IMG_2847.jpg",
    thumbnail: "/api/thumb/abc123",
    width: 4032,
    height: 3024,
    format: "JPEG",
    filesize: 12400000,
    tagged: true,
    tagCount: 23,               // For grid display
    results: null,              // Full results loaded on demand
    processing: false,
}
```

### 3.2 State Transitions

```
┌─────────────────────────────────────────────────────────────┐
│                        GRID STATE                           │
│  selectedImageId: null, panelOpen: false                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Click card
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PANEL LOADING STATE                       │
│  selectedImageId: "abc", panelOpen: true, panelLoading: true│
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Fetch complete
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      PANEL OPEN STATE                       │
│  selectedImageId: "abc", panelOpen: true, panelLoading: false│
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Escape / Click outside / X
                              ▼
                        Back to GRID STATE
```

---

## Part 4: Grid Cards (Simplified)

### 4.1 Card Anatomy

```
┌─────────────────────────────────┐
│                                 │
│         THUMBNAIL               │
│           (1:1)                 │
│                                 │
├─────────────────────────────────┤
│ IMG_2847.jpg              ✓ 23 │  ← Filename + status + count
└─────────────────────────────────┘
```

### 4.2 Card States

| State | Visual |
|-------|--------|
| **Unprocessed** | No badge, dimmed status area |
| **Processing** | Spinner icon, "Processing..." text |
| **Tagged** | ✓ checkmark + tag count (e.g., "23") |
| **Selected** | Dark border, subtle shadow |
| **Error** | ⚠ warning icon, red tint |

### 4.3 Card CSS

```css
.image-card {
    position: relative;
    background: #FFFFFF;
    border: 1px solid #E2E1DE;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 150ms ease, box-shadow 150ms ease;
}

.image-card:hover {
    border-color: #C0BFBC;
}

.image-card.selected {
    border-color: #1C1C1A;
    box-shadow: 0 0 0 1px #1C1C1A;
}

.image-card-thumb {
    aspect-ratio: 1;
    width: 100%;
    object-fit: cover;
}

.image-card-info {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px;
    border-top: 1px solid #E2E1DE;
}

.image-card-name {
    flex: 1;
    font-size: 13px;
    font-weight: 400;
    color: #1C1C1A;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.image-card-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #8A8A86;
}

.image-card-status.tagged {
    color: #22c55e;
}
```

### 4.4 Grid CSS

```css
.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
    padding: 24px;
    /* When panel open, grid shrinks */
    transition: padding-right 200ms ease;
}

.image-grid.panel-open {
    padding-right: calc(400px + 24px);  /* Panel width + gap */
}

@media (max-width: 900px) {
    .image-grid.panel-open {
        padding-right: 24px;  /* Panel overlays on mobile */
    }
}
```

---

## Part 5: Detail Panel

### 5.1 Panel Anatomy

```
┌─────────────────────────────────────────┐
│ ←  IMG_2847.jpg                     ✕   │  ← Header (56px)
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │        PREVIEW THUMBNAIL            │ │  ← Click for lightbox
│ │           (16:9 crop)               │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ 4032 × 3024  •  JPEG  •  12.4 MB        │  ← Metadata
│                                         │
├─────────────────────────────────────────┤
│ RAM++                        23 tags    │  ← Plugin header
├─────────────────────────────────────────┤
│ sunset                  92%  ●●●●● 5/5  │
│ ocean                   89%  ●●●●○ 4/5  │
│ beach                   85%  ●●●●○ 4/5  │
│ person                  71%  ●●●○○ 2/5  │
│ sky                     68%  ●●●○○ 2/5  │
│ clouds                  65%  ●●○○○ 1/5  │
│ sand                    61%  ●●○○○ 1/5  │
│ waves                   58%  ●●○○○ 1/5  │
│ ... (scrollable)                        │
│                                         │
├─────────────────────────────────────────┤  ← If multiple plugins
│ SigLIP                       12 tags    │
├─────────────────────────────────────────┤
│ golden hour             94%  ●●●●●      │
│ warm tones              88%  ●●●●○      │
│ ...                                     │
│                                         │
├─────────────────────────────────────────┤
│ [  View Full Size  ] [  Re-tag  ]       │  ← Actions (56px)
└─────────────────────────────────────────┘
```

### 5.2 Panel CSS

```css
.detail-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 400px;
    height: 100vh;
    background: #FFFFFF;
    border-left: 1px solid #E2E1DE;
    box-shadow: -4px 0 16px rgba(0, 0, 0, 0.06);
    z-index: 100;
    display: flex;
    flex-direction: column;

    /* Animation */
    transform: translateX(100%);
    transition: transform 200ms ease;
}

.detail-panel.open {
    transform: translateX(0);
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    padding: 0 16px;
    border-bottom: 1px solid #E2E1DE;
    flex-shrink: 0;
}

.panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

.panel-actions {
    display: flex;
    gap: 8px;
    padding: 16px;
    border-top: 1px solid #E2E1DE;
    flex-shrink: 0;
}
```

### 5.3 Tag Row CSS

```css
.tag-row {
    display: flex;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #EEEEED;
}

.tag-label {
    flex: 1;
    font-size: 15px;
    font-weight: 400;
    color: #1C1C1A;
}

.tag-confidence {
    width: 40px;
    text-align: right;
    font-size: 13px;
    font-weight: 500;
    margin-right: 8px;
}

.tag-confidence.high   { color: #22c55e; }  /* 80%+ */
.tag-confidence.medium { color: #5C5C58; }  /* 60-79% */
.tag-confidence.low    { color: #8A8A86; }  /* <60% */

.tag-dots {
    display: flex;
    gap: 2px;
    margin-right: 8px;
}

.tag-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #E2E1DE;
}

.tag-dot.filled {
    background: #1C1C1A;
}

.tag-sources {
    width: 32px;
    text-align: right;
    font-size: 11px;
    font-weight: 300;
    color: #8A8A86;
}
```

### 5.4 Multi-Plugin Display

**Decision**: Stacked sections with headers (not tabs, not accordion)

```css
.plugin-section {
    margin-bottom: 24px;
}

.plugin-section:last-child {
    margin-bottom: 0;
}

.plugin-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: #F4F4F2;
    border-radius: 4px;
    margin-bottom: 8px;
}

.plugin-name {
    font-size: 13px;
    font-weight: 500;
    color: #1C1C1A;
}

.plugin-count {
    font-size: 11px;
    color: #8A8A86;
}
```

---

## Part 6: Empty, Loading, and Error States

### 6.1 Empty States

| Context | Message | Action |
|---------|---------|--------|
| No images | "Drop images here to get started" | Drop zone visible |
| No tags (unprocessed) | "Click 'Tag Image' to analyze" | Tag button |
| No tags (processed, 0 results) | "No tags found" | Re-tag button |
| Panel loading | Skeleton rows | None |

### 6.2 Loading States

**Grid card processing:**
```html
<div class="image-card-status processing">
    <svg class="spinner">...</svg>
</div>
```

**Panel loading:**
```html
<div class="panel-body">
    <div class="skeleton-preview"></div>
    <div class="skeleton-row"></div>
    <div class="skeleton-row"></div>
    <div class="skeleton-row"></div>
</div>
```

**Skeleton CSS (static, not animated per Braun):**
```css
.skeleton-row {
    height: 32px;
    background: #F4F4F2;
    border-radius: 4px;
    margin-bottom: 8px;
}

.skeleton-preview {
    aspect-ratio: 16/9;
    background: #F4F4F2;
    border-radius: 8px;
    margin-bottom: 16px;
}
```

### 6.3 Error States

**Tagging failed:**
```html
<div class="plugin-section error">
    <div class="plugin-section-header">
        <span class="plugin-name">RAM++</span>
        <span class="plugin-error">Failed</span>
    </div>
    <p class="error-message">Model not available. Check GPU memory.</p>
    <button class="btn btn-secondary">Retry</button>
</div>
```

**Error CSS:**
```css
.plugin-error {
    font-size: 11px;
    font-weight: 500;
    color: #ef4444;
}

.error-message {
    font-size: 13px;
    color: #5C5C58;
    padding: 8px 0;
}
```

---

## Part 7: Interaction & Keyboard Navigation

### 7.1 Keyboard Shortcuts

| Key | Context | Action |
|-----|---------|--------|
| `Arrow keys` | Grid focused | Move selection |
| `Enter` | Card selected | Open panel (if closed) or lightbox |
| `Space` | Card selected | Tag image |
| `Escape` | Panel open | Close panel |
| `Escape` | Lightbox open | Close lightbox |
| `Tab` | Any | Move focus |
| `L` | Card selected | Open lightbox directly |

### 7.2 Focus Management

```javascript
function openPanel(imageId) {
    state.selectedImageId = imageId;
    state.panelOpen = true;

    // Move focus to panel close button for accessibility
    requestAnimationFrame(() => {
        document.querySelector('.panel-close-btn').focus();
    });
}

function closePanel() {
    const previouslySelected = state.selectedImageId;
    state.panelOpen = false;

    // Return focus to the card that was selected
    requestAnimationFrame(() => {
        document.querySelector(`[data-id="${previouslySelected}"]`).focus();
    });
}
```

### 7.3 Click Behavior

| Target | Action |
|--------|--------|
| Card (panel closed) | Select card, open panel |
| Card (panel open, same card) | No change |
| Card (panel open, different card) | Change selection, update panel |
| Panel overlay (mobile) | Close panel |
| Panel X button | Close panel |
| Panel thumbnail | Open lightbox |
| Panel "View Full Size" | Open lightbox |
| Tag row | Nothing (future: filter) |

### 7.4 Tag Click Behavior

**Decision**: No action for now. Future options:

| Option | Complexity | Value |
|--------|------------|-------|
| Nothing | None | Safe default |
| Copy tag | Low | Useful for search |
| Filter grid | Medium | Power user feature |
| Show similar | High | Requires search index |

Recommended: **Copy tag on click** (low effort, clear value)

---

## Part 8: Responsive & Mobile

### 8.1 Breakpoints

| Width | Grid | Panel |
|-------|------|-------|
| > 1200px | Shrinks to accommodate panel | 400px, slides in |
| 900-1200px | Full width | 400px, overlays with backdrop |
| < 900px | Full width | Full width, slides up from bottom |

### 8.2 Mobile Panel (Bottom Sheet)

```css
@media (max-width: 900px) {
    .detail-panel {
        top: auto;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        height: 70vh;
        border-left: none;
        border-top: 1px solid #E2E1DE;
        border-radius: 16px 16px 0 0;

        transform: translateY(100%);
    }

    .detail-panel.open {
        transform: translateY(0);
    }
}
```

### 8.3 Touch Gestures

| Gesture | Action |
|---------|--------|
| Tap card | Open panel |
| Swipe panel down | Close panel |
| Swipe left/right on grid | Navigate cards |

---

## Part 9: Dark Mode

### 9.1 Panel Colors (Dark)

```css
@media (prefers-color-scheme: dark) {
    .detail-panel {
        background: #111113;
        border-left-color: #2D2D30;
    }

    .panel-header {
        border-bottom-color: #2D2D30;
    }

    .tag-row {
        border-bottom-color: #1f1f23;
    }

    .tag-label {
        color: rgba(250, 250, 248, 0.87);
    }

    .tag-confidence.high { color: #22c55e; }
    .tag-confidence.medium { color: rgba(250, 250, 248, 0.60); }
    .tag-confidence.low { color: rgba(250, 250, 248, 0.38); }

    .tag-dot {
        background: #3A3A3E;
    }

    .tag-dot.filled {
        background: rgba(250, 250, 248, 0.87);
    }

    .plugin-section-header {
        background: #1A1A1C;
    }

    .skeleton-row,
    .skeleton-preview {
        background: #1A1A1C;
    }
}
```

---

## Part 10: Accessibility

### 10.1 ARIA Labels

```html
<!-- Grid -->
<div class="image-grid" role="grid" aria-label="Image gallery">
    <div class="image-card"
         role="gridcell"
         tabindex="0"
         aria-selected="false"
         aria-label="IMG_2847.jpg, tagged with 23 labels">
        ...
    </div>
</div>

<!-- Panel -->
<aside class="detail-panel"
       role="complementary"
       aria-label="Image details"
       aria-hidden="true">
    <header class="panel-header">
        <button class="panel-back" aria-label="Close panel">←</button>
        <h2 id="panel-title">IMG_2847.jpg</h2>
        <button class="panel-close" aria-label="Close panel">✕</button>
    </header>
    <div class="panel-body" aria-labelledby="panel-title">
        ...
    </div>
</aside>
```

### 10.2 Contrast Requirements

| Element | Foreground | Background | Ratio | WCAG |
|---------|------------|------------|-------|------|
| Primary text | #1C1C1A | #FFFFFF | 16.1:1 | AAA |
| Secondary text | #5C5C58 | #FFFFFF | 5.7:1 | AA |
| Tertiary text | #8A8A86 | #FFFFFF | 3.5:1 | Fail* |
| High confidence | #22c55e | #FFFFFF | 3.0:1 | Fail* |

*Tertiary and functional colors are supplementary, not sole indicators.

---

## Part 11: URL Routing (Optional)

### 11.1 URL Structure

| URL | State |
|-----|-------|
| `/` | Grid only, no selection |
| `/image/abc123` | Grid + panel for image abc123 |
| `/image/abc123/full` | Lightbox for image abc123 |

### 11.2 History Management

```javascript
function openPanel(imageId) {
    state.selectedImageId = imageId;
    state.panelOpen = true;
    history.pushState({ imageId }, '', `/image/${imageId}`);
}

function closePanel() {
    state.panelOpen = false;
    state.selectedImageId = null;
    history.pushState({}, '', '/');
}

window.addEventListener('popstate', (e) => {
    if (e.state?.imageId) {
        openPanel(e.state.imageId);
    } else {
        closePanel();
    }
});
```

---

## Part 12: Animation Timing

### 12.1 Transitions

| Element | Duration | Easing |
|---------|----------|--------|
| Panel slide | 200ms | ease |
| Card hover | 150ms | ease |
| Card selection | 150ms | ease |
| Lightbox fade | 200ms | ease |
| Focus ring | 0ms | none |

### 12.2 CSS Variables

```css
:root {
    --transition-fast: 150ms ease;
    --transition-base: 200ms ease;
    --transition-slow: 300ms ease;
}
```

---

## Part 13: Implementation Phases

### Phase 1: Backend (Boosted Confidence)
- [ ] Add `boost_confidence()` function
- [ ] Update `MergedTag` dataclass
- [ ] Update `merge_tags()` function
- [ ] Update API responses
- [ ] Update SME documentation
- [ ] Add unit tests for boosting logic

### Phase 2: Panel UI
- [ ] Add panel HTML structure
- [ ] Add panel CSS (light mode)
- [ ] Add panel open/close logic
- [ ] Add keyboard navigation (Escape)
- [ ] Wire up card click → panel open
- [ ] Fetch and display tag data in panel

### Phase 3: Card Simplification
- [ ] Remove tag chips from cards
- [ ] Add tag count badge
- [ ] Add selected state styling
- [ ] Update grid shrinking when panel open

### Phase 4: Multi-Plugin Display
- [ ] Stacked plugin sections in panel
- [ ] Plugin headers with counts
- [ ] Error states per plugin
- [ ] Handle discovery mode results

### Phase 5: Polish
- [ ] Loading skeletons
- [ ] Empty states
- [ ] Dark mode colors
- [ ] Mobile/responsive (bottom sheet)
- [ ] Touch gestures (swipe to close)

### Phase 6: Accessibility & Routing
- [ ] ARIA labels
- [ ] Focus management
- [ ] Full keyboard navigation
- [ ] URL routing (optional)

---

## Decisions Summary

| Question | Decision |
|----------|----------|
| Boost algorithm | Log-odds, 0.15 per source |
| Panel width | 400px |
| Panel position | Right slide (desktop), bottom sheet (mobile) |
| Multi-plugin display | Stacked sections |
| Tag click action | Copy to clipboard |
| Confidence visualization | Percentage + dots + sources |
| URL routing | Phase 6 (optional) |
| Card content | Thumbnail + filename + status + count |

---

## Approval Checklist

- [ ] Backend boost algorithm
- [ ] API response shape
- [ ] Panel anatomy
- [ ] Card simplification
- [ ] Multi-plugin stacked sections
- [ ] Mobile bottom sheet
- [ ] Implementation phase order
