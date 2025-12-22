# GUI vs CLI/Engine Parity Audit

> **Audit Date**: 2025-12-21
> **Auditor**: Claude (audit)
> **Scope**: Compare GUI capabilities against CLI/Engine features

---

## Executive Summary

**Overall Parity: 70%** - GUI is missing pipeline mode and has a bug in batch endpoint

| Component | Status | Notes |
|-----------|--------|-------|
| Per-plugin settings | **GOOD** | threshold, limit, quality all work |
| Standard tagging | **GOOD** | Single image works correctly |
| Batch tagging | **BUG** | Wrong function signature |
| Pipeline mode | **MISSING** | Not implemented |
| Hardware detection | **GOOD** | Works |
| Settings persistence | **GOOD** | Saves to config file |

---

## Critical Issues

### 1. Pipeline Mode Not Implemented

**Severity: HIGH**

The new pipeline feature (`--pipeline` in CLI) is completely missing from GUI.

**Missing Backend:**
- No `/api/tag-pipeline/{image_id}` endpoint
- Engine has `tag_image_pipeline()` but server doesn't use it

**Missing Frontend:**
- No pipeline toggle in settings
- No pipeline results display (candidate_sources, vocabulary_size, etc.)

**Impact:** Users cannot use the best tagging mode (RAM++ + Florence-2 → SigLIP) from the GUI.

---

### 2. Batch Endpoint Bug

**Severity: MEDIUM**

`server.py` line 531:
```python
result = await tag_image(image_id, plugins, threshold, limit)
```

**Problem:** `tag_image()` expects `TagRequest` object, not positional args.

**Fix needed:**
```python
request = TagRequest(plugins=plugins, plugin_configs=...)
result = await tag_image(image_id, request)
```

---

## Feature Comparison Matrix

| Feature | CLI | Engine | GUI Server | GUI Frontend |
|---------|-----|--------|------------|--------------|
| Tag single image | `tag photo.jpg` | `tag_image()` | `/api/tag/{id}` | Tag button |
| Tag batch | `tag *.jpg` | `tag_batch()` | `/api/tag-batch` (buggy) | Auto-tag queue |
| **Pipeline mode** | `--pipeline` | `tag_image_pipeline()` | **MISSING** | **MISSING** |
| Plugin selection | `-p plugin` | `plugin_names=[]` | `plugins` field | Plugin toggles |
| Threshold | `--threshold 0.5` | `threshold=0.5` | `plugin_configs` | Slider per plugin |
| Limit | `--limit 50` | `limit=50` | `plugin_configs` | Slider per plugin |
| Quality | (via config) | `quality='standard'` | `plugin_configs` | Dropdown per plugin |
| Recursive folder | `--recursive` | N/A (CLI only) | N/A (upload) | Folder upload |
| Output file | `-o results.json` | N/A | N/A | N/A (in-memory) |

---

## GUI-Specific Features (Not in CLI)

| Feature | Notes |
|---------|-------|
| Image grid view | Visual thumbnail grid |
| Lightbox preview | Full image with tags |
| Drag & drop upload | Folders and files |
| Settings persistence | Auto-saves to config |
| Hardware display | Shows GPU/CPU info |
| RAW file preview | Converts to JPEG for display |

---

## Recommendations

### Must Fix (Critical)

1. **Add Pipeline Mode to GUI**

   **Backend** - Add endpoint to `server.py`:
   ```python
   @app.post("/api/tag-pipeline/{image_id}")
   async def tag_image_pipeline_api(image_id: str, request: TagRequest | None = None):
       # Similar to tag_image but calls engine.tag_image_pipeline()
   ```

   **Frontend** - Add to `app.js`:
   - Pipeline toggle in settings modal
   - Pipeline results display showing:
     - Candidate sources (ram_plus, florence_2)
     - Candidate count (e.g., "90 candidates")
     - Vocabulary size scored
     - Source results summary

2. **Fix Batch Endpoint Bug**

   Replace line 531 in `server.py`:
   ```python
   # Old (broken):
   result = await tag_image(image_id, plugins, threshold, limit)

   # New (fixed):
   request = TagRequest(
       plugins=plugins,
       plugin_configs={p: PluginConfig(threshold=threshold, limit=limit) for p in plugins} if plugins else None
   )
   result = await tag_image(image_id, request)
   ```

### Should Fix (Important)

3. **Add pipeline toggle to settings UI**
   ```html
   <div class="setting">
       <label>Tagging Mode</label>
       <select id="taggingMode">
           <option value="standard">Standard (all plugins independently)</option>
           <option value="pipeline">Pipeline (RAM++ + Florence → SigLIP)</option>
       </select>
   </div>
   ```

4. **Display pipeline-specific results in lightbox**
   - Show "Candidates: 90 from ram_plus, florence_2"
   - Show source plugin results in collapsible section
   - Highlight that confidence scores are from SigLIP validation

### Consider (Nice to Have)

5. Add export to JSON button in lightbox
6. Add "Re-tag with Pipeline" option for images already tagged
7. Show timing breakdown (RAM++ time + Florence time + SigLIP time)

---

## Files to Modify

| File | Changes Needed |
|------|----------------|
| `src/visual_buffet/gui/server.py` | Add `/api/tag-pipeline/{image_id}`, fix batch bug |
| `src/visual_buffet/gui/static/app.js` | Add pipeline toggle, pipeline results display |
| `src/visual_buffet/gui/static/index.html` | Add pipeline mode selector to settings |
| `src/visual_buffet/gui/static/styles.css` | Style pipeline results section |

---

## Audit Metadata

### Files Reviewed
- `src/visual_buffet/gui/server.py` (598 lines)
- `src/visual_buffet/gui/static/app.js` (893 lines)
- `src/visual_buffet/gui/static/index.html` (135 lines)
- `src/visual_buffet/core/engine.py` (compared)
- `src/visual_buffet/cli.py` (compared)

### Confidence: HIGH
- Clear feature comparison
- Bug identified with exact line number
- Recommendations are specific and actionable
