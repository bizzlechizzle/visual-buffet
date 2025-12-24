# RAM++ Threshold Quality Benchmark

Benchmark suite to measure tag quality at different threshold levels.

## Critical Insight

**Resolution does NOT matter for RAM++.** The model internally resizes all images to 384×384 regardless of input size. Testing proved this empirically:

| Size | Avg Time | Avg Tags | Avg Confidence |
|------|----------|----------|----------------|
| 480px | 55ms | 139 | 0.606 |
| 1080px | 60ms | 133 | 0.607 |
| 2048px | 72ms | 133 | 0.606 |

**Same tags, same confidence, just slower.** Only **threshold** matters for output quality.

## Threshold Test Matrix

All tests use 480px (little) with **UNCAPPED** output to measure true tag quality:

| Threshold | Description | Use Case |
|-----------|-------------|----------|
| 0.4 | Maximum discovery | Capture everything possible |
| 0.5 | Comprehensive coverage | Thorough tagging |
| **0.6** | **Balanced (recommended)** | **Default for most use cases** |
| 0.7 | High confidence only | Fewer, more confident tags |
| 0.8 | Very selective | Top tags only |

## Running the Benchmark

```bash
# From project root, with venv activated
source .venv/bin/activate && python tests/benchmark/ram_plus_settings_test.py

# Or with uv
uv run python tests/benchmark/ram_plus_settings_test.py
```

## Test Images

Uses 10 test images in `images/` (urban exploration photography):

| Image | Subject | Key Expected Tags |
|-------|---------|-------------------|
| testimage01.jpg | Abandoned diner | bar, stool, floor, ceiling fan, restaurant, counter |
| testimage02.jpg | Abandoned dry cleaner | interior, room, window, clothing, hanger |
| testimage03.jpg | Abandoned church | church, cross, chapel, bench, row |
| testimage04.jpg | Abandoned house interior | room, floor, window, wall, door |
| testimage05.jpg | Industrial building | building, factory, window, brick |
| testimage06.jpg | Abandoned factory | industrial, machine, factory, window |
| testimage07.jpg | Building exterior | building, sky, architecture, window |
| testimage08.jpg | Interior with peeling paint | wall, paint, decay, room, floor |
| testimage09.jpg | Parking lot with cars | car, parking lot, tree, vehicle, forest |
| testimage10.jpg | Factory by river | building, factory, water, river, chimney |

## Metrics Collected

- **Number of tags**: Tags returned per image (uncapped)
- **Tag range**: Min/max tags across images
- **Average confidence**: Mean confidence score at each threshold
- **Top tags**: First 10 tags per image (for quality review)

## Expected Results

Based on empirical testing:

| Threshold | Expected Tags | Avg Confidence |
|-----------|---------------|----------------|
| 0.4 | 150-200 | ~0.58 |
| 0.5 | 130-175 | ~0.60 |
| 0.6 | 100-150 | ~0.65 |
| 0.7 | 60-100 | ~0.72 |
| 0.8 | 30-60 | ~0.82 |

## Output

Results saved to `ram_plus_threshold_quality_YYYYMMDD_HHMMSS.json`

## Related Documentation

- **SME Document**: `/docs/sme/ram_plus.sme.md`
- **Settings Reference**: `/docs/reference/ram_plus_settings.md`
- **Test Schema**: `/tests/fixtures/schemas/test_image_expectations.json`
