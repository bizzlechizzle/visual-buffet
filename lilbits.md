# IMLAGE Script Registry (lilbits)

Every utility script with purpose, usage, and line count. Keep scripts focused and under ~300 lines.

## Registry

| Script | Purpose | Lines | Usage |
|--------|---------|-------|-------|
| `scripts/generate_variants.py` | Generate test image variants at different resolutions/formats | 199 | `python scripts/generate_variants.py images/ output/` |
| `scripts/analyze_results.py` | Analyze tagging results, calculate stability metrics, generate master tags | 402 | `python scripts/analyze_results.py test-results/` |
| `scripts/run_quality_test.py` | Run full quality test pipeline (generate, tag, analyze) | 147 | `python scripts/run_quality_test.py images/ test-results/` |

## Adding a Script

When adding a new utility script:

1. Create script in appropriate location (`scripts/`, `src/imlage/utils/`, etc.)
2. Add entry to this registry with:
   - Script name/path
   - One-line purpose
   - Line count
   - Usage example
3. Keep under 300 lines; split if larger

## Template Entry

```markdown
| `scripts/example.py` | Brief description of what it does | 45 | `python scripts/example.py --arg value` |
```

## Guidelines

- **One function per script** — If it does two things, make two scripts
- **Clear naming** — Name should indicate purpose (`download_models.py`, not `util.py`)
- **Document in registry** — If it's not here, it doesn't exist
- **Executable** — Scripts should be runnable standalone with clear CLI args
