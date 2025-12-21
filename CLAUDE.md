# IMLAGE

Image Machine Learning Aggregate — A universal CLI/GUI that compares visual tagging results from local ML tools.

## Three-Doc Stack

This project uses three core instruction files. Read them in order before any task:

| File | Purpose | Modify? |
|------|---------|---------|
| **CLAUDE.md** (this file) | Project rules, architecture, constraints, reference index | Never |
| **@techguide.md** | Implementation details, build setup, environment config, deep troubleshooting | Never |
| **@lilbits.md** | Script registry — every utility script with purpose, usage, line count | Never |

These three files are the complete instruction set. All other docs are reference material consulted on-demand.

**If any of these files are missing, empty, or unreadable: STOP and report to human. Do not proceed.**

## Quick Context

- **Mission**: Compare visual tagging results across local ML tools
- **Current**: Pre-release, foundation phase
- **Target**: CLI-first tool with optional web GUI for comparing ML tagger outputs
- **Persona**: Researcher/creator comparing which tagger works best for their images
- **Runtime**: Python >=3.11

## Core Principles

1. **CLI First, GUI Second** — CLI is the primary interface; GUI is a convenience layer
2. **Nothing Hardcoded** — Tools are plugins with their own settings; paths, models, options all configurable
3. **Flexible Input** — Single file, single folder, multi-folder; drag-drop supported in GUI
4. **Plugin Architecture** — Each ML tool is a plugin that can be installed, configured, updated, removed
5. **SME Files** — Each plugin gets a Subject Matter Expert file documenting its capabilities and quirks
6. **Performance Scaling** — Detect user hardware, adjust batch sizes and model variants accordingly
7. **Use Existing Tools** — Bundle proven binaries rather than reinvent; check hardware with established utilities

## Output Contract

All plugins return:
```json
{"tags": [{"label": "string", "confidence": 0.0}]}
```

CLI aggregates into:
```json
{"file": "path", "results": {"plugin_name": {"tags": [...]}}}
```

## Data Ownership

All processing local. No cloud APIs. Images never leave machine.

## Boot Sequence

1. Read this file (CLAUDE.md) completely
2. Read @techguide.md for implementation details
3. Read @lilbits.md for script registry
4. Read the task request
5. **Then** touch code — not before

## Commands

```bash
# Package management
uv sync                  # Install dependencies
uv run imlage --help     # Run CLI

# Development
uv run pytest            # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
```

> **Note**: Verify these commands match `pyproject.toml` scripts before relying on them.

## Development Rules

1. **Scope Discipline** — Only implement what the current request describes; no surprise features
2. **Plugin-First** — Every change must serve the plugin architecture or comparison workflows
3. **Prefer Open Source + Verify Licenses** — Default to open tools, log every dependency license
4. **Local-First** — Assume zero connectivity; no cloud ML services
5. **One Script = One Function** — Keep each script focused, under ~300 lines, recorded in lilbits.md
6. **No AI in Docs** — Never mention Claude, ChatGPT, Codex, or similar in user-facing docs or UI
7. **Keep It Simple** — Favor obvious code, minimal abstraction, fewer files
8. **Binary Dependencies Welcome** — App size is not a concern; bundle ML models and binaries freely
9. **Verify Build Before Done** — After any implementation work, run tests and confirm the CLI works

## Do Not

- Add cloud/remote ML services
- Hardcode plugin paths or model locations
- Process images without user action
- Store images outside user-specified locations
- Invent new features beyond what the task authorizes
- Add dependencies without logging licenses
- Mention AI assistants in UI, user docs, or exports
- Leave TODOs or unexplained generated code in production branches
- **Modify or remove core instruction files** — CLAUDE.md, techguide.md, and lilbits.md are protected
- **Assume when uncertain** — If a task is ambiguous or conflicts with these rules, stop and ask

## Stop and Ask When

- Adding a new plugin
- Changing output schema
- Task requires modifying CLAUDE.md, techguide.md, or lilbits.md
- Task conflicts with a rule in this file
- Referenced file or path doesn't exist
- Task scope is unclear or seems to exceed "one feature"
- You're about to delete code without understanding why it exists

## Critical Gotchas

| Gotcha | Details |
|--------|---------|
| **Plugin isolation** | Each plugin runs in its own subprocess to prevent model conflicts |
| **Output normalization** | All plugins MUST return the standard output contract format |
| **Hardware detection** | Run once at startup, cache results, let user override |
| **Model files** | Large model files live outside git; document download/setup in plugin SME |

## File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Python modules | snake_case | `tag_service.py` |
| CLI commands | kebab-case | `imlage tag-image` |
| Plugins | snake_case directory | `plugins/ram_plus/` |
| SME files | plugin name + .sme.md | `ram_plus.sme.md` |
| Tests | test_ prefix | `test_tag_service.py` |

## Change Protocols

| Change Type | Required Steps |
|-------------|----------------|
| New plugin | Create plugin dir, implement interface, add SME file |
| Schema change | Update output contract in this file, update all plugins |
| New dependency | Log license in commit message; verify offline functionality |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Plugin not found | Check `plugins/` directory and `plugin.toml` config |
| Model missing | Run plugin setup command, check SME file for download instructions |
| Hardware not detected | Check `~/.imlage/hardware.json`, delete to re-detect |

## Contact Surface

All prompts funnel through this CLAUDE.md. Do not copy instructions elsewhere.
