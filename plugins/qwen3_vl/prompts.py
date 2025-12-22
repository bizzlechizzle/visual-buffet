"""Built-in prompt templates for Qwen3-VL plugin.

These prompts are optimized for extracting tags and descriptions
from vision-language models via Ollama.

NOTE: Qwen3-VL has a prompt length sensitivity issue where prompts
over ~200 characters may return empty responses. All prompts here
are kept short, with quality differentiated by generation parameters.
"""

# Universal tagging prompt - works reliably with Qwen3-VL
# Quality is controlled by generation params (max_tokens, temperature)
TAGGING_PROMPT = """Return JSON with image tags: {"tags": [...], "description": "..."}
Include: objects, colors, textures, setting, mood, activities."""

# Description-focused prompt (kept short)
DESCRIBE_PROMPT = """Describe this image in detail. Cover main subjects, colors, setting, mood, and composition."""

# Aliases for backward compatibility
TAGGING_PROMPT_QUICK = TAGGING_PROMPT
TAGGING_PROMPT_MAX = TAGGING_PROMPT

# All quality levels use same prompt - quality from generation params
PROMPTS_BY_QUALITY = {
    "quick": TAGGING_PROMPT,
    "standard": TAGGING_PROMPT,
    "max": TAGGING_PROMPT,
}

# Generation settings by quality level
# Quality is primarily controlled here, not by prompt length
GENERATION_SETTINGS = {
    "quick": {
        "max_tokens": 384,
        "temperature": 0.1,
    },
    "standard": {
        "max_tokens": 768,
        "temperature": 0.15,
    },
    "max": {
        "max_tokens": 1536,
        "temperature": 0.2,
    },
}
