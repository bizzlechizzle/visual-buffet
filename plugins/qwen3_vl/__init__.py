"""Qwen3-VL Vision-Language Model plugin.

Connects to Ollama to run Qwen3-VL (or other vision models) for
image tagging and description generation.

Supports multiple modes:
- tagging: Returns structured tags (JSON parsed)
- describe: Returns detailed description
- custom: Uses user-defined prompt
"""

import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

# Add src to path for imports during plugin loading
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from visual_buffet.exceptions import ModelNotFoundError, PluginError  # noqa: E402
from visual_buffet.plugins.base import PluginBase  # noqa: E402
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult  # noqa: E402

from .prompts import (  # noqa: E402
    DESCRIBE_PROMPT,
    GENERATION_SETTINGS,
    PROMPTS_BY_QUALITY,
    TAGGING_PROMPT,
)

PLUGIN_VERSION = "1.0.0"

# Default configuration
DEFAULT_CONFIG = {
    # Connection
    "server_url": "http://localhost:11434",
    "model_name": "qwen3-vl:latest",
    "timeout_seconds": 120,
    "keep_alive": "5m",
    # Generation
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "mode": "tagging",
    "custom_prompt": "",
    # Image processing
    "max_size": 1280,
    "jpeg_quality": 90,
    # Output
    "include_description": True,
    "expand_compounds": True,
    "min_tag_length": 2,
}


class Qwen3VLPlugin(PluginBase):
    """Qwen3-VL vision-language model plugin via Ollama."""

    def __init__(self, plugin_dir: Path):
        """Initialize the Qwen3-VL plugin."""
        super().__init__(plugin_dir)
        self._config = DEFAULT_CONFIG.copy()
        self._ollama = None  # Lazy loaded
        self._available_models: list[str] | None = None

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="qwen3_vl",
            version=PLUGIN_VERSION,
            description="Qwen3 Vision-Language model for image tagging via Ollama",
            hardware_reqs={
                "gpu": False,  # Runs on Ollama server
                "min_ram_gb": 2,
            },
            provides_confidence=False,  # VLMs generate text, not scores
            recommended_threshold=0.0,
        )

    def is_available(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            import httpx
        except ImportError:
            return False

        try:
            # Check Ollama server is running
            url = self._config["server_url"].rstrip("/")
            response = httpx.get(
                f"{url}/api/tags",
                timeout=5.0,
            )
            if response.status_code != 200:
                return False

            # Check if our model is available
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            self._available_models = models

            model_name = self._config["model_name"]
            # Check exact match or prefix match (qwen3-vl matches qwen3-vl:latest)
            for m in models:
                if m == model_name or m.startswith(model_name.split(":")[0]):
                    return True

            return False

        except Exception:
            return False

    def setup(self) -> bool:
        """Pull the model from Ollama if not present."""
        try:
            self._ensure_ollama()
            model_name = self._config["model_name"]
            print(f"Pulling model {model_name} from Ollama...")
            self._ollama.pull(model_name)
            print(f"Model {model_name} ready!")
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def configure(self, **kwargs) -> None:
        """Update plugin configuration.

        Args:
            server_url: Ollama server URL
            model_name: Model to use (qwen3-vl:latest, etc.)
            timeout_seconds: Request timeout
            mode: Output mode (tagging, describe, custom)
            custom_prompt: Custom prompt for custom mode
            max_tokens: Max generation tokens
            temperature: Sampling temperature
            max_size: Max image dimension
        """
        # Connection settings - skip None values from Pydantic models
        if "server_url" in kwargs and kwargs["server_url"] is not None:
            self._config["server_url"] = str(kwargs["server_url"]).rstrip("/")

        if "model_name" in kwargs and kwargs["model_name"] is not None:
            self._config["model_name"] = str(kwargs["model_name"])

        if "timeout_seconds" in kwargs and kwargs["timeout_seconds"] is not None:
            self._config["timeout_seconds"] = int(kwargs["timeout_seconds"])

        if "keep_alive" in kwargs and kwargs["keep_alive"] is not None:
            self._config["keep_alive"] = str(kwargs["keep_alive"])

        # Generation settings
        if "mode" in kwargs and kwargs["mode"] is not None:
            mode = str(kwargs["mode"])
            if mode not in ("tagging", "describe", "custom", "both"):
                raise PluginError(f"Unknown mode: {mode}. Use: tagging, describe, custom, both")
            self._config["mode"] = mode

        if "custom_prompt" in kwargs and kwargs["custom_prompt"] is not None:
            self._config["custom_prompt"] = str(kwargs["custom_prompt"])

        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            self._config["max_tokens"] = int(kwargs["max_tokens"])

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            temp = float(kwargs["temperature"])
            if not 0.0 <= temp <= 2.0:
                raise PluginError("temperature must be between 0.0 and 2.0")
            self._config["temperature"] = temp

        if "top_p" in kwargs and kwargs["top_p"] is not None:
            self._config["top_p"] = float(kwargs["top_p"])

        # Image settings
        if "max_size" in kwargs:
            self._config["max_size"] = int(kwargs["max_size"])

        # Output settings
        if "include_description" in kwargs:
            self._config["include_description"] = bool(kwargs["include_description"])

        if "expand_compounds" in kwargs:
            self._config["expand_compounds"] = bool(kwargs["expand_compounds"])

    def tag(self, image_path: Path, quality: str = "standard") -> TagResult:
        """Tag an image using the VLM.

        Args:
            image_path: Path to image file
            quality: Quality level (quick, standard, max)

        Returns:
            TagResult with tags and metadata
        """
        if not self.is_available():
            raise ModelNotFoundError(
                f"Ollama server not available at {self._config['server_url']} "
                f"or model {self._config['model_name']} not found.\n"
                f"Available models: {self._available_models}"
            )

        self._ensure_ollama()

        mode = self._config["mode"]

        # Handle "both" mode - run tagging AND describe
        if mode == "both":
            return self._run_both_modes(image_path, quality)

        # Get quality-specific settings
        gen_settings = GENERATION_SETTINGS.get(quality, GENERATION_SETTINGS["standard"])

        # Build prompt based on mode
        if mode == "tagging":
            prompt = PROMPTS_BY_QUALITY.get(quality, TAGGING_PROMPT)
        elif mode == "describe":
            prompt = DESCRIBE_PROMPT
        elif mode == "custom":
            prompt = self._config["custom_prompt"] or TAGGING_PROMPT
        else:
            prompt = TAGGING_PROMPT

        # Preprocess and encode image
        image_data = self._prepare_image(image_path)

        # Run inference
        start_time = time.perf_counter()

        try:
            response = self._ollama.chat(
                model=self._config["model_name"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data],
                    }
                ],
                options={
                    "temperature": gen_settings.get("temperature", self._config["temperature"]),
                    "num_predict": gen_settings.get("max_tokens", self._config["max_tokens"]),
                    "top_p": self._config["top_p"],
                },
                keep_alive=self._config["keep_alive"],
            )
        except Exception as e:
            raise PluginError(f"Ollama inference failed: {e}") from e

        inference_time = (time.perf_counter() - start_time) * 1000

        # Extract response text
        response_text = response.get("message", {}).get("content", "")

        # Parse response into tags
        if mode == "describe":
            tags = self._parse_description(response_text)
            metadata = {"description": response_text, "mode": "describe"}
        else:
            tags, description = self._parse_tagging_response(response_text)
            metadata = {
                "mode": mode,
                "quality": quality,
                "raw_response": response_text[:500],  # Truncate for storage
            }
            if description and self._config["include_description"]:
                metadata["description"] = description

        return TagResult(
            tags=tags,
            model=self._config["model_name"],
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
            metadata=metadata,
        )

    def _run_both_modes(self, image_path: Path, quality: str) -> TagResult:
        """Run both tagging and describe modes, combining results."""
        gen_settings = GENERATION_SETTINGS.get(quality, GENERATION_SETTINGS["standard"])
        image_data = self._prepare_image(image_path)

        all_tags: list[Tag] = []
        seen: set[str] = set()
        total_time = 0.0

        # Run 1: Tagging mode
        tagging_prompt = PROMPTS_BY_QUALITY.get(quality, TAGGING_PROMPT)
        start_time = time.perf_counter()

        try:
            tagging_response = self._ollama.chat(
                model=self._config["model_name"],
                messages=[
                    {
                        "role": "user",
                        "content": tagging_prompt,
                        "images": [image_data],
                    }
                ],
                options={
                    "temperature": gen_settings.get("temperature", self._config["temperature"]),
                    "num_predict": gen_settings.get("max_tokens", self._config["max_tokens"]),
                    "top_p": self._config["top_p"],
                },
                keep_alive=self._config["keep_alive"],
            )
        except Exception as e:
            raise PluginError(f"Ollama tagging inference failed: {e}") from e

        total_time += (time.perf_counter() - start_time) * 1000
        tagging_text = tagging_response.get("message", {}).get("content", "")
        tags_from_tagging, tag_description = self._parse_tagging_response(tagging_text)

        for tag in tags_from_tagging:
            if tag.label not in seen:
                seen.add(tag.label)
                all_tags.append(tag)

        # Run 2: Describe mode
        start_time = time.perf_counter()

        try:
            describe_response = self._ollama.chat(
                model=self._config["model_name"],
                messages=[
                    {
                        "role": "user",
                        "content": DESCRIBE_PROMPT,
                        "images": [image_data],
                    }
                ],
                options={
                    "temperature": gen_settings.get("temperature", self._config["temperature"]),
                    "num_predict": gen_settings.get("max_tokens", self._config["max_tokens"]),
                    "top_p": self._config["top_p"],
                },
                keep_alive=self._config["keep_alive"],
            )
        except Exception as e:
            raise PluginError(f"Ollama describe inference failed: {e}") from e

        total_time += (time.perf_counter() - start_time) * 1000
        describe_text = describe_response.get("message", {}).get("content", "")

        # Build metadata with both responses
        metadata = {
            "mode": "both",
            "quality": quality,
            "tagging_response": tagging_text[:500],
            "description": describe_text,
        }
        if tag_description:
            metadata["tagging_description"] = tag_description

        return TagResult(
            tags=all_tags,
            model=self._config["model_name"],
            version=PLUGIN_VERSION,
            inference_time_ms=round(total_time, 2),
            metadata=metadata,
        )

    def get_available_models(self) -> list[str]:
        """Get list of vision models available on Ollama server."""
        if self._available_models is None:
            self.is_available()  # Populates _available_models
        return self._available_models or []

    def _ensure_ollama(self) -> None:
        """Ensure ollama client is loaded."""
        if self._ollama is None:
            try:
                import ollama

                # Configure client with custom host if needed
                server_url = self._config["server_url"]
                if server_url != "http://localhost:11434":
                    # Set environment variable for ollama client
                    import os
                    os.environ["OLLAMA_HOST"] = server_url

                self._ollama = ollama
            except ImportError as e:
                raise PluginError(
                    "ollama package not installed. Install with:\n"
                    "pip install ollama"
                ) from e

    def _prepare_image(self, image_path: Path) -> str:
        """Load, resize if needed, and encode image to base64."""
        from PIL import Image

        img = Image.open(image_path)

        # Convert to RGB if needed (handle RGBA, P modes)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize if too large
        max_size = self._config["max_size"]
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(d * ratio) for d in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Encode to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=self._config["jpeg_quality"])
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_tagging_response(self, response: str) -> tuple[list[Tag], str | None]:
        """Parse VLM response into tags.

        Handles both JSON and plain text responses.

        Returns:
            Tuple of (tags list, description or None)
        """
        tags: list[Tag] = []
        description: str | None = None
        seen: set[str] = set()

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # Extract tags from JSON
                tag_list = data.get("tags", [])
                if isinstance(tag_list, list):
                    for tag in tag_list:
                        if isinstance(tag, str):
                            self._add_tag(tag, tags, seen)

                # Extract from categories if present
                categories = data.get("categories", {})
                if isinstance(categories, dict):
                    for category_tags in categories.values():
                        if isinstance(category_tags, list):
                            for tag in category_tags:
                                if isinstance(tag, str):
                                    self._add_tag(tag, tags, seen)

                # Get description
                description = data.get("description")

            except json.JSONDecodeError:
                pass  # Fall through to text parsing

        # If no tags from JSON, parse as plain text
        if not tags:
            tags, description = self._parse_text_response(response)

        return tags, description

    def _parse_text_response(self, response: str) -> tuple[list[Tag], str | None]:
        """Parse plain text response into tags."""
        tags: list[Tag] = []
        seen: set[str] = set()

        lines = response.strip().split("\n")
        description = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this looks like a description sentence
            if len(line) > 100 and "." in line:
                description = line
                continue

            # Remove common prefixes
            line = re.sub(r"^[\-\*\•\d\.\)\]]+\s*", "", line)
            line = line.strip()

            # Skip if looks like a header or instruction
            if line.endswith(":") or line.startswith("#"):
                continue

            # Split on commas if present
            if "," in line:
                parts = line.split(",")
            else:
                parts = [line]

            for part in parts:
                part = part.strip()
                if part:
                    self._add_tag(part, tags, seen)

        return tags, description

    def _add_tag(self, tag: str, tags: list[Tag], seen: set[str]) -> None:
        """Process and add a tag if valid."""
        # Normalize
        tag = tag.lower().strip()

        # Remove quotes and extra punctuation
        tag = re.sub(r'^["\']|["\']$', "", tag)
        tag = re.sub(r"[^\w\s\-]", "", tag)
        tag = tag.strip()

        # Check minimum length
        min_len = self._config["min_tag_length"]
        if len(tag) < min_len:
            return

        # Skip if already seen
        if tag in seen:
            return

        seen.add(tag)
        tags.append(Tag(label=tag, confidence=None))

        # Expand compound tags if enabled
        if self._config["expand_compounds"] and " " in tag:
            words = tag.split()
            if len(words) <= 3:  # Only expand short compounds
                # Add underscore version
                compound = "_".join(words)
                if compound not in seen:
                    seen.add(compound)
                    tags.append(Tag(label=compound, confidence=None))

                # Add individual words
                for word in words:
                    word = word.strip()
                    if len(word) >= min_len and word not in seen:
                        seen.add(word)
                        tags.append(Tag(label=word, confidence=None))

    def _parse_description(self, response: str) -> list[Tag]:
        """Parse describe mode response into tags."""
        # For describe mode, return the full description as a single tag
        # Plus extract key terms
        tags: list[Tag] = []
        seen: set[str] = set()

        # Add full description as special tag
        description = response.strip()
        if description:
            # Truncate if very long
            if len(description) > 500:
                description = description[:500] + "..."
            tags.append(Tag(label=f"description:{description}", confidence=None))

        # Also extract individual terms from the description
        # Extract nouns and adjectives using simple heuristics
        words = re.findall(r"\b[a-zA-Z]{3,}\b", response.lower())

        # Common stop words to skip
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "her", "was", "one", "our", "out", "has", "have", "been", "were",
            "being", "there", "their", "this", "that", "with", "from", "they",
            "will", "would", "could", "should", "which", "what", "when", "where",
            "image", "shows", "appears", "visible", "seen", "looks", "seems",
        }

        for word in words:
            if word not in stop_words and word not in seen:
                seen.add(word)
                tags.append(Tag(label=word, confidence=None))

        return tags[:100]  # Limit
