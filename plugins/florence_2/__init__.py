"""Florence-2 vision foundation model plugin.

Uses Microsoft's Florence-2 model from HuggingFace for image tagging.
Supports multiple task types including object detection, captioning, and OCR.

See: https://huggingface.co/microsoft/Florence-2-large-ft
"""

import sys
import time
from pathlib import Path

# Add src to path for imports during plugin loading
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from visual_buffet.exceptions import ModelNotFoundError, PluginError
from visual_buffet.plugins.base import PluginBase
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult

PLUGIN_VERSION = "1.0.0"

# Model variants available on HuggingFace
# Note: We use multimodalart's no-flash-attn variants for Mac/MPS compatibility
MODEL_VARIANTS = {
    "base": "microsoft/Florence-2-base",
    "large": "multimodalart/Florence-2-large-no-flash-attn",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large-ft": "multimodalart/Florence-2-large-no-flash-attn",
    "florence-2-base": "microsoft/Florence-2-base",
    "florence-2-large": "multimodalart/Florence-2-large-no-flash-attn",
    "florence-2-base-ft": "microsoft/Florence-2-base-ft",
    "florence-2-large-ft": "multimodalart/Florence-2-large-no-flash-attn",
    # Original Microsoft variants (may require CUDA)
    "large-ft-original": "microsoft/Florence-2-large-ft",
    "large-original": "microsoft/Florence-2-large",
}

# Supported task prompts
TASK_PROMPTS = {
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR>",
    "<OCR_WITH_REGION>",
}

# Florence-2 does not provide native confidence scores.
# Tags are returned without confidence values.

# Default configuration
# NOTE: <MORE_DETAILED_CAPTION> returns rich descriptions (20-50 tags)
# <OD> only returns detected objects (2-5 tags typically)
DEFAULT_CONFIG = {
    "model_variant": "florence-2-large-ft",
    "task_prompt": "<MORE_DETAILED_CAPTION>",
    "max_new_tokens": 1024,
}


class Florence2Plugin(PluginBase):
    """Florence-2 vision foundation model plugin."""

    def __init__(self, plugin_dir: Path):
        """Initialize the Florence-2 plugin."""
        super().__init__(plugin_dir)
        self._model = None
        self._processor = None
        self._device = None
        self._torch_dtype = None
        self._config = DEFAULT_CONFIG.copy()

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="florence_2",
            version=PLUGIN_VERSION,
            description="Microsoft Florence-2 vision foundation model for image tagging and captioning",
            hardware_reqs={
                "gpu": False,
                "min_ram_gb": 4,
            },
            provides_confidence=False,
            recommended_threshold=0.0,
        )

    def is_available(self) -> bool:
        """Check if transformers and torch are available."""
        try:
            import torch
            import transformers

            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Download model from HuggingFace.

        Florence-2 models are cached in HuggingFace's default cache directory
        (~/.cache/huggingface/hub/), not in the plugin's models/ directory.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = self._get_model_id()
            print(f"Downloading Florence-2 model: {model_id}")
            print("Models will be cached in ~/.cache/huggingface/hub/")

            # Download processor
            print("Downloading processor...")
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

            # Download model
            print("Downloading model (this may take a while)...")
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

            print("Download complete!")
            return True

        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def configure(self, **kwargs) -> None:
        """Update plugin configuration.

        Args:
            model_variant: Model to use (base, large, base-ft, large-ft)
            task_prompt: Task prompt (<OD>, <CAPTION>, etc.)
            max_new_tokens: Max tokens to generate (256-4096)
        """
        if "model_variant" in kwargs:
            variant = kwargs["model_variant"]
            if variant not in MODEL_VARIANTS:
                raise PluginError(
                    f"Unknown model variant: {variant}. "
                    f"Available: {list(MODEL_VARIANTS.keys())}"
                )
            self._config["model_variant"] = variant
            # Reset model so it reloads with new variant
            self._model = None
            self._processor = None

        if "task_prompt" in kwargs:
            prompt = kwargs["task_prompt"]
            if prompt not in TASK_PROMPTS:
                raise PluginError(
                    f"Unknown task prompt: {prompt}. " f"Available: {TASK_PROMPTS}"
                )
            self._config["task_prompt"] = prompt

        if "max_new_tokens" in kwargs:
            tokens = int(kwargs["max_new_tokens"])
            if not 256 <= tokens <= 4096:
                raise PluginError("max_new_tokens must be between 256 and 4096")
            self._config["max_new_tokens"] = tokens

    def tag(self, image_path: Path) -> TagResult:
        """Tag an image using Florence-2."""
        if not self.is_available():
            raise ModelNotFoundError(
                "Florence-2 dependencies not found. Install with:\n"
                "pip install torch torchvision transformers>=4.40.0"
            )

        # Lazy load model
        if self._model is None:
            self._load_model()

        # Run inference
        start_time = time.perf_counter()
        tags = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        model_name = self._config["model_variant"].replace("florence-2-", "")
        return TagResult(
            tags=tags,
            model=f"florence2_{model_name}",
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
        )

    def _get_model_id(self) -> str:
        """Get HuggingFace model ID from variant name."""
        variant = self._config["model_variant"]
        return MODEL_VARIANTS.get(variant, MODEL_VARIANTS["large-ft"])

    def _load_model(self) -> None:
        """Load the Florence-2 model from HuggingFace."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            # Determine device and dtype
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._torch_dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                self._torch_dtype = torch.float32
            else:
                self._device = torch.device("cpu")
                self._torch_dtype = torch.float32

            model_id = self._get_model_id()
            print(f"Loading Florence-2 model ({model_id}) on {self._device}...")

            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self._torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager",  # Required for transformers 4.47+
            ).to(self._device)

            self._model.eval()

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install torch torchvision transformers>=4.40.0\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise PluginError(f"Failed to load model: {e}")

    def _run_inference(self, image_path: Path) -> list[Tag]:
        """Run inference on an image."""
        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            task_prompt = self._config["task_prompt"]

            # Prepare inputs
            inputs = self._processor(
                text=task_prompt, images=image, return_tensors="pt"
            ).to(self._device, self._torch_dtype)

            # Generate with greedy decoding (beam search has compatibility issues)
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self._config["max_new_tokens"],
                    num_beams=1,
                    do_sample=False,
                )

            # Decode output
            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            # Post-process based on task
            parsed = self._processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
            )

            # Convert to tags
            return self._parse_output(parsed, task_prompt)

        except Exception as e:
            raise PluginError(f"Inference failed: {e}")

    def _parse_output(self, parsed: dict, task_prompt: str) -> list[Tag]:
        """Parse Florence-2 output into Tag objects."""
        tags = []

        if task_prompt == "<OD>":
            # Object detection returns {'<OD>': {'bboxes': [...], 'labels': [...]}}
            # Labels are single words like "furniture", "person"
            key = task_prompt
            if key in parsed and "labels" in parsed[key]:
                labels = parsed[key]["labels"]
                seen = set()
                for label in labels:
                    label_lower = label.lower().strip()
                    if label_lower and label_lower not in seen:
                        seen.add(label_lower)
                        tags.append(Tag(label=label_lower))

        elif task_prompt == "<DENSE_REGION_CAPTION>":
            # Dense region returns phrases like "abandoned bar counter with stools"
            # Parse each phrase into individual words for better tagging
            key = task_prompt
            if key in parsed and "labels" in parsed[key]:
                all_text = " ".join(parsed[key]["labels"])
                tags = self._extract_tags_from_caption(all_text)

        elif task_prompt in ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"):
            # Caption returns {task: "caption text"}
            caption = parsed.get(task_prompt, "")
            if caption:
                tags = self._extract_tags_from_caption(caption)

        elif task_prompt == "<OCR>" or task_prompt == "<OCR_WITH_REGION>":
            # OCR returns text content
            key = task_prompt
            if key in parsed:
                ocr_text = parsed[key]
                if isinstance(ocr_text, str) and ocr_text.strip():
                    tags.append(Tag(label=f"text:{ocr_text.strip()}"))
                elif isinstance(ocr_text, dict) and "labels" in ocr_text:
                    for label in ocr_text["labels"]:
                        tags.append(Tag(label=f"text:{label}"))

        elif task_prompt == "<REGION_PROPOSAL>":
            # Region proposal doesn't have labels, not useful for tagging
            pass

        return tags

    def _extract_tags_from_caption(self, caption: str) -> list[Tag]:
        """Extract meaningful tags from a caption string.

        Extracts both:
        - Individual words (after removing stopwords)
        - Compound phrases like "white_house", "swimming_pool" (slugified bigrams)

        Uses a minimal stop word list to preserve descriptive adjectives
        like 'red', 'wooden', 'empty', 'large', 'abandoned', etc.

        Note: Florence-2 does not provide confidence scores. Tags are returned
        without confidence values.
        """
        import re

        # Minimal stop words: articles, prepositions, conjunctions, pronouns
        # KEEP adjectives and descriptive words
        stop_words = {
            # Articles
            "a", "an", "the",
            # Prepositions
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "up", "down", "out", "off", "over", "about",
            # Conjunctions
            "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
            # Pronouns
            "it", "its", "this", "that", "these", "those", "which", "who", "whom",
            # Be verbs
            "is", "are", "was", "were", "be", "been", "being",
            # Common verbs that don't add meaning
            "has", "have", "had", "do", "does", "did", "can", "could", "will",
            "would", "should", "may", "might", "must",
            # Image description filler
            "image", "shows", "showing", "appears", "seen", "visible", "there",
        }

        # Split into phrases on punctuation (keep word sequences together)
        phrases = re.split(r"[,.;:!?\-\(\)\[\]\"']", caption.lower())

        seen = set()
        compound_tags = []
        single_tags = []

        for phrase in phrases:
            # Extract words from this phrase
            words = re.findall(r"\b[a-zA-Z]{2,}\b", phrase)

            # Extract compound phrases (bigrams) BEFORE filtering stopwords
            # This keeps "white house" together even though we'd filter "the"
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                # Only create compound if BOTH words are meaningful (not stopwords)
                if w1 not in stop_words and w2 not in stop_words:
                    compound = f"{w1}_{w2}"
                    if compound not in seen:
                        seen.add(compound)
                        compound_tags.append(compound)

            # Also extract individual words
            for word in words:
                if word not in stop_words and word not in seen:
                    seen.add(word)
                    single_tags.append(word)

        # Build final tag list (compound phrases first, then singles)
        tags = []
        for label in compound_tags:
            tags.append(Tag(label=label))
        for label in single_tags:
            tags.append(Tag(label=label))

        # Limit to 100 tags
        return tags[:100]
