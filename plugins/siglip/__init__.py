"""SigLIP (Sigmoid Loss for Language Image Pre-Training) plugin.

Google's vision-language model that uses sigmoid loss instead of softmax,
enabling independent per-label confidence scores (ideal for multi-label tagging).

See SME: docs/sme/siglip.sme.md for full documentation.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports during plugin loading
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from visual_buffet.exceptions import ModelNotFoundError, PluginError  # noqa: E402
from visual_buffet.plugins.base import PluginBase  # noqa: E402
from visual_buffet.plugins.schemas import PluginInfo, Tag, TagResult  # noqa: E402

PLUGIN_VERSION = "1.0.0"

# Model variants with HuggingFace IDs
# Using original SigLIP v1 as default for better compatibility
MODEL_VARIANTS = {
    # Original SigLIP variants (stable, recommended)
    "base": "google/siglip-base-patch16-224",
    "so400m": "google/siglip-so400m-patch14-384",
    # SigLIP 2 variants (experimental, may have compatibility issues)
    "base-v2": "google/siglip2-base-patch16-224",
    "large-v2": "google/siglip2-large-patch16-512",
    "so400m-v2": "google/siglip2-so400m-patch14-384",
    "giant-v2": "google/siglip2-giant-opt-patch16-384",
    # NaFlex variants (variable aspect ratio)
    "base-naflex": "google/siglip2-base-patch16-naflex",
}

# Default model (best balance of speed/accuracy)
DEFAULT_MODEL = "so400m"

# CRITICAL: SigLIP was trained on lowercase text
# Prompt template that matches training distribution
PROMPT_TEMPLATE = "This is a photo of {}."


class SigLIPPlugin(PluginBase):
    """SigLIP zero-shot image tagging plugin.

    Provides actual confidence scores via sigmoid activation,
    unlike RAM++ and Florence-2 which only return tag names.
    """

    def __init__(self, plugin_dir: Path):
        """Initialize the SigLIP plugin."""
        super().__init__(plugin_dir)
        self._model = None
        self._processor = None
        self._device = None
        self._torch_dtype = None
        self._labels = None
        self._config = self._load_default_config()
        # Discovery plugins injected by engine
        self._discovery_plugins: dict[str, PluginBase] = {}

    def _load_default_config(self) -> dict:
        """Load default configuration."""
        return {
            "model_variant": DEFAULT_MODEL,
            "attention": "auto",
            "dtype": "auto",
            "quantization": "none",
            "max_num_patches": 256,
            # Discovery mode: use other plugins to discover vocabulary
            "discovery_mode": False,
            "use_ram_plus": True,      # General tagging (4500+ categories)
            "use_florence_2": True,    # Detailed captioning
            "use_yolo": False,         # Object detection (80 COCO classes)
            "use_paddle_ocr": False,   # Text detection (OCR)
            "use_easyocr": False,      # Scene text recognition (photos/signs)
        }

    def get_info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name="siglip",
            version=PLUGIN_VERSION,
            description="Google SigLIP vision-language model for zero-shot image tagging",
            hardware_reqs={
                "gpu": False,
                "min_ram_gb": 4,
            },
            provides_confidence=True,
            # CRITICAL: SigLIP uses sigmoid activation, outputs are typically 0.01-0.30
            # A threshold of 0.5 would filter out almost everything!
            recommended_threshold=0.01,
        )

    def is_available(self) -> bool:
        """Check if transformers and torch are available."""
        try:
            import importlib.util

            # Check torch is available
            if importlib.util.find_spec("torch") is None:
                return False

            # Check transformers is available and version is sufficient
            if importlib.util.find_spec("transformers") is None:
                return False

            import transformers
            from packaging import version

            if version.parse(transformers.__version__) < version.parse("4.47.0"):
                return False

            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Download model from HuggingFace.

        Models are cached in HuggingFace's default cache directory
        (~/.cache/huggingface/hub/).
        """
        try:
            from transformers import AutoModel, AutoProcessor

            model_id = self._get_model_id()
            print(f"Downloading SigLIP model: {model_id}")
            print("Models will be cached in ~/.cache/huggingface/hub/")

            # Download processor
            print("Downloading processor...")
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

            # Download model
            print("Downloading model (this may take a while)...")
            AutoModel.from_pretrained(model_id, trust_remote_code=True)

            print("Download complete!")
            return True

        except Exception as e:
            print(f"Setup failed: {e}")
            return False

    def configure(self, **kwargs) -> None:
        """Update plugin configuration.

        Args:
            model_variant: Model to use (base, large, so400m, giant)
            attention: Attention implementation (auto, sdpa, flash_attention_2, eager)
            dtype: Data type (auto, float16, bfloat16, float32)
            quantization: Quantization mode (none, 8bit, 4bit)
            max_num_patches: NaFlex patch count (64-512)
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

        if "attention" in kwargs:
            attn = kwargs["attention"]
            if attn not in ("auto", "sdpa", "flash_attention_2", "eager"):
                raise PluginError(f"Unknown attention: {attn}")
            self._config["attention"] = attn
            self._model = None

        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
            if dtype not in ("auto", "float16", "bfloat16", "float32"):
                raise PluginError(f"Unknown dtype: {dtype}")
            self._config["dtype"] = dtype
            self._model = None

        if "quantization" in kwargs:
            quant = kwargs["quantization"]
            if quant not in ("none", "8bit", "4bit"):
                raise PluginError(f"Unknown quantization: {quant}")
            self._config["quantization"] = quant
            self._model = None

        if "max_num_patches" in kwargs:
            patches = int(kwargs["max_num_patches"])
            if not 64 <= patches <= 512:
                raise PluginError("max_num_patches must be between 64 and 512")
            self._config["max_num_patches"] = patches

        # Discovery mode settings
        if "discovery_mode" in kwargs:
            self._config["discovery_mode"] = bool(kwargs["discovery_mode"])
        if "use_ram_plus" in kwargs:
            self._config["use_ram_plus"] = bool(kwargs["use_ram_plus"])
        if "use_florence_2" in kwargs:
            self._config["use_florence_2"] = bool(kwargs["use_florence_2"])
        if "use_yolo" in kwargs:
            self._config["use_yolo"] = bool(kwargs["use_yolo"])
        if "use_paddle_ocr" in kwargs:
            self._config["use_paddle_ocr"] = bool(kwargs["use_paddle_ocr"])
        if "use_easyocr" in kwargs:
            self._config["use_easyocr"] = bool(kwargs["use_easyocr"])

    def tag(self, image_path: Path) -> TagResult:
        """Tag an image using SigLIP zero-shot classification.

        If discovery_mode is enabled, first runs RAM++ and/or Florence-2 to
        discover candidate tags, then scores them with SigLIP. Returns all
        model results in the metadata.

        If discovery_mode is disabled, uses the builtin vocabulary.
        """
        if not self.is_available():
            raise ModelNotFoundError(
                "SigLIP dependencies not found. Install with:\n"
                "pip install torch torchvision transformers>=4.47.0 accelerate"
            )

        # Check if discovery mode is enabled
        if self._config["discovery_mode"]:
            return self._tag_with_discovery(image_path)

        return self._tag_standard(image_path)

    def _tag_standard(self, image_path: Path) -> TagResult:
        """Tag using builtin vocabulary (standard mode)."""
        # Lazy load model and labels
        if self._model is None:
            self._load_model()

        if self._labels is None:
            self._load_labels()

        # Run inference
        start_time = time.perf_counter()
        tags = self._run_inference(image_path)
        inference_time = (time.perf_counter() - start_time) * 1000

        model_name = f"siglip_{self._config['model_variant']}"
        return TagResult(
            tags=tags,
            model=model_name,
            version=PLUGIN_VERSION,
            inference_time_ms=round(inference_time, 2),
        )

    def _tag_with_discovery(self, image_path: Path) -> TagResult:
        """Tag using discovered vocabulary from RAM++ and/or Florence-2.

        Runs discovery plugins first to get candidate tags, then uses SigLIP
        to score each candidate with real sigmoid confidence values.

        Returns:
            TagResult with scored tags and discovery metadata containing
            full results from all models that ran.
        """
        # Determine which discovery sources are available and enabled
        available_sources = []
        for source_name, config_key in [
            ("ram_plus", "use_ram_plus"),
            ("florence_2", "use_florence_2"),
            ("yolo", "use_yolo"),
            ("paddle_ocr", "use_paddle_ocr"),
            ("easyocr", "use_easyocr"),
        ]:
            if self._config.get(config_key) and source_name in self._discovery_plugins:
                plugin = self._discovery_plugins[source_name]
                if plugin.is_available():
                    available_sources.append(source_name)

        if not available_sources:
            raise PluginError(
                "Discovery mode enabled but no discovery plugins available. "
                "Either enable ram_plus or florence_2 in settings, ensure they "
                "are installed, or disable discovery mode."
            )

        # Collect candidates from discovery plugins
        all_candidates: set[str] = set()
        discovery_results: dict[str, dict] = {}
        total_discovery_time = 0.0

        for name in available_sources:
            plugin = self._discovery_plugins[name]
            result = plugin.tag(image_path)

            # Collect unique lowercase tags
            for tag in result.tags:
                all_candidates.add(tag.label.lower().strip())

            # Store full results for display
            discovery_results[name] = {
                "tags": [t.to_dict() for t in result.tags],
                "model": plugin.get_info().name,
                "inference_time_ms": result.inference_time_ms or 0,
                "total_tags": len(result.tags),
            }
            total_discovery_time += result.inference_time_ms or 0

        # Set vocabulary to discovered tags
        candidate_list = sorted(all_candidates)
        self.set_vocabulary(candidate_list)

        # Lazy load SigLIP model
        if self._model is None:
            self._load_model()

        # Run SigLIP inference to score candidates
        start_time = time.perf_counter()
        tags = self._run_inference(image_path)
        siglip_inference_time = (time.perf_counter() - start_time) * 1000

        # Reset vocabulary for next call
        self.reset_vocabulary()

        model_name = f"siglip_{self._config['model_variant']}"
        return TagResult(
            tags=tags,
            model=model_name,
            version=PLUGIN_VERSION,
            inference_time_ms=round(siglip_inference_time, 2),
            metadata={
                "discovery_mode": True,
                "discovery_sources": available_sources,
                "vocabulary_size": len(candidate_list),
                "discovery_results": discovery_results,
                "total_discovery_time_ms": round(total_discovery_time, 2),
            },
        )

    def _get_model_id(self) -> str:
        """Get HuggingFace model ID from variant name."""
        variant = self._config["model_variant"]
        return MODEL_VARIANTS.get(variant, MODEL_VARIANTS[DEFAULT_MODEL])

    def _determine_device_and_dtype(self):
        """Determine optimal device and dtype for this system."""
        import torch

        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Determine dtype based on config and device
        dtype_config = self._config["dtype"]

        if dtype_config == "auto":
            if self._device.type == "cuda":
                # Check for bfloat16 support (Ampere+)
                if torch.cuda.get_device_capability()[0] >= 8:
                    self._torch_dtype = torch.bfloat16
                else:
                    self._torch_dtype = torch.float16
            elif self._device.type == "mps":
                # MPS has limited bfloat16 support - use float32 for compatibility
                self._torch_dtype = torch.float32
            else:
                self._torch_dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self._torch_dtype = dtype_map[dtype_config]

    def _get_attention_implementation(self) -> str | None:
        """Get attention implementation based on config and availability."""
        attn_config = self._config["attention"]

        if attn_config == "auto":
            # Check for SDPA (PyTorch 2.0+)
            try:
                import torch
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    return "sdpa"
            except Exception:
                pass
            return None  # Use default

        if attn_config == "flash_attention_2":
            # Check if flash-attn is installed
            import importlib.util

            if importlib.util.find_spec("flash_attn") is not None:
                return "flash_attention_2"
            else:
                print("Warning: flash_attention_2 requested but not installed. Using SDPA.")
                return "sdpa"

        if attn_config == "eager":
            return "eager"

        return "sdpa" if attn_config == "sdpa" else None

    def _load_model(self) -> None:
        """Load the SigLIP model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoProcessor

            self._determine_device_and_dtype()

            model_id = self._get_model_id()
            attn_impl = self._get_attention_implementation()

            print(f"Loading SigLIP model ({model_id}) on {self._device}...")

            # Build model kwargs
            model_kwargs = {
                "torch_dtype": self._torch_dtype,
                "trust_remote_code": True,
            }

            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl

            # Handle quantization
            quant = self._config["quantization"]
            if quant != "none" and self._device.type == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    if quant == "4bit":
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=self._torch_dtype,
                        )
                    elif quant == "8bit":
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                except ImportError:
                    print("Warning: bitsandbytes not installed. Skipping quantization.")

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )

            # Load model with device_map for proper placement
            if quant == "none":
                model_kwargs["device_map"] = self._device
            else:
                model_kwargs["device_map"] = "auto"

            self._model = AutoModel.from_pretrained(model_id, **model_kwargs)
            self._model.eval()

        except ImportError as e:
            raise PluginError(
                f"Missing dependencies. Install with:\n"
                f"pip install torch torchvision transformers>=4.47.0 accelerate\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise PluginError(f"Failed to load model: {e}") from e

    def _load_labels(self) -> None:
        """Load the label vocabulary for zero-shot classification."""
        # Use comprehensive label set for general image tagging
        # These are common tags that work well with SigLIP
        self._labels = self._get_tag_vocabulary()

    def set_vocabulary(self, labels: list[str]) -> None:
        """Set a custom vocabulary for zero-shot classification.

        This allows using SigLIP to score tags from other models (RAM++, Florence-2)
        with real confidence values.

        Args:
            labels: List of tag labels to score against
        """
        # Normalize to lowercase (SigLIP was trained on lowercase)
        self._labels = [label.lower().strip() for label in labels if label.strip()]

    def get_vocabulary(self) -> list[str] | None:
        """Get the current vocabulary.

        Returns:
            Current label list, or None if not loaded
        """
        return self._labels

    def reset_vocabulary(self) -> None:
        """Reset vocabulary to default labels.

        Call this before standard tagging to ensure the default vocabulary
        is used instead of a previously set custom vocabulary from pipeline mode.
        """
        self._labels = None

    def set_discovery_plugins(self, plugins: dict[str, PluginBase]) -> None:
        """Set references to plugins used for vocabulary discovery.

        Called by TaggingEngine after loading all plugins. When discovery_mode
        is enabled, SigLIP will use these plugins to discover candidate tags
        before scoring them.

        Args:
            plugins: Dict mapping plugin name to plugin instance
                     (e.g., {"ram_plus": <plugin>, "florence_2": <plugin>})
        """
        self._discovery_plugins = plugins

    def get_discovery_plugins(self) -> dict[str, PluginBase]:
        """Get the currently configured discovery plugins.

        Returns:
            Dict of discovery plugin references
        """
        return self._discovery_plugins

    def _get_tag_vocabulary(self) -> list[str]:
        """Get comprehensive tag vocabulary for zero-shot classification.

        Returns a curated list of common visual concepts that work well
        with SigLIP's zero-shot classification.
        """
        # Categories: objects, scenes, actions, attributes, concepts
        labels = [
            # Animals
            "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig",
            "chicken", "duck", "elephant", "lion", "tiger", "bear", "wolf",
            "fox", "deer", "rabbit", "squirrel", "mouse", "butterfly",
            "bee", "spider", "snake", "frog", "turtle", "owl", "eagle",
            "penguin", "dolphin", "whale", "shark", "octopus", "crab",

            # People and body
            "person", "man", "woman", "child", "baby", "face", "hand",
            "eye", "smile", "portrait", "crowd", "group", "family",

            # Vehicles
            "car", "truck", "bus", "motorcycle", "bicycle", "airplane",
            "helicopter", "boat", "ship", "train", "subway",

            # Buildings and places
            "house", "building", "skyscraper", "bridge", "tower", "church",
            "castle", "palace", "temple", "mosque", "school", "hospital",
            "restaurant", "cafe", "bar", "hotel", "office", "factory",
            "warehouse", "barn", "cabin", "tent",

            # Nature and landscapes
            "mountain", "hill", "valley", "cliff", "cave", "volcano",
            "forest", "jungle", "desert", "beach", "ocean", "sea", "lake",
            "river", "waterfall", "pond", "stream", "island",
            "field", "meadow", "garden", "park", "farm",
            "sky", "cloud", "sun", "moon", "star", "rainbow", "storm",
            "rain", "snow", "fog", "mist",

            # Plants
            "tree", "flower", "grass", "plant", "bush", "leaf", "branch",
            "rose", "tulip", "sunflower", "palm tree", "cactus", "mushroom",

            # Food and drink
            "food", "fruit", "vegetable", "meat", "bread", "cake", "pizza",
            "burger", "sandwich", "salad", "soup", "pasta", "rice", "sushi",
            "apple", "orange", "banana", "grape", "strawberry", "watermelon",
            "coffee", "tea", "wine", "beer", "water", "juice",

            # Objects
            "table", "chair", "sofa", "bed", "desk", "lamp", "mirror",
            "clock", "phone", "computer", "laptop", "keyboard", "monitor",
            "television", "camera", "book", "newspaper", "magazine",
            "bag", "backpack", "suitcase", "umbrella", "glasses", "hat",
            "shoe", "dress", "shirt", "pants", "jacket", "coat",
            "ball", "toy", "doll", "game", "puzzle",
            "guitar", "piano", "drum", "violin", "microphone",
            "knife", "fork", "spoon", "plate", "bowl", "cup", "bottle",
            "pen", "pencil", "scissors", "key", "lock", "door", "window",

            # Actions and activities
            "walking", "running", "jumping", "sitting", "standing", "lying",
            "eating", "drinking", "cooking", "sleeping", "reading", "writing",
            "talking", "laughing", "crying", "singing", "dancing", "playing",
            "working", "studying", "driving", "flying", "swimming", "climbing",
            "skiing", "surfing", "cycling", "hiking", "camping", "fishing",

            # Attributes
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "gray", "gold", "silver",
            "big", "small", "tall", "short", "long", "wide", "narrow",
            "old", "new", "young", "modern", "vintage", "antique",
            "beautiful", "ugly", "clean", "dirty", "bright", "dark",
            "hot", "cold", "warm", "cool", "wet", "dry",

            # Scenes and contexts
            "indoor", "outdoor", "urban", "rural", "suburban",
            "daytime", "nighttime", "sunrise", "sunset", "dusk", "dawn",
            "spring", "summer", "autumn", "winter",
            "wedding", "party", "celebration", "festival", "concert",
            "sport", "game", "competition", "race",
            "travel", "vacation", "adventure", "exploration",

            # Art and style
            "art", "painting", "drawing", "sculpture", "photography",
            "abstract", "realistic", "minimalist", "colorful", "monochrome",
            "portrait", "landscape", "still life", "street photography",

            # Emotions and moods
            "happy", "sad", "angry", "peaceful", "dramatic", "romantic",
            "mysterious", "nostalgic", "serene", "energetic", "calm",

            # Technical photography terms
            "close-up", "macro", "wide angle", "panorama", "aerial",
            "bokeh", "silhouette", "reflection", "shadow", "texture",
            "pattern", "symmetry", "contrast", "depth of field",
        ]

        return labels

    def _run_inference(self, image_path: Path) -> list[Tag]:
        """Run zero-shot classification inference on an image."""
        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # CRITICAL: SigLIP was trained on lowercase text
            labels_lower = [label.lower() for label in self._labels]

            # Create prompts using the training template
            texts = [PROMPT_TEMPLATE.format(label) for label in labels_lower]

            # Process inputs
            # CRITICAL: padding="max_length" is REQUIRED for SigLIP
            inputs = self._processor(
                text=texts,
                images=image,
                padding="max_length",
                max_length=64,  # SigLIP 2 requirement
                return_tensors="pt",
            )

            # Move to device and convert pixel_values to model dtype
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device)
                # Convert pixel_values to model dtype (bfloat16/float16/float32)
                if hasattr(inputs, "pixel_values") and self._torch_dtype is not None:
                    inputs.pixel_values = inputs.pixel_values.to(self._torch_dtype)
            else:
                inputs = {k: v.to(self._device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
                # Convert pixel_values to model dtype
                if "pixel_values" in inputs and self._torch_dtype is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get logits - SigLIP outputs logits_per_image
            if hasattr(outputs, "logits_per_image"):
                logits = outputs.logits_per_image
            else:
                # Fallback for different model output formats
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # CRITICAL: Use sigmoid, NOT softmax!
            # SigLIP's sigmoid outputs independent probabilities per label
            probs = torch.sigmoid(logits).squeeze(0)

            # Convert to list for processing (float32 required for numpy)
            probs_list = probs.cpu().float().numpy().tolist()

            # Create Tag objects with confidence scores
            tags = []
            for label, prob in zip(labels_lower, probs_list, strict=True):
                # SigLIP provides real confidence scores
                tags.append(Tag(label=label, confidence=float(prob)))

            # Sort by confidence (highest first)
            tags.sort(key=lambda t: t.confidence or 0, reverse=True)

            return tags

        except Exception as e:
            raise PluginError(f"Inference failed: {e}") from e
